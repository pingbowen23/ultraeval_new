import argparse
import datetime
import json
import os
import random
import sys
import re
import numpy as np
# from langdetect import detect
sys.path.append("..")
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import torch
from models import get_model
from tasks import eval_task
from transformers import AutoTokenizer, AutoModelForCausalLM

class Evaluator:
    def __init__(
        self,
        args,
        seed=1234,
    ):
        self.args = args

        model = args.model
        model_args = args.model_args
        self.set_seed(seed)

        if isinstance(model, str):
            if not model_args:
                exit("model args is required")

            model = get_model(model)(model_args) 
            # model =  AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model, device_map="cuda" if torch.cuda.is_available() else "cpu")
        else:
            print("Incorrect model format")
            exit()

        self.model = model
        data_config = self.process_config(args.config_path)

        self.build_tasks(data_config)

    def process_config(self, config_path: str):
        if config_path.endswith(".json"):
            with open(config_path, "r", encoding="utf-8") as f:
                data_config = json.load(f)
        else:
            exit("config file must be json format")
        return data_config

    def build_tasks(self, configs):
        tasks_map = {v["task_name"]: v for v in configs}

        selected_task_objects = []
        for name in tasks_map:
            task_cfg = tasks_map[name]

            if not os.path.exists(task_cfg["path"]):
                print(f'{task_cfg["path"]} not exist!')
                exit()

            if len(task_cfg["metric"]) == 0:
                raise ValueError(
                    "No metric selected for task `{}`".format(task_cfg["task_name"])
                )

            selected_task_objects.append(
                eval_task.EvalTask(
                    task_name=task_cfg["task_name"],
                    task_path=task_cfg["path"],
                    description=task_cfg.get("description", ""),
                    transform_script_path=task_cfg["transform"],
                    num_fewshot=self.args.num_fewshot
                    if self.args.num_fewshot is not None
                    else task_cfg["fewshot"],
                    metric_config=task_cfg["metric"],
                    sample_config=task_cfg.get("generate"),
                    model_postprocess=self.args.postprocess,
                    task_postprocess=task_cfg["postprocess"],
                    log_dir=self.args.output_base_path,
                    params=self.args.params,
                    limit=self.args.limit,
                    batch_size=self.args.batch_size,
                )
            )

        self.tasks = selected_task_objects

    def set_seed(self, seed=1234):
        random.seed(seed)
        np.random.seed(seed)

    def run(
        self,
    ):
        for task in self.tasks:
            task.run(self.model)

    def write_out(self):
        def dump_task(task, base_path):
            for ins in task.dataset[: task.limit]:
                ins.dump(os.path.join(base_path, task.task_name))

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(dump_task, task, self.args.output_base_path)
                for task in self.tasks
            ]

            for future in futures:
                future.result()

    def make_table(
        self,
    ):
        from pytablewriter import MarkdownTableWriter

        md_writer = MarkdownTableWriter()
        md_writer.headers = ["Task", "Metric", "Value"]

        values = []
        dataset_result = {}
        for task in self.tasks:
            dataset_name = task.task_name.split("_")[0]
            if dataset_name not in dataset_result:
                dataset_result[dataset_name] = dict()
            dataset_result[dataset_name][task.task_name] = task.final_metrics

            for k, v in task.final_metrics.items():
                values.append([task.task_name, k, "%.4f" % v])

        md_writer.value_matrix = values

        print(md_writer.dumps())

        for dataset, tasks in dataset_result.items():
            sums = defaultdict(float)
            counts = defaultdict(int)
            for key, value in tasks.items():
                for k, v in value.items():
                    sums[k] += v
                    counts[k] += 1
            dataset_result[dataset]["mean_result"] = {
                key: sums[key] / counts[key] for key in sums
            }

        with open(
            os.path.join(self.args.output_base_path, "_all_results.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(dataset_result, f, indent=4, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--output_base_path", type=str, default="logs")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_fewshot", type=int)
    parser.add_argument("--postprocess", type=str, default="")
    parser.add_argument("--params", type=str, default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--language", type=str,default="fr")
    return parser.parse_args()

def contains_basic_chinese(text):
    basic_chinese_pattern = "[\u4e00-\u9fff]"
    return re.search(basic_chinese_pattern, text) is not None


def main():
    starting = time.time()
    args = parse_args()

    now = datetime.datetime.now()
    dir_name = now.strftime("%Y-%m-%d_%H-%M-%S")
    args.output_base_path = os.path.join(args.output_base_path, dir_name)

    evaluator = Evaluator(args)
    evaluator.run()
    evaluator.make_table()
    running = time.time()
    print(f"Running time: {running - starting} seconds")

    if args.write_out:
        evaluator.write_out()

    ending = time.time()
    
    # cnt_total , cnt_ac = 0, 0
    # with open(f"{args.output_base_path}/gsm8k_gsm8k_gen/instance.jsonl","r",encoding="utf-8") as lines:
    #     for line in lines:
    #         line = json.loads(line)
    #         text , ans = line["raw_outputs"][0] , line["eval_results"]["accuracy"]
            
    #         if detect(text) == "en":
    #             print("-" * 20)
    #             print(f"{text}")
    #             print("*" * 20)
    #         # if contains_basic_chinese(text) is False:
    #         #     print("-" * 20)
    #         #     print(f"{text}")
    #         #     print("*" * 20)
    #             cnt_total += 1
    # print(f"英文 {cnt_total} ")
    print(
        f"Running time: {running - starting} seconds, the whole time: {ending - starting} seconds"
    )


if __name__ == "__main__":
    main()
