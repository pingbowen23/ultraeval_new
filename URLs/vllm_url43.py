import argparse
import re
"""
reference:https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
"""

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str,default="/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/continue_learning/code_0123_epoch1",  help="Model name on hugginface")
parser.add_argument("--gpuid", type=str, default="0", help="GPUid to be deployed")
parser.add_argument("--port", type=int, default=5031, help="the port")
parser.add_argument("--weight", type=float, default=0, help="linear")
parser.add_argument("--gate_path", type=str,help="linear")
parser.add_argument("--use_vllm", action="store_true",help="linear")
parser.add_argument("--temperature", type=float, default=0, help="linear")
parser.add_argument("--language_model", type=str, default=0, help="linear")
parser.add_argument("--task_model", type=str, default=0, help="linear")
args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
import torch
from flask import Flask, jsonify, request
# from optimum.bettertransformer import BetterTransformer
from vllm import LLM, SamplingParams
from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token

use_vllm=args.use_vllm
# use_vllm=False

# llm = LLM(
#     model=args.model_name,
#     trust_remote_code=True,
#     tensor_parallel_size=len(args.gpuid.split(",")),
# )

print("model load finished")

app = Flask(__name__)

# 模型的模型参数
params_dict = {
    "n": 1,
    "best_of": None,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1,
    "use_beam_search": False,
    "length_penalty": 1.0,
    "early_stopping": False,
    "stop": None,
    "stop_token_ids": None,
    "ignore_eos": False,
    "max_tokens": 16,
    "logprobs": None,
    "prompt_logprobs": None,
    "skip_special_tokens": True,
}

def Generate(prompts,model):
    inputs = tokenizer(
        prompts,
        max_length=2048,
        return_tensors="pt",
        padding=True,
    ).to(device)
    
    outputs = model.generate(
        input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"], 
        max_new_tokens=512,
        do_sample=True,
        temperature=0.0001,
        top_p=1, 
    )
        
    outputs = tokenizer.batch_decode(
        outputs.to("cpu"), skip_special_tokens=True
    )
    
    for i in range(len(outputs)):
        if outputs[i].startswith(prompts[i]):
            outputs[i] = outputs[i][len(prompts[i]):]
    
    return outputs
def load_base_model(args):
    # pretrained_model = AutoModelForCausalLM.from_pretrained(
    # "/data/public/wangshuo/exp/ft-en-magicoder-llama-2-7b/ckpts/checkpoints/epoch_2_hf"
    #     ).to(device)
    pretrained_model = AutoModelForCausalLM.from_pretrained(
    args.model_name
        ).to(device)
    
    # pretrained_model = AutoModelForCausalLM.from_pretrained(
    # "/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/code_0120"
    #     ).to(device)
    return pretrained_model
def load_peft_model():
    pretrained_model = AutoModelForCausalLM.from_pretrained(
    "/home/wanghanqing/projects/models/Llama-2-7b-hf"
        )
    lora_model_name_or_path = "/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/ru/lora"

    #### 初始化PeftModel, 并且load第一个adapter
    lora_model = PeftModel.from_pretrained(pretrained_model, model_id = lora_model_name_or_path, adapter_name = "zh")
    lora_model.load_adapter(model_id = "//home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/trim_lora/dim256/code/lora",adapter_name = "math")
    
    lora_model.base_model.add_weighted_adapter(adapters = ['zh','math'],weights = [0.1,0.9],adapter_name = "zh-math",combination_type='linear')
    
    lora_model.base_model.set_adapter(["zh-math"])
    merged_model = lora_model.base_model.merge_and_unload(adapter_names = ["zh-math"])
    
    merged_model.to(device)
    # merged_model.save_pretrained("/data/public")
    return merged_model

def load_model():
    def find_numbers_in_string(text):
    # 使用正则表达式找到所有数字
        numbers = re.findall(r'\d+', text)
        return int(numbers[0])

    def load_fusion_gate(model=None,gate_path=args.gate_path):
        gate_weight_dict = torch.load(gate_path)
        gate_dict = {k:v for k,v in gate_weight_dict.items() if "lora_fusion_gate" in k}
        weight_bias_dict = {k:v for k,v in gate_weight_dict.items() if "weight_bias" in k}
        
        gate_weight_list , weight_bias_list = [] ,[]
        for i in range(len(gate_dict.keys())):
            layer_gate_key = f"encoder.layers.{i}.lora_fusion_gate.weight"
            weight_bias_key = f"encoder.layers.{i}.weight_bias"
            
            weight_bias_list.append(weight_bias_dict[weight_bias_key])
            gate_weight_list.append(gate_dict[layer_gate_key])
        
        for n,p in model.named_parameters():
            if "lora_fusion_gate" in n or "weight_bias" in n:
                layer = find_numbers_in_string(n)
                if "weight_bias" in n: 
                    p.data = weight_bias_list[layer]
                    
                    if "0" in n:
                        print(f"weight_bias_list[0]: {weight_bias_list[0]}")
                else:
                    p.data = gate_weight_list[layer]
    

    #### 载入pretrained_model
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        args.model_name
            ).to(device)
    
    for i in range(32):
        pretrained_model.model.layers[i].temperature = args.temperature
    
    lora_model_name_or_path = args.language_model

    #### 初始化PeftModel, 并且load第一个adapter
    lora_model = PeftModel.from_pretrained(pretrained_model, model_id = lora_model_name_or_path, adapter_name = "zh")
    lora_model = lora_model.to(device)

    #### 读取另外两个adapter
    lora_model.load_adapter(model_id = args.task_model,adapter_name = "math")

    lora_model.base_model.set_adapter(["zh","math"])
    lora_model.base_model.model.model.to(device)
    
    # import pdb
    # pdb.set_trace()
    
    # load_fusion_gate(lora_model) 
    # load_fusion_gate(lora_model,"/home/wanghanqing/projects/exp/lora_fusion_exp/best/new_zh_prompt/new_zh_prompt.pt")
    load_fusion_gate(lora_model,args.gate_path)

    
    lora_model.to(device).to(pretrained_model.dtype)
    return lora_model


def load_code_model():
    pretrained_model = AutoModelForCausalLM.from_pretrained(
    "/home/wanghanqing/projects/models/Llama-2-7b-hf"
        )
    
    #### 初始化PeftModel, 并且load第一个adapter
    # lora_model = PeftModel.from_pretrained(pretrained_model, model_id = lora_model_name_or_path, adapter_name = "fr")
    lora_model = PeftModel.from_pretrained(pretrained_model,model_id = "/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/trim_lora/dim256/code/lora",adapter_name = "math")
    
    lora_model.base_model.add_weighted_adapter(adapters = ['math'],weights = [0.9],adapter_name = "math2",combination_type='linear')
    
    lora_model.base_model.set_adapter(["math2"])
    merged_model = lora_model.base_model.merge_and_unload(adapter_names = ["math2"])
    
    merged_model.to(device)
    return merged_model


def load_zh_model(pretrained_model): # check 一下class
    # print(f"class2 {type(pretrained_model)}")
    lora_model_name_or_path = "/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/ru/lora"

    #### 初始化PeftModel, 并且load第一个adapter
    lora_model = PeftModel.from_pretrained(pretrained_model, model_id = lora_model_name_or_path, adapter_name = "zh")
    # lora_model.load_adapter(model_id = "/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/zh/lora",adapter_name = "zh")
    
    lora_model.base_model.add_weighted_adapter(adapters = ['zh'],weights = [0.1],adapter_name = "zh2",combination_type='linear')
    
    lora_model.base_model.set_adapter(["zh2"])
    merged_model = lora_model.base_model.merge_and_unload(adapter_names = ["zh2"])
    
    merged_model.to(device)
    
    # merged_model.save_pretrained("/data/public/multilingual/whq/model/saved_model/zh1_code9/")
    
    return merged_model


device = "cuda" if torch.cuda.is_available() else "cpu"


if use_vllm:
    llm = LLM(
    model=args.model_name,
    trust_remote_code=True,
    tensor_parallel_size=len(args.gpuid.split(",")),
)
else:
    # model = load_base_model(args=args)
    model = load_model()
    # model = load_code_model()
    # model = load_zh_model(model)

# model.load_state_dict(torch.load("/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/trim_lora/code/other_param/other.pt"),strict=False)
# if use_vllm:
#     llm.llm_engine.workers[0].model = model

@app.route("/vllm-url43-infer", methods=["POST"])
def main():
    datas = request.get_json()
    params = datas["params"]
    prompts = datas["instances"]
    
    for key, value in params.items():
        if key in params_dict:
            params_dict[key] = value
            
    # outputs = llm.generate(prompts, SamplingParams(**params_dict))
    # model = load_peft_model()
    if use_vllm:
        outputs = llm.generate(prompts, SamplingParams(**params_dict))
    else:
        outputs = Generate(prompts=prompts,model=model)

    res = []
    if "prompt_logprobs" in params and params["prompt_logprobs"] is not None:
        for output in outputs:
            prompt_logprobs = output.prompt_logprobs
            logp_list = [list(d.values())[0] for d in prompt_logprobs[1:]]
            res.append(logp_list)
        return jsonify(res)
    else:
        for output in outputs:
            if use_vllm:
                generated_text = output.outputs[0].text
            else:
                generated_text = output
            res.append(generated_text)

        return jsonify(res)


if __name__ == "__main__":
    app.run(port=args.port, debug=False)