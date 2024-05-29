import random

from UltraEval.tasks.postprocess import GSM8KPost


def transform(data, num_sample: int, r: random.Random, dataset_name: str):
    text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n ### Instruction:\n{data['question']}\n\n### Response: Let's think step by step."
    # text = f"[INST] <<SYS>>\nBelow is an instruction that describes a task. Write a response that appropriately completes the request.\n<</SYS>>\n\n{data['question']} [/INST]"
    # text = f"Question: {data['question']}\nAnswer: "
    # text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{data['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nLet's think step by step."
    # text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{data['question']}. Let's think step by step. <|eot_id|> "
    correct_answer = data["answer"]
    gsm8kp = GSM8KPost()
    _, processed_correct_answer = gsm8kp([], correct_answer)
    return {
        "input": text,
        "output": correct_answer,
        "processed_output": processed_correct_answer,
    }
