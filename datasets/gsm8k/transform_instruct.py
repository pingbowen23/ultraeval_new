import random

from UltraEval.tasks.postprocess import GSM8KPost

def transform(data, num_sample: int, r: random.Random, dataset_name: str):
    text = f"<s>[INST] Question: {data['question']}. Answer: [/INST]"
    # text = f"[INST] <<SYS>>\nBelow is an instruction that describes a task. Write a response that appropriately completes the request.\n<</SYS>>\n\n{data['question']} [/INST]"
    #wizardmath template
    # text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n ### Instruction:\n{data['question']}\n\n### Response: Let's think step by step."
    # text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n ### Instruction:\n{data['question']}\n\n### Response: Let's think step by step."
    # text = f"<|system|>\nBelow is an instruction that describes a task. Write a response that appropriately completes the request.</s>\n<|user|>\n{data['question']}</s>\n<|assistant|>"
    # text = f"[INST] <<SYS>>\n你是中文人工智能助手，你的职责是帮助用户解决他们的问题。回答问题时，为了让中文用户更好地理解你的回答，你只能使用中文作答。答案尽量具有较高帮助性、可读性，并且不要输出有害内容。请注意一定不要使用英文作答。\n<</SYS>>\n\n{data['question']} [/INST]"
    text = f"<s>Math Correct User: {data['question']}<|end_of_turn|>"
    correct_answer = data["answer"]
    gsm8kp = GSM8KPost()
    _, processed_correct_answer = gsm8kp([], correct_answer)
    return {
        "input": text,
        "output": correct_answer,
        "processed_output": processed_correct_answer,
    }