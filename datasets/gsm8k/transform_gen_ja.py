import random

from UltraEval.tasks.postprocess import GSM8KPost


def transform(data, num_sample: int, r: random.Random, dataset_name: str):
    text = f"[INST] <<SYS>>\nあなたは日本語の人工知能アシスタントで、ユーザーの問題を解決するのを助けることがあなたの責務です。質問に答える際には、日本語のユーザーがより理解しやすいように、日本語のみで回答してください。答えはできるだけ有用で、読みやすく、有害な内容を含まないようにしてください。英語での回答は絶対に避けてください。\n<</SYS>>\n\n{data['question']} [/INST]"
    correct_answer = data["answer"]
    gsm8kp = GSM8KPost()
    _, processed_correct_answer = gsm8kp([], correct_answer)
    return {
        "input": text,
        "output": correct_answer,
        "processed_output": processed_correct_answer,
    }
