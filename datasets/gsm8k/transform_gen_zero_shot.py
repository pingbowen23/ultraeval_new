import random

from UltraEval.tasks.postprocess import GSM8KPost

def transform(data, num_sample: int, r: random.Random, dataset_name: str):
    # text = f"[INST] <<SYS>>\nBelow is an instruction that describes a task. Write a response that appropriately completes the request.\n<</SYS>>\n\n{data['question']} [/INST]"
    # text = f"[INST] <<SYS>>\n你是中文人工智能助手，你的职责是帮助用户解决他们的问题。回答问题时，为了让中文用户更好地理解你的回答，你只能使用中文作答。答案尽量具有较高帮助性、可读性，并且不要输出有害内容。请注意一定不要使用英文作答。\n<</SYS>>\n\n{data['question']} [/INST]"
    
    text = f"[INST] <<SYS>>\nVous êtes un assistant d'intelligence artificielle en français, votre responsabilité est d'aider les utilisateurs à résoudre leurs problèmes. Lorsque vous répondez aux questions, afin de permettre aux utilisateurs francophones de mieux comprendre vos réponses, vous ne pouvez répondre qu'en français. Les réponses doivent être aussi utiles et lisibles que possible, et ne doivent pas contenir de contenu nuisible. Veuillez noter qu'il ne faut absolument pas répondre en anglais.\n<</SYS>>\n\n{data['question']} [/INST]"
    correct_answer = data["answer"]
    gsm8kp = GSM8KPost()
    _, processed_correct_answer = gsm8kp([], correct_answer)
    return {
        "input": text,
        "output": correct_answer,
        "processed_output": processed_correct_answer,
    }

#