import random


def rand(n: int, r: random.Random):
    return int(r.random() * n)


def transform(data, num_sample: int, r: random.Random, dataset_name: str):
    # ru, Создайте Python скрипт для этой задачи
    # es, Crear un script de Python para este problema 
    # zh, 为这个问题创建一个Python脚本
    # description = "[INST]为这个问题创建一个Python脚本: " + data["text"] # + "[/INST]"
    tests = "\n".join(data["test_list"])
#     description = f"""You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

# @@ Instruction
# {data["text"]}

# @@ Response
# """
    description = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data["text"]}

### Response:
"""
    return {
        "input": f'"""{description}\n{tests}"""',
        "output": data["code"],
        "processed_output": data["code"],
    }
