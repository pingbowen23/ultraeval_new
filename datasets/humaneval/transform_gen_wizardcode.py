import random


def transform(data, num_sample: int, r: random.Random, dataset_name: str):
    prompt = data['prompt'].strip().replace("    ", "\t")
#     temp_input = f"""[INST]Create a Python script for this problem:
# {prompt}

# [/INST]"""

#     temp_input = f"""[INST]为这个问题创建一个Python脚本:
# {prompt}

# [/INST]"""

    temp_input = f"""[INST]Создайте Python скрипт для этой задачи.:
{prompt}

[/INST]"""


    return {"input": temp_input, "output": "", "processed_output": ""}