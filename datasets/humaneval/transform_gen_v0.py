import random


def transform(data, num_sample: int, r: random.Random, dataset_name: str):
    prompt = data["prompt"].strip()
    temp_input = "[INST] Create a python script for this problem: " + prompt + " [/INST]" # Mistral format
    # temp_input = f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>" 
    # temp_input = "Create a python script for this problem: " + prompt 
#     temp_input = f"""You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

# @@ Instruction
# {prompt}

# @@ Response
# """
    # import pdb; pdb.set_trace()
    return {"input": temp_input, "output": "", "processed_output": ""}
