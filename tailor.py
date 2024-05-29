import argparse
import jsonlines
import sys
import shutil
import logging
import os
import time
from tqdm import tqdm
import glob
import json
import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
# from vllm import LLM, SamplingParams
import re
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_model_name', type=str, required=True, help='pretrained model name')
parser.add_argument('--finetuned_model_name', type=str, required=True, help='finetuned model name')
parser.add_argument('--save_dir', type=str, required=True, help='finetuned model name')
parser.add_argument('--dim', type=int, required=True, help='finetuned model name')
parser.add_argument('--scale_factor', type=float, required=True, default=1.45, help='finetuned model name')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

pretrained_model_name = args.pretrained_model_name 

finetuned_model_name = args.finetuned_model_name 
pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=pretrained_model_name,torch_dtype=torch.bfloat16).to(device)
pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name)

finetuned_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=finetuned_model_name,torch_dtype=torch.bfloat16).to(device)
finetuned_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=finetuned_model_name)

def set_random_seed(seed: int = 0):
    """
    set random seed
    :param seed: int, random seed
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(seed=0)
scale_factor = finetuned_model.config.intermediate_size / finetuned_model.config.hidden_size

def decomposition(masked_input_tensor,dim):

    U , S , V = torch.svd(masked_input_tensor.to(torch.float32))
    U , S , V = U[:, :dim],S[:dim],V[:, :dim]
    # return torch.mm(U, torch.diag(S)), V.t()
    return torch.mm(U, torch.mm(torch.diag(S), V.t()))   #return lora_B, lora_A


with torch.no_grad():
    for k,v in finetuned_model.state_dict().items():
        dim = args.dim
        if "self_attn" in k or "mlp" in k:
            if "mlp" in k:
                dim = int(dim * scale_factor)
            p = pretrained_model.get_submodule(k.replace(".weight", "")).weight
            delta = decomposition(v - p,dim).to(v.dtype)
            finetuned_model.get_submodule(k.replace(".weight", "")).weight.copy_(p + delta)

finetuned_model.save_pretrained(save_directory=args.save_dir)
finetuned_tokenizer.save_pretrained(save_directory=args.save_dir)


print("--end--")
