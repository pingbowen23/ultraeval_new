import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="Model name on hugginface")
parser.add_argument("--gpuid", type=str, default="0", help="GPUid to be deployed")
parser.add_argument("--port", type=int, default=5002, help="the port")
parser.add_argument("--weight", type=float, default=0, help="linear")
parser.add_argument("--use_peft", default=None,action="store_true",help="linear")
parser.add_argument("--use_vllm", default=None,action="store_true",help="linear")
parser.add_argument("--use_gate", default=None,action="store_true",help="linear")
parser.add_argument("--gate_path", type=str,help="linear")
parser.add_argument("--temperature", type=float, default=1.0, help="linear")
parser.add_argument("--decoding_temp", type=float, default=0.0001, help="linear")
parser.add_argument("--language_model", type=str, default="", help="linear")
parser.add_argument("--task_model", type=str, default="", help="linear")
parser.add_argument("--layer_wise", action="store_true", help="linear")
parser.add_argument("--model_wise", action="store_true", help="linear")
args = parser.parse_args()

import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
from flask import Flask, jsonify, request
# from optimum.bettertransformer import BetterTransformer
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftConfig, PeftModel

"""
reference:https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
"""

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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
        temperature=args.decoding_temp,
        top_p=1, 
    )
        
    outputs = tokenizer.batch_decode(
        outputs.to("cpu"), skip_special_tokens=True
    )
    
    for i in range(len(outputs)):
        if outputs[i].startswith(prompts[i]):
            outputs[i] = outputs[i][len(prompts[i]):]
    
    return outputs

def load_peft_model():
    pretrained_model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    device_map="auto",
        )
    
    for i in range(32):
        pretrained_model.model.layers[i].temperature = args.temperature
    return pretrained_model

def load_param_wise_gate(model,gate_path):
    param_dict = torch.load(gate_path)
    new_param_dict = dict()
    
    fusion_gate_mapping_dict = {
        "q_lora_fusion_gate": "base_model.model.model.layers.*.self_attn.q_lora_fusion_gate.weight",
        "k_lora_fusion_gate": "base_model.model.model.layers.*.self_attn.k_lora_fusion_gate.weight",
        "v_lora_fusion_gate": "base_model.model.model.layers.*.self_attn.v_lora_fusion_gate.weight",
        "o_lora_fusion_gate": "base_model.model.model.layers.*.self_attn.o_lora_fusion_gate.weight",
        "w_0_lora_fusion_gate":"base_model.model.model.layers.*.mlp.gate_lora_fusion_gate.weight ",
        "w_1_lora_fusion_gate":"base_model.model.model.layers.*.mlp.up_lora_fusion_gate.weight",
        "w_out_lora_fusion_gate":"base_model.model.model.layers.*.mlp.down_lora_fusion_gate.weight",
    }
    
    weight_bias_mapping_dict = {
        "q_weight_bias": "base_model.model.model.layers.*.self_attn.q_weight_bias",
        "k_weight_bias": "base_model.model.model.layers.*.self_attn.k_weight_bias",
        "v_weight_bias": "base_model.model.model.layers.*.self_attn.v_weight_bias",
        "o_weight_bias": "base_model.model.model.layers.*.self_attn.o_weight_bias",
        "w_0_weight_bias":"base_model.model.model.layers.*.mlp.gate_weight_bias",
        "w_1_weight_bias":"base_model.model.model.layers.*.mlp.up_weight_bias",
        "w_out_weight_bias":"base_model.model.model.layers.*.mlp.down_weight_bias",        
    }
    
    for k,v in param_dict.items():
        tmp = dict()
        if "fusion_gate" in k:
            tmp = fusion_gate_mapping_dict        
        elif "weight_bias" in k:
            tmp = weight_bias_mapping_dict
        
        for key,value in tmp.items():
            if key in k:
                match = re.search(r'layers\.(\d+)', k)
                number = match.group(1)
                
                new_param_dict[value.replace("*",number)] = v
                
    model.load_state_dict(new_param_dict,strict=False)
    return model

def load_model_wise_gate(model,gate_path):
    param_dict = torch.load(gate_path)
    new_param_dict = dict()
    new_param_dict["base_model.model.model.lora_fusion_gate.weight"] = param_dict["lora_fusion_gate.weight"]
    new_param_dict["base_model.model.model.weight_bias"] = param_dict["weight_bias"]
    
    model.load_state_dict(new_param_dict,strict=False)
    # import pdb;pdb.set_trace()
    return model

def load_gate_model():
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
    lora_model.load_adapter(model_id = args.task_model ,adapter_name = "math") # /home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/math_epoch3/lora

    lora_model.base_model.set_adapter(["zh","math"])
    lora_model.base_model.model.model.to(device)
    if args.layer_wise == True:
        load_fusion_gate(lora_model)
    elif args.model_wise == True:
        lora_model = load_model_wise_gate(lora_model,args.gate_path)
    else:
        lora_model = load_param_wise_gate(lora_model,gate_path=args.gate_path)
    
    lora_model.to(device).to(pretrained_model.dtype)
    return lora_model

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    
    if args.use_vllm:
        model = LLM(
            model=args.model_name,
            trust_remote_code=True,
            tensor_parallel_size=len(args.gpuid.split(",")),
            )
    elif args.use_gate:
        print(f"------loading gate-----")
        model = load_gate_model()
    else:
        model = load_peft_model()
    return model


model = load_model()

@app.route("/vllm-url2-infer", methods=["POST"])
def main():
    datas = request.get_json()
    params = datas["params"]
    prompts = datas["instances"]
    
    for key, value in params.items():
        if key in params_dict:
            params_dict[key] = value
            
    
    if args.use_vllm:
        # params_dict["temperature"] = 0.1
        outputs = model.generate(prompts, SamplingParams(**params_dict))
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
            if args.use_vllm:
                generated_text = output.outputs[0].text
            else:
                generated_text = output
            res.append(generated_text)
        return jsonify(res)


if __name__ == "__main__":
    app.run(port=args.port, debug=False)