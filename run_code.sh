URL="http://127.0.0.1:5031/vllm-url31-infer"

TASK_NAME=humaneval 
NUMBER_OF_THREAD=1
HF_MODEL_NAME=/data/public/opensource_models/meta-llama/Llama-2-7b-hf/
language_model="/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/zh/lora/"
gate_model=/data/public/multilingual/whq/model/bsz_2_lr_1e-3_epoch_5_zh_code/checkpoints/epoch_1/gate.pt
task_model=/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/7b_loras/code_lora/  
config=configs/code_zh_confg.json


directories=$(find /data/groups/QY_LLM_Other/pingbowen/models/magicoder/genetic/ -type d -name "*_full*") 
gpu_count=8  # 假设有8个GPU
pids=()

# for dir in $directories; do

gpu_id=$(( (${#pids[@]} % gpu_count))) 

nohup python URLs/vllm_url31.py --task_model $task_model --language_model $language_model --temperature 1.0 --gpuid 7  --port $(($gpu_id + 5031))  --model_name $HF_MODEL_NAME --gate_path $gate_model &


sleep 60

python main.py \
    --model general \
    --model_args url=$URL,concurrency=$NUMBER_OF_THREAD \
    --config_path $config \
    --output_base_path code_result/ \
    --batch_size 8 \
    --postprocess general_torch \
    --params models/model_params/vllm_sample_wizardcode.json \
    --write_out \

# pids+=($!)

# if (( ${#pids[@]} >= gpu_count )); then
#     for pid in "${pids[@]}"; do
#     wait $pid
#     done
#     # 清空进程ID数组
#     pids=()
# fi

PID=$(pgrep -f "python URLs/vllm_url31.py --use_vllm --task_model $task_model --language_model $language_model --temperature 1.0 --gpuid 7 --port $(($gpu_id + 5031)) --model_name $dir --gate_path $gate_model")
kill $PID
# done


# PID=$(ps -ef | grep "31.py" | grep -v grep | awk '{print $2}')

# kill $PID