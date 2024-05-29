#!/bin/bash

# hyperparameters
TASK_NAME=humaneval # 需要评测的任务，多个用,隔开
HF_MODEL_NAME=/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/trim_lora/dim256/code/  # huggingface上的模型名
URL="http://127.0.0.1:5002/vllm-url-infer"  # 这里是固定的
NUMBER_OF_THREAD=1  # 线程数，一般设为 gpu数/per-proc-gpus
CONFIG_PATH=configs/humaneval_config.json  # 评测文件路径
OUTPUT_BASE_PATH=llama-2-7b-hf  # 结果保存路径，与HF_MODEL_NAME一致

# # 步骤1
# # 选择评测的任务，生成评测 config文件。其中method=gen，表示生成式
# python configs/make_config.py --datasets $TASK_NAME --method gen



# 步骤2
# 启动 gunicorn 并保存 PID
bash URLs/start_gunicorn.sh --hf-model-name $HF_MODEL_NAME --per-proc-gpus 1 & 
echo $! > gunicorn.pid


# 步骤3
# 检查服务是否已启动
MAX_RETRIES=6  # 最大尝试次数，相当于等待30分钟
COUNTER=0

while [ $COUNTER -lt $MAX_RETRIES ]; do
    sleep 30
    curl -s $URL > /dev/null
    if [ $? -eq 0 ]; then
        echo "Service is up!"
        break
    fi
    COUNTER=$((COUNTER+1))
    if [ $COUNTER -eq $MAX_RETRIES ]; then
        echo "Service did not start in time. Exiting."
        exit 1
    fi
done


# 步骤4
# 执行 Python 脚本
python main.py \
    --model general \
    --model_args url=$URL,concurrency=$NUMBER_OF_THREAD \
    --config_path $CONFIG_PATH \
    --output_base_path $OUTPUT_BASE_PATH \
    --batch_size 64 \
    --postprocess general_torch \
    --params models/model_params/vllm_sample.json \
    --write_out \
    # --limit 2


# 步骤5
# 结束 gunicorn 进程及其 worker 进程
MAIN_PID=$(cat gunicorn.pid)
pgrep -P $MAIN_PID | xargs kill -9
kill -9 $MAIN_PID
rm gunicorn.pid