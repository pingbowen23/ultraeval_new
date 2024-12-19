TASK_NAME=gsm8k 
NUMBER_OF_THREAD=1
# python configs/make_config.py --datasets $TASK_NAME --method gen

# HF_MODEL_NAME="/data/public/opensource_models/meta-llama/Llama-2-7b-hf/"  # huggingface上的模型名
# URL="http://127.0.0.1:5006/vllm-url4-infer"
NUMBER_OF_THREAD=1  # 线程数，一般设为 gpu数/per-proc-gpus


LANGUAGE=$1  # 语言 
Gate_path=$2
port=$3
gpuid=$4
eval_task=$5
model_size=$6
bsz=$7
type=$8
temperature=$9


HF_MODEL_NAME=/data/public/opensource_models/meta-llama/Llama-2-${model_size}-hf/

OUTPUT_ROOT_PATH=result_parallel_infer

OUTPUT_BASE_PATH=$OUTPUT_ROOT_PATH/${eval_task}

if [ "$eval_task" = "math" ]; then
    # 执行一些操作
    echo "math"
    CONFIG_PATH=configs/eval_config_${LANGUAGE}.json
    # PARAMS=models/model_params/vllm_sample.json
    PARAMS=models/model_params/vllm_sample_wizardcode.json

    if [ "$model_size" = "7b" ]; then
        TASK_MODEL_PATH=/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/math_epoch3/lora
    fi
    if [ "$model_size" = "13b" ]; then
        TASK_MODEL_PATH=/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/13b/math/lora
    fi
    if [ "$model_size" != "7b" ] && [ "$model_size" != "13b" ]; then echo "error, model_size must be 7b or 13b"; exit 1; fi


fi

if [ "$eval_task" = "code" ]; then
    # 执行一些操作
    echo "code"
    CONFIG_PATH=configs/code_${LANGUAGE}_confg.json
    PARAMS=models/model_params/vllm_sample_wizardcode.json
    
    if [ "$model_size" = "7b" ]; then
        TASK_MODEL_PATH=/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/trim_lora/dim256/code/lora
    fi
    if [ "$model_size" = "13b" ]; then
        TASK_MODEL_PATH=/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/trim_lora/dim256_2/code
    fi
    if [ "$model_size" != "7b" ] && [ "$model_size" != "13b" ]; then echo "error, model_size must be 7b or 13b"; exit 1; fi
fi


if [ "$eval_task" != "math" ] && [ "$eval_task" != "code" ]; then echo "error, eval_task must be math or code"; exit 1; fi


# CONFIG_PATH=configs/eval_config_ru.json  # 评测文件路径
# CONFIG_PATH=configs/eval_config_${LANGUAGE}.json  # 评测文件路径

# OUTPUT_BASE_PATH=llama-2-7b-hf  # 结果保存路径，与HF_MODEL_NAME一致

if [ "$model_size" = "7b" ]; then
    LANGUAGE_MODEL_PATH=/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/${LANGUAGE}/lora
fi
if [ "$model_size" = "13b" ]; then
    LANGUAGE_MODEL_PATH=/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/13b/${LANGUAGE}/lora
fi




##如果 Gate_path中不包含LANGUAGE，则报错，并打印：Gate_path中不包含LANGUAGE的信息
## 代码

# if [[ $Gate_path != *$LANGUAGE* ]]; then echo "Gate_path中不包含LANGUAGE"; exit 1; fi

URL=http://127.0.0.1:${port}/vllm-url2-infer


# nohup python URLs/url_parallel.py --use_gate --model_name $HF_MODEL_NAME --language_model $LANGUAGE_MODEL_PATH --gate_path $Gate_path --gpuid $gpuid --port $port --temperature  1.0  >> result_parallel_infer/port_${port}.log &

# nohup python URLs/url_parallel.py --use_vllm --model_name $HF_MODEL_NAME --language_model $LANGUAGE_MODEL_PATH --task_model $TASK_MODEL_PATH --gate_path $Gate_path --gpuid $gpuid --port $port --temperature  1.0  >> result_parallel_infer/port_${port}.log 2>&1  &
# nohup python URLs/url_parallel.py --use_gate --model_name $HF_MODEL_NAME --language_model $LANGUAGE_MODEL_PATH --task_model $TASK_MODEL_PATH --gate_path $Gate_path --gpuid $gpuid --port $port --temperature  1.0  >> result_parallel_infer/port_${port}.log 2>&1  &
nohup python URLs/url_parallel.py --use_gate --model_name $HF_MODEL_NAME --language_model $LANGUAGE_MODEL_PATH --task_model $TASK_MODEL_PATH --gate_path $Gate_path --gpuid $gpuid --port $port --temperature  $temperature --type $type &


# 步骤3  
# 检查服务是否已启动  ./../model/bsz_2_lr_1e-2_epoch_5_500/checkpoints/epoch_2/gate.pt
MAX_RETRIES=6  # 最大尝试次数，相当于等待30分钟
COUNTER=0

sleep 180

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

CONFIG_PATH=configs/eval_config.json

echo "URL:${URL}"
echo "CONFIG_PATH${CONFIG_PATH}"
echo "LANGUAGE:${LANGUAGE}"
echo "Gate_path:${Gate_path}"
echo "port:${port}"
echo "gpuid:${gpuid}"
echo "eval_task:${eval_task}"
echo "model_size:${model_size}"
echo "bsz:${bsz}"



python main.py \
    --model general \
    --model_args url=$URL,concurrency=$NUMBER_OF_THREAD \
    --config_path $CONFIG_PATH \
    --output_base_path $OUTPUT_BASE_PATH \
    --batch_size $bsz \
    --postprocess general_torch \
    --params $PARAMS \
    --write_out \
    # --limit 60 \



lsof -i :${port} -t | xargs kill -9
# lsof -i :5041 -t | xargs kill -9
# lsof -i :5042 -t | xargs kill -9



# PID=$(ps -ef | grep "python URLs/vllm_url4.py " | grep -v grep | awk '{print $2}')

# kill $PID