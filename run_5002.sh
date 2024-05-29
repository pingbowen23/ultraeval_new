TASK_NAME=gsm8k 
# python configs/make_config.py --datasets $TASK_NAME --method gen 


HF_MODEL_NAME=/home/pingbowen/workspace/delta-compression/save/delta_512_mix_8_4_full
NUMBER_OF_THREAD=1  # 线程数，一般设为 gpu数/per-proc-gpus 
CONFIG_PATH=configs/eval_config_zh.json  # 评测文件路径
OUTPUT_BASE_PATH=llama-2-13b-hf  # 结果保存路径，与HF_MODEL_NAME一致
language_MODEL_Path="/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/zh/lora/"
Gate_path=/data/public/multilingual/whq/model/bsz_2_lr_1e-3_epoch_5_200_softmax/checkpoints/epoch_1/gate.pt
task_model=/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/math_epoch3/lora
models=(/home/pingbowen/workspace/delta-compression/save/wizardmath-svd/ /home/pingbowen/workspace/delta-compression/saved_model/WizardMath-7B-V1.0_bitdelta/ /data/groups/QY_LLM_Other/pingbowen/models/wizardmath/delta_1024_mix_32_8_3_2_full/ /data/public/opensource_models/WizardLM/WizardMath-7B-V1.0/)
names=("0.8bit" "1.2bit" "1.4bit" "1.6bit" "1.8bit" "2bit")
for (( i=0; i<8; i++ )); do # ${#values[@]} ${#names[@]}
# /home/pingbowen/workspace/delta-compression/save/delta_1024_mix_32_8_3_2_full
HF_MODEL_NAME=/data/public/opensource_models/meta-llama/Llama-2-7b-hf/
port=$((5008 + i))
URL="http://127.0.0.1:${port}/vllm-url2-infer"

nohup python URLs/vllm_url2.py --use_gate --task_model $task_model --gpuid $i  --port $port  --decoding_temp 0.1 --temperature 1.1  --model_name $HF_MODEL_NAME --gate_path $Gate_path --language_model $language_MODEL_Path &

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

python main.py \
    --model general \
    --model_args url=$URL,concurrency=$NUMBER_OF_THREAD \
    --config_path $CONFIG_PATH \
    --output_base_path $OUTPUT_BASE_PATH \
    --batch_size 8 \
    --postprocess general_torch \
    --params models/model_params/vllm_sample_math_zero.json \
    --write_out \
    >> test_interval.log &

# PID=$(ps -ef | grep "python URLs/vllm_url2.py " | grep -v grep | awk '{print $2}')

# kill $PID 
done
wait