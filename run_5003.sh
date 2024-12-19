TASK_NAME=gsm8k 
# python configs/make_config.py --datasets $TASK_NAME --method gen 
#/home/pingbowen/workspace/delta-compression/save/wizard_math_256_V8_U4_full
#/home/pingbowen/workspace/delta-compression/save/wizard_math_256_qkv8_o4_full
#/home/pingbowen/workspace/delta-compression/save/wizard_math_rank_256_gptq_4_bf16
# /home/pingbowen/workspace/delta-compression/save/wizard_math_rank_512_gptq_4_bf16

URL="http://127.0.0.1:5010/vllm-url3-infer"
HF_MODEL_NAME=/home/pingbowen/workspace/delta-compression/save/delta_512_mix_8_4_full
NUMBER_OF_THREAD=1  # 线程数，一般设为 gpu数/per-proc-gpus 
# CONFIG_PATH=configs/eval_config_en.json  # 评测文件路径
CONFIG_PATH=configs/instruct_math_config.json
OUTPUT_BASE_PATH=llama-2-7b-hf  # 结果保存路径，与HF_MODEL_NAME一致
language_MODEL_Path="/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/zh/lora/"
Gate_path=/data/public/multilingual/whq/model/es/bsz_8_lr_1e-4_epoch_5_es/checkpoints/epoch_0/gate.pt
task_model=/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/math_epoch3/lora
# values=(delta_1024_mix_0_8_2_full delta_1024_mix_2_8_2_full delta_1024_mix_0_8_3_full delta_1024_mix_32_8_3_2_full_test delta_1024_mix_2_8_3_full delta_1024_mix_8_8_4_full_best) # 3 4 5 6 7 8 16
# values=(delta_1024_mix_3_8_add_3_2_full) # 
values=(672 679 880)
for (( i=0; i<1; i++ )); do # ${#values[@]}

HF_MODEL_NAME=/home/pingbowen/workspace/delta-compression/save/Mistral-7B-Instruct-svd/

nohup python URLs/vllm_url3.py --use_vllm --task_model $task_model --gpuid 6  --port 5010  --temperature 1.0  --model_name $HF_MODEL_NAME --gate_path $Gate_path --language_model $language_MODEL_Path &

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
    --batch_size 512 \
    --postprocess general_torch \
    --params models/model_params/vllm_sample_math_zero.json \
    --write_out \
    # >> test_svd.log

PID=$(ps -ef | grep "python URLs/vllm_url3.py " | grep -v grep | awk '{print $2}')

kill $PID
done