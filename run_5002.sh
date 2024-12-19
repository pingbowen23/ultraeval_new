TASK_NAME=gsm8k 
# python configs/make_config.py --datasets $TASK_NAME --method gen 
HF_MODEL_NAME=/home/pingbowen/workspace/delta-compression/save/delta_512_mix_8_4_full
NUMBER_OF_THREAD=1  # 线程数，一般设为 gpu数/per-proc-gpus 
OUTPUT_BASE_PATH=llama-2-13b-hf  
language=en
CONFIG_PATH=configs/eval_config_${language}.json  
Gate_path=(epoch_0/gate.pt epoch_0/gate.pt)
models=(/home/pingbowen/workspace/delta-compression/save/wizardmath-svd)
names=("0.8bit" "1.2bit" "1.4bit" "1.6bit" "1.8bit" "2bit")
language_MODEL_Path="/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/${language}/lora/"
task_model=/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/math_epoch3/lora
# /data/public/multilingual/whq/model/bsz_2_lr_1e-3_epoch_5_zh_math_not_share_reproduce_our/checkpoints/

for (( i=0; i<1; i++ )); do # ${#values[@]} ${#names[@]}
# directories=$(find /data/groups/QY_LLM_Other/pingbowen/models/wizardmath/genetic/ -type d -name "*_full*")
# for dir in $directories; do

HF_MODEL_NAME=/home/pingbowen/workspace/delta-compression/gptq-delta/save_compressed/test_wizardmath_full
port=$((5088 + i))
URL="http://127.0.0.1:${port}/vllm-url2-infer"

nohup python URLs/vllm_url2.py --use_vllm \
  --task_model $task_model \
  --gpuid 0  \
  --port $port  \
  --decoding_temp 0.0001 \
  --temperature 1.0  \
  --model_name $HF_MODEL_NAME \
  --gate_path "/data/public/multilingual/whq/model/bsz_8_lr_1e-4_epoch_5_es/checkpoints/${Gate_path[$i]}" \
  --language_model $language_MODEL_Path &
  # --layer_wise &
# 步骤3  
# 检查服务是否已启动  ./../model/bsz_2_lr_1e-2_epoch_5_500/checkpoints/epoch_2/gate.pt
# MAX_RETRIES=6  # 最大尝试次数，相当于等待30分钟
# COUNTER=0

sleep 30

# while [ $COUNTER -lt $MAX_RETRIES ]; do
#     sleep 30
#     curl -s $URL > /dev/null
#     if [ $? -eq 0 ]; then
#         echo "Service is up!"
#         break
#     fi
#     COUNTER=$((COUNTER+1))
#     if [ $COUNTER -eq $MAX_RETRIES ]; then
#         echo "Service did not start in time. Exiting."
#         exit 1
#     fi
# done
echo "Waiting for $dir to finish..."
python main.py \
    --model general \
    --model_args url=$URL,concurrency=$NUMBER_OF_THREAD \
    --config_path $CONFIG_PATH \
    --output_base_path $OUTPUT_BASE_PATH \
    --batch_size 512 \
    --postprocess general_torch \
    --params models/model_params/vllm_sample_math_zero.json \
    --write_out \

# echo "$dir finished." >> results_gsm8k.txt

PID=$(ps -ef | grep "python URLs/vllm_url2.py " | grep -v grep | awk '{print $2}')

kill $PID 
done
# wait