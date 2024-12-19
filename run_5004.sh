TASK_NAME=truthfulqa 
NUMBER_OF_THREAD=1
# python configs/make_config.py --datasets $TASK_NAME --method gen
# for i in {5..6}; do
# datasets/gsm8k/transform_gen_en.py
HF_MODEL_NAME="/home/pingbowen/models/llama2_7b_4bit"  
URL="http://127.0.0.1:5006/vllm-url4-infer"
NUMBER_OF_THREAD=1  # 线程数，一般设为 gpu数/per-proc-gpus
CONFIG_PATH=configs/eval_config_en.json
#.json
OUTPUT_BASE_PATH=llama-2-7b-hf  # 结果保存路径，与HF_MODEL_NAME一致
language_MODEL_Path="/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/13b/zh/lora/"
Gate_path=/data/public/multilingual/whq/model/bsz_2_lr_1e-3_epoch_5_13b/checkpoints/epoch_4/gate.pt
task_model=/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/13b/math/lora/
names=("2" "3" "4" "8_2" "8_3" "8_4" "8_3_2")
gpu_id=0

for (( i=0; i<1; i++ )); do # ${#names[@]}
HF_MODEL_NAME=/home/pingbowen/workspace/delta-compression/gptq-delta/save_compressed/test_wizardmath_full
#/data/public/wangshuo/exp/ft-en-magicoder-llama-2-7b/ckpts/checkpoints/epoch_2_hf
# /home/pingbowen/workspace/delta-compression/save/Wizardcoder13B_bitdelta/
#/home/wanghanqing/projects/models/model_ver2/Mistral-7B-Instruct-v0.2
# HF_MODEL_NAME="/home/pingbowen/workspace/delta-compression/save/magicoder_mix_2_full"
nohup python URLs/vllm_url4.py --temperature 1.0 --task_model $task_model  --use_vllm --gpuid $gpu_id --model_name $HF_MODEL_NAME --language_model $language_MODEL_Path --gate_path $Gate_path  --port 5006    &
# wait

sleep 30
echo "service is up"

python main.py \
    --model general \
    --model_args url=$URL,concurrency=$NUMBER_OF_THREAD \
    --config_path $CONFIG_PATH \
    --output_base_path $OUTPUT_BASE_PATH \
    --batch_size 512 \
    --postprocess general_torch \
    --params models/model_params/vllm_sample_wizardcode.json \
    --write_out \

PID=$(ps -ef | grep "python URLs/vllm_url4.py " | grep -v grep | awk '{print $2}')

kill $PID
done