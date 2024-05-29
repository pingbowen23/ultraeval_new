URL="http://127.0.0.1:5031/vllm-url31-infer"

TASK_NAME=humaneval 
NUMBER_OF_THREAD=1
HF_MODEL_NAME="/data/public/multilingual/whq/model/saved_model/ru-code-lorahub-7b/"
language_model="/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/13b/zh/lora/"
gate_model=/data/public/multilingual/whq/model/bsz_2_lr_1e-3_epoch_5_13b_code/checkpoints/epoch_1/gate.pt
task_model=/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/trim_lora/dim256_2/code  # /home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/trim_lora/dim256/code/lora
# python configs/make_config.py --datasets $TASK_NAME --method gen 
nohup python URLs/vllm_url31.py --use_vllm --task_model $task_model --language_model $language_model --temperature 1.0 --gpuid 3  --port 5031  --model_name $HF_MODEL_NAME --gate_path $gate_model &

#  /data/public/multilingual/whq/model/bsz_2_lr_1e-3_epoch_5_code/checkpoints/epoch_3/gate.pt

# /data/public/multilingual/whq/UltraEval/code_result/2024-01-25_10-57-41

sleep 30

python main.py \
    --model general \
    --model_args url=$URL,concurrency=$NUMBER_OF_THREAD \
    --config_path configs/code_eval_confg.json \
    --output_base_path code_result/ru \
    --batch_size 256 \
    --postprocess general_torch \
    --params models/model_params/vllm_sample_wizardcode.json \
    --write_out \
    # --limit 1 \

PID=$(ps -ef | grep "31.py" | grep -v grep | awk '{print $2}')

kill $PID

# # 要测的模型路径
# /home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/trim_lora/dim128/code/with_ft_other
# /home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/trim_lora/dim256/code
# /home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/trim_lora/dim256/code/with_ft_other
# /home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/trim_lora/dim64/code
# /home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/trim_lora/dim64/code/with_ft_other

# /home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/code_0121_epoch1

# /home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/code_0120_epoch3