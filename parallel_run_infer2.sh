

# LANGUAGE=$1  # 语言 
# Gate_path=$2
# port=$3
# gpuid=$4

# /home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/ru
# /home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/es

# bash run_5041.sh ru /data/public/multilingual/whq/model/bsz_2_lr_1e-3_epoch_5_ru/checkpoints/epoch_1/gate.pt 5041 0 /home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/ru &
# sleep 30
# bash run_5041.sh es /data/public/multilingual/whq/model/bsz_2_lr_1e-3_epoch_5_es/checkpoints/epoch_1/gate.pt 5042 1 /home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/peft_ver/es &

# bash run_5041.sh ru /data/public/multilingual/whq/model/ru/bsz_2_lr_1e-3_epoch_5_ru/checkpoints/epoch_4/gate.pt 5041 0 math "7b" 8 > ru-bsz_2_lr_1e-3_epoch_5_ru-checkpoints-epoch_4.log 2>&1 &
# sleep 30
# bash run_5041.sh zh /data/public/multilingual/whq/model/bsz_2_lr_1e-3_epoch_5_13b_code/checkpoints/epoch_4/gate.pt 5042 1 code "13b" 4 > zh-bsz_2_lr_1e-3_epoch_5_13b_code-checkpoints-epoch_4.log 2>&1 &

# bash run_para.sh zh /data/public/multilingual/whq/model/bsz_2_lr_1e-3_epoch_5_200_softmax/checkpoints/epoch_1/gate.pt 5041 0 math "7b" 8 >> result_parallel_infer/port_5041.log 2>&1 &
# sleep 30
# bash run_para.sh zh /data/public/multilingual/whq/model/bsz_2_lr_1e-3_epoch_5_zh_code/checkpoints/epoch_1/gate.pt 5042 1 code "7b" 4 >> result_parallel_infer/port_5042.log 2>&1 &
# sleep 30

# wait


# bash run_para.sh ru /data/public/multilingual/whq/model/ru/bsz_2_lr_1e-3_epoch_5_ru/checkpoints/epoch_4/gate.pt 5041 0 math "7b" 8 >> result_parallel_infer/port_5041.log 2>&1 &
# sleep 30
# bash run_para.sh ru /data/public/multilingual/whq/model/ru/bsz_8_lr_1e-4_epoch_5_ru_code/checkpoints/epoch_0/gate.pt 5042 1 code "7b" 4 >> result_parallel_infer/port_5042.log 2>&1 &
# sleep 30
# wait


# bash run_para.sh es /data/public/multilingual/whq/model/es/bsz_8_lr_1e-4_epoch_5_es/checkpoints/epoch_0/gate.pt 5041 0 math "7b" 8 >> result_parallel_infer/port_5041.log 2>&1 &
# sleep 30
# bash run_para.sh es /data/public/multilingual/whq/model/es/bsz_2_lr_1e-3_epoch_5_es_code/checkpoints/epoch_2/gate.pt 5042 1 code "7b" 4 >> result_parallel_infer/port_5042.log 2>&1 &
# sleep 30
# wait

# ru gate math exp

# bash run_para.sh es  /data/public/multilingual/whq/model/ru/bsz_2_lr_1e-3_epoch_5_ru/checkpoints/epoch_4/gate.pt 5040 0 math "7b" 8 >> result_parallel_infer/port_5040.log 2>&1 &
# sleep 30
# bash run_para.sh zh  /data/public/multilingual/whq/model/ru/bsz_2_lr_1e-3_epoch_5_ru/checkpoints/epoch_4/gate.pt 5041 1 math "7b" 8 >> result_parallel_infer/port_5041.log 2>&1 &
# sleep 30
# bash run_para.sh zh  /data/public/multilingual/whq/model/ru/bsz_2_lr_1e-3_epoch_5_ru/checkpoints/epoch_4/gate.pt 5042 2 code "7b" 8 >> result_parallel_infer/port_5042.log 2>&1 &
# sleep 30
# bash run_para.sh es  /data/public/multilingual/whq/model/ru/bsz_2_lr_1e-3_epoch_5_ru/checkpoints/epoch_4/gate.pt 5043 3 code "7b" 8 >> result_parallel_infer/port_5043.log 2>&1 &
# sleep 30
# bash run_para.sh ru  /data/public/multilingual/whq/model/ru/bsz_2_lr_1e-3_epoch_5_ru/checkpoints/epoch_4/gate.pt 5044 4 code "7b" 8 >> result_parallel_infer/port_5044.log 2>&1 &
# sleep 30

# wait

# # es gate math exp
# bash run_para.sh ru  /data/public/multilingual/whq/model/es/bsz_8_lr_1e-4_epoch_5_es/checkpoints/epoch_0/gate.pt 5040 0 math "7b" 8 >> result_parallel_infer/port_5040.log 2>&1 &
# sleep 30
# bash run_para.sh zh  /data/public/multilingual/whq/model/es/bsz_8_lr_1e-4_epoch_5_es/checkpoints/epoch_0/gate.pt 5041 1 math "7b" 8 >> result_parallel_infer/port_5041.log 2>&1 &
# sleep 30
# bash run_para.sh zh  /data/public/multilingual/whq/model/es/bsz_8_lr_1e-4_epoch_5_es/checkpoints/epoch_0/gate.pt 5042 2 code "7b" 8 >> result_parallel_infer/port_5042.log 2>&1 &
# sleep 30
# bash run_para.sh es  /data/public/multilingual/whq/model/es/bsz_8_lr_1e-4_epoch_5_es/checkpoints/epoch_0/gate.pt 5043 3 code "7b" 8 >> result_parallel_infer/port_5043.log 2>&1 &
# sleep 30
# bash run_para.sh ru  /data/public/multilingual/whq/model/es/bsz_8_lr_1e-4_epoch_5_es/checkpoints/epoch_0/gate.pt 5044 4 code "7b" 8 >> result_parallel_infer/port_5044.log 2>&1 &
# sleep 30

# wait


# bash run_para.sh es  /data/public/multilingual/whq/model/bsz_2_lr_1e-3_epoch_5_200_softmax/checkpoints/epoch_1/gate.pt 5043 3 code "7b" 8 >> result_parallel_infer/port_5043.log 2>&1 &
# sleep 30
# bash run_para.sh ru  /data/public/multilingual/whq/model/bsz_2_lr_1e-3_epoch_5_200_softmax/checkpoints/epoch_1/gate.pt 5044 4 code "7b" 8 >> result_parallel_infer/port_5044.log 2>&1 &
# sleep 30

# wait

# # zh code gate exp

# bash run_para.sh ru  /data/public/multilingual/whq/model/bsz_2_lr_1e-3_epoch_5_zh_code/checkpoints/epoch_1/gate.pt 5040 0 math "7b" 8 >> result_parallel_infer/port_5040.log 2>&1 &
# sleep 30
# bash run_para.sh zh  /data/public/multilingual/whq/model/bsz_2_lr_1e-3_epoch_5_zh_code/checkpoints/epoch_1/gate.pt 5041 1 math "7b" 8 >> result_parallel_infer/port_5041.log 2>&1 &
# sleep 30
# bash run_para.sh es  /data/public/multilingual/whq/model/bsz_2_lr_1e-3_epoch_5_zh_code/checkpoints/epoch_1/gate.pt 5042 2 math "7b" 8 >> result_parallel_infer/port_5042.log 2>&1 &
# sleep 30
# bash run_para.sh es  /data/public/multilingual/whq/model/bsz_2_lr_1e-3_epoch_5_zh_code/checkpoints/epoch_1/gate.pt 5043 3 code "7b" 8 >> result_parallel_infer/port_5043.log 2>&1 &
# sleep 30
# bash run_para.sh ru  /data/public/multilingual/whq/model/bsz_2_lr_1e-3_epoch_5_zh_code/checkpoints/epoch_1/gate.pt 5044 4 code "7b" 8 >> result_parallel_infer/port_5044.log 2>&1 &
# sleep 30

# wait

# # es code gate exp
# bash run_para.sh ru  /data/public/multilingual/whq/model/es/bsz_2_lr_1e-3_epoch_5_es_code/checkpoints/epoch_2/gate.pt 5040 0 math "7b" 8 >> result_parallel_infer/port_5040.log 2>&1 &
# sleep 30
# bash run_para.sh zh  /data/public/multilingual/whq/model/es/bsz_2_lr_1e-3_epoch_5_es_code/checkpoints/epoch_2/gate.pt 5041 1 math "7b" 8 >> result_parallel_infer/port_5041.log 2>&1 &
# sleep 30
# bash run_para.sh es  /data/public/multilingual/whq/model/es/bsz_2_lr_1e-3_epoch_5_es_code/checkpoints/epoch_2/gate.pt 5042 2 math "7b" 8 >> result_parallel_infer/port_5042.log 2>&1 &
# sleep 30
# bash run_para.sh zh  /data/public/multilingual/whq/model/es/bsz_2_lr_1e-3_epoch_5_es_code/checkpoints/epoch_2/gate.pt 5043 3 code "7b" 8 >> result_parallel_infer/port_5043.log 2>&1 &
# sleep 30
# bash run_para.sh ru  /data/public/multilingual/whq/model/es/bsz_2_lr_1e-3_epoch_5_es_code/checkpoints/epoch_2/gate.pt 5044 4 code "7b" 8 >> result_parallel_infer/port_5044.log 2>&1 &
# sleep 30

# wait

# # ru code gate exp
# bash run_para.sh zh /data/public/multilingual/whq/model/bsz_2_lr_1.5e-3_epoch_5_13b/checkpoints/epoch_1/gate.pt 5021 0 math "13b" 4 >> result_parallel_infer/port_5043.log 2>&1 &
# sleep 30

# bash run_para.sh zh /data/public/multilingual/whq/model/bsz_2_lr_1.5e-3_epoch_5_13b/checkpoints/epoch_2/gate.pt 5022 1 math "13b" 4 >> result_parallel_infer/port_5042.log 2>&1 &
# sleep 30

# bash run_para.sh zh /data/public/multilingual/whq/model/bsz_2_lr_1.5e-3_epoch_5_13b/checkpoints/epoch_3/gate.pt 5023 2 math "13b" 4 >> result_parallel_infer/port_5041.log 2>&1 &
# sleep 30

# # bash run_para.sh zh /data/public/multilingual/whq/model/bsz_2_lr_2e-3_epoch_5_13b/checkpoints/epoch_1/gate.pt 5024 3 math "13b" 4 >> result_parallel_infer/port_5040.log 2>&1 &
# # sleep 30

# wait
# # ja code gate exp
# bash run_para.sh jp /data/public/multilingual/whq/model/bsz_2_lr_1e-3_epoch_5_ja/checkpoints/epoch_0/gate.pt 5019 0 math "7b" 8 >> result_parallel_infer2/port_5019.log 2>&1 &
# sleep 30

# bash run_para.sh jp /data/public/multilingual/whq/model/bsz_2_lr_1e-3_epoch_5_ja/checkpoints/epoch_1/gate.pt 5020 1 math "7b" 8 >> result_parallel_infer2/port_5020.log 2>&1 &
# sleep 30

# bash run_para.sh jp /data/public/multilingual/whq/model/bsz_2_lr_1e-3_epoch_5_ja/checkpoints/epoch_2/gate.pt 5021 2 math "7b" 8 >> result_parallel_infer2/port_5021.log 2>&1 &
# sleep 30

# bash run_para.sh jp /data/public/multilingual/whq/model/bsz_2_lr_1e-3_epoch_5_ja/checkpoints/epoch_3/gate.pt 5022 3 math "7b" 8 >> result_parallel_infer2/port_5022.log 2>&1 &
# sleep 30

# bash run_para.sh jp /data/public/multilingual/whq/model/bsz_2_lr_1e-3_epoch_5_ja/checkpoints/epoch_4/gate.pt 5023 4 math "7b" 8 >> result_parallel_infer2/port_5023.log 2>&1 &
# sleep 30


# bash run_para.sh zh /data/public/multilingual/whq/all_share_mAlign/scripts_whq/try_exp/bsz_2_lr_1e-3_epoch_5_zh_math_share/checkpoints/epoch_1/gate.pt 5041 0 math "7b" 8 share_gate >> result_parallel_infer/port_5041.log 2>&1 &
# sleep 30

# bash run_para.sh ru /data/public/multilingual/whq/all_share_mAlign/scripts_whq/try_exp/bsz_2_lr_1e-3_epoch_5_ru_math_share/checkpoints/epoch_4/gate.pt 5042 1 math "7b" 8 share_gate >> result_parallel_infer/port_5042.log 2>&1 &
# sleep 30

bash run_para.sh es /data/public/multilingual/whq/all_share_mAlign/scripts_whq/try_exp/bsz_8_lr_1e-4_epoch_5_es_math_share/checkpoints/epoch_2/gate.pt 5041 2 math "7b" 8 share_gate 1.1 >> result_parallel_infer/port_5041.log 2>&1 &
sleep 30

bash run_para.sh es /data/public/multilingual/whq/all_share_mAlign/scripts_whq/try_exp/bsz_8_lr_1e-4_epoch_5_es_math_share/checkpoints/epoch_1/gate.pt 5042 3 math "7b" 8 share_gate 1.1 >> result_parallel_infer/port_5042.log 2>&1 &
sleep 30

bash run_para.sh es /data/public/multilingual/whq/all_share_mAlign/scripts_whq/try_exp/bsz_8_lr_1e-4_epoch_5_es_math_share/checkpoints/epoch_3/gate.pt 5043 4 math "7b" 8 share_gate 1.1 >> result_parallel_infer/port_5043.log 2>&1 &
sleep 30

bash run_para.sh es /data/public/multilingual/whq/all_share_mAlign/scripts_whq/try_exp/bsz_8_lr_1e-4_epoch_5_es_math_share/checkpoints/epoch_4/gate.pt 5044 5 math "7b" 8 share_gate 1.1 >> result_parallel_infer/port_5044.log 2>&1 &
sleep 30

wait

