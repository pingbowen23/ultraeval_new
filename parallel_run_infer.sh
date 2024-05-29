

# LANGUAGE=$1  # 语言 
# Gate_path=$2
# port=$3
# gpuid=$4

# /home/pingbowen/workspace/mgsm/mgsm_ru.jsonl


bash run_para.sh zh /data/public/multilingual/whq/model/bsz_2_lr_1e-3_epoch_5_zh_code/checkpoints/epoch_0/gate.pt 5040 0 code "7b" 8 gate 1.0 >> result_parallel_infer/port_5040.log 2>&1 &
sleep 30

bash run_para.sh zh /data/public/multilingual/whq/model/bsz_2_lr_1e-3_epoch_5_zh_code/checkpoints/epoch_1/gate.pt 5041 1 code "7b" 8 gate 1.0 >> result_parallel_infer/port_5041.log 2>&1 &
sleep 30

bash run_para.sh zh /data/public/multilingual/whq/model/bsz_2_lr_1e-3_epoch_5_zh_code/checkpoints/epoch_2/gate.pt 5042 2 code "7b" 8 gate 1.0 >> result_parallel_infer/port_5042.log 2>&1 &
sleep 30

bash run_para.sh zh /data/public/multilingual/whq/model/bsz_2_lr_1e-3_epoch_5_zh_code/checkpoints/epoch_3/gate.pt 5043 3 code "7b" 8 gate 1.0 >> result_parallel_infer/port_5043.log 2>&1 &
sleep 30

bash run_para.sh zh /data/public/multilingual/whq/model/bsz_2_lr_1e-3_epoch_5_zh_code/checkpoints/epoch_4/gate.pt 5044 4 code "7b" 8 gate 1.0 >> result_parallel_infer/port_5044.log 2>&1 &
sleep 30

wait
