#!/bin/bash

run_name=${1:-temp}

output_dir=Exps/${run_name}
mkdir -p $output_dir
cp "$0" ${output_dir}/$(date +"%Y-%m-%d-%H-%M-%S").sh

torchrun --nproc_per_node=1 --master_port=1688 ./finetune_val.py \
    --base_model '../text2hpc/pretrained_models/Llama-3.1-8B-Instruct' \
    --data_path 'librilight_colm.jsonl' \
    --data_val_path 'none' \
    --output_dir $output_dir \
    --disable_tqdm False \
    --batch_size 8 \
    --micro_batch_size 1 \
    --num_epochs 1 \
    --warmup_ratio 0.1 \
    --logging_steps 1 \
    --eval_steps 100 \
    --save_steps 10 \
    --save_start_steps 10 \
    --save_total_limit 500 \
    --learning_rate 1e-4 \
    --cutoff_len 4500 \
    --val_set_size 100 \
    --num_proc 8 \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]' \
    --train_on_inputs False \
    --group_by_length True \
    --wandb_project '' \
    --wandb_run_name ${run_name}

pkill -f wandb