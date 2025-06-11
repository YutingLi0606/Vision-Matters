#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
MAX_PIXELS=1204224 \
swift rlhf \
    --rlhf_type dpo \
    --beta 0.1 \
    --rpo_alpha 0.1 \
    --model_type qwen2_5_vl \
    --model /DATA/SFT-DPO-Training/ms-swift/run/math8k_dpo/v0-20250412-135136/checkpoint-70-merged \
    --dataset /DATA/Rejection-Sampling/math-8k-concat/0_25.json /DATA/Rejection-Sampling/math-8k-mixup/0_25.json /DATA/Rejection-Sampling/math-8k-rotated/0_25.json \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps $(expr 16 / 4) \
    --eval_steps 5 \
    --save_steps 5 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 16384 \
    --warmup_ratio 0.05 \
    --output_dir math-8k_dpo \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --deepspeed zero2 \



