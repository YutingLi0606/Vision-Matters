#!/bin/bash



MAX_PIXELS=1204224 CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 swift sft \
  --model_type qwen2_5_vl \
  --model /DATA/SFT-DPO-Training/ms-swift/run/output/v0-20250401-215652/checkpoint-130-merged \
  --train_type lora \
  --dataset /DATA/Rejection-Sampling/M3CoT-concat/0_25.json /DATA/Rejection-Sampling/M3CoT-mixup/0_25.json /DATA/yuting/sample/M3CoT-mixup-0-5/0_25.json /DATA/Rejection-Sampling/M3CoT-rotated/0_25.json \
  --num_train_epochs 3 \
  --gradient_accumulation_steps 16 \
  --eval_steps 10 \
  --save_steps 10 \
  --save_total_limit 2 \
  --output_dir m3cot-concat \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --max_length 2048 \

