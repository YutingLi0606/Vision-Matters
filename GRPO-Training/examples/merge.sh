#!/bin/bash

export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=0,1


python3 scripts/model_merger.py \
    --local_dir /DATA/GRPO-Training/checkpoints/easy_r1/math8k-augmentation-Episode15/global_step_5/actor
