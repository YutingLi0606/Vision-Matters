#!/bin/bash

export MASTER_PORT=29507
export MKL_SERVICE_FORCE_INTEL=1


models=(
    "/DATA/GRPO-Training/checkpoints/easy_r1/geoqa-blur-Episode15/global_step_95/actor/huggingface"
    "/DATA/GRPO-Training/checkpoints/easy_r1/geoqa-blur-Episode15/global_step_90/actor/huggingface"
    "/DATA/GRPO-Training/checkpoints/easy_r1/geoqa-blur-Episode15/global_step_85/actor/huggingface"
    "/DATA/GRPO-Training/checkpoints/easy_r1/geoqa-blur-Episode15/global_step_80/actor/huggingface"
)   

models_names=(
    "geoqa-blur-95"
    "geoqa-blur-90"
    "geoqa-blur-85"
    "geoqa-blur-80"
)

benchmarks=("mathvision" "mathverse" "mathvista" "wemath")

for i in "${!models[@]}"; do
    model_path="${models[i]}"
    model_name="${models_names[i]}"
    
    for benchmark in "${benchmarks[@]}"; do
        echo "Running inference for model: $model_name on benchmark: $benchmark"
        CUDA_VISIBLE_DEVICES=0,1 python inference.py --benchmark "$benchmark" --model "$model_name" --model_path "$model_path"
        
        echo "Running judge for model: $model_name on benchmark: $benchmark"
        CUDA_VISIBLE_DEVICES=0,1 python judge.py --benchmark "$benchmark" --model "$model_name"

        echo "=========="
        echo "$model_name"
    done
done
