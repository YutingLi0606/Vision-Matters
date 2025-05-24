#!/bin/bash
#SBATCH --gpus=2

module load CUDA/12.2
module load anaconda3/2023.09
source  activate /HOME/paratera_xy/pxy368/.conda/envs/lyt


model_info=(
   
    "math8k-Episode15-5:/HOME/paratera_xy/pxy368/HDD_POOL/yuting/EasyR1/checkpoints/easy_r1/math8k-Episode15/global_step_5/actor/huggingface"
    "math8k-Episode15-10:/HOME/paratera_xy/pxy368/HDD_POOL/yuting/EasyR1/checkpoints/easy_r1/math8k-Episode15/global_step_10/actor/huggingface"
    "math8k-Episode15-15:/HOME/paratera_xy/pxy368/HDD_POOL/yuting/EasyR1/checkpoints/easy_r1/math8k-Episode15/global_step_15/actor/huggingface"
)

benchmarks=("mathvision" "mathvista" "mathverse" "wemath")


for model_entry in "${model_info[@]}"; do
    IFS=':' read -r model_name model_path <<< "$model_entry"
    for benchmark in "${benchmarks[@]}"; do
        echo "Running TQA model: $model_name on $benchmark"
        python simple_inference.py --benchmark "$benchmark" --model "$model_name" --model_path "$model_path"
        python judge_qwen.py --benchmark "$benchmark" --model "$model_name"
    done
done