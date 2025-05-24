#!/bin/bash
#SBATCH --gpus=2

module load CUDA/12.2
module load anaconda3/2023.09
source activate easyr1
export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=0,1




python3 scripts/model_merger.py \
    --local_dir /HOME/paratera_xy/pxy368/HDD_POOL/yuting/EasyR1/checkpoints/easy_r1/math8k-augmentation-Episode15/global_step_5/actor
