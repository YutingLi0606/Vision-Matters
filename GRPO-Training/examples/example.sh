#!/bin/bash
#SBATCH --gpus=8

module load CUDA/12.2
module load anaconda3/2023.09
source activate easyr1

export PYTHONUNBUFFERED=1

set -x

MODEL_PATH=/HOME/paratera_xy/pxy368/HDD_POOL/CODE/models/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

FORMAT_PROMPT="Please reason step by step, and put your final answer within \\boxed{}."

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/HOME/paratera_xy/pxy368/HDD_POOL/yuting/datasets/geoqa-r1v-blur@train \
    data.val_files=/HOME/paratera_xy/pxy368/HDD_POOL/yuting/huggingface/geometry3k@test \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.enable_chunked_prefill=false \
    trainer.logger=['tensorboard'] \
    trainer.experiment_name=geoqa-r1v-blur-Episode15 \
    trainer.n_gpus_per_node=8 \
    trainer.total_episodes=15 \
    trainer.save_limit=50 \
    data.max_pixels=1204224 
