#!/bin/bash
#SBATCH --gpus=2

module load CUDA/12.2
module load anaconda3/2023.09
source activate mllm

python step-1.py \
--dataset_path /HOME/paratera_xy/pxy368/HDD_POOL/yuting/datasets/geoqa-r1v-noise/geoqa-r1v-noise.json \
--generation_model_path /HOME/paratera_xy/pxy368/HDD_POOL/CODE/models/Qwen2.5-VL-7B-Instruct \
--judgment_model_path /HOME/paratera_xy/pxy368/HDD_POOL/yuting/models/Qwen2.5-32B-Instruct \
--out_dir /HOME/paratera_xy/pxy368/HDD_POOL/yuting/sample \
--dataset geoqa-r1v-noise \
--model qwen2.5vl-7b-sample16 \
--num_samples 16

python step-2.py \
--dataset_path /HOME/paratera_xy/pxy368/HDD_POOL/yuting/sample/geoqa-r1v-noise/qwen2.5vl-7b-sample16_geoqa-r1v-noise-rejection_sampling.json \
--model_path /HOME/paratera_xy/pxy368/HDD_POOL/yuting/models/Qwen2.5-32B-Instruct \
--out_dir /HOME/paratera_xy/pxy368/HDD_POOL/yuting/sample \
--dataset geoqa-r1v-noise \

python step-3.py \
--dataset_path /HOME/paratera_xy/pxy368/HDD_POOL/datasets/K12/K12-formatted.json \
--generation_model_path /HOME/paratera_xy/pxy368/HDD_POOL/CODE/models/Qwen2.5-VL-7B-Instruct \
--judgment_model_path /HOME/paratera_xy/pxy368/HDD_POOL/yuting/models/Qwen2.5-32B-Instruct \
--out_dir /HOME/paratera_xy/pxy368/HDD_POOL/yuting/sample \
--dataset geoqa-r1v-noise \
--model qwen2.5vl-7b-sample16 \
--num_samples 16

