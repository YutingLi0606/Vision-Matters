#!/bin/bash

python step-1.py \
--dataset_path /DATA/datasets/geoqa-r1v-noise/geoqa-r1v-noise.json \
--generation_model_path /DATA/models/Qwen2.5-VL-7B-Instruct \
--judgment_model_path /DATA/models/Qwen2.5-32B-Instruct \
--out_dir /DATA/Rejection-Sampling \
--dataset geoqa-r1v-noise \
--model qwen2.5vl-7b-sample16 \
--num_samples 16

python step-2.py \
--dataset_path /DATA/Rejection-Sampling/geoqa-r1v-noise/qwen2.5vl-7b-sample16_geoqa-r1v-noise-rejection_sampling.json \
--model_path /DATA/models/Qwen2.5-32B-Instruct \
--out_dir /DATA/Rejection-Sampling \
--dataset geoqa-r1v-noise \

python step-3.py \
--dataset_path /DATA/datasets/K12/K12-formatted.json \
--generation_model_path /DATA/models/Qwen2.5-VL-7B-Instruct \
--judgment_model_path /DATA/models/Qwen2.5-32B-Instruct \
--out_dir /DATA/Rejection-Sampling \
--dataset geoqa-r1v-noise \
--model qwen2.5vl-7b-sample16 \
--num_samples 16

