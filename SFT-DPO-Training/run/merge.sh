#!/bin/bash


paths=(
    "/DATA/SFT-DPO-Training/ms-swift/run/math8k-concat/v1-20250511-170926/checkpoint-207"
    "/DATA/SFT-DPO-Training/ms-swift/run/math8k-concat/v2-20250511-182015/checkpoint-70"
)

for path in "${paths[@]}"; do
    echo "Processing: $path"
    CUDA_VISIBLE_DEVICES=0 swift export \
        --adapters "$path" \
        --merge_lora true
done