#!/bin/bash

datasets=("dataset1")
# datasets=("dataset1" "dataset2")

for dataset in "${datasets[@]}"; do
    python chunk_json/step1_chunk_json.py \
        --input_video_json demo_result/step0/merge_final_json/${dataset}.json \
        --output_chunk_json_folder step1/chunk_input_json/${dataset} \
        --resume_dir step1/final_output/${dataset} \
        --total_part 1 \
        --num_workers 64
done