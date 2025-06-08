#!/bin/bash

datasets=("dataset1")
# datasets=("dataset1" "dataset2")

for dataset in "${datasets[@]}"; do
    python chunk_json/step3_chunk_json.py \
        --input_video_json demo_result/step2/merge_final_json/${dataset}.json \
        --output_chunk_json_folder demo_result/step3/chunk_input_json/${dataset} \
        --resume_dir demo_result/step3/final_output/${dataset} \
        --total_part 1 \
        --num_workers 64
done