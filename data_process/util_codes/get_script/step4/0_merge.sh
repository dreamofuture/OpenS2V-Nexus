#!/bin/bash

datasets=("dataset1")
# datasets=("dataset1" "dataset2")

for dataset in "${datasets[@]}"; do
    python step4-0_merge_json.py \
        --input_json_folder demo_result/step3/final_output/${dataset} \
        --output_json_file demo_result/step3/merge_final_json/${dataset}.json \
        --num_workers 64
done