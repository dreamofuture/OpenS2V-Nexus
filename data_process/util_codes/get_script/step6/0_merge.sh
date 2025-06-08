#!/bin/bash

datasets=("dataset1")
# datasets=("dataset1" "dataset2")

for dataset in "${datasets[@]}"; do
    python step6-0_merge_json.py \
        --input_json_folder demo_result/step5/final_output/${dataset} \
        --output_json_file demo_result/step5/merge_final_json/${dataset}.json \
        --num_workers 64
done