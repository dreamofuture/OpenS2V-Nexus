#!/bin/bash

datasets=("dataset1")
# datasets=("dataset1" "dataset2")

for dataset in "${datasets[@]}"; do
    python step2_get_pure_person_clip.py \
        --input_json_folder demo_result/step1/final_output/${dataset} \
        --output_json_folder demo_result/step2/final_output/${dataset} \
        --num_workers 64
done