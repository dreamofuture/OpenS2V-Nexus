#!/bin/bash
python chunk_json/step6_chunk_json.py \
    --input_video_json demo_result/step6/cross-frames-images/cluster_videos.json \
    --output_chunk_json_folder demo_result/step6/cross-frames-images/chunk_input_json \
    --resume_dir demo_result/step6/cross-frames-images/final_output \
    --total_part 1 \
    --num_workers 64
