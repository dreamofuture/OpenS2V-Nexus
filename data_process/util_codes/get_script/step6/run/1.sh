INPUT_VIDEO_ROOT=/mnt/workspace/ysh/Code/ConsisID-X/1_main_codes/upload/OpenS2V-Nexus-main/data_process/demo_result/step0/videos
OUTPUT_JSON_FOLDER=/mnt/workspace/ysh/Code/ConsisID-X/1_main_codes/upload/OpenS2V-Nexus-main/data_process/demo_result/step6/cross-frames-pairs/final_output/
INPUT_CLUSTER_JSON_FOLDER=/mnt/workspace/ysh/Code/ConsisID-X/1_main_codes/upload/OpenS2V-Nexus-main/data_process/demo_result/step6/cross-frames-pairs/chunk_input_json
INPUT_VIDEO_JSON_FOLDER=/mnt/workspace/ysh/Code/ConsisID-X/1_main_codes/upload/OpenS2V-Nexus-main/data_process/demo_result/step5/merge_final_json

for i in {0..0}; do
    INPUT_CLUSTER_JSON="${INPUT_CLUSTER_JSON_FOLDER}/cluster_videos_part$((i+1)).json"
    CUDA_VISIBLE_DEVICES=$((i % 1)) python step6-2_get_cross-frame.py --input_video_root ${INPUT_VIDEO_ROOT} --input_cluster_json ${INPUT_CLUSTER_JSON} --input_video_json_folder ${INPUT_VIDEO_JSON_FOLDER} --output_json_folder ${OUTPUT_JSON_FOLDER} & \
done
wait



