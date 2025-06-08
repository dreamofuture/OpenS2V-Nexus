import os


def generate_bash_scripts(
    input_video_root,
    output_json_folder,
    input_cluster_json_folder,
    input_video_json_folder,
    num_gpus,
    batch_size,
    num_machines,
    output_scripts_dir,
):
    tasks_per_machine = num_gpus * batch_size

    if not os.path.exists(output_scripts_dir):
        os.makedirs(output_scripts_dir)

    for machine_num in range(num_machines):
        bash_script = ""
        bash_script += f"INPUT_VIDEO_ROOT={input_video_root}\n"
        bash_script += f"OUTPUT_JSON_FOLDER={output_json_folder}\n"
        bash_script += f"INPUT_CLUSTER_JSON_FOLDER={input_cluster_json_folder}\n"
        bash_script += f"INPUT_VIDEO_JSON_FOLDER={input_video_json_folder}\n\n"

        start_task = machine_num * tasks_per_machine
        end_task = (machine_num + 1) * tasks_per_machine - 1
        bash_script += f"for i in {{{start_task}..{end_task}}}; do\n"

        bash_script += (
            '    INPUT_CLUSTER_JSON="${INPUT_CLUSTER_JSON_FOLDER}/' + 'cluster_videos_part$((i+1)).json"\n'
        )
        bash_script += f"    CUDA_VISIBLE_DEVICES=$((i % {num_gpus})) python step6-2_get_cross-frame.py --input_video_root ${{INPUT_VIDEO_ROOT}} --input_cluster_json ${{INPUT_CLUSTER_JSON}} --input_video_json_folder ${{INPUT_VIDEO_JSON_FOLDER}} --output_json_folder ${{OUTPUT_JSON_FOLDER}} & \\\n"

        bash_script += "done\n"
        bash_script += "wait\n\n\n\n"

        script_filename = f"{output_scripts_dir}/{machine_num + 1}.sh"
        with open(script_filename, "w") as f:
            f.write(bash_script)

        print(f"Script generated at: {script_filename}")


input_video_root = "demo_result/step0/videos"
input_cluster_json_folder = "demo_result/step6/cross-frames-images/chunk_input_json"
input_video_json_folder = "demo_result/step5/merge_final_json"
output_json_folder = "demo_result/step6/cross-frames-images/final_output/"

output_scripts_dir = "run"
num_gpus = 1
batch_size = 1
num_machines = 1


generate_bash_scripts(
    input_video_root,
    output_json_folder,
    input_cluster_json_folder,
    input_video_json_folder,
    num_gpus,
    batch_size,
    num_machines,
    output_scripts_dir,
)
