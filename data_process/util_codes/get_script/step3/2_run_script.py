import os


def generate_bash_scripts(
    input_video_root,
    output_json_folder,
    input_video_json_dir,
    num_gpus,
    batch_size,
    num_machines,
    dataset_names,
    output_scripts_dir,
):
    tasks_per_machine = num_gpus * batch_size

    if not os.path.exists(output_scripts_dir):
        os.makedirs(output_scripts_dir)

    for machine_num in range(num_machines):
        bash_script = ""
        for dataset_name in dataset_names:
            bash_script += f"INPUT_VIDEO_ROOT={os.path.join(input_video_root, dataset_name)}\n"
            bash_script += f"OUTPUT_JSON_FOLDER={os.path.join(output_json_folder, dataset_name)}\n"
            bash_script += f"INPUT_VIDEO_JSON_DIR={os.path.join(input_video_json_dir, dataset_name)}\n\n"

            start_task = machine_num * tasks_per_machine
            end_task = (machine_num + 1) * tasks_per_machine - 1
            bash_script += f"for i in {{{start_task}..{end_task}}}; do\n"

            bash_script += (
                '    INPUT_VIDEO_JSON="${INPUT_VIDEO_JSON_DIR}/' + f"{dataset_name}" + '_part$((i+1)).json"\n'
            )
            bash_script += f"    CUDA_VISIBLE_DEVICES=$((i % {num_gpus})) python step3-1_get_caption.py --input_video_root ${{INPUT_VIDEO_ROOT}} --input_video_json ${{INPUT_VIDEO_JSON}} --output_json_folder ${{OUTPUT_JSON_FOLDER}} & \\\n"

            bash_script += "done\n"
            bash_script += "wait\n\n\n\n"

        script_filename = f"{output_scripts_dir}/{machine_num + 1}.sh"
        with open(script_filename, "w") as f:
            f.write(bash_script)

        print(f"Script generated at: {script_filename}")


dataset_names = ["dataset1"]
input_video_root = "demo_result/step0/videos"
input_video_json_dir = "demo_result/step3/chunk_input_json"
output_json_folder = "demo_result/step3/final_output"

output_scripts_dir = "run"
num_gpus = 8
batch_size = 1
num_machines = 3


generate_bash_scripts(
    input_video_root,
    output_json_folder,
    input_video_json_dir,
    num_gpus,
    batch_size,
    num_machines,
    dataset_names,
    output_scripts_dir,
)
