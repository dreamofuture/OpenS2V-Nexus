import os
import json
import numpy as np
from PIL import Image
import argparse
from utils.gme.gme_model import GmeQwen2VL

import decord

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def sample_video_frames(video_path, num_frames=16):
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)

    if num_frames is None:
        frame_indices = np.arange(total_frames)
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        frame = vr[int(idx)].asnumpy()
        pil_image = Image.fromarray(frame)
        frames.append(pil_image)

    return frames


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_video_folder",
        type=str,
        default="demo_result/model_name_input_video",
    )
    parser.add_argument(
        "--input_json_file",
        type=str,
        default="demo_result/input_json/Open-Domain_Eval.json",
    )
    parser.add_argument(
        "--output_json_folder",
        type=str,
        default="demo_result/model_name_output_json",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Alibaba-NLP/gme-Qwen2-VL-7B-Instruct/",
    )
    parser.add_argument("--num_frames", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda"

    input_video_folder = args.input_video_folder
    input_json_file = args.input_json_file
    output_json_folder = args.output_json_folder
    model_path = args.model_path
    num_frames = args.num_frames

    output_json_file = os.path.join(output_json_folder, "gmescore.json")
    os.makedirs(output_json_folder, exist_ok=True)

    with open(input_json_file, "r") as f:
        prompts = json.load(f)
    gme = GmeQwen2VL(model_path, attn_model="flash_attention_2", device=device)

    results_dict = {}
    for video_filename in os.listdir(input_video_folder):
        if video_filename.endswith(".mp4"):
            video_name = os.path.splitext(video_filename)[0]

            if video_name in prompts.keys():
                text_prompt = prompts[video_name].get("prompt", "")
            else:
                print(f"No prompt found for video: {video_name}. Skipping.")
                continue

            video_path = os.path.join(input_video_folder, video_filename)
            frames = sample_video_frames(video_path, num_frames=num_frames)

            e_query = gme.get_text_embeddings(
                texts=[text_prompt] * len(frames),
                instruction="Find an image that matches the given text.",
            )
            e_corpus = gme.get_image_embeddings(
                images=frames, is_query=False, show_progress_bar=False
            )
            gme_scores = (e_query * e_corpus).sum(-1)
            gme_score = gme_scores.mean().detach().item()

            results_dict[video_name] = {"gme_score": gme_score}

    with open(output_json_file, "w") as results_file:
        json.dump(results_dict, results_file, indent=4)

    print(f"All results have been saved to {output_json_file}")


if __name__ == "__main__":
    main()
