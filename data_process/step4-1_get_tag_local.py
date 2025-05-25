from torch.utils.data import DataLoader, Dataset

import argparse
import json
import os
import re

from tqdm import tqdm
from vllm import LLM, SamplingParams

os.environ["TOKENIZERS_PARALLELISM"] = "false"

prompt_format = (
    "Given an image caption, please retrieve the entity words that indicate background, subject, and visually separable objects. "
    "[Definition of background] The background spaces that appear in most of the image area. "
    "[Definition of subject] Human or animal subjects that appear in the image. "
    "[Definition of object] Entities that are visually separable, tangible, and physically present in part of the image. "
    "Attention! All entity words need to strictly follow the rules below: "
    "1) The entity word is a singular or plural noun without any quantifier or descriptive phrase. "
    "2) The entity word must be an exact subset of the caption, including its characters, words, and symbols. (e.g, 'red top' better than 'top', 'martial arts uniforms' better than 'uniforms') "
    "3) Exclude any part of the body (e.g., 'hands', 'legs', 'feet', 'head'). "
    "4) Exclude abstract or non-physical concepts (e.g., 'facial expressions', 'gestures', 'stance'). "
    "5) Exclude actions or descriptions (e.g., 'adjusting', 'imitating'). "
    "6) Do not modify or interpret any part of the caption. "
    "Here is an example, follow this format to output the results: "
    "Caption: A woman in a mask and coat, with long brown hair, shows a small green-capped bottle to the camera. "
    "Output: {'background': [''], 'subject': ['woman'], 'object': ['mask', 'coat', 'long brown hair', 'green-capped bottle']} "
    "Here is the input: "
    "Caption: {{{}}} "
    "Output: "
)

keywords_to_remove = [
    "person",
    "people",
    "human",
    "individual",
    "individuals",
    "woman",
    "man",
    "child",
    "children",
    "adult",
    "teenager",
    "infant",
    "youth",
    "elder",
    "boy",
    "girl",
    "male",
    "female",
]


def filter_keywords(items, keywords):
    return [item for item in items if item.lower() not in keywords]


def extract_data_from_response(response):
    response = response.replace('"', "'")

    background_pattern = r"'background'\s*:\s*\[(.*?)\]"
    subject_pattern = r"'subject'\s*:\s*\[(.*?)\]"
    object_pattern = r"'object'\s*:\s*\[(.*?)\]"

    background_match = re.search(background_pattern, response)
    subject_match = re.search(subject_pattern, response)
    object_match = re.search(object_pattern, response)

    result = {
        "pre_define": ["person", "human head"],
        "background": [],
        "subject": [],
        "object": [],
    }

    if background_match:
        background_items = [
            item.strip("' ") for item in background_match.group(1).split("', '")
        ]
        # result["background"] = filter_keywords(background_items, keywords_to_remove)
        result["background"] = background_items

    if subject_match:
        subject_items = [
            item.strip("' ") for item in subject_match.group(1).split("', '")
        ]
        # result["subject"] = filter_keywords(subject_items, keywords_to_remove)
        result["subject"] = subject_items

    if object_match:
        object_items = [
            item.strip("' ") for item in object_match.group(1).split("', '")
        ]
        # result["object"] = filter_keywords(object_items, keywords_to_remove)
        result["object"] = object_items

    if background_match and subject_match and object_match:
        flag = True
    else:
        flag = False

    return result, flag


class CaptionData(Dataset):
    def __init__(self, video_data, output_json_folder):
        super().__init__()
        self.output_json_folder = output_json_folder
        video_keys = [i["video_key"] for i in video_data]
        caps = [i["cap"] for i in video_data]
        save_paths = [
            os.path.join(output_json_folder, (i["video_key"] + "_step4.json"))
            for i in video_data
        ]
        print("part x origin num", len(save_paths))
        self.paths = [[i, c, m] for i, c, m in zip(save_paths, caps, video_keys)]
        print("part x need to process num", len(self.paths))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        try:
            save_path, inputs, video_key = self.paths[index]
            return save_path, inputs, video_key
        except Exception as e:
            print("error", e)
            return False, False, False


def collate_fn(batch):
    save_paths, inputs, video_key = zip(*batch)
    return save_paths, inputs, video_key


def split_list(data, nums, part):
    items = list(data.items())
    size = len(items)
    part_size = size // nums
    remainder = size % nums

    start = part * part_size + min(part, remainder)
    end = start + part_size + (1 if part < remainder else 0)

    return dict(items[start:end])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id_or_path",
        type=str,
        default="Qwen/Qwen3-32B",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--input_video_json",
        type=str,
        default="demo_result/step3/merge_final_json/dataset1.json",
    )
    parser.add_argument(
        "--output_json_folder",
        type=str,
        default="demo_result/step4/final_output/dataset1",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--total_part", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    model_id_or_path = args.model_id_or_path
    input_video_json = args.input_video_json
    output_json_folder = args.output_json_folder
    batch_size = args.batch_size
    num_workers = args.num_workers

    os.makedirs(output_json_folder, exist_ok=True)

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=512,
    )

    llm = LLM(
        model_id_or_path,
        # max_model_len=32768 if process_vision_info is None else 4096,
        # tensor_parallel_size=4,
        # distributed_executor_backend="mp",
    )

    with open(input_video_json, "r") as f:
        origin_video_data = json.load(f)
    print("total data", len(origin_video_data))

    video_data = [
        {
            "video_key": key,
            "cap": prompt_format.replace(
                "{{{}}}",
                (
                    i["metadata"].get("face_cap_glm", None)
                    or i["metadata"].get("face_cap_Aria", None)
                    or i["metadata"].get("face_cap_qwen", None)
                ).strip(),
            )
            .replace("\n", "")
            .replace("\t", "")
            .strip(),
        }
        for key, i in tqdm(origin_video_data.items())
    ]

    print("after filter data", len(video_data))
    video_data = video_data[args.part :: args.total_part]
    data = CaptionData(video_data, output_json_folder)
    loader = DataLoader(
        data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    for save_paths, prompts, video_keys in tqdm(loader):
        unprocess_save_paths = []
        unprocess_prompts = []
        unprocess_video_keys = []
        for i, save_path in enumerate(save_paths):
            if not os.path.exists(save_path):
                unprocess_save_paths.append(save_path)
                unprocess_prompts.append(prompts[i])
                unprocess_video_keys.append(video_keys[i])
            else:
                print(f"{save_path} is already exists")
        save_paths = unprocess_save_paths
        prompts = unprocess_prompts
        video_keys = unprocess_video_keys

        outputs = llm.generate(prompts, sampling_params)
        for output, save_path, video_key in zip(outputs, save_paths, video_keys):
            response = output.outputs[0].text

            try:
                response_data = json.loads(response)
            except json.JSONDecodeError:
                response_data, flag = extract_data_from_response(response)

            if video_key in origin_video_data:
                value = origin_video_data[video_key]

                value["word_tags"] = response_data

                with open(save_path, "w") as f:
                    json.dump(value, f, indent=4)
            else:
                print(f"[Warning] video_key {video_key} not found in origin_video_data")

    print("Done")


if __name__ == "__main__":
    main()
