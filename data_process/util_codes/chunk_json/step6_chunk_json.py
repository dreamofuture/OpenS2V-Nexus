import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


def should_include_item(item, resume_dir):
    key, value = item
    file_name = key + "_step6.json"
    file_path = os.path.join(resume_dir, file_name)
    if not os.path.exists(file_path):
        return key, value
    return None


def write_chunk_to_file(chunk, output_file):
    with open(output_file, "w") as f:
        json.dump(chunk, f, indent=2)
    return output_file


def split_json_file(
    input_video_json,
    output_chunk_json_folder,
    total_part,
    resume_dir,
    num_workers,
    size_estimate_fn=None,
):
    os.makedirs(output_chunk_json_folder, exist_ok=True)

    with open(input_video_json, "r") as f:
        json_data = json.load(f)

    print(f"{input_video_json} Total items before: {len(json_data)}")

    if not isinstance(json_data, (list, dict)):
        raise ValueError("JSON data must be an array or object")

    data = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = executor.map(
            lambda item: should_include_item(item, resume_dir),
            json_data.items() if isinstance(json_data, dict) else enumerate(json_data),
        )
        for result in futures:
            if result:
                key, value = result
                data[key] = value

    items = list(data.items())
    print(f"{input_video_json} Total items after: {len(items)}")

    # Default size estimator if not provided
    if size_estimate_fn is None:

        def size_estimate_fn(item):
            # Simple size estimation by converting to JSON string
            return len(json.dumps(item[1]))

    # Estimate sizes for all items
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        _ = list(executor.map(lambda item: (item[0], size_estimate_fn(item)), items))

    # Sort items by descending size for bin packing
    sorted_items = sorted(items, key=lambda x: size_estimate_fn(x), reverse=True)
    sorted_sizes = [size_estimate_fn(item) for item in sorted_items]

    # Bin packing algorithm to distribute items
    bins = [[] for _ in range(total_part)]
    bin_sizes = [0] * total_part

    for item, size in zip(sorted_items, sorted_sizes):
        # Find bin with smallest current size
        min_bin = min(range(total_part), key=lambda i: bin_sizes[i])
        bins[min_bin].append(item)
        bin_sizes[min_bin] += size

    # Sort items within each part from shortest to longest
    for i in range(total_part):
        bins[i].sort(key=lambda x: size_estimate_fn(x))

    # Calculate statistics for debugging
    avg_size = sum(bin_sizes) / total_part
    size_variance = sum((s - avg_size) ** 2 for s in bin_sizes) / total_part
    print(
        f"Part size stats - Avg: {avg_size:.2f}, Min: {min(bin_sizes)}, Max: {max(bin_sizes)}, Variance: {size_variance:.2f}"
    )

    # Write chunks in parallel
    base_name = os.path.splitext(os.path.basename(input_video_json))[0]
    tasks = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for i, chunk in enumerate(bins):
            if not chunk:
                continue

            output_file = os.path.join(
                output_chunk_json_folder, f"{base_name}_part{i + 1}.json"
            )

            # Convert to original data type
            chunk_data = (
                dict(chunk) if isinstance(data, dict) else [item[1] for item in chunk]
            )

            tasks.append(executor.submit(write_chunk_to_file, chunk_data, output_file))

        for future in as_completed(tasks):
            future.result()


def main():
    parser = argparse.ArgumentParser(
        description="Split a large JSON file into smaller chunks."
    )
    parser.add_argument(
        "--input_video_json",
        type=str,
        default="../../demo_result/step6/cross-frames-images/cluster_videos.json",
    )
    parser.add_argument(
        "--output_chunk_json_folder",
        type=str,
        default="../../demo_result/step6/cross-frames-images/chunk_input_json",
    )
    parser.add_argument(
        "--resume_dir",
        type=str,
        default="../../demo_result/step6/cross-frames-images/final_output",
    )
    parser.add_argument("--total_part", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=64)

    args = parser.parse_args()

    os.makedirs(args.output_chunk_json_folder, exist_ok=True)
    os.makedirs(args.resume_dir, exist_ok=True)

    split_json_file(
        args.input_video_json,
        args.output_chunk_json_folder,
        args.total_part,
        args.resume_dir,
        args.num_workers,
    )


if __name__ == "__main__":
    main()
