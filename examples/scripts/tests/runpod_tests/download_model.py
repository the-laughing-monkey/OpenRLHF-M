#!/usr/bin/env python3
import argparse
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(
        description="Download a large HuggingFace model repository to the cache, without loading it."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-72B-Instruct",
        help="The repository for the model to download (default: Qwen/Qwen2.5-VL-72B-Instruct)"
    )
    args = parser.parse_args()

    print(f"Downloading model repository for {args.model_name}...")
    # This will download the repository of the model into the HuggingFace cache directory
    snapshot_download(repo_id=args.model_name)
    print("Download complete. The model repository has been cached in the HuggingFace cache directory.")


if __name__ == '__main__':
    main()
