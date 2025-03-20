#!/usr/bin/env python3
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(
        description="Download a large HuggingFace model and its tokenizer, caching them automatically."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-72B-Instruct",
        help="The name or path of the model to download (default: Qwen/Qwen2.5-VL-72B-Instruct)"
    )
    args = parser.parse_args()

    print(f"Downloading model and tokenizer for {args.model_name}...")
    # This will automatically download and cache the model in /root/.cache/huggingface
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    print("Download complete. The model and tokenizer have been cached in the HuggingFace cache directory.")


if __name__ == '__main__':
    main()
