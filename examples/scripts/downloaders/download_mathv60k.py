#!/usr/bin/env python3
"""
Download and prepare the mathv60k dataset, replacing paths with a configurable root directory.
Usage: python download_mathv60k.py [--root_dir /path/to/dataset/root]
"""

import argparse
import os
import shutil
import tarfile
from huggingface_hub import hf_hub_download

print("=================================================================")
print("MathV60K Dataset Downloader and Preparation Tool")
print("=================================================================")

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Download and prepare mathv60k dataset')
parser.add_argument('--root_dir', type=str, default="/data/datasets/VerMulti",
                    help='Root directory for storing the dataset')
args = parser.parse_args()

# Set root directory and ensure it exists
DIR_ROOT = args.root_dir
print(f"[1/5] Creating directory: {DIR_ROOT}")
os.makedirs(DIR_ROOT, exist_ok=True)
print(f"      Using root directory: {DIR_ROOT}")

# Download files from Hugging Face
print(f"[2/5] Downloading files from Hugging Face...")
print(f"      Downloading JSONL file (mathv60k_message.jsonl)...")
jsonl_path = hf_hub_download(repo_id="VLM-Reasoner/VerMulti",
                           filename="mathv60k_message.jsonl", repo_type="dataset")
print(f"      Downloading image archive (mathv60k_img.tar.gz)...")
tar_path = hf_hub_download(repo_id="VLM-Reasoner/VerMulti",
                         filename="mathv60k_img.tar.gz", repo_type="dataset")
print(f"      Downloaded JSONL to: {jsonl_path}")
print(f"      Downloaded tar.gz to: {tar_path}")

# Extract images
print(f"[3/5] Extracting images to {DIR_ROOT}...")
print(f"      This may take a while depending on the archive size...")
with tarfile.open(tar_path, "r:gz") as tar:
    tar.extractall(path=DIR_ROOT)
print(f"      Extraction complete!")

# Process JSONL file to replace paths
original_path_prefix = "/apdcephfs_gy2/share_302735770/berlinni/zgr/data/mathv60k/"
output_jsonl_path = os.path.join(DIR_ROOT, "mathv60k_message.jsonl")

print(f"[4/5] Processing JSONL file to replace paths...")
print(f"      Replacing: {original_path_prefix}")
print(f"      With:      {DIR_ROOT}/")
line_count = 0
with open(jsonl_path, 'r') as f_in, open(output_jsonl_path, 'w') as f_out:
    for line in f_in:
        # Replace all instances of the original path with the new root directory
        modified_line = line.replace(original_path_prefix, f"{DIR_ROOT}/")
        f_out.write(modified_line)
        line_count += 1
        if line_count % 10000 == 0:
            print(f"      Processed {line_count} lines...")
print(f"      Processed a total of {line_count} lines")

# Copy the tar file for reference
output_tar_path = os.path.join(DIR_ROOT, "mathv60k_img.tar.gz")
print(f"[5/5] Copying original tar archive to destination...")
shutil.copy(tar_path, output_tar_path)

print("=================================================================")
print(f"Dataset preparation complete!")
print(f"Files saved to {DIR_ROOT}")
print(f"- JSONL file: {output_jsonl_path}")
print(f"- Original tar archive: {output_tar_path}")
print(f"- Extracted images in: {DIR_ROOT}")
print(f"To use this dataset with OpenRLHF-M, set:")
print(f"  DATASET_PATH={output_jsonl_path}")
print("=================================================================")