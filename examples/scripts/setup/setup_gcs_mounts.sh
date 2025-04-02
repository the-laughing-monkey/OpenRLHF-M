#!/bin/bash
# Setup model caching with GCS bucket
GCS_BUCKET="${GCS_BUCKET:-gs://[YOUR_BUCKET]}"
GCS_BUCKET_NAME=$(echo ${GCS_BUCKET} | sed 's|gs://||')

echo "Setting up GCS mount and HuggingFace cache symlinks..."
# Mount GCS bucket if not already mounted
if ! grep -q "/mnt/gcs-cache" /proc/mounts; then
    echo "Mounting GCS bucket ${GCS_BUCKET} to /mnt/gcs-cache"
    if ! gcsfuse --implicit-dirs ${GCS_BUCKET_NAME} /mnt/gcs-cache; then
        echo "Error: Failed to mount GCS bucket. Please ensure the container is running with proper privileges (e.g., using --privileged or --cap-add SYS_ADMIN with /dev/fuse access)."
        exit 1
    fi
else
    echo "GCS bucket already mounted at /mnt/gcs-cache"
fi

# Create directories for model cache if they don't exist in GCS
mkdir -p /mnt/gcs-cache/model-cache/huggingface
mkdir -p /mnt/gcs-cache/model-cache/ray

# Create symlinks for huggingface and ray caches
echo "Creating symlinks for model caches"
rm -rf ~/.cache/huggingface
ln -sf /mnt/gcs-cache/model-cache/huggingface ~/.cache/huggingface
rm -rf ~/.cache/ray
ln -sf /mnt/gcs-cache/model-cache/ray ~/.cache/ray

# Create and set up dataset directories
echo "Setting up dataset directories"
mkdir -p /mnt/gcs-cache/datasets
mkdir -p /app/datasets

# Create symlink for datasets
ln -sf /mnt/gcs-cache/datasets /app/datasets/VerMulti

# Ensure the model checkpoint directory exists
mkdir -p /mnt/gcs-cache/checkpoints