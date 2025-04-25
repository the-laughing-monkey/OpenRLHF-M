# Running OpenRLHF-M on GCP Vertex AI

This document provides comprehensive instructions for deploying and running OpenRLHF-M on GCP Vertex AI using a Docker container approach. This methodology ensures reproducible environments and follows GCP best practices for ML workloads.

## Overview

In this setup, we'll use:
- **Container Registry**: For storing our custom Docker image with all dependencies
- **Filestore (NFS)**: For high-performance access to datasets and model checkpoints
- **Cloud Storage**: Optionally used for initial data staging or backups (not for direct training access)
- **Vertex AI VMs**: for training
- **Ray**: For distributed multinode training

- **Prerequisites for running OpenRLHF-M:**
   - A Google Cloud Platform account with a project.
   - Billing enabled for your project.
   - Google Cloud SDK (gcloud) installed and configured on your local machine.
   - Docker installed on your local machine.
   - Sufficient quota for a2-ultragpu-2g instances in your preferred region.
   - A service account with the necessary permissions:
       • Vertex AI User (roles/aiplatform.user)
       • Storage Admin (roles/storage.admin)
       • Container Registry Writer (roles/containerregistry.writer)
       • Filestore Editor (roles/file.editor)

## Preliminary Steps

Before proceeding, ensure your local environment is properly set up:

1. Update the Google Cloud CLI:
```bash
   gcloud components update
```

2. Authenticate with Google Cloud:
```bash
   gcloud auth application-default login
```

3. Select or create a Google Cloud project:
   - List projects: gcloud projects list
   - Create a new project: gcloud projects create YOUR_PROJECT_ID
   - Set the active project: gcloud config set project YOUR_PROJECT_ID

4. Enable billing for your project:
   - Ensure that a billing account is attached to your project.
   - To attach a billing account:
       a. Open the Google Cloud Console and navigate to the Billing section.
       b. Select your project and choose to link or set up a billing account. For more details, see https://cloud.google.com/billing/docs/how-to/modify-project.

5. Now we'll install the various services we need for running OpenRLHF-M on GCP:
```bash
   gcloud services enable aiplatform.googleapis.com storage.googleapis.com containerregistry.googleapis.com compute.googleapis.com filestore.googleapis.com vpcaccess.googleapis.com
```


## Setup Process

### 1. Create a Filestore Instance and VPC Network

First, set up a VPC network (if you don't have one suitable) and a Filestore instance for shared file storage. Filestore requires a VPC network for access.

```bash
# Define variables (replace with your values)
export PROJECT_ID=[YOUR-PROJECT-ID]
export REGION=us-central1
export ZONE=us-central1-a
export NETWORK_NAME=openrlhf-vpc
export SUBNET_NAME=openrlhf-subnet
export FILESTORE_NAME=openrlhf-nfs
export FILE_SHARE_NAME=vol1 # Default share name for basic tier
export NFS_MOUNT_POINT=/mnt/nfs # Choose a mount point

# Create a VPC Network
gcloud compute networks create ${NETWORK_NAME} --project=${PROJECT_ID} --subnet-mode=custom --mtu=1460 --bgp-routing-mode=regional

# Create a Subnet within the VPC
gcloud compute networks subnets create ${SUBNET_NAME} --project=${PROJECT_ID} --network=${NETWORK_NAME} --region=${REGION} --range=10.0.0.0/24

# Create Firewall Rules for the VPC
# Allow internal traffic within the subnet (for Ray, NFS, etc.)
gcloud compute firewall-rules create ${NETWORK_NAME}-allow-internal \
    --project=${PROJECT_ID} \
    --network=projects/${PROJECT_ID}/global/networks/${NETWORK_NAME} \
    --direction=INGRESS \
    --priority=1000 \
    --allow=tcp,udp,icmp \
    --source-ranges=10.0.0.0/24 # Adjust if your subnet range is different

# Allow SSH (tcp:22) and ICMP (ping) from anywhere (restrict source range in production)
gcloud compute firewall-rules create ${NETWORK_NAME}-allow-ssh-icmp \
    --project=${PROJECT_ID} \
    --network=projects/${PROJECT_ID}/global/networks/${NETWORK_NAME} \
    --direction=INGRESS \
    --priority=1000 \
    --allow=tcp:22,icmp \
    --source-ranges=0.0.0.0/0 # <-- IMPORTANT: Restrict this IP range in production environments!

# Create a Filestore instance (Basic HDD tier, adjust tier and capacity as needed)
# Note: Filestore IP range must not overlap with the subnet range used for VMs.
# Filestore uses a /29 range internally. Let's reserve 10.1.0.0/24 for it.
gcloud filestore instances create ${FILESTORE_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --tier=BASIC_HDD \
    --file-share=name="${FILE_SHARE_NAME}",capacity=1TB \
    --network=name="${NETWORK_NAME}",reserved-ip-range="10.1.0.0/29"

# Get the Filestore instance IP address (Needed for mounting)
# Wait a few minutes for the instance to be created before running this
FILESTORE_IP=$(gcloud filestore instances describe ${FILESTORE_NAME} --zone=${ZONE} --format='value(networks[0].ipAddresses[0])')
echo "Filestore IP Address: ${FILESTORE_IP}"
# Store this IP address, you'll need it later. You can set it as an environment variable:
export FILESTORE_IP=${FILESTORE_IP}
```

### 2. Build a Custom Docker Container

Create a Dockerfile that includes all necessary dependencies for OpenRLHF-M, including the NFS client.

```bash
# Create a directory for your Dockerfile
mkdir -p dockerfile/gcp && cd dockerfile/gcp

# Create the Dockerfile
cat > Dockerfile << 'EOF'
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

# Upgrade pip
RUN pip install --upgrade pip

# Install system dependencies, including NFS client
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    vim \
    build-essential \
    net-tools \
    iputils-ping \
    netcat \
    lsof \
    nfs-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    wheel \
    packaging \
    setuptools \
    huggingface_hub \
    ray[default] \
    vllm==0.7.3 \
    flash-attn --no-build-isolation \
    wandb

# Clone OpenRLHF-M repository
RUN git clone https://github.com/the-laughing-monkey/OpenRLHF-M.git /app/OpenRLHF-M

# Install OpenRLHF-M
WORKDIR /app/OpenRLHF-M
RUN pip install -e .

# The training script is already in the repository at:
# /app/OpenRLHF-M/examples/scripts/tests/train_grpo_ray_qwen2_5_vl_mathv60k_multinode_gcp.sh

# Create directories for NFS mounting (inside container) and datasets symlink target
# Note: The actual mount point /mnt/nfs will be created on the host VM and mounted into the container
RUN mkdir -p /app/datasets

# Set working directory
WORKDIR /app/OpenRLHF-M

# Default command
CMD ["/bin/bash"]
EOF

# Build the Docker image for the amd64 architecture
docker buildx build --platform linux/amd64 -t gcr.io/[YOUR-PROJECT-ID]/openrlhf-m-nfs:latest .

# Initialize Docker Container Registry: Configure docker to use gcloud as a credential helper
# This step is required to push your image to GCP Container Registry
# If you encounter an error regarding the Artifact Registry API (e.g., "denied: Artifact Registry API has not been used in project [YOUR-PROJECT-ID] before or it is disabled"),
# enable it by visiting and clicking "Enable": https://console.developers.google.com/apis/api/artifactregistry.googleapis.com/overview?project=[YOUR-PROJECT-ID]
# then wait a few minutes for the action to propagate and retry.
gcloud auth configure-docker

# Push the Docker image to GCP Container Registry
docker push gcr.io/[YOUR-PROJECT-ID]/openrlhf-m-nfs:latest
```

### 3. Prepare Models and Datasets on Filestore

Use a temporary VM connected to the same VPC network as your Filestore instance to prepare models and datasets directly on the NFS share.

```bash
# Create a temporary VM in the same VPC/Subnet
gcloud compute instances create data-prep-vm \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --machine-type=n1-standard-8 \
    --network=${NETWORK_NAME} \
    --subnet=${SUBNET_NAME} \
    --boot-disk-size=200GB \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --scopes=cloud-platform # Ensure it has scope for GCS if downloading source data from there

# SSH into the VM
gcloud compute ssh data-prep-vm --zone=${ZONE}

# --- Inside the data-prep-vm ---

# Install NFS client and other tools
sudo apt-get update
sudo apt-get install -y nfs-common python3-pip git wget curl

# Set the filestore IP and mount point that you used in step 1
export FILESTORE_IP=[YOUR_FILESTORE_IP_ADDRESS] # Make sure this is set
export NFS_MOUNT_POINT=/mnt/nfs # Choose a mount point
export FILE_SHARE_NAME=vol1 # Default share name for basic tier

# Create mount point directory
sudo mkdir -p ${NFS_MOUNT_POINT}

# Mount the Filestore share
sudo mount ${FILESTORE_IP}:/${FILE_SHARE_NAME} ${NFS_MOUNT_POINT}

# Verify mount
df -h | grep ${NFS_MOUNT_POINT}
# Expected output should show the NFS mount

# Make the mount point accessible
sudo chmod go+rw ${NFS_MOUNT_POINT}

# Create directories on the NFS share for models, datasets, checkpoints etc.
sudo mkdir -p ${NFS_MOUNT_POINT}/model-cache/huggingface
sudo mkdir -p ${NFS_MOUNT_POINT}/model-cache/ray
sudo mkdir -p ${NFS_MOUNT_POINT}/datasets
sudo mkdir -p ${NFS_MOUNT_POINT}/checkpoints
sudo chown -R $USER:$USER ${NFS_MOUNT_POINT} # Change ownership to current user for easier access

# Define paths on NFS
export NFS_MODEL_CACHE_PATH=${NFS_MOUNT_POINT}/model-cache
export NFS_DATASET_PATH=${NFS_MOUNT_POINT}/datasets

# --- 3.1 Prepare Hugging Face Model Cache on NFS ---

# Install Python packages for download
pip install huggingface_hub transformers accelerate torch # Added torch

# Configure Hugging Face cache to use the NFS path directly
export HF_HOME=${NFS_MODEL_CACHE_PATH}/huggingface
echo "Using HuggingFace cache directory: ${HF_HOME}"
mkdir -p ${HF_HOME} # Ensure directory exists

# Download the model to the NFS-backed cache
# Example for Qwen/Qwen2.5-VL-3B-Instruct:
python3 -c "
from huggingface_hub import snapshot_download
import os
model_name = 'Qwen/Qwen2.5-VL-3B-Instruct'
target_dir = os.environ.get('HF_HOME')
print(f'Downloading {model_name} to HuggingFace cache at {target_dir}...')
snapshot_download(repo_id=model_name, cache_dir=target_dir, local_files_only=False)
print('Download complete!')
"
# Repeat for other models as needed...

# --- 3.2 Prepare the MathV60K Dataset on NFS ---

# Clone OpenRLHF-M repo if needed (or copy scripts)
# git clone https://github.com/the-laughing-monkey/OpenRLHF-M.git /tmp/OpenRLHF-M

# Define target dataset directory on NFS
export MATHV60K_TARGET_DIR=${NFS_DATASET_PATH}/VerMulti/MathV60K
mkdir -p ${MATHV60K_TARGET_DIR}

# Clone OpenRLHF-M repository
git clone https://github.com/the-laughing-monkey/OpenRLHF-M.git /tmp/OpenRLHF-M

# Download and prepare the MathV60K dataset directly to NFS
# Adjust path to the download script as needed
python3 /tmp/OpenRLHF-M/examples/scripts/downloaders/download_mathv60k.py --root_dir ${MATHV60K_TARGET_DIR}

# Verify dataset download
echo "Verifying MathV60K dataset on NFS:"
ls -lh ${MATHV60K_TARGET_DIR}

# --- Finished Prep ---

# Unmount NFS before exiting (optional, but good practice)
# sudo umount ${NFS_MOUNT_POINT}

# Exit the VM
exit

# Delete the temporary VM
gcloud compute instances delete data-prep-vm --zone=${ZONE} --quiet

# --- End of Prep ---
```

### 4. Create VMs for Training

Create head and worker nodes in the same VPC network as the Filestore instance. Use the Deep Learning VM image.

```bash
# Choose the desired PyTorch image family
export IMAGE_FAMILY="pytorch-latest-gpu"
export DL_PROJECT="deeplearning-platform-release"
export WANDB_API_KEY={YOUR_WANDB_API_KEY}

# Define variables (ensure these are set from step 1)
# export PROJECT_ID=[YOUR-PROJECT-I
# export ZONE=us-central1-a
# export NETWORK_NAME=openrlhf-vpc
# export SUBNET_NAME=openrlhf-subnet
# export FILESTORE_IP=[YOUR_FILESTORE_IP_ADDRESS]
# export FILE_SHARE_NAME=vol1
# export NFS_MOUNT_POINT=/mnt/nfs

# --- Option A: Create VMs with A100 GPUs (a2-ultragpu-2g) --- 
# Use this if a2-ultragpu-2g is available in your selected ZONE and you have A100 quota

# Create the head node VM with A100
gcloud compute instances create openrlhf-head \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --machine-type=a2-ultragpu-2g \
    --network=${NETWORK_NAME} \
    --subnet=${SUBNET_NAME} \
    --maintenance-policy=TERMINATE \
    --image-family="${IMAGE_FAMILY}" \
    --image-project="${DL_PROJECT}" \
    --boot-disk-size=200GB \
    --accelerator=type=nvidia-a100-80gb,count=2 \
    --scopes=cloud-platform \
    --metadata=wandb-api-key=${WANDB_API_KEY},filestore-ip=${FILESTORE_IP},file-share-name=${FILE_SHARE_NAME},nfs-mount-point=${NFS_MOUNT_POINT}

# Create the worker node VM with A100
gcloud compute instances create openrlhf-worker1 \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --machine-type=a2-ultragpu-2g \
    --network=${NETWORK_NAME} \
    --subnet=${SUBNET_NAME} \
    --maintenance-policy=TERMINATE \
    --image-family="${IMAGE_FAMILY}" \
    --image-project="${DL_PROJECT}" \
    --boot-disk-size=200GB \
    --accelerator=type=nvidia-a100-80gb,count=2 \
    --scopes=cloud-platform \
    --metadata=wandb-api-key=${WANDB_API_KEY},filestore-ip=${FILESTORE_IP},file-share-name=${FILE_SHARE_NAME},nfs-mount-point=${NFS_MOUNT_POINT}

# --- Option B: Create VMs with H100 GPUs (a3-highgpu-2g) --- 
# Use this as an alternative if A100s (a2-ultragpu-2g) are unavailable but a3-highgpu-2g exists 
# in your selected ZONE and you have H100 quota.
# Note: Ensure ${ZONE} is set correctly for this option (e.g., us-central1-a)

# # Create the head node VM with H100 
gcloud compute instances create openrlhf-head \
     --project=${PROJECT_ID} \
     --zone=${ZONE} \
     --machine-type=a3-highgpu-2g \
     --network=${NETWORK_NAME} \
     --subnet=${SUBNET_NAME} \
     --maintenance-policy=TERMINATE \
     --image-family="${IMAGE_FAMILY}" \
     --image-project="${DL_PROJECT}" \
     --boot-disk-size=200GB \
     --accelerator=type=nvidia-h100-80gb,count=2 \
     --scopes=cloud-platform \
     --metadata=wandb-api-key=${WANDB_API_KEY},filestore-ip=${FILESTORE_IP},file-share-name=${FILE_SHARE_NAME},nfs-mount-point=${NFS_MOUNT_POINT}

# # Create the worker node VM with H100 
gcloud compute instances create openrlhf-worker1 \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --machine-type=a3-highgpu-2g \
    --network=${NETWORK_NAME} \
    --subnet=${SUBNET_NAME} \
    --maintenance-policy=TERMINATE \
    --image-family="${IMAGE_FAMILY}" \
    --image-project="${DL_PROJECT}" \
    --boot-disk-size=200GB \
    --accelerator=type=nvidia-h100-80gb,count=2 \
    --scopes=cloud-platform \
    --metadata=wandb-api-key=${WANDB_API_KEY},filestore-ip=${FILESTORE_IP},file-share-name=${FILE_SHARE_NAME},nfs-mount-point=${NFS_MOUNT_POINT}
```
*Note: We pass Filestore info via metadata for easy access within the VM.*

### 5. Connect to VMs, Mount NFS, and Setup Training Environment

#### Common Steps for BOTH Head and Worker Nodes:

```bash
# SSH into the node (replace openrlhf-head with openrlhf-worker1 for worker)
gcloud compute ssh openrlhf-head --zone=${ZONE}
# OR
# gcloud compute ssh openrlhf-worker1 --zone=${ZONE}

# --- Inside the VM ---

# 1. Install NVIDIA drivers if prompted (Say Yes)

# 2. Mount the Filestore NFS share
# Retrieve metadata
export FILESTORE_IP=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/filestore-ip)
export FILE_SHARE_NAME=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/file-share-name)
export NFS_MOUNT_POINT=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/nfs-mount-point)

# Install NFS client (should be present on DLVM, but verify)
sudo apt-get update
sudo apt-get install -y nfs-common

# Create mount point directory
sudo mkdir -p ${NFS_MOUNT_POINT}

# Mount the Filestore share
sudo mount ${FILESTORE_IP}:/${FILE_SHARE_NAME} ${NFS_MOUNT_POINT}

# Verify mount
df -h | grep ${NFS_MOUNT_POINT}

# Make the mount point accessible (optional, depends on permissions set during prep)
# sudo chmod go+rw ${NFS_MOUNT_POINT}

# Optional: Configure automatic mounting on reboot (add to /etc/fstab)
# echo "${FILESTORE_IP}:/${FILE_SHARE_NAME} ${NFS_MOUNT_POINT} nfs defaults 0 0" | sudo tee -a /etc/fstab

# 3. Authenticate Docker with gcloud
gcloud auth configure-docker

# 4. Pull your custom container image
sudo docker pull gcr.io/${PROJECT_ID}/openrlhf-m-nfs:latest

# 5. Retrieve W&B API key from metadata
export WANDB_API_KEY=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/wandb-api-key)

# 6. Start the container
#    Mount the host's NFS directory into the container
#    Set HF_HOME inside the container to point to the NFS cache
#    Remove FUSE-specific flags (--device, --privileged might still be needed for GPUs/NCCL)
sudo docker run -d --name openrlhf-$(hostname)-container \
    --gpus all \
    --net=host \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --privileged \
    -v ${NFS_MOUNT_POINT}:${NFS_MOUNT_POINT} `# Mount NFS from host` \
    --env WANDB_API_KEY="${WANDB_API_KEY}" \
    --env HF_HOME="${NFS_MOUNT_POINT}/model-cache/huggingface" `# Point HF cache to NFS` \
    --env NFS_MOUNT_POINT="${NFS_MOUNT_POINT}" `# Pass mount point info` \
    gcr.io/${PROJECT_ID}/openrlhf-m-nfs:latest sleep infinity

# 7. Login to the container to check setup (optional)
# sudo docker exec -it openrlhf-$(hostname)-container bash
# Inside container: check `ls ${NFS_MOUNT_POINT}`, `echo $HF_HOME`, `nvidia-smi`
# exit

# 8. Setup Symlinks Inside the Container (for convenience)
# Link the NFS dataset path to the expected /app/datasets/VerMulti
sudo docker exec openrlhf-$(hostname)-container bash -c "ln -sf ${NFS_MOUNT_POINT}/datasets /app/datasets/VerMulti"
# Link the NFS Ray cache path (optional, Ray might use ~/.cache/ray by default)
# sudo docker exec openrlhf-$(hostname)-container bash -c "mkdir -p ~/.cache && ln -sf ${NFS_MOUNT_POINT}/model-cache/ray ~/.cache/ray"

# --- End of common steps for the node ---
```

#### Specific Steps for Head Node:

```bash
# (After common steps on head node VM)

# 9. Start Ray head inside the container
sudo docker exec openrlhf-head-container ray start --head --node-ip-address 0.0.0.0 --port=6379 --dashboard-port=8265
```

#### Specific Steps for Worker Node:

```bash
# (After common steps on worker node VM)

# 9. Get the head node's internal IP address
# Run this on your local machine or on the worker VM after SSHing into head
HEAD_IP=$(gcloud compute instances describe openrlhf-head --zone=${ZONE} --format='get(networkInterfaces[0].networkIP)')
echo "Head node IP: $HEAD_IP"
# Set it as an environment variable inside the worker VM shell
export HEAD_IP=${HEAD_IP}

# 10. Connect worker Ray process to the head inside the container
sudo docker exec openrlhf-worker1-container ray start --address=${HEAD_IP}:6379

# --- Exit worker VM SSH session ---
# exit
```

### 6. Run the Training Job

Connect to the **head node VM** via SSH. Then, execute the training script **inside the running head container**. Update the script to use NFS paths.

```bash
# SSH into head node
# gcloud compute ssh openrlhf-head --zone=${ZONE}

# Login to the head container
sudo docker exec -it openrlhf-head-container bash

# --- Inside the head container ---

# 1. Navigate to the scripts directory
cd /app/OpenRLHF-M/examples/scripts/tests

# 2. Edit the training script to use NFS paths and remove GCS variables
# vi train_grpo_ray_qwen2_5_vl_mathv60k_multinode_gcp.sh
#
# Make the following changes within the script:
#   - Remove or comment out: GCS_BUCKET="${GCS_BUCKET:-gs://[YOUR_BUCKET]}"
#   - Change DATASET_PATH to use the NFS mount point (passed via env var or use symlink)
#     Example using symlink: DATASET_PATH="${DATASET_PATH:-/app/datasets/VerMulti/MathV60K/mathv60k_message.jsonl}"
#     Example using direct NFS path: DATASET_PATH="${DATASET_PATH:-${NFS_MOUNT_POINT}/datasets/VerMulti/MathV60K/mathv60k_message.jsonl}"
#   - Change SAVE_PATH to use the NFS mount point
#     Example: SAVE_PATH="${SAVE_PATH:-${NFS_MOUNT_POINT}/checkpoints}"
#   - Ensure PRETRAIN_MODEL_PATH points to the model name (HuggingFace cache is handled by HF_HOME env var)
#     Example: PRETRAIN_MODEL_PATH="${PRETRAIN_MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}" (No change needed here if using HF cache)
#   - Adjust MODEL_NAME and EXPECTED_WORKERS as needed.

# Example of modified variables in the script:
# FILE: train_grpo_ray_qwen2_5_vl_mathv60k_multinode_gcp.sh
# NFS_MOUNT_POINT="${NFS_MOUNT_POINT:-/mnt/nfs}" # Get from env or default
# DATASET_PATH="${DATASET_PATH:-${NFS_MOUNT_POINT}/datasets/VerMulti/MathV60K/mathv60k_message.jsonl}"
# PRETRAIN_MODEL_PATH="${PRETRAIN_MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
# SAVE_PATH="${SAVE_PATH:-${NFS_MOUNT_POINT}/checkpoints}"
# MODEL_NAME="${MODEL_NAME:-qwen2.5-vl-3b-ins-mathvista-grpo-nfs}"
# EXPECTED_WORKERS="${EXPECTED_WORKERS:-1}"

# 3. Execute the training script
bash ./train_grpo_ray_qwen2_5_vl_mathv60k_multinode_gcp.sh

# --- Exit container when done ---
# exit
```

## 7. Cleaning Up Resources

Once you have finished your training and saved any necessary results from the NFS share, you should delete the compute resources to avoid incurring further charges.

```bash
# Ensure your ZONE and PROJECT_ID variables are still set correctly
# export ZONE=us-central1-a # Or your chosen zone
# export PROJECT_ID=[YOUR-PROJECT-ID]

# Delete the head node VM
echo "Deleting head node VM (openrlhf-head)..."
gcloud compute instances delete openrlhf-head --zone=${ZONE} --project=${PROJECT_ID} --quiet

# Delete the worker node VM
echo "Deleting worker node VM (openrlhf-worker1)..."
gcloud compute instances delete openrlhf-worker1 --zone=${ZONE} --project=${PROJECT_ID} --quiet

# --- Optional: Delete the Filestore Instance --- 
WARNING: This will permanently delete the NFS share and all data stored on it (datasets, models, checkpoints).
Only run this if you are certain you no longer need the data on the Filestore instance.

export FILESTORE_NAME=openrlhf-nfs # Ensure Filestore name is set

echo "WARNING: About to delete Filestore instance ${FILESTORE_NAME} and ALL its data!"
echo "Press Ctrl+C to cancel, or Enter to continue..."
read -p ""
gcloud filestore instances delete ${FILESTORE_NAME} --zone=${ZONE} --project=${PROJECT_ID} --quiet
echo "Filestore instance ${FILESTORE_NAME} deleted."

--- Optional: Delete VPC Network Resources ---
Only do this if the VPC, subnet, and firewall rules are no longer needed for other purposes.

export NETWORK_NAME=openrlhf-vpc # Ensure network name is set
export SUBNET_NAME=openrlhf-subnet
export REGION=us-central1 # Ensure region is set

echo "Deleting firewall rules..."
gcloud compute firewall-rules delete ${NETWORK_NAME}-allow-internal --project=${PROJECT_ID} --quiet
gcloud compute firewall-rules delete ${NETWORK_NAME}-allow-ssh-icmp --project=${PROJECT_ID} --quiet
echo "Deleting subnet..."
gcloud compute networks subnets delete ${SUBNET_NAME} --region=${REGION} --project=${PROJECT_ID} --quiet
echo "Deleting VPC network..."
gcloud compute networks delete ${NETWORK_NAME} --project=${PROJECT_ID} --quiet
echo "VPC network resources deleted."
```

## Deployment Options

This guide now describes three different deployment options that you can choose from, depending on your management and scaling preferences, adapted for NFS.

---

### **Option 1: Standard Deployment Using Individual VMs**

This option uses the manual VM provisioning described in Sections 4 & 5 above. It gives full control but requires manual NFS mounting and container setup on each node.

*(Content essentially covered in Sections 4 & 5, no need to repeat VM creation commands here. Refer back to those sections)*

---

### **Option 2: Alternative Deployment Using Google Kubernetes Engine (GKE)**

Deploy on GKE using an NFS volume for shared storage. This requires setting up the Filestore CSI driver or manually configuring NFS volumes.

#### Step 1: Create a GPU-Enabled GKE Cluster (Same as before, ensure VPC-native)

```bash
# Create a GKE cluster (ensure VPC-native for Filestore access)
gcloud container clusters create openrlhf-cluster \
    --zone ${ZONE} \
    --num-nodes=0 \
    --enable-ip-alias \
    --network ${NETWORK_NAME} \
    --subnetwork ${SUBNET_NAME}

# Create a GPU-enabled node pool
gcloud container node-pools create gpu-pool \
    --cluster=openrlhf-cluster \
    --zone=${ZONE} \
    --num-nodes=4 \
    --machine-type=a2-ultragpu-2g \
    --accelerator=type=nvidia-a100-80gb,count=8 \
    --metadata=disable-legacy-endpoints=true
    # Note: GKE nodes will automatically get nfs-common if needed by CSI driver or pods
```

#### Step 2: Configure NFS Access (Option A: Filestore CSI Driver - Recommended)

Install the [Google Cloud Filestore CSI driver](https://cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/filestore-csi-driver) to your cluster. This allows dynamic or static provisioning of Filestore volumes.

```bash
# Enable Filestore CSI driver on the cluster
gcloud container clusters update openrlhf-cluster --update-addons=GcpFilestoreCsiDriver=ENABLED --zone=${ZONE}

# Create a StorageClass (e.g., filestore-sc.yaml)
# apiVersion: storage.k8s.io/v1
# kind: StorageClass
# metadata:
#   name: filestore-basic-hdd
# provisioner: filestore.csi.storage.gke.io
# parameters:
#   tier: BASIC_HDD # Or STANDARD, PREMIUM
#   network: projects/${PROJECT_ID}/global/networks/${NETWORK_NAME}
#   # reserved-ip-range: [Optional] Specify if needed, otherwise uses default Filestore range logic
# volumeBindingMode: Immediate # Or WaitForFirstConsumer

# kubectl apply -f filestore-sc.yaml

# Create a PersistentVolumeClaim (e.g., nfs-pvc.yaml)
# apiVersion: v1
# kind: PersistentVolumeClaim
# metadata:
#   name: openrlhf-nfs-pvc
# spec:
#   accessModes:
#   - ReadWriteMany
#   storageClassName: filestore-basic-hdd
#   resources:
#     requests:
#       storage: 1Ti # Match Filestore capacity or request needed size

# kubectl apply -f nfs-pvc.yaml
```
```bash
# kubectl apply -f nfs-pv.yaml
# kubectl apply -f nfs-pvc.yaml
```

#### Step 3: Deploy Training Pods with NFS Volume Mount

Modify the Kubernetes manifest to use the `openrlhf-nfs-pvc` claim.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: openrlhf-head
spec:
  replicas: 1
  selector:
    matchLabels:
      app: openrlhf-head
  template:
    metadata:
      labels:
        app: openrlhf-head
    spec:
      # Removed initContainer for gcsfuse
      containers:
      - name: head-container
        image: gcr.io/${PROJECT_ID}/openrlhf-m:latest
        env:
        - name: WANDB_API_KEY
          value: "[YOUR_WANDB_API_KEY]"
        - name: NFS_MOUNT_POINT # Inform container where NFS is mounted
          value: "/mnt/nfs"
        - name: HF_HOME
          value: "/mnt/nfs/model-cache/huggingface" # Point HF cache to NFS
        # Removed securityContext for gcsfuse if not needed for other reasons
        resources:
          limits:
            nvidia.com/gpu: "8"
        volumeMounts:
        - name: nfs-storage # Mount the PVC
          mountPath: "/mnt/nfs" # Mount point inside the container
        # Removed fuse volume mount
      volumes:
      - name: nfs-storage
        persistentVolumeClaim:
          claimName: openrlhf-nfs-pvc # Reference the PVC
      # Removed fuse volume
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: openrlhf-worker
spec:
  replicas: 3 # Adjust as needed
  selector:
    matchLabels:
      app: openrlhf-worker
  template:
    metadata:
      labels:
        app: openrlhf-worker
    spec:
      # Removed initContainer for gcsfuse
      containers:
      - name: worker-container
        image: gcr.io/${PROJECT_ID}/openrlhf-m:latest
        env:
        - name: WANDB_API_KEY
          value: "[YOUR_WANDB_API_KEY]"
        - name: NFS_MOUNT_POINT
          value: "/mnt/nfs"
        - name: HF_HOME
          value: "/mnt/nfs/model-cache/huggingface"
        - name: GCP_WORKER # Keep if used by application logic
          value: "1"
        # Removed securityContext for gcsfuse
        resources:
          limits:
            nvidia.com/gpu: "8"
        volumeMounts:
        - name: nfs-storage
          mountPath: "/mnt/nfs"
        # Removed fuse volume mount
      volumes:
      - name: nfs-storage
        persistentVolumeClaim:
          claimName: openrlhf-nfs-pvc
      # Removed fuse volume
```

Deploy this manifest with:

```bash
kubectl apply -f your-nfs-deployment.yaml
```
*Note: You'll need to adapt the commands inside the pods (e.g., Ray start, training script execution) similarly to Section 6, ensuring paths point to `/mnt/nfs`.*

---

### **Option 3: Managed Deployment Using Vertex AI Ray Clusters**

Configure the Ray cluster YAML to automatically mount the Filestore NFS share.

#### Step 1: Create a Ray Cluster Configuration File (`ray_cluster_config_nfs.yaml`)

```yaml
# cluster_name: openrlhf-ray-cluster # Optional: If creating new
# project_id: [YOUR-PROJECT-ID]
# region: us-central1
# network: projects/${PROJECT_ID}/global/networks/${NETWORK_NAME} # Specify the VPC network
# min_replicas: 2 # Head + 1 Worker min
# max_replicas: 2 # Example: 1 Head + 1 Worker
# head_node_type:
#   machine_type: a2-ultragpu-2g
#   accelerator_type: nvidia-a100-80gb
#   accelerator_count: 8
# worker_node_types:
#   - machine_type: a2-ultragpu-2g
#     accelerator_type: nvidia-a100-80gb
#     accelerator_count: 8
#     min_replicas: 1
#     max_replicas: 1
# container_image: gcr.io/[YOUR-PROJECT-ID]/openrlhf-m:latest
# env_vars:
#   WANDB_API_KEY: "[YOUR_WANDB_API_KEY]"
#   NFS_MOUNT_POINT: "/mnt/nfs/ray-share" # Define the mount point for Ray Cluster
#   HF_HOME: "/mnt/nfs/ray-share/model-cache/huggingface" # Point HF cache to NFS

# Configure NFS Mount using network_filesystem
network_filesystem:
  - server: "[YOUR_FILESTORE_IP_ADDRESS]" # Filestore IP
    path: "/${FILE_SHARE_NAME}"          # Filestore share name (e.g., /vol1)
    mount_point: "/mnt/nfs/ray-share"   # Mount point inside the node/container

# No startup_script needed for mounting if using network_filesystem
# Optional startup_script for symlinks or other setup inside container:
# startup_script: |
#   #!/bin/bash
#   echo "Setting up symlinks..."
#   ln -sf /mnt/nfs/ray-share/datasets /app/datasets/VerMulti
#   mkdir -p ~/.cache
#   ln -sf /mnt/nfs/ray-share/model-cache/ray ~/.cache/ray
```

#### Step 2: Create or Update the Ray Cluster

```bash
# Install Vertex AI SDK if needed
# pip install google-cloud-aiplatform

# Create the cluster
python -c "
from google.cloud import aiplatform
aiplatform.init(
    project='[YOUR-PROJECT-ID]',
    location='us-central1',
)
# Load config, ensuring paths and required fields (like network) are correct
cluster = aiplatform.RayCluster.create_from_yaml('ray_cluster_config_nfs.yaml')
print(f'Ray cluster dashboard URL: {cluster.get_dashboard_uri()}')
"
```
*Note: When submitting jobs to a Vertex Ray cluster (e.g., via `ray job submit`), ensure your training script uses the correct NFS paths (e.g., `/mnt/nfs/ray-share/...`) defined in the config.*

---

## Monitoring and Managing Your Training Job

### Ray Dashboard

Access remains the same (Port 8265 via SSH tunnel or direct access if configured).

### View Training Logs

Logs are typically stored within the container or potentially written to the NFS share if configured.

```bash
# SSH into the head node (Option 1) or find the head pod (Option 2/3)
# Example for Option 1:
# gcloud compute ssh openrlhf-head --zone=${ZONE}
# sudo docker exec -it openrlhf-head-container bash
# cd /app/OpenRLHF-M
# tail -f logs/qwen2.5-vl-3b-ins-mathvista-grpo-nfs/*/train.log

# If logs are written to NFS:
# tail -f ${NFS_MOUNT_POINT}/logs/...
```

### Monitor GPU Usage

```bash
# On the VM/Node:
watch -n1 nvidia-smi
# Or inside the container:
# sudo docker exec openrlhf-[head|worker]-container nvidia-smi
```

### Accessing Saved Models

Models are saved to the NFS share.

```bash
# Access via any machine mounting the NFS share (e.g., the head node)
ls -lh ${NFS_MOUNT_POINT}/checkpoints/qwen2.5-vl-3b-ins-mathvista-grpo-nfs/
# Or list via the container
# sudo docker exec openrlhf-head-container ls -lh ${NFS_MOUNT_POINT}/checkpoints/...
```

## Troubleshooting

### NFS Mounting Issues

1.  **Verify Filestore Instance**: Check the Filestore instance status in the GCP console. Ensure it's `READY`.
2.  **Check VM/Pod Network**: Ensure the VM/Pod is in the same VPC network specified during Filestore instance creation.
3.  **Verify NFS Client**: Ensure `nfs-common` is installed on the VM/Node (`sudo apt-get install -y nfs-common`).
4.  **Check Mount Command**: Double-check the Filestore IP address and share name (`sudo mount <IP>:/<share> <mount_point>`).
5.  **Check Mount Status**: Use `df -h | grep <mount_point>` or `mount | grep <mount_point>` to verify the mount.
6.  **Firewall Rules**: Ensure VPC firewall rules allow NFS traffic (TCP/UDP port 2049 and others like 111 if using NFSv3/portmapper) between the Filestore IP range and the VM/GKE node subnet range. Standard VPCs often allow internal traffic by default, but check custom rules.
7.  **Permissions**: If mounting works but access is denied, check permissions set on the Filestore share (during prep step: `sudo chmod go+rw /mnt/nfs`) and ensure user/group IDs match if necessary (though `chmod` often suffices).

### Cache and Model Loading Issues (NFS Context)

1.  **Verify NFS Mount**: Check `df -h` inside the *container* to ensure the volume mount from the host is working.
2.  **Check `HF_HOME`**: Inside the container, run `echo $HF_HOME`. It should point to the NFS path (e.g., `/mnt/nfs/model-cache/huggingface`).
3.  **Verify Symlinks**: If using symlinks (e.g., `/app/datasets/VerMulti`), check they point correctly to the NFS path (`ls -la /app/datasets/`).
4.  **Verify Files Exist**: Check if model/dataset files actually exist on the NFS share at the expected path (`ls -lh $HF_HOME/...` or `ls -lh /mnt/nfs/datasets/...`). Permissions errors during the data prep phase could lead to missing files.

### Disk Space Management

Monitor space on the *Filestore instance* itself, not the local VM disk, for shared data.
1.  Check Filestore usage in the GCP Console or using `df -h ${NFS_MOUNT_POINT}` on a mounted machine.
2.  Increase Filestore capacity if needed (can usually be done online for most tiers).

### Ray Cluster Issues (General)
// ... existing code ...

### GPU Issues (General)
// ... existing code ...

### NCCL Communication Errors (General)
// ... existing code ...

## Cost Optimization

Filestore adds cost based on tier and capacity. A2 GPUs remain expensive.
1.  **Choose Appropriate Filestore Tier**: Basic HDD/SSD vs. High Scale SSD depends on performance needs and budget.
2.  Shut down VMs/Clusters when not in use. Filestore continues to incur costs while provisioned.
3.  Delete Filestore instances if no longer needed (backup data first!).
4.  Store checkpoints frequently on NFS to resume from interruptions.
5.  Consider smaller GPU types for testing.

## Conclusion

This guide provides a comprehensive approach to running OpenRLHF-M on GCP Vertex AI using Docker containers and **NFS (Google Cloud Filestore)** for high-performance shared storage. This setup replaces GCS FUSE, aiming for faster data access during training, which is crucial for I/O intensive ML workloads. The container-based approach offers reproducibility and scalability, while NFS provides efficient data handling.

For multinode training workloads, GCP's environment combined with Filestore offers excellent networking performance and the ability to easily scale resources.

## References

- [GCP Vertex AI Documentation](https://cloud.google.com/vertex-ai)
- [Google Cloud Filestore Documentation](https://cloud.google.com/filestore/docs)
- [Using Filestore with GKE (CSI Driver)](https://cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/filestore-csi-driver)
- [Ray on Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs/open-source/ray-on-vertex-ai/overview)
- [Mounting NFS shares on Compute Engine](https://cloud.google.com/compute/docs/file-servers/mounting-nfs-shares)
- [Ray Documentation](https://docs.ray.io/)
- [OpenRLHF Documentation](https://openrlhf.readthedocs.io/)
- [Containerizing ML Workloads](https://cloud.google.com/architecture/ml-on-gcp-best-practices)



</rewritten_file> 