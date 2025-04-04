# Running OpenRLHF-M on GCP Vertex AI

This document provides comprehensive instructions for deploying and running OpenRLHF-M on GCP Vertex AI using a Docker container approach. This methodology ensures reproducible environments and follows GCP best practices for ML workloads.

## Overview

In this setup, we'll use:
- **Container Registry**: For storing our custom Docker image with all dependencies
- **Cloud Storage**: For datasets and model checkpoints
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

5. Now weill install the various services we need for running OpenRLHF-M on GCP:
```bash
   gcloud services enable aiplatform.googleapis.com storage.googleapis.com containerregistry.googleapis.com compute.googleapis.com
```


## Setup Process

### 1. Create a GCS Bucket

First, create a Google Cloud Storage bucket to store your datasets, model checkpoints, and logs:

```bash
# Create a GCS bucket (replace [YOUR-PROJECT] and [YOUR-BUCKET] with your values)
gsutil mb -p [YOUR-PROJECT] -l us-central1 gs://[YOUR-BUCKET]
```

# Sanity Check: Verify that the bucket was created
```bash
gsutil ls gs://[YOUR-BUCKET]
```

### 2. Build a Custom Docker Container

Create a Dockerfile that includes all necessary dependencies for OpenRLHF-M:

```bash
# Create a directory for your Dockerfile
mkdir -p dockerfile/gcp && cd dockerfile/gcp

# Create the Dockerfile
cat > Dockerfile << 'EOF'
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

# Upgrade pip
RUN pip install --upgrade pip

# Install system dependencies
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
    fuse \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install gcsfuse for GCS bucket mounting
RUN curl -L https://github.com/GoogleCloudPlatform/gcsfuse/releases/download/v1.1.0/gcsfuse_1.1.0_amd64.deb > gcsfuse.deb && \
    dpkg -i gcsfuse.deb && \
    rm gcsfuse.deb

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

# Create directories for GCS mounting
RUN mkdir -p /mnt/gcs-cache /app/datasets

# Set working directory
WORKDIR /app/OpenRLHF-M

# Default command
CMD ["/bin/bash"]
EOF

# Build the Docker image for the amd64 architecture
docker buildx build --platform linux/amd64 -t gcr.io/[YOUR-PROJECT-ID]/openrlhf-m:latest .

# Initialize Docker Container Registry: Configure docker to use gcloud as a credential helper
# This step is required to push your image to GCP Container Registry
# If you encounter an error regarding the Artifact Registry API (e.g., "denied: Artifact Registry API has not been used in project [YOUR-PROJECT-ID] before or it is disabled"),
# enable it by visiting and clicking "Enable": https://console.developers.google.com/apis/api/artifactregistry.googleapis.com/overview?project=[YOUR-PROJECT-ID]
# then wait a few minutes for the action to propagate and retry.
gcloud auth configure-docker

# Push the Docker image to GCP Container Registry
docker push gcr.io/[YOUR-PROJECT-ID]/openrlhf-m:latest
```

### 3. Prepare Models and Datasets

For large language models and datasets, downloading them to your local machine and then uploading to GCS is inefficient. Instead, we'll use temporary VMs to prepare both models and datasets directly in the cloud.

### 3.0 Create GCS Mount Script

To access the prepared models and datasets from your training nodes, create a script that will mount the GCS bucket and set up the necessary cache symlinks:

```bash
# Create directory if it doesn't exist
mkdir -p ./examples/scripts/setup

# Create a GCS mount and cache setup script
cat > ./examples/scripts/setup/setup_gcs_mounts.sh << 'EOF'
#!/bin/bash
# Setup model caching with GCS bucket
GCS_BUCKET="${GCS_BUCKET:-gs://[YOUR_BUCKET]}"
GCS_BUCKET_NAME=$(echo ${GCS_BUCKET} | sed 's|gs://||')

echo "Setting up GCS mount and HuggingFace cache symlinks..."
# Mount GCS bucket if not already mounted
if ! grep -q "/mnt/gcs-cache" /proc/mounts; then
    echo "Mounting GCS bucket ${GCS_BUCKET} to /mnt/gcs-cache"
    gcsfuse --implicit-dirs ${GCS_BUCKET_NAME} /mnt/gcs-cache
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
EOF

chmod +x ./examples/scripts/setup/setup_gcs_mounts.sh
```

This script will be used on both head and worker nodes before starting the training process.

#### 3.1 Prepare the Hugging Face Model Cache

```bash
# Create a VM to prepare the model cache.
# Choose the datacenter that you want for model cache and make sure it matches the region of your GCS bucket:
# For example, us-central1-a is Iowa, us-east1-a is South Carolina, us-west1-a is Oregon

gcloud compute instances create model-cache-prep \
    --machine-type=n1-standard-8 \
    --boot-disk-size=200GB \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --scopes=cloud-platform \
    --zone=us-central1-a


# SSH into the VM
gcloud compute ssh model-cache-prep

# Then on the VM:
# Install required packages
sudo apt-get update && sudo apt-get install -y python3-pip git
pip install huggingface_hub transformers accelerate

# Download the model to cache
# For example to download the Qwen2.5-VL-3B-Instruct model:
python3 -c "
from huggingface_hub import snapshot_download
# Replace with your model name
model_name = 'Qwen/Qwen2.5-VL-3B-Instruct'
print(f'Downloading {model_name} to HuggingFace cache...')
snapshot_download(repo_id=model_name, local_files_only=False)
print('Download complete!')
"


# Or to download the 7B Qwen2.5-VL-7B-Instruct model:
python3 -c "
from huggingface_hub import snapshot_download
model_name = 'Qwen/Qwen2.5-VL-7B-Instruct'
print(f'Downloading {model_name} to HuggingFace cache...')
snapshot_download(repo_id=model_name, local_files_only=False)
print('Download complete!')
"


# Or to download the 32B Qwen2.5-VL-32B-Instruct model:
python3 -c "
from huggingface_hub import snapshot_download
model_name = 'Qwen/Qwen2.5-VL-32B-Instruct'
print(f'Downloading {model_name} to HuggingFace cache...')
snapshot_download(repo_id=model_name, local_files_only=False)
print('Download complete!')
"


# Or to download the 72B Qwen2.5-VL-72B-Instruct model:
python3 -c "
from huggingface_hub import snapshot_download
model_name = 'Qwen/Qwen2.5-VL-72B-Instruct'
print(f'Downloading {model_name} to HuggingFace cache...')
snapshot_download(repo_id=model_name, local_files_only=False)
print('Download complete!')
"


# Upload the model cache to the models directory in your bucket
echo "Uploading model cache to gs://[YOUR-BUCKET]/model-cache/"
gsutil -m cp -r ~/.cache/huggingface gs://[YOUR-BUCKET]/model-cache/

# Once the upload is complete, exit the VM and delete the instance
exit
gcloud compute instances delete model-cache-prep --quiet
```

#### 3.2 Prepare the MathV60K Dataset

Now use the same VM to prepare the MathV60K dataset:

```bash
# Ensure the target dataset directory has proper permissions
sudo mkdir -p /app/datasets/VerMulti
sudo chown $USER:$USER /app/datasets/VerMulti

# Download and prepare the MathV60K dataset
python3 examples/scripts/downloaders/download_mathv60k.py --root_dir /app/datasets/VerMulti

# Verify bucket exists before uploading
gsutil ls gs://[YOUR-BUCKET] > /dev/null 2>&1
if [ $? -ne 0 ]; then
  echo "Error: Bucket gs://[YOUR-BUCKET] does not exist. Please create it first."
  exit 1
fi

# Upload the prepared dataset to GCS
echo "Uploading MathV60K dataset to gs://[YOUR-BUCKET]/datasets/"
gsutil -m cp -r /app/datasets/VerMulti gs://[YOUR-BUCKET]/datasets/

# Once the upload is complete, exit the VM and delete the instance
exit
gcloud compute instances delete model-cache-prep --quiet
```

### 4. Create VMs for Training

Now, create head and worker nodes using the Deep Learning VM image, which comes with NVIDIA drivers and Docker pre-installed.

```bash
# Choose the desired PyTorch image family (check Deep Learning Images documentation for latest)
export IMAGE_FAMILY="pytorch-latest-gpu" 
export DL_PROJECT="deeplearning-platform-release"

# Create the head node VM using a Deep Learning VM Image
gcloud compute instances create openrlhf-head \
    --zone=us-central1-a \
    --machine-type=a2-ultragpu-2g \
    --maintenance-policy=TERMINATE \
    --image-family="${IMAGE_FAMILY}" \
    --image-project="${DL_PROJECT}" \
    --boot-disk-size=200GB \
    --accelerator=type=nvidia-a100-80gb,count=2 \
    --scopes=cloud-platform \
    --metadata=GCS_BUCKET=gs://[YOUR-BUCKET],WANDB_API_KEY=your-wandb-api-key-here # Metadata accessible via metadata server

# Create the worker node VM using a Deep Learning VM Image
gcloud compute instances create openrlhf-worker1 \
    --zone=us-central1-a \
    --machine-type=a2-ultragpu-2g \
    --maintenance-policy=TERMINATE \
    --image-family="${IMAGE_FAMILY}" \
    --image-project="${DL_PROJECT}" \
    --boot-disk-size=200GB \
    --accelerator=type=nvidia-a100-80gb,count=2 \
    --scopes=cloud-platform \
    --metadata=GCS_BUCKET=gs://[YOUR-BUCKET],WANDB_API_KEY=your-wandb-api-key-here # Metadata accessible via metadata server
```
*Note: Deep Learning VM images include necessary drivers and the NVIDIA Container Toolkit.*

### 5. Connect to the VMs and set up the training environment

#### On the Head Node:

```bash
# SSH into the head node
gcloud compute ssh openrlhf-head --zone=us-central1-a
```

CRITICAL NOTE: This will prompt you to install the NVIDIA drivers. Say yes.

# Inside the head node VM:

# 1. Authenticate Docker with gcloud (needed to pull from GCR)
```bash
gcloud auth configure-docker
```

# 2. Pull your custom container image
```bash
sudo docker pull gcr.io/[YOUR-PROJECT-ID]/openrlhf-m:latest
```

# 3. Retrieve metadata for environment variables
```bash
export GCS_BUCKET_METADATA=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/GCS_BUCKET)
export WANDB_API_KEY_METADATA=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/WANDB_API_KEY)
```

# 4. Start your container in detached mode
#    --gpus all: Exposes GPUs using NVIDIA Container Toolkit (pre-installed on DLVM)
#    --net=host: Allows easy Ray communication between containers on different nodes
#    --shm-size, --ulimit: Recommended settings for distributed training
#    --device, --cap-add: Needed for gcsfuse inside the container
#    --env: Pass metadata variables to the container
```bash
sudo docker run -d --name openrlhf-head-container \
    --gpus all \
    --net=host \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --device=/dev/fuse \
    --privileged \
    --env GCS_BUCKET="${GCS_BUCKET_METADATA}" \
    --env WANDB_API_KEY="${WANDB_API_KEY_METADATA}" \
    gcr.io/[YOUR-PROJECT-ID]/openrlhf-m:latest sleep infinity
```

# 5. Login to the container to check it is running and good:
```bash
sudo docker exec -it openrlhf-head-container bash
```

# 6. Check that the GPUs are visible:
```bash
nvidia-smi
```

# 7. Update the code:
```bash
cd /app/OpenRLHF-M
git pull
```

# 8. Exit the contiainer back to the base VM:
```bash
exit
```

# 9. Run the GCS setup script inside the container
```bash
sudo docker exec openrlhf-head-container bash /app/OpenRLHF-M/examples/scripts/setup/setup_gcs_mounts.sh
```

# 10. Start Ray head inside the container
```bash
sudo docker exec openrlhf-head-container ray start --head --node-ip-address 0.0.0.0 --port=6379 --dashboard-port=8265
```

#### On the Worker Node:

First, get the head node's internal IP address **on your local machine**: 

```bash
# Run this on your local machine
HEAD_IP=$(gcloud compute instances describe openrlhf-head \
    --zone=us-central1-a \
    --format='get(networkInterfaces[0].networkIP)')
echo "Head node IP: $HEAD_IP" 
```

Now, SSH into the worker node and set up the container:

```bash
# SSH into the worker node (run on your local machine)
gcloud compute ssh openrlhf-worker1 --zone=us-central1-a 

# Inside the worker node VM:

# 1. Authenticate Docker with gcloud
gcloud auth configure-docker

# 2. Pull your custom container image
sudo docker pull gcr.io/[YOUR-PROJECT-ID]/openrlhf-m:latest

# 3. Retrieve metadata for environment variables
export GCS_BUCKET_METADATA=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/GCS_BUCKET)
export WANDB_API_KEY_METADATA=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/WANDB_API_KEY)

# 4. Start your container in detached mode (similar flags as head node)
sudo docker run -d --name openrlhf-worker-container \
    --gpus all \
    --net=host \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --device=/dev/fuse \
    --privileged \
    --env GCS_BUCKET="${GCS_BUCKET_METADATA}" \
    --env WANDB_API_KEY="${WANDB_API_KEY_METADATA}" \
    gcr.io/[YOUR-PROJECT-ID]/openrlhf-m:latest sleep infinity

# 5. Run the GCS setup script inside the container
sudo docker exec openrlhf-worker-container bash /app/OpenRLHF-M/examples/scripts/setup/setup_gcs_mounts.sh

# 6. Connect to the head node's Ray process inside the container
#    Replace <HEAD_IP> below with the actual IP address printed by the 'echo' command above
export HEAD_IP=$HEAD_IP # Set this manually inside the VM shell before running docker exec
sudo docker exec openrlhf-worker-container ray start --address=${HEAD_IP}:6379
```

### 6. Run the Training Job

Connect to the **head node VM** via SSH if you aren't already connected. Then, execute the training script **inside the running head container**:

Edit these lines to the correct values for your environment:

# Login to the container to check it is running and good:
```bash
sudo docker exec -it openrlhf-head-container bash
vi /app/OpenRLHF-M/examples/scripts/tests/train_grpo_ray_qwen2_5_vl_mathv60k_multinode_gcp.sh

```

# Configure the following parameters as needed:
GCS_BUCKET="${GCS_BUCKET:-gs://[YOUR_BUCKET]}"
DATASET_PATH="${DATASET_PATH:-${GCS_BUCKET}/datasets/VerMulti/mathv60k_message.jsonl}"
PRETRAIN_MODEL_PATH="${PRETRAIN_MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
SAVE_PATH="${SAVE_PATH:-${GCS_BUCKET}/checkpoints}"
MODEL_NAME="${MODEL_NAME:-qwen2.5-vl-3b-ins-mathvista-grpo}"
EXPECTED_WORKERS="${EXPECTED_WORKERS:-1}"

# NCCL configuration for GCP networking
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ens7


```bash
# SSH into head node (if not already connected)
# gcloud compute ssh openrlhf-head --zone=us-central1-a

# Execute training script inside the head container
sudo docker exec openrlhf-head-container bash /app/OpenRLHF-M/examples/scripts/tests/train_grpo_ray_qwen2_5_vl_mathv60k_multinode_gcp.sh
```




## Deployment Options

This guide now describes three different deployment options that you can choose from, depending on your management and scaling preferences.

---

### **Option 1: Standard Deployment Using Individual VMs**

In this option, you manually provision your training environment by creating individual VM instances using gcloud. This gives you complete control over each VM's configuration (machine type, GPU accelerators, boot disk size, container images, etc.).

#### Creating the VMs

For example, to create a head node and a worker node using the `a2-ultragpu-2g` machine type with 8 A100 GPUs (adjust the accelerator count as needed for your GPU requirements), run:

```bash
# Choose the desired PyTorch image family (check Deep Learning Images documentation for latest)
IMAGE_FAMILY="pytorch-latest-gpu" 
DL_PROJECT="deeplearning-platform-release"

# Create the head node VM using a Deep Learning VM Image
gcloud compute instances create openrlhf-head \
    --zone=us-central1-a \
    --machine-type=a2-ultragpu-2g \
    --maintenance-policy=TERMINATE \
    --image-family="${IMAGE_FAMILY}" \
    --image-project="${DL_PROJECT}" \
    --boot-disk-size=200GB \
    --accelerator=type=nvidia-a100-80gb,count=8 \
    --scopes=cloud-platform \
    --metadata=GCS_BUCKET=gs://[YOUR-BUCKET],WANDB_API_KEY=your-wandb-api-key-here

# Create the worker node VM using a Deep Learning VM Image
gcloud compute instances create openrlhf-worker1 \
    --zone=us-central1-a \
    --machine-type=a2-ultragpu-2g \
    --maintenance-policy=TERMINATE \
    --image-family="${IMAGE_FAMILY}" \
    --image-project="${DL_PROJECT}" \
    --boot-disk-size=200GB \
    --accelerator=type=nvidia-a100-80gb,count=8 \
    --scopes=cloud-platform \
    --metadata=GCS_BUCKET=gs://[YOUR-BUCKET],WANDB_API_KEY=your-wandb-api-key-here
```
*Note:* After creating these VMs, you would follow steps similar to Section 5 (Connect to the VMs...) to configure Docker, pull/run your container, and start Ray on each node before launching the training job as shown in Section 6.

---

### **Option 2: Alternative Deployment Using Google Kubernetes Engine (GKE)**

Deploying your training environment on GKE lets you benefit from a fully managed Kubernetes service. This option makes scaling easier by managing a GPU-enabled node pool and deploying your application as pods.

#### Step 1: Create a GPU-Enabled GKE Cluster

Use these commands to create a new cluster with no default nodes and then add a GPU-enabled node pool with 4 nodes (each configured with 8 GPUs):

```bash
# Create a GKE cluster (with no default nodes)
gcloud container clusters create openrlhf-cluster \
    --zone us-central1-a \
    --num-nodes=0 \
    --enable-ip-alias

# Create a GPU-enabled node pool with 4 nodes
gcloud container node-pools create gpu-pool \
    --cluster=openrlhf-cluster \
    --zone=us-central1-a \
    --num-nodes=4 \
    --machine-type=a2-ultragpu-2g \
    --accelerator=type=nvidia-a100-80gb,count=8 \
    --metadata=disable-legacy-endpoints=true
```

#### Step 2: Deploy Training Pods with a Kubernetes Manifest

Below is an example YAML manifest that deploys one head pod and three worker pods (totaling 4 nodes). Note the inclusion of an init container that handles GCS mounting:

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
      initContainers:
      - name: gcs-mount-setup
        image: gcr.io/[YOUR-PROJECT-ID]/openrlhf-m:latest
        command: ["/bin/bash", "/app/OpenRLHF-M/setup_gcs_mounts.sh"]
        env:
        - name: GCS_BUCKET
          value: "gs://[YOUR-BUCKET]"
        securityContext:
          privileged: true  # Required for gcsfuse
        volumeMounts:
        - name: fuse
          mountPath: /dev/fuse
        - name: cache-volume
          mountPath: /root/.cache
      containers:
      - name: head-container
        image: gcr.io/[YOUR-PROJECT-ID]/openrlhf-m:latest
        env:
        - name: GCS_BUCKET
          value: "gs://[YOUR-BUCKET]"
        - name: WANDB_API_KEY
          value: "your-wandb-api-key-here"
        securityContext:
          privileged: true  # Required for gcsfuse
        resources:
          limits:
            nvidia.com/gpu: "8"
        volumeMounts:
        - name: fuse
          mountPath: /dev/fuse
        - name: cache-volume
          mountPath: /root/.cache
      volumes:
      - name: fuse
        hostPath:
          path: /dev/fuse
      - name: cache-volume
        emptyDir: {}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: openrlhf-worker
spec:
  replicas: 3
  selector:
    matchLabels:
      app: openrlhf-worker
  template:
    metadata:
      labels:
        app: openrlhf-worker
    spec:
      initContainers:
      - name: gcs-mount-setup
        image: gcr.io/[YOUR-PROJECT-ID]/openrlhf-m:latest
        command: ["/bin/bash", "/app/OpenRLHF-M/setup_gcs_mounts.sh"]
        env:
        - name: GCS_BUCKET
          value: "gs://[YOUR-BUCKET]"
        securityContext:
          privileged: true  # Required for gcsfuse
        volumeMounts:
        - name: fuse
          mountPath: /dev/fuse
        - name: cache-volume
          mountPath: /root/.cache
      containers:
      - name: worker-container
        image: gcr.io/[YOUR-PROJECT-ID]/openrlhf-m:latest
        env:
        - name: GCS_BUCKET
          value: "gs://[YOUR-BUCKET]"
        - name: WANDB_API_KEY
          value: "your-wandb-api-key-here"
        - name: GCP_WORKER
          value: "1"
        securityContext:
          privileged: true  # Required for gcsfuse
        resources:
          limits:
            nvidia.com/gpu: "8"
        volumeMounts:
        - name: fuse
          mountPath: /dev/fuse
        - name: cache-volume
          mountPath: /root/.cache
      volumes:
      - name: fuse
        hostPath:
          path: /dev/fuse
      - name: cache-volume
        emptyDir: {}
```

Deploy this manifest with:

```bash
kubectl apply -f your-deployment.yaml
```

*Note:* Adjust the replica counts and resource limits as needed.

---

### **Option 3: Managed Deployment Using Vertex AI Ray Clusters**

Vertex AI Ray Clusters offer a managed solution where you define your cluster configuration in a YAML file and let Vertex AI handle resource provisioning and cluster management.

#### Step 1: Create a Ray Cluster Configuration File

Create a file named `ray_cluster_config.yaml` with the following content:

```yaml
cluster_name: openrlhf-ray-cluster
project_id: [YOUR-PROJECT-ID]
region: us-central1
min_replicas: 2
max_replicas: 2
head_node_type: a2-ultragpu-2g
worker_node_types:
  - machine_type: a2-ultragpu-2g
    accelerator_type: nvidia-a100-80gb
    accelerator_count: 8
    min_replicas: 1
    max_replicas: 1
container_image: gcr.io/[YOUR-PROJECT-ID]/openrlhf-m:latest
env_vars:
  GCS_BUCKET: gs://[YOUR-BUCKET]
  WANDB_API_KEY: "your-wandb-api-key-here"
# Add startup script to run for each node on cluster
# This will ensure GCS buckets are mounted and caches properly set up
startup_script: |
  #!/bin/bash
  # Setup model caching with GCS bucket
  GCS_BUCKET="${GCS_BUCKET:-gs://[YOUR_BUCKET]}"
  GCS_BUCKET_NAME=$(echo ${GCS_BUCKET} | sed 's|gs://||')

  echo "Setting up GCS mount and HuggingFace cache symlinks..."
  # Mount GCS bucket if not already mounted
  if ! grep -q "/mnt/gcs-cache" /proc/mounts; then
      echo "Mounting GCS bucket ${GCS_BUCKET} to /mnt/gcs-cache"
      gcsfuse --implicit-dirs ${GCS_BUCKET_NAME} /mnt/gcs-cache
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

  # Ensure the model checkpoint directory exists
  mkdir -p /mnt/gcs-cache/checkpoints
```

#### Step 2: Create the Ray Cluster via Command Line

Install the Vertex AI SDK if you haven't already:

```bash
pip install google-cloud-aiplatform
```

Then run the following Python command:

```bash
python -c "
from google.cloud import aiplatform
aiplatform.init(
    project='[YOUR-PROJECT-ID]',
    location='us-central1',
)
cluster = aiplatform.RayCluster.create_from_yaml('ray_cluster_config.yaml')
print(f'Ray cluster dashboard URL: {cluster.get_dashboard_uri()}')
"
```

This will create a managed Ray cluster and print out the URL for the Ray dashboard.

For additional details on using Ray on Vertex AI, refer to the [Ray on Vertex AI documentation](https://cloud.google.com/vertex-ai/docs/open-source/ray-on-vertex-ai/create-cluster) and the [Vertex AI Custom Training documentation](https://cloud.google.com/vertex-ai/docs/training/create-custom-job).

---

## Monitoring and Managing Your Training Job

### Ray Dashboard

The Ray dashboard is available on port 8265. To access it:

```bash
# Create an SSH tunnel to the head node's Ray dashboard
gcloud compute ssh openrlhf-head -- -L 8265:localhost:8265
```

Then open `http://localhost:8265` in your browser.

### View Training Logs

```bash
# SSH into the head node
gcloud compute ssh openrlhf-head

# View the training logs
cd /app/OpenRLHF-M
tail -f logs/qwen2.5-vl-3b-ins-mathvista-grpo/*/train.log
```

### Monitor GPU Usage

```bash
# On either node
watch -n1 nvidia-smi
```

### Accessing Saved Models

Your trained models will be saved to the GCS bucket specified in the script:

```bash
# List saved checkpoints
gsutil ls gs://[YOUR-BUCKET]/checkpoints/qwen2.5-vl-3b-ins-mathvista-grpo/
```

## Troubleshooting

### Cache and Model Loading Issues

If you encounter issues with model loading:
1. Verify the GCS bucket is properly mounted with `df -h | grep gcs`
2. Check symlinks are correctly set up with `ls -la ~/.cache/`
3. Verify the model files exist in the cache with `find ~/.cache/huggingface -name "*.bin" | head`
4. Check GCS permissions - ensure the service account has Storage Object Viewer role

### Disk Space Management

With the caching system in place, monitor disk space usage:
1. Check local disk usage: `df -h`
2. Check cache size: `du -sh ~/.cache/*`
3. If necessary, clear non-essential cache: `rm -rf ~/.cache/torch/hub` (not huggingface cache)
4. If disk space is still an issue, increase the boot disk size when creating VMs

### Ray Cluster Issues

If nodes fail to connect:
1. Verify that the firewall allows traffic on ports 6379 (Ray) and 8265 (dashboard)
2. Check that the IP address used for the head node is correct
3. Ensure the worker node can reach the head node (test with `ping` or `nc`)

### GPU Issues

If you encounter GPU-related errors:
1. Verify GPU drivers are installed and working with `nvidia-smi`
2. Check CUDA compatibility between your code and the driver
3. Adjust memory utilization parameters if you encounter OOM errors

### NCCL Communication Errors

For NCCL-related errors:
1. Verify network settings are correctly configured
2. Try adjusting `NCCL_SOCKET_IFNAME` to match your actual network interface
3. Increase verbosity with `export NCCL_DEBUG=TRACE` to get more detailed logs

## Cost Optimization

A2 GPU instances are expensive. To minimize costs:
1. Use preemptible VMs when possible for non-critical workloads
2. Shut down VMs when not in use
3. Store checkpoints frequently to resume from interruptions
4. Consider using smaller GPU types for testing your setup before scaling to A100s

## Conclusion

This guide provides a comprehensive approach to running OpenRLHF-M on GCP Vertex AI using Docker containers. The container-based approach offers reproducibility, scalability, and alignment with GCP best practices for ML workloads.

For multinode training workloads, GCP's environment provides excellent networking performance and the ability to easily scale resources based on your training needs.

## References

- [GCP Vertex AI Documentation](https://cloud.google.com/vertex-ai)
- [Ray on Vertex AI](https://cloud.google.com/vertex-ai/docs/open-source/ray-on-vertex-ai/overview)
- [Ray Documentation](https://docs.ray.io/)
- [OpenRLHF Documentation](https://openrlhf.readthedocs.io/)
- [Containerizing ML Workloads](https://cloud.google.com/architecture/ml-on-gcp-best-practices) 

