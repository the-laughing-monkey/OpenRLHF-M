# Running OpenRLHF-M on GCP Vertex AI

This document provides comprehensive instructions for deploying and running OpenRLHF-M on GCP Vertex AI using a Docker container approach. This methodology ensures reproducible environments and follows GCP best practices for ML workloads.

## Overview

In this setup, we'll use:
- **Container Registry**: For storing our custom Docker image with all dependencies
- **Cloud Storage**: For datasets and model checkpoints
- **Vertex AI VMs**: a2-ultragpu-2g instances (2x A100 GPUs) for training
- **Ray**: For distributed multinode training

## Prerequisites

- A Google Cloud Platform account with Vertex AI API enabled
- Google Cloud SDK (`gcloud`) installed and configured on your local machine
- Docker installed on your local machine
- Sufficient quota for a2-ultragpu-2g instances in your preferred region
- A service account with necessary permissions:
  - Vertex AI User
  - Storage Admin
  - Container Registry Writer

## Setup Process

### 1. Create a GCS Bucket

First, create a Google Cloud Storage bucket to store your datasets, model checkpoints, and logs:

```bash
# Create a GCS bucket (replace [YOUR-PROJECT] and [YOUR-BUCKET] with your values)
gsutil mb -p [YOUR-PROJECT] -l us-central1 gs://[YOUR-BUCKET]
```

### 2. Build a Custom Docker Container

Create a Dockerfile that includes all necessary dependencies for OpenRLHF-M:

```bash
# Create a directory for your Dockerfile
mkdir -p openrlhf-docker && cd openrlhf-docker

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

# Copy the training script
COPY train_grpo_ray_qwen2_5_vl_mathv60k_multinode_gcp.sh /app/OpenRLHF-M/examples/scripts/tests/

# Set proper permissions
RUN chmod +x /app/OpenRLHF-M/examples/scripts/tests/train_grpo_ray_qwen2_5_vl_mathv60k_multinode_gcp.sh

# Set working directory
WORKDIR /app/OpenRLHF-M

# Default command
CMD ["/bin/bash"]
EOF

# Create entrypoint script as a wrapper for the training script
cat > train_grpo_ray_qwen2_5_vl_mathv60k_multinode_gcp.sh << 'EOF'
#!/bin/bash
#=============================================================================
# GCP Vertex AI Simplified Multinode Training Script for OpenRLHF-M MathV60K
#=============================================================================
# Usage:
#   On the head node: simply run the script (without GCP_WORKER set).
#   On worker nodes: set GCP_WORKER=1 and GCP_HEAD_IP to the head node's IP address.
#
# Configure the following parameters as needed:
GCS_BUCKET="${GCS_BUCKET:-gs://[YOUR_BUCKET]}"
DATASET_PATH="${DATASET_PATH:-${GCS_BUCKET}/datasets/VerMulti/mathv60k_message.jsonl}"
PRETRAIN_MODEL_PATH="${PRETRAIN_MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
SAVE_PATH="${SAVE_PATH:-${GCS_BUCKET}/checkpoints}"
MODEL_NAME="${MODEL_NAME:-qwen2.5-vl-3b-ins-mathvista-grpo}"
EXPECTED_WORKERS="${EXPECTED_WORKERS:-1}"

# NCCL configuration for GCP networking
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0

RAY_PORT=6379
DASHBOARD_PORT=8265
REWARD_MODEL_PORT=5000

# Check WandB API key
if [ -z "${WANDB_API_KEY}" ]; then
  echo "[INFO] WANDB_API_KEY not set. WandB logging will be disabled."
  WANDB_ARGS=""
else
  echo "[INFO] WANDB_API_KEY found. WandB logging enabled."
  WANDB_ARGS="--use_wandb ${WANDB_API_KEY} --wandb_run_name ${MODEL_NAME} --wandb_group \"openrlhf-m-gcp\""
fi

# Setup logging directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/${MODEL_NAME}"
CUR_LOG_DIR="${LOG_DIR}/${TIMESTAMP}"
mkdir -p "${CUR_LOG_DIR}"

if [ -z "${GCP_WORKER}" ]; then
    echo "=========================================================="
    echo "Starting as HEAD NODE"
    echo "=========================================================="
    ray stop || true
    ray start --head --node-ip-address=0.0.0.0 --port=${RAY_PORT} --dashboard-port=${DASHBOARD_PORT} --num-gpus=2
    echo "Ray head started. Dashboard available at http://localhost:${DASHBOARD_PORT}"

    # Start the remote reward model server
    echo "Starting remote reward model server..."
    python -m openrlhf.models.remote_rm.math_verifier \
        --dataset "${DATASET_PATH}" \
        --input_key message \
        --prompt-template chatml 2>&1 | tee -a "${CUR_LOG_DIR}/remote_rm.log" &
    REMOTE_RM_PID=$!
    
    echo "Waiting for remote reward model server to initialize..."
    sleep 10
    
    # Get this node's IP address for reward model URL
    HEAD_IP=$(hostname -i)
    REWARD_MODEL_URL="http://${HEAD_IP}:${REWARD_MODEL_PORT}/get_reward"
    
    # Wait for worker nodes
    echo "Waiting for worker nodes to join the cluster..."
    WORKER_RETRY=0
    MAX_WORKER_RETRIES=20
    
    while true; do
      WORKER_COUNT=$(ray status 2>/dev/null | grep "alive" | wc -l)
      WORKER_COUNT=$((WORKER_COUNT - 1)) # Subtract 1 for head node
      
      if [ $WORKER_COUNT -ge $EXPECTED_WORKERS ]; then
        echo "All expected worker nodes ($EXPECTED_WORKERS) have joined."
        break
      fi
      
      echo "Waiting for workers... ($WORKER_COUNT/$EXPECTED_WORKERS joined)"
      sleep 10
      WORKER_RETRY=$((WORKER_RETRY+1))
      
      if [ $WORKER_RETRY -ge $MAX_WORKER_RETRIES ]; then
        echo "[WARNING] Not all workers joined. Proceeding with $WORKER_COUNT workers."
        break
      fi
    done
    
    # Calculate GPU parameters based on detected worker count
    TOTAL_NODES=$((WORKER_COUNT + 1)) # Including head node
    TOTAL_GPUS_PER_NODE=8
    TOTAL_GPUS=$((TOTAL_GPUS_PER_NODE * TOTAL_NODES))

    # Total actor GPUs available (actor_num_nodes * actor_num_gpus_per_node) equals TOTAL_GPUS
    ACTOR_TOTAL_GPUS=${TOTAL_GPUS}

    # Set VLLM_NUM_ENGINES to 8 if sufficient GPUs are available; otherwise use all available GPUs
    if [ ${ACTOR_TOTAL_GPUS} -ge 8 ]; then
        VLLM_NUM_ENGINES=8
    else
        VLLM_NUM_ENGINES=${ACTOR_TOTAL_GPUS}
    fi

    # Set VLLM_TENSOR_PARALLEL_SIZE so that its product with VLLM_NUM_ENGINES equals ACTOR_TOTAL_GPUS
    VLLM_TENSOR_PARALLEL_SIZE=$(( ACTOR_TOTAL_GPUS / VLLM_NUM_ENGINES ))

    echo "Submitting training job..."
    ray job submit --address="http://127.0.0.1:${DASHBOARD_PORT}" \
      --runtime-env-json='{"working_dir": "$(pwd)"}' \
      -- python3 -m openrlhf.cli.train_ppo_ray \
         --ref_num_nodes 1 \
         --ref_num_gpus_per_node ${TOTAL_GPUS} \
         --remote_rm_url "${REWARD_MODEL_URL}" \
         --actor_num_nodes 1 \
         --actor_num_gpus_per_node ${TOTAL_GPUS} \
         --vllm_num_engines ${VLLM_NUM_ENGINES} \
         --vllm_tensor_parallel_size ${VLLM_TENSOR_PARALLEL_SIZE} \
         --colocate_all_models \
         --vllm_enable_sleep \
         --vllm_gpu_memory_utilization 0.4 \
         --vllm_sync_backend nccl \
         --enable_prefix_caching \
         --pretrain ${PRETRAIN_MODEL_PATH} \
         --save_path ${SAVE_PATH}/${MODEL_NAME} \
         --micro_train_batch_size 1 \
         --train_batch_size 128 \
         --micro_rollout_batch_size 1 \
         --rollout_batch_size 128 \
         --temperature 1.0 \
         --n_samples_per_prompt 4 \
         --max_epochs 1 \
         --num_episodes 2 \
         --prompt_max_len 128000 \
         --max_samples 1000 \
         --generate_max_len 8000 \
         --advantage_estimator group_norm \
         --use_kl_loss \
         --kl_estimator k3 \
         --init_kl_coef 1e-3 \
         --zero_stage 3 \
         --bf16 \
         --actor_learning_rate 5e-7 \
         --prompt_data ${DATASET_PATH} \
         --input_key message \
         --normalize_reward \
         --flash_attn \
         --lambd 1 \
         --gamma 1 \
         --gradient_checkpointing \
         --save_steps 10 \
         --max_ckpt_num 2 \
         --ckpt_path ${SAVE_PATH}/${MODEL_NAME}/ckpt \
         --save_hf_ckpt \
         ${WANDB_ARGS} \
         --use_tensorboard ${LOG_DIR} > "${CUR_LOG_DIR}/train.log" 2>&1
else
    echo "=========================================================="
    echo "Starting as WORKER NODE"
    echo "=========================================================="
    if [ -z "${GCP_HEAD_IP}" ]; then
        echo "[ERROR] GCP_HEAD_IP must be set for worker nodes."
        exit 1
    fi
    ray stop || true
    echo "Waiting for head node at ${GCP_HEAD_IP}:${RAY_PORT}..."
    until nc -z ${GCP_HEAD_IP} ${RAY_PORT} 2>/dev/null; do
       echo "Waiting..."
       sleep 5
    done
    ray start --address=${GCP_HEAD_IP}:${RAY_PORT} --num-gpus=2
    echo "Worker node connected to Ray cluster."
    while true; do sleep 60; done
fi
EOF

# Build the Docker image (replace [YOUR-PROJECT-ID] with your GCP project ID)
docker build -t gcr.io/[YOUR-PROJECT-ID]/openrlhf-m:latest .

# Push the Docker image to GCP Container Registry
docker push gcr.io/[YOUR-PROJECT-ID]/openrlhf-m:latest
```

### 3. Prepare the MathV60K Dataset

You'll need to download and prepare the MathV60K dataset for use with OpenRLHF-M.

#### Option 1: Prepare Locally and Upload to GCS

```bash
# Clone the OpenRLHF-M repository locally if you haven't already
git clone https://github.com/the-laughing-monkey/OpenRLHF-M.git
cd OpenRLHF-M

# Install requirements for dataset preparation
pip install -r requirements.txt

# Download and prepare the dataset
python examples/scripts/data_downloaders/download_mathv60k.py --root_dir ./datasets/VerMulti

# Upload the prepared dataset to GCS (replace with your bucket)
gsutil -m cp -r ./datasets/VerMulti gs://[YOUR-BUCKET]/datasets/
```

#### Option 2: Prepare Directly on a Temporary VM

```bash
# Create a VM to prepare the dataset
gcloud compute instances create dataset-prep \
    --machine-type=n1-standard-8 \
    --boot-disk-size=100GB \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --scopes=cloud-platform

# SSH into the VM
gcloud compute ssh dataset-prep

# Then on the VM:
git clone https://github.com/the-laughing-monkey/OpenRLHF-M.git
cd OpenRLHF-M
pip install -r requirements.txt
python examples/scripts/data_downloaders/download_mathv60k.py --root_dir ./datasets/VerMulti
gsutil -m cp -r ./datasets/VerMulti gs://[YOUR-BUCKET]/datasets/

# Exit and delete the VM when done
exit
gcloud compute instances delete dataset-prep
```

### 4. Create a2-ultragpu-2g VMs for Training

Now, create head and worker nodes using the a2-ultragpu-2g machine type:

```bash
# Create the head node VM
gcloud compute instances create openrlhf-head \
    --machine-type=a2-ultragpu-2g \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=200GB \
    --accelerator=type=nvidia-a100-80gb,count=2 \
    --container-image=gcr.io/[YOUR-PROJECT-ID]/openrlhf-m:latest \
    --scopes=cloud-platform \
    --metadata=GCS_BUCKET=gs://[YOUR-BUCKET]

# Create the worker node VM
gcloud compute instances create openrlhf-worker1 \
    --machine-type=a2-ultragpu-2g \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=200GB \
    --accelerator=type=nvidia-a100-80gb,count=2 \
    --container-image=gcr.io/[YOUR-PROJECT-ID]/openrlhf-m:latest \
    --scopes=cloud-platform \
    --metadata=GCS_BUCKET=gs://[YOUR-BUCKET]
```

### 5. Run the Training Job

#### On the Head Node:

```bash
# SSH into the head node
gcloud compute ssh openrlhf-head

# Start the training process
cd /app/OpenRLHF-M
bash examples/scripts/tests/train_grpo_ray_qwen2_5_vl_mathv60k_multinode_gcp.sh
```

#### On the Worker Node:

```bash
# SSH into the worker node
gcloud compute ssh openrlhf-worker1

# Get the head node's internal IP
HEAD_IP=$(gcloud compute instances describe openrlhf-head \
    --format='get(networkInterfaces[0].networkIP)')

# Start the worker process
cd /app/OpenRLHF-M
GCP_WORKER=1 GCP_HEAD_IP=$HEAD_IP bash examples/scripts/tests/train_grpo_ray_qwen2_5_vl_mathv60k_multinode_gcp.sh
```

## Using Vertex AI Ray Clusters (Alternative Approach)

GCP also offers Ray on Vertex AI, which can simplify cluster management:

```bash
# Install the Vertex AI SDK
pip install google-cloud-aiplatform

# Create a Ray cluster configuration file
cat > ray_cluster_config.yaml << EOF
cluster_name: openrlhf-ray-cluster
project_id: [YOUR-PROJECT-ID]
region: us-central1
min_replicas: 2
max_replicas: 2
head_node_type: a2-ultragpu-2g
worker_node_types:
  - machine_type: a2-ultragpu-2g
    accelerator_type: nvidia-a100-80gb
    accelerator_count: 2
    min_replicas: 1
    max_replicas: 1
container_image: gcr.io/[YOUR-PROJECT-ID]/openrlhf-m:latest
env_vars:
  GCS_BUCKET: gs://[YOUR-BUCKET]
EOF

# Import and run the Python script to create the Ray cluster
python -c "
from google.cloud import aiplatform

# Initialize the Vertex AI SDK
aiplatform.init(
    project='[YOUR-PROJECT-ID]',
    location='us-central1',
)

# Create the Ray cluster
cluster = aiplatform.RayCluster.create_from_yaml('ray_cluster_config.yaml')
print(f'Ray cluster dashboard URL: {cluster.get_dashboard_uri()}')
"
```

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