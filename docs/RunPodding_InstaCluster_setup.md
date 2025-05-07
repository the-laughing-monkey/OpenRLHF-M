# Running OpenRLHF-M on RunPod: The Ultimate Deployment Guide

Welcome to the OpenRLHF-M RunPodding guide!  
Here, we show you how to unleash OpenRLHF-M on a RunPod instance. Whether you're working on language model alignment or multimodal reinforcement learning, this guide will get you set up with a system that boasts 1000GB storage and the best distributed training practices in town.

---

## Prerequisites

- A RunPod account with sufficient credits
- Basic familiarity with Linux commands and SSH
- An SSH client on your local machine  
- Patience—and maybe a snack!
- When using Ray for distributed training, **ensure that port 6379 is open** on the Ray head node for worker join and **port 8265 is open** for job submission and dashboard access. These ports are used for:
  - **Worker Join:** The head node listens on `http://<HEAD_NODE_IP>:6379` for worker nodes connecting to the cluster.
  - **Job Submission & Dashboard:** The head node listens on `http://<HEAD_NODE_IP>:8265` for job submissions and dashboard communications.
- **Before running any training or job submission scripts, be sure to start Ray on your pod.**  
  Please refer to the [Ray documentation](https://docs.ray.io/).

---

## Step-by-Step Instructions

### 1. Create Your Storage Volume

1. Log in to your RunPod account.
2. Navigate to the **Volumes** section and click **Create Volume**.
3. Name your volume (we recommend `data`) and set the size (500GB or 1000GB).
4. Select your preferred datacenter and click **Create**.

### 2. Launch Your Pod

1. In the **Insta Cluster** section, click **+ Deploy**.

Choose the number of nodes you want to deploy.

2. Choose a GPU-based template—our favorite is the Ubuntu-22.04 PyTorch template.

3. Set your GPU count (more than one if you like parallel power!).

4. Choose the correct template:
RunPod Pytorch 2.4.0  (by default it pickes 2.2.1)

5. Click **Edit Template** to adjust:
   - Container disk size (1000GB is a good start). # Ray cache is large
   - Attach your volume by mounting it to `/data`.
   - Enable a public IP.

6. Ensure that "ssh" and "start Jupyter Notebook" are checked, then click **Deploy**.

### 3. Configure Your SSH Access

1. Generate your SSH key locally:
```bash
   ssh-keygen -t my_runpod_key -C "your_email@example.com"
   cat ~/.ssh/my_runpod_key.pub
```
2. Log into your RunPod account and paste your public key under **SSH Public Keys**.

3. Once your pod is live, note its IP and SSH port. Then connect using:
```bash
   ssh -i ~/.ssh/my_runpod_key root@<POD_IP_ADDRESS> -p <SSH_PORT>
```

### 4. Updates your OS, Set Up Your Python Environment and Install OpenRLHF-M

1. Update the system and install Python tools:
```bash
   apt update && apt upgrade -y && apt install -y python3-pip python3-venv python3-dev build-essential git curl vim lsof net-tools rsync libopenmpi-dev build-essential dkms dnsutils dnsutils iputils-ping
```


2. Create a virtual environment in your data directory:


```bash
   mkdir /workspace
   cd /workspace
   python3 -m venv openrlhf-env
   source openrlhf-env/bin/activate
```

### 5. Clone the repository OpenRLHF-M repository and install it

```bash
  git clone https://github.com/the-laughing-monkey/OpenRLHF-M.git
```

### 6. Pip install huggingface hub so you can download models and data while the rest installs
```bash
  pip install huggingface_hub
```

# Now you can use the examples/scripts/downloaders/download_model.py or examples/scripts/downloaders/download_mathv60k.py to download models and a sample dataset in another terminal to work in parallel.

### 7. Run setup script to install everything else

On both nodes:
```bash
  cd ./OpenRLHF-M
  bash ./examples/scripts/setup/setup.sh
```

### 8. (Optional) Set Your WandB API Key

If you wish to use Weights & Biases (wandb) for experiment tracking, set your API key:

```bash
    export WANDB_API_KEY=YOUR_WANDB_API_KEY
```
This step is optional but recommended for more integrated experiment monitoring.


### 9. Set the file descriptor limit to 65536 to avoid "Too many open files" error
```bash
ulimit -n 65536
```
---

### 10. Prepare Your Cache

Move model caches to your larger `/data` volume to conserve space:
```bash
    mkdir -p /data/cache-models/huggingface/hub /data/cache-models/modelscope/hub /data/cache-ray
    rm -rf /root/.cache/huggingface && ln -s /data/cache-models/huggingface /root/.cache/huggingface
    rm -rf /root/.cache/modelscope && ln -s /data/cache-models/modelscope /root/.cache/modelscope
    # DO NOT DO THIS PART ON A NETWORK DISK. Insread make your local /root disk LARGE. Like 2GB.
    rm -rf /root/.cache/ray && ln -s /data/cache-ray /root/.cache/ray
    # Verify symlinks
    ls -la /root/.cache/
```

This is a critical step because:
- Model training checkpoints can be large (multiple GB each)
- The default container disk (50GB) will quickly fill up during training
- Moving these to your data volume (500GB-1000GB) prevents "No space left on device" errors

---

### 11. Download and Prepare the MathV60K Dataset

Before running a training job, you'll need to prepare the dataset:

1. Create the datasets directory:
```bash
    mkdir -p /data/datasets
```

2. Download and prepare the MathV60K dataset:

On the head node:
```bash
    cd /workspace/OpenRLHF-M
    python3 examples/scripts/downloaders/download_mathv60k.py --root_dir /data/datasets/VerMulti
```

This script will:
- Download the dataset files from Hugging Face
- Extract images to the specified directory
- Process the JSONL file to update paths
- Make the dataset ready for training

The script provides detailed progress information and will tell you when the dataset is ready.


### 12. Set your NCCL environment variables to use the eth1 interface:

CRITICAL: RunPod only allows internode communication over eth1. So you need to set your NCCL to use the eth1 IP or NCCL will fail to update weights across nodes.

```bash
    export NCCL_DEBUG=INFO
    export NCCL_SOCKET_IFNAME=eth1
```


### 13. Set your Ray head node to use the eth1 IP

CRITICAL: RunPod only allows internode communication over eth1. So you need to set your Ray head node to use the eth1 IP.

```bash
    export ETH1_IP=$(ip addr show eth1 | grep -oP 'inet \K[\d.]+')
    echo "Using ETH1 IP: ${ETH1_IP}"
```

2. Stop any running Ray instances (if any):
```bash
    ray stop
```

3. Start the Ray head node bound to all available IPs on eth1:
```bash
    ray start --head --node-ip-address $ETH1_IP --port=6379 --dashboard-port=8265   --temp-dir ~/.cache/ray
```

4. To connect a worker node to the Ray head node, run:
```bash
    export HEAD_NODE_IP={YOUR_HEAD_NODE_IP}
ray start \
  --address="${HEAD_NODE_IP}:6379" \
  --temp-dir ~/.cache/ray
```


### 14. Run Your First OpenRLHF-M Training Job with MathV60K

Now you're ready to launch a training job using the MathV60K dataset and the Qwen2.5-VL-3B model:

1. First, copy the script to your pod:

```bash
    cd /workspace/OpenRLHF-M
    mkdir -p ./scripts
    cp examples/scripts/tests/train_grpo_ray_qwen2_5_vl_mathv60k_multinode_simple.sh ./scripts/my_train_script.sh
```

2. Edit the script to match your pod's GPU configuration:

```bash
    nano my_train_script.sh
```

Make the following adjustments:


a. Update the GPU distribution in the training command. For example, if you have 2 GPUs:

# vllm_num_engines x vllm_tensor_parallel_size = number of actor_num_nodes x actor_num_gpus_per_node in one way or another. So this works:

```bash
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 1 \
```

# So does this:

```bash
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 2 \
```

# If you want to split a model across multiple nodes, you can do something like this, which would split it across 2 nodes with 8 GPUs each for a total of 16 GPUs:

```bash
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 16 \
    --vllm_num_engines 8 \
    --vllm_tensor_parallel_size 2 \
```

# Here is an example of a more complex GPU distribution:

```bash
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 8 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 8 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 2 \
    --colocate_all_models \
```

b. Make sure the working directory is set correctly:
```bash
    --runtime-env-json="{\"working_dir\": \"/data/OpenRLHF-M\"}" \
```

3. Run the adjusted training script:
```bash
    bash ./scripts/my_train_script.sh
```

4. **Important Disk Space Considerations:**

Disk space issues can cause training to fail when saving checkpoints. Adjust these parameters in your training script to prevent disk space problems:

```bash
# Save less frequently
--save_steps 50 \  # Default is often too frequent (e.g., 10)

# Limit number of checkpoints kept (default is 3)
--max_ckpt_num 2 \

# Control checkpoint format (smaller but less compatible)
--disable_ds_ckpt \ # Skip DeepSpeed format checkpoints which are larger
--save_hf_ckpt \    # While still saving HuggingFace format checkpoints (recommended)
```


### 15. Monitoring NVIDIA GPU Memory

To monitor the NVIDIA GPU memory usage while the script loads and runs, open a new terminal session (or use a multiplexer like tmux/screen) and run:

```bash
watch -n 1 nvidia-smi
```

# or

```bash
watch -n 1 "nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,temperature.gpu,fan.speed,memory.total,memory.used,memory.free --format=csv,noheader,nounits"
```

# or

```bash
watch -n 1 "echo 'GPU   Total(MiB)   Used(MiB)'; nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | awk -F',' '{printf \"%-3s %-12s %-10s\n\", \$1, \$2, \$3}'"
```

### 16. Monitoring and Managing Disk Space

Running out of disk space is a common issue during training. To monitor disk usage:

```bash
# Check overall disk usage
df -h

# Find largest directories and files in /root
du -h --max-depth=2 /root | sort -hr | head -20

# Find largest directories in Ray cache
du -h --max-depth=2 /data/cache-ray | sort -hr | head -20

# Find large checkpoint files
find /data -name "*.pt" -size +1G | xargs ls -lh
```

If you're running low on disk space despite using the data volume:

```bash
# Clear Triton cache (safe to delete)
rm -rf /root/.triton/autotune

# Clear older Ray session directories (if not using the symlink setup)
find /root/.cache/ray/session_* -maxdepth 0 -type d | sort | head -n -2 | xargs rm -rf

# Reduce checkpoint frequency in training scripts
# Example: Change --save_steps 10 to --save_steps 50

# Limit the number of checkpoints kept
# Example: Add --max_ckpt_num 2 to training arguments
```

For critical low disk situations, you can safely clear caches:

```bash
# Clear PyTorch hub cache
rm -rf /root/.cache/torch/hub/*

# Remove older checkpoints if needed
find /data/checkpoints -name "global_step*" | sort | head -n -2 | xargs rm -rf
```

### Troubleshooting

#### "Too many open files" error (Raylet or other Ray process)

This error occurs when a Ray process (like the raylet) exceeds the operating system's limit on the number of open file descriptors. Ray uses file descriptors for various resources, including network sockets for internal communication (gRPC). With a large number of tasks or actors, Ray can open many connections, hitting the default limit.

To resolve this, increase the file descriptor limit (`ulimit -n`) for the user running the Ray processes *before* starting Ray. A common value is 65536.

```bash
ulimit -n 65536
# Then run your ray start or training command
ray start --head ... # or ray start --address=...
```

You should apply this command in the terminal session *before* executing the Ray start command on both head and worker nodes.
