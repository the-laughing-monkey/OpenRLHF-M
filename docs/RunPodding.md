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

1. In the **Pods** section, click **+ Deploy**.

2. Choose a GPU-based template—our favorite is the Ubuntu-22.04 PyTorch template.

3. Set your GPU count (more than one if you like parallel power!).

4. Click **Edit Template** to adjust:
   - Container disk size (50GB is a good start).
   - Attach your volume by mounting it to `/data`.
   - Enable a public IP.
   - Set Expose TCP Ports to expose the default port and the Ray ports: `22,6379,8265`

5. Ensure that "ssh" and "start Jupyter Notebook" are checked, then click **Deploy**.

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

### 4. Set Up Your Python Environment and Install Packages

1. Update the system and install Python tools:
```bash
   apt update && apt upgrade -y && apt install -y python3-pip python3-venv python3-dev build-essential git curl vim lsof net-tools rsync libopenmpi-dev
```

2. Create a virtual environment in your data directory:
```bash
   cd /data
   python3 -m venv openrlhf-env
   source openrlhf-env/bin/activate
   
```

3. Clone the OpenRLHF-M repository and install:
```bash
   git clone https://github.com/OpenRLHF/OpenRLHF-M.git
   cd OpenRLHF-M
   pip install -e .[vllm]
```

4. Install Flash Attention Nightly Build for CUDA 12.7:
```bash
   pip uninstall -y flash-attn
   wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
   pip install flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
   rm -f flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```


### 6. Prepare Your Cache so it uses the larger network /data volume

Move model caches to your larger `/data` volume to conserve space:
```bash
mkdir -p /data/cache-models/huggingface/hub /data/cache-models/modelscope/hub
rm -rf /root/.cache/huggingface && ln -s /data/cache-models/huggingface /root/.cache/huggingface
rm -rf /root/.cache/modelscope && ln -s /data/cache-models/modelscope /root/.cache/modelscope
# Verify symlinks
ls -la /root/.cache/
```

---

### 7. (Optional) Set Your WandB API Key

If you wish to use Weights & Biases (wandb) for experiment tracking, consider setting your API key:
1. Sign up or log in at [Weights & Biases](https://wandb.ai/site) and obtain your API key.
2. On your pod, run:
```bash
   wandb login YOUR_WANDB_API_KEY
```
This step is optional but recommended for more integrated experiment monitoring.

---

### 8. Run Your First OpenRLHF-M Training Job

Now you're ready to launch a training job. For example, to train a Qwen2.5‑VL‑3B model with RLOO (Reinforcement Learning with Off-policy Updates):

1. First edit the script:

a. Change this: 
```bash
   PRETRAIN_MODEL="/root/projects/OpenRLHF/ckpts/Qwen2.5-VL-3B-Instruct"
```

b. To this so you can use the model from HuggingFace:
```bash
   PRETRAIN_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
```

c. Make sure to match the number of nodes and GPUs on your pod:
```bash
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 1 \
```

d. Set your dataset path to your actual dataset path:
```bash
   export DATASET="hiyouga/math12k"
```

e. Change the GPU number in Ray to match your pod:
```bash
   ray start --head --node-ip-address 0.0.0.0 --num-gpus 8 --temp-dir ~/.cache/ray
```

f. Finally set your working directory to your actual working directory:
```bash
   --runtime-env-json='{"working_dir": "/data/OpenRLHF-M"}' \
```


Now run the script:
```bash
   bash examples/scripts/train_rloo_qwenvl2_5_math.sh
```

### 9. Monitoring NVIDIA GPU Memory

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

---

ARCHIVE: Nightly build docs if you want specific version of packages:

- **NightlyFlash Attention:**
```bash
   pip uninstall -y flash-attn
   wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
   pip install flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
   rm -f flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

- **Nightly vLLM build:**
```bash
  pip uninstall -y vllm
  pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
```
  
For Transformers support (if using Qwen2.5‑VL models) until the next release:
```bash
pip install git+https://github.com/huggingface/transformers.git

OR it could be with a specific commit:

pip install --upgrade --force-reinstall git+https://github.com/huggingface/transformers@f3f6c86582611976e72be054675e2bf0abb5f775
```

### 6. Start Ray on Your Nodes Manually if you Want Instead of via Script

**Before launching any training scripts, you must start Ray on your pod.**

- **Head Node:**  
  On the head node, start Ray with:
```bash
  ray start --head --dashboard-host=0.0.0.0
```
  By default, this will configure job submission and the dashboard on port **8265**. (You can verify this by checking the output.)
  
- **Worker Nodes:**  
  Join the Ray cluster by running:
```bash
  ray start --address="203.0.113.5:6379"
```
  Replace `203.0.113.5` with your head node's actual public IP.

After executing the appropriate command on each node, verify that Ray is running using:
```bash
ray status
```

### 7. (Optional) Monitoring Ray Metrics with Prometheus and Grafana

In addition to monitoring GPUs and storage, you can set up Prometheus and Grafana in your Ray cluster so you can view detailed metrics.

#### Installing Prometheus

1. **Download Prometheus:**  
   Visit the [Prometheus Downloads page](https://prometheus.io/download/) and download the appropriate binary for your operating system.

2. **Configure Prometheus:**  
   Create a file named `prometheus.yml` with the following content. This configuration instructs Prometheus to scrape Ray's metrics from the service discovery file:
   ```yaml
   global:
     scrape_interval:     2s
     evaluation_interval: 2s

   scrape_configs:
   - job_name: 'ray'
     file_sd_configs:
     - files:
       - '/tmp/ray/prom_metrics_service_discovery.json'
   ```

3. **Start Prometheus:**  
   From the directory where you extracted Prometheus, run:
   ```bash
   ./prometheus --config.file=prometheus.yml
   ```
   By default, Prometheus will run on port 9090.

#### Installing Grafana

1. **Download Grafana:**  
   Visit the [Grafana Downloads page](https://grafana.com/grafana/download) and install Grafana for your operating system.

2. **Start Grafana:**  
   After installation, start Grafana (for example, on a Linux system you might run):
   ```bash
   sudo systemctl start grafana-server
   ```
   Grafana usually runs on port 3000. Open your browser at `http://localhost:3000` and log in (the default username and password are both `admin`).

3. **Configure Grafana to Use Prometheus:**  
   - In Grafana, add a new data source of type **Prometheus**.
   - Set the URL to point to your Prometheus server (for example, `http://localhost:9090`).
   - Optionally, you may import the Ray default dashboard JSON from your Ray session directory (typically at `/tmp/ray/session_latest/metrics/grafana/dashboards/default_grafana_dashboard.json`).

#### Configuring Ray to Export Metrics

When starting your Ray cluster, include the `--metrics-export-port=8080` flag to have Ray export metrics to that port. For example:
```bash
ray start --head --node-ip-address=0.0.0.0 --num-gpus 2 --metrics-export-port=8080 --temp-dir ~/.cache/ray
```
Prometheus (using the configuration above) will use the file-based service discovery (located at `/tmp/ray/prom_metrics_service_discovery.json`) to locate and scrape your Ray metrics.

Once you have Prometheus and Grafana running, you will have multiple options:
- **Using Ray Dashboard:** Access `http://127.0.0.1:8265` to view embedded Grafana visualizations.
- **Using Grafana directly:** Open `http://localhost:3000` and use your configured dashboard.

---

## 2. Update Your Shell Scripts to Use Metrics

You need to update the Ray start command in your job scripts so that Ray is launched with the metrics-export-port configured. For example, update the two scripts as follows:

### Updated `train_grpo_ray_hybrid_engine.sh`

Replace the existing Ray start command with one that includes the metrics-export-port flag. For example:

```shell
#!/bin/bash
# Use the math12k dataset from HF and our Qwen2.5‑VL–3B model with a formatted prompt.
export DATASET="hiyouga/math12k"

MODEL_CPK_NAME="qwenvl25_3B_ins_grpo_math"
PRETRAIN_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
SAVE_PATH="./ckpts"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}"

# Start the remote reward verifier (this will log its output for debugging)
python3 -m openrlhf.models.remote_rm.math_verifier \
    --dataset $DATASET \
    --input_key problem \
    --prompt-template "Question: {}\nAnswer:" \
    > "${SAVE_PATH}/${MODEL_CPK_NAME}/remote_rm.log" 2>&1 &
childpid=$!

# Start Ray on the head node with 2 GPUs and export metrics on port 8080.
ray start --head --node-ip-address 0.0.0.0 --num-gpus 2 --metrics-export-port=8080 --temp-dir ~/.cache/ray

# Submit the job using a runtime working directory of /data/OpenRLHF-M.
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/data/OpenRLHF-M"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 2 \
   --colocate_all_models \
   --vllm_enable_sleep \
   --vllm_gpu_memory_utilization 0.5 \
   --vllm_sync_backend gloo \
   --enable_prefix_caching \
   --pretrain $PRETRAIN_MODEL \
   --reward_pretrain $PRETRAIN_MODEL \
   --save_path $SAVE_PATH/$MODEL_CPK_NAME \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 256 \
   --temperature 1 \
   --n_samples_per_prompt 16 \
   --max_epochs 1 \
   --num_episodes 30 \
   --prompt_max_len 1024 \
   --max_samples 100000 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 1e-3 \
   --gamma 1.0 \
   --use_kl_loss \
   --kl_estimator k3 \
   --advantage_estimator group_norm \
   --prompt_data $DATASET \
   --input_key problem \
   --input_template "Question: {}\nAnswer:" \
   --normalize_reward \
   --flash_attn \
   --gradient_checkpointing \
   --packing_samples \
   --enforce_eager \
   --deepspeed_enable_sleep \
   --save_steps 10 \
   --ckpt_path $SAVE_PATH/$MODEL_CPK_NAME/ckpt \
   --save_hf_ckpt \
   --use_tensorboard $SAVE_PATH/$MODEL_CPK_NAME/logs

ray stop
```

### Updated `train_rloo_qwenvl2_5_math.sh`

Likewise, update the Ray start line:

```shell
export DATASET="hiyouga/math12k"

MODEL_CPK_NAME="qwenvl25_3B_ins_rloo_math"
PRETRAIN_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
SAVE_PATH="./ckpts"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}"

python3 -m openrlhf.models.remote_rm.math_verifier \
   --dataset $DATASET \
   --input_key problem \
   --prompt-template "Question: {}\nAnswer:" \
   > "${SAVE_PATH}/${MODEL_CPK_NAME}/remote_rm.log" 2>&1 &
childpid=$!

# Start Ray on the head node with 2 GPUs and export metrics on port 8080.
ray start --head --node-ip-address 0.0.0.0 --num-gpus 2 --metrics-export-port=8080 --temp-dir ~/.cache/ray

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/data/OpenRLHF-M"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --remote_rm_url http://127.0.0.1:5000/get_reward \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 2 \
   --colocate_all_models \
   --vllm_enable_sleep \
   --vllm_gpu_memory_utilization 0.5 \
   --vllm_sync_backend gloo \
   --enable_prefix_caching \
   --pretrain $PRETRAIN_MODEL \
   --save_path $SAVE_PATH/$MODEL_CPK_NAME \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 256 \
   --temperature 1 \
   --n_samples_per_prompt 16 \
   --max_epochs 1 \
   --num_episodes 30 \
   --prompt_max_len 1024 \
   --max_samples 100000 \
   --generate_max_len 3000 \
   --advantage_estimator rloo \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 1e-6 \
   --init_kl_coef 0.0 \
   --prompt_data $DATASET \
   --input_key problem \
   --input_template "Question: {}\nAnswer:" \
   --normalize_reward \
   --flash_attn \
   --gradient_checkpointing \
   --save_steps 10 \
   --ckpt_path $SAVE_PATH/$MODEL_CPK_NAME/ckpt \
   --save_hf_ckpt \
   --use_tensorboard $SAVE_PATH/$MODEL_CPK_NAME/logs

ray stop
```

---

## References

- [Ray Metrics Documentation](https://docs.ray.io/en/latest/cluster/metrics.html)
- [Prometheus Downloads](https://prometheus.io/download/)
- [Grafana Downloads](https://grafana.com/grafana/download)

With these updates, your Ray cluster will export metrics on port 8080, which you can scrape with Prometheus and visualize in Grafana. The updated documentation now explains the installation and configuration process, and your shell scripts are modified to set up Ray with these metrics.

---

## Final Thoughts

This guide ensures you have all the tools to deploy OpenRLHF-M on RunPod. OpenRLHF-M's architecture enables efficient distributed training of large language and multimodal models using various reinforcement learning techniques.

Happy RunPodding, and may your training sessions be as rewarding as they are fun!
