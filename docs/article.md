Conquering the 72B Parameter Beast: How To Train Qwen's Monster Multimodal Model on a 4-Node Cluster
Massive models used to be the exclusive domain of AI labs with billions in funding. Not anymore.
The democratization of AI continues at breakneck speed, and today I'm going to show you how to wield power that Google and OpenAI could only dream about just a few years ago – training a colossal 72B parameter multimodal model with nothing but some RunPod credits and the determination to see it through.
And you won't need $10 million in hardware or a special invitation to some closed-source AI priesthood. You just need grit, a bit of technical know-how, and this guide.
The Big AI Revolution Isn't Just About Models – It's About Who Gets To Train Them
Look, I'm going to be blunt: for too long, we've been told that training powerful AI models requires the resources of nation-states or megacorps. That's bullshit.
What if I told you that you could train a model that rivals Claude or GPT-4 in capabilities on your own hardware? That you could modify it, tune it for your specific use case, and control your own AI destiny?
This is what truly scares the monopolists. Not that AI will become sentient and destroy the world – but that their stranglehold on the technology will evaporate as distributed clusters of machines tear down the artificial barriers they've erected.
So let's stick it to them, shall we? Here's how.
Setting Up Your Distributed AI Beast-Mode Cluster
First things first – you need a cluster with enough GPU muscle to handle a 72B parameter model. With RunPod's InstaCluster, you can spin up a 4-node powerhouse in minutes rather than spending weeks begging your IT department for access to the corporate cluster.
Here's your shopping list:
RunPod Account with credits (Figure $50-150 depending on how long you train)
4 nodes with 8xA100s (or whatever monster GPUs you can get your hands on)
1000GB volume (Trust me, you'll thank me when you're not frantically deleting files mid-training)
Step 1: Create Your Storage Volume
Log in to your RunPod account.
Navigate to the Volumes section and click Create Volume.
Name your volume (I recommend data) and set the size (1000GB).
Select your preferred datacenter and click Create.
Step 2: Launch Your Cluster
Head to the Insta Cluster section and click + Deploy.
Choose 4 nodes with 8x A100s each.
Select the Ubuntu-22.04 PyTorch template.
Edit the template to:
Set container disk size to 50GB
Attach your volume to /data
Enable a public IP
Make sure SSH access is enabled, then click Deploy.
(What you've just done used to require a PhD, a million-dollar grant, and a year of planning. You did it in 10 minutes. Feel that power?)
Step 3: Set Up Your Environment
Once your cluster is up, SSH into the head node:

ssh -i ~/.ssh/your_key root@<HEAD_NODE_IP> -p <SSH_PORT>

Now, let's set up the environment:

# Update system and install dependencies
apt update && apt upgrade -y && apt install -y python3-pip python3-venv python3-dev build-essential git curl vim lsof net-tools rsync libopenmpi-dev build-essential dkms dnsutils iputils-ping

# Create and activate virtual environment
cd /data
python3 -m venv openrlhf-env
source openrlhf-env/bin/activate

# Clone the repo
git clone https://github.com/the-laughing-monkey/OpenRLHF-M.git
cd OpenRLHF-M

# Install dependencies
pip install huggingface_hub
bash ./examples/scripts/tests/setup/setup.sh

Step 4: Prepare Your Cache
This is the part that separates the pros from the wannabes. Move those cache directories to your data volume:

mkdir -p /data/cache-models/huggingface/hub /data/cache-models/modelscope/hub /data/cache-ray
rm -rf /root/.cache/huggingface && ln -s /data/cache-models/huggingface /root/.cache/huggingface
rm -rf /root/.cache/modelscope && ln -s /data/cache-models/modelscope /root/.cache/modelscope
rm -rf /root/.cache/ray && ln -s /data/cache-ray /root/.cache/ray

You'd be shocked how many enterprise AI projects fail because some PhD didn't think about cache management. Not you though. You're better than that.
Step 5: The Secret Sauce - Network Configuration
But hardware alone won't cut it. The secret sauce is in the configuration:

# Get the IP from eth1 - this is CRITICAL for multi-node training
export ETH1_IP=$(ip addr show eth1 | grep -oP 'inet \K[\d.]+')
echo "Your cluster will communicate over: ${ETH1_IP}"

# This is what the monopolists don't tell you - NCCL needs to use eth1!
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth1

# Stop any existing Ray instances
ray stop

# Start Ray with the right IP
ray start --head --node-ip-address ${ETH1_IP} --port=6379 --dashboard-port=8265

Why does this matter? Because 90% of distributed training failures happen at the networking layer. The big labs have entire teams solving these problems, but we're doing it with a couple of environment variables and pure force of will.
(Don't you love it when David can throw stones at Goliath without a team of network engineers?)

## Connecting Worker Nodes to Your Beast

Now for the real distributed power - adding worker nodes to your cluster. SSH into each worker node and run:

```bash
# On your worker nodes, set the head node's eth1 IP as an environment variable
export HEAD_NODE_IP={YOUR_HEAD_NODE_IP}  # Replace with your head node's actual eth1 IP from above

# Configure NCCL for eth1 or whatever nic your nodes connect on locally  - absolutely critical for multi-node training
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth1

# Stop any existing Ray instances just to be safe
ray stop

# Join the Ray cluster by connecting to the head node
export HEAD_NODE_IP={YOUR_HEAD_NODE_IP}
ray start --address=${HEAD_NODE_IP}:6379
```

Back on your head node, verify the connection with:

```bash
ray status
```

You should see all your worker nodes appear in the output. This is your supercomputer taking shape - a distributed AI training monster that would make the monopolists nervous.

It will look like this:

======== Autoscaler status: 2025-03-22 13:04:38.058023 ========
Node status
---------------------------------------------------------------
Active:
 1 node_44217b144c577d28786c822da9f4c18256d58b7b1dcb5d5393775e37
 1 node_ee352e747d1e4759df63703ece93b403c89ec3f0d088ca845c4cdec5
 1 node_d886fbc2822cc7b4b95b3e8bf813d8aaa88592e5f8c8e1c6806bfec1
 1 node_66b0e31ce8ecd00abf7b03d4ea813623751f2fee69761d71903b0510
Pending:
 (no pending nodes)
Recent failures:
 (no failures)

Resources
---------------------------------------------------------------
Usage:
 0.0/544.0 CPU
 0.0/32.0 GPU
 0B/4.75TiB memory
 0B/745.06GiB object_store_memory

Step 6: Download and Prepare the MathV60K Dataset
Let's get our training data ready:

mkdir -p /data/datasets
cd /data/OpenRLHF-M
python3 examples/scripts/data_downloaders/download_mathv60k.py --root_dir /data/datasets/VerMulti

This script handles all the dirty work – downloading from Hugging Face, extracting images, updating file paths – everything you need to get that sweet, sweet mathematics data ready for your 72B behemoth to learn from.
The Heart of the Beast: Your Training Script
The training script below isn't just code – it's a revolution in portable form. This is how you tell 32 A100 GPUs across 4 nodes to work in perfect harmony to create an AI that could have only existed in research papers just 18 months ago.

# Put this in a file called train_qwen2_5_vl_72b.sh
set -x

# Set environment variables and paths
NODE_HOSTNAME=$(hostname)
WORKSPACE_DIR="/data/OpenRLHF-M"
DATASET_PATH="/data/datasets/VerMulti/mathv60k_message.jsonl"
PRETRAIN_MODEL_PATH="Qwen/Qwen2.5-VL-72B-Instruct"
SAVE_PATH="./checkpoints"
MODEL_NAME="qwen2.5-vl-72b-ins-mathvista-grpo"
WANDB_DIR="${WORKSPACE_DIR}"

# NCCL Commands
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth1

# Suppress unhandled errors in Ray
export RAY_IGNORE_UNHANDLED_ERRORS=1

# Get the IP address of eth1 interface
ETH1_IP=$(ip addr show eth1 | grep -oP 'inet \K[\d.]+')
echo "Using eth1 IP address: ${ETH1_IP}"

# Start the remote reward model server and test connectivity
echo "[HEAD NODE] Starting remote reward model server..."
python3 -m openrlhf.models.remote_rm.math_verifier \
    --dataset "${DATASET_PATH}" \
    --input_key message \
    --prompt-template chatml &
REMOTE_RM_PID=$!

# Submit the training job
echo "[HEAD NODE] Submitting training job via Ray job submit..."
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="{\"working_dir\": \"${WORKSPACE_DIR}\"}" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
       --ref_num_nodes 1 \
       --ref_num_gpus_per_node 32 \
       --remote_rm_url http://${ETH1_IP}:5000/get_reward \
       --actor_num_nodes 1 \
       --actor_num_gpus_per_node 32 \
       --vllm_num_engines 8 \
       --vllm_tensor_parallel_size 8 \
       --vllm_enable_sleep \
       --vllm_sync_backend nccl \
       --vllm_gpu_memory_utilization 0.4 \
       --colocate_all_models \
       --enable_prefix_caching \
       --pretrain ${PRETRAIN_MODEL_PATH} \
       --save_path ${SAVE_PATH}/${MODEL_NAME} \
       --micro_train_batch_size 1 \
       --train_batch_size 64 \
       --micro_rollout_batch_size 1 \
       --rollout_batch_size 64 \
       --temperature 1.0 \
       --n_samples_per_prompt 4 \
       --max_epochs 1 \
       --num_episodes 2 \
       --prompt_max_len 4096 \
       --max_samples 1000 \
       --generate_max_len 8000 \
       --advantage_estimator group_norm \
       --use_kl_loss \
       --kl_estimator k3 \
       --init_kl_coef 1e-3 \
       --bf16 \
       --actor_learning_rate 5e-7 \
       --prompt_data ${DATASET_PATH} \
       --input_key message \
       --normalize_reward \
       --zero_stage 3 \
       --flash_attn \
       --lambd 1 \
       --gamma 1 \
       --gradient_checkpointing \
       --save_steps 50 \
       --max_ckpt_num 2 \
       --ckpt_path ${SAVE_PATH}/${MODEL_NAME}/ckpt \
       --save_hf_ckpt

Make it executable, then run it:

```bash
chmod +x train_qwen2_5_vl_72b.sh
./train_qwen2_5_vl_72b.sh
```

What's happening here is nothing short of miraculous – we're using Ray to orchestrate a complex dance of tensor parallel processing across multiple machines, with NCCL handling the low-level communication. 

This isn't just a script – **it's a declaration of independence from the AI gatekeepers.**

## The Math Behind the Madness

When you're training a 72B parameter model, the mathematics gets downright brutal. At 2 bytes per parameter for bfloat16, you're looking at 144GB just to store the model weights! Then factor in optimizer states, gradients, and forward activations, and you quickly realize why the monopolists thought they could keep this game to themselves.

But here's the trick they don't want you to know: **with the right tensor parallelism strategy, you can slice this monster into manageable chunks**.

By setting `vllm_tensor_parallel_size 8`, we're telling the system to split each layer across 8 GPUs. Combined with ZeRO stage 3 optimizer sharding (`--zero_stage 3`), we can train a model that would normally require a single GPU with over 500GB of memory (which doesn't exist) across commodity hardware.

## Monitoring Your Training Beast

While your training is running, you'll want to keep an eye on a few things:

1. **GPU Utilization**: Open a new terminal and run:

```bash
watch -n 1 "nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"
```

2. **Disk Space**: Monitor this religiously or weep later:

```bash
watch -n 300 "df -h && echo '---' && du -h --max-depth=1 /data | sort -hr | head -10"
```

3. **Ray Dashboard**: Access it at http://HEAD_NODE_IP:8265 in your browser.

If you see disk space getting tight, don't panic. Use these commands to clear space:

```bash
# Clear Ray cache (safe during training)
find /data/cache-ray -name "*.ray" -mtime +1 -delete

# Remove older checkpoints but keep the latest two
ls -t ${SAVE_PATH}/${MODEL_NAME}/ckpt/global_step* | tail -n +3 | xargs rm -rf
```

## The Open Source Revolution in Action

Let's be absolutely clear about what we're doing here. We're taking Qwen2.5-VL-72B, a model that rivals or exceeds GPT-4V and Claude 3 Opus in many capabilities, and we're fine-tuning it on specialized mathematics data using open-source tools that anyone can access.

Just five years ago, this would have required millions in specialized hardware, a team of PhDs, and a corporate budget. Today, it requires a credit card and a weekend.

**This is precisely why monopolists, jingoists, and doomers are so terrified of open-source AI.** Because when powerful tools get democratized, they lose control of the narrative and the margins.

## When It All Goes Wrong (And It Will)

Here's something the polished corporate blogs won't tell you: distributed training is a nightmare of random errors, inexplicable crashes, and moments where you'll question your life choices.

You'll see things like:

```
(LLMRayActor pid=32499, ip=10.65.0.3) ERROR 03-21 14:01:57 worker_base.py:581] socketStartConnect: Connect to 172.18.0.2<44801> failed : Software caused connection abort
```

And you'll want to throw your computer out the window. **Don't**.

These errors happen because we're pushing hardware to its absolute limits and asking software components that were never designed to work together to suddenly become best friends.

When (not if) this happens, try these steps:

1. **Check if NCCL is using the right interface**:
   ```bash
   export NCCL_DEBUG=TRACE  # More verbose than INFO
   export NCCL_SOCKET_IFNAME=eth1
   ```

2. **Ensure Ray is using the correct IP address**:
   ```bash
   ray stop
   export ETH1_IP=$(ip addr show eth1 | grep -oP 'inet \K[\d.]+')
   ray start --head --node-ip-address ${ETH1_IP} --port=6379 --dashboard-port=8265
   ```

3. **Test network connectivity between nodes**:
   ```bash
   # On head node
   nc -l -p 9999
   # On worker node
   echo "test" | nc ${ETH1_IP} 9999
   ```

Remember: **the "big boys" deal with these exact same issues**. They just have more people to throw at the problem.

## The Future Is Distributed or It Isn't

Let me make a prediction: within 5 years, training trillion-parameter models will be something hobbyists do on the weekend. Just like Linux grew from a Finnish student's side project to running 100% of the world's top supercomputers, open-source distributed AI training will evolve from "barely works if you're lucky" to "so robust a child could do it."

If that future is going to happen – if we're going to topple the AI monopolies and create a world where intelligence amplification is as universally available as smartphone apps – then we need people like you running scripts like these on clusters like those.

**The future of AI won't be built by the most well-funded labs. It will be built by the most determined individuals.**

So clone that repo. Launch those instances. Set those environment variables. And let's show the world what the democratization of AI really looks like.

Because in the end, the monopolists, the jingoists, the doomers, and the idiots will all be proven wrong by one thing and one thing only: working code in the hands of passionate people.

And that's something no amount of fearmongering, regulation, or capital can stop.