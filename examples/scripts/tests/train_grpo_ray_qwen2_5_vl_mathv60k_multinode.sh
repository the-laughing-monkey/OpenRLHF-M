#!/bin/bash
#=============================================================================
# OpenRLHF-M MathV60K Training Script - GRPO Version - MULTINODE
#=============================================================================
# This script trains the Qwen2.5-VL-3B-Instruct model on the MathV60K dataset
# using Group Reward Policy Optimization (GRPO) with Ray for distributed training
# across multiple nodes.
#
# The script:
# 1. Sets up environment variables and directories
# 2. Starts a Ray cluster with a head node
# 3. Provides instructions for worker nodes to join the cluster
# 4. Launches a remote reward model verifier
# 5. Submits a GRPO training job with the appropriate configuration
#
# The training uses a multimodal training approach with:
# - Qwen2.5-VL-3B-Instruct as the foundation model
# - Ray for distributed training across multiple nodes (2 nodes x 2 GPUs each)
# - vLLM for efficient inference
# - GRPO algorithm (no critic network needed)
#=============================================================================

# Get hostname for node-specific paths
NODE_HOSTNAME=$(hostname)

# =============== NETWORKING CONFIGURATION ===============
# This script uses RunPod's Global Networking for communication between nodes
# Each pod is assigned a DNS name: $podid.runpod.internal
# For more information, see: https://docs.runpod.io/pods/networking

# Check required environment variables
if [ -z "${RAY_WORKER}" ] || [ "${RAY_WORKER}" = "0" ]; then
  # This is the head node
  export HEAD_NODE=1
  
  # Check required environment variables
  if [ -z "${HEAD_POD_ID}" ]; then
    echo "ERROR: HEAD_POD_ID environment variable must be set for head node"
    echo "Example: export HEAD_POD_ID=abc123 (your pod's ID)"
    exit 1
  fi
  
  # Set up internal DNS name
  export HEAD_NODE_IP="${HEAD_POD_ID}.runpod.internal"
  echo "Using head node internal DNS: ${HEAD_NODE_IP}"
  
  # Internal ports are used directly with Global Networking (no port mapping needed)
  export RAY_PORT="6379"
  export DASHBOARD_PORT="8265"
  export REWARD_MODEL_PORT="5000"
else
  # This is a worker node, need the head node's pod ID
  if [ -z "${HEAD_POD_ID}" ]; then
    echo "ERROR: HEAD_POD_ID environment variable must be set for worker nodes"
    echo "Example: export HEAD_POD_ID=abc123 (head node's pod ID)"
    exit 1
  fi
  
  # Set up internal DNS name for the head node
  export HEAD_NODE_IP="${HEAD_POD_ID}.runpod.internal"
  export RAY_PORT="6379" # Internal port is used directly
  echo "Using head node internal DNS: ${HEAD_NODE_IP} and Ray port: ${RAY_PORT}"
fi

# Set environment variables for paths and model configuration
export WORKSPACE_DIR="$(pwd)"
export DATASET_PATH="/data/datasets/VerMulti/mathv60k_message.jsonl"
export PRETRAIN_MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"

# Create node-specific Ray cache directory to prevent conflicts
export RAY_TEMP_DIR="/data/cache-ray-${NODE_HOSTNAME}"
mkdir -p $RAY_TEMP_DIR

# Set up node-specific paths for checkpoints and logs
# Only the head node will save the final checkpoints
if [ -z "${RAY_WORKER}" ] || [ "${RAY_WORKER}" = "0" ]; then
  # Head node uses main checkpoint directory
  export SAVE_PATH="./checkpoints"
  export MODEL_NAME="qwen2.5-vl-3b-ins-mathvista-grpo"
else
  # Worker nodes use separate directories to avoid conflicts
  export SAVE_PATH="./checkpoints_worker_${NODE_HOSTNAME}"
  export MODEL_NAME="qwen2.5-vl-3b-ins-mathvista-grpo-worker-${NODE_HOSTNAME}"
fi

export WANDB_DIR="${WORKSPACE_DIR}"  

# Check if WANDB_API_KEY exists in environment
# If it doesn't exist, disable WandB logging
if [ -z "${WANDB_API_KEY}" ]; then
  echo "WANDB_API_KEY environment variable not set. WandB logging will be disabled."
  WANDB_ARGS=""
else
  echo "WANDB_API_KEY found. WandB logging will be enabled."
  # Add node-specific run name to avoid WandB conflicts
  WANDB_ARGS="--use_wandb ${WANDB_API_KEY} --wandb_run_name ${MODEL_NAME}_${NODE_HOSTNAME} --wandb_group \"openrlhf-m-training\""
fi

# Get process ID for later termination and set up log directories
SCRIPT_PID=$$
export TIMESTAMP=$(date +%Y%m%d_%H%M%S)_${NODE_HOSTNAME}
export LOG_DIR="${SAVE_PATH}/${MODEL_NAME}/logs"
export CUR_LOG_DIR="${LOG_DIR}/${TIMESTAMP}"

# Stop any existing Ray instances
ray stop

# Create necessary directories
mkdir -p "${SAVE_PATH}/${MODEL_NAME}"
mkdir -p "${LOG_DIR}"
mkdir -p "${CUR_LOG_DIR}"

# Print information about the training run
echo "================================================================"
echo "OpenRLHF-M Math Vista Training with GRPO - MULTINODE"
echo "================================================================"
echo "Node: ${NODE_HOSTNAME}"
echo "Model name: ${MODEL_NAME}"
echo "Dataset: ${DATASET_PATH}"
echo "Pretrained model: ${PRETRAIN_MODEL_PATH}"
echo "Logs will be saved to: ${CUR_LOG_DIR}"
echo "Ray temp directory: ${RAY_TEMP_DIR}"
echo

if [ -z "${RAY_WORKER}" ] || [ "${RAY_WORKER}" = "0" ]; then
  echo "Head node configuration (Global Networking):"
  echo "Pod ID: ${HEAD_POD_ID}"
  echo "Internal DNS: ${HEAD_NODE_IP}"
  echo "Ray port: ${RAY_PORT}"
  echo "Dashboard port: ${DASHBOARD_PORT}"
  echo "Reward model port: ${REWARD_MODEL_PORT}"
else
  echo "Worker node configuration:"
  echo "Head node Pod ID: ${HEAD_POD_ID}"
  echo "Head node DNS: ${HEAD_NODE_IP}"
  echo "Head node Ray port: ${RAY_PORT}"
fi

echo
echo "To monitor logs:"
echo "  tail -f ${CUR_LOG_DIR}/train.log"
echo
echo "To stop training:"
echo "  kill ${SCRIPT_PID}"
echo "  or: pkill -f openrlhf"
echo "  followed by: ray stop"
echo "================================================================"

# =============== MULTINODE CONFIGURATION ===============
# Determine if this is the head node (default) or worker node
if [ -z "${RAY_WORKER}" ] || [ "${RAY_WORKER}" = "0" ]; then
  # This is the head node
  echo "Starting Ray head node with Global Networking on ${HEAD_NODE_IP}"
  ray start --head --node-ip-address ${HEAD_NODE_IP} --num-gpus 2 --port=6379 --dashboard-port=8265 --temp-dir ${RAY_TEMP_DIR} --dashboard-host=0.0.0.0 --include-dashboard=true
  
  # Check which IPs Ray is listening on for dashboard port 8265
  echo "================================================================"
  echo "Checking all IPs that Ray is listening on port 8265..."
  echo "Using netstat:"
  netstat -tulpn | grep ":8265" || echo "netstat command failed or no matching ports found"
  
  echo "Using ss command:"
  ss -tulpn | grep ":8265" || echo "ss command failed or no matching ports found"
  
  echo "Listening sockets from /proc/net/tcp (decimal port 8265 = hex port 2049):"
  cat /proc/net/tcp | grep ":2049" || echo "No TCP socket info found for port 8265"
  
  echo "Dashboard process info:"
  ps aux | grep -E "ray.*dashboard" || echo "No dashboard process found"
  
  echo "================================================================"
  
  echo "Head node started. Worker nodes can join using:"
  echo "ray start --address=${HEAD_NODE_IP}:${RAY_PORT} --num-gpus=2 --temp-dir $RAY_TEMP_DIR"
  echo "================================================================"
  
  # Wait for worker nodes to join (optional)
  echo "Waiting for worker nodes to join..."
  sleep 30
  
else
  echo "Joining Ray cluster at ${HEAD_NODE_IP}:${RAY_PORT} as worker node"
  # Wait for the head node's Ray service to become reachable using bash built-in /dev/tcp
  while ! (echo > /dev/tcp/${HEAD_NODE_IP}/${RAY_PORT}) 2>/dev/null; do
    echo "Waiting for head node ${HEAD_NODE_IP}:${RAY_PORT} to be reachable..."
    sleep 5
  done
  echo "Head node is reachable. Waiting an additional 10 seconds for full initialization..."
  sleep 10
  ray start --address=${HEAD_NODE_IP}:${RAY_PORT} --num-gpus=2
fi

# Check if Ray is running properly
if ! ray status > /dev/null 2>&1; then
  echo "ERROR: Ray failed to start properly. Check the logs for details."
  exit 1
fi

# Get current Ray cluster status
echo "Current Ray cluster status:"
ray status

# =============== REMOTE REWARD MODEL ===============
# Only start the remote reward model server on the head node
if [ "${HEAD_NODE:-0}" -eq "1" ]; then
  echo "Starting remote reward model server on head node (port ${REWARD_MODEL_PORT})..."
  echo "Binding to all interfaces (0.0.0.0) instead of specific host for wider accessibility"
  
  python3 -m openrlhf.models.remote_rm.math_verifier \
      --dataset "${DATASET_PATH}" \
      --input_key message \
      --prompt-template chatml 2>&1 | tee -a "${CUR_LOG_DIR}/remote_rm.log" &
  REMOTE_RM_PID=$!
  
  # Verify the reward model server is running
  echo "Waiting for reward model server to start (PID: ${REMOTE_RM_PID})..."
  sleep 15
  if ! ps -p $REMOTE_RM_PID > /dev/null; then
    echo "ERROR: Reward model server failed to start"
    exit 1
  fi
  
  # On head node: Use 127.0.0.1 for local access to the reward model
  # Workers will use the DNS name via REMOTE_RM_URL_FOR_WORKER
  export REMOTE_RM_URL="http://127.0.0.1:${REWARD_MODEL_PORT}/get_reward"
  export REMOTE_RM_URL_FOR_WORKER="http://${HEAD_NODE_IP}:${REWARD_MODEL_PORT}/get_reward"
  
  echo "Reward model server started with PID: $REMOTE_RM_PID"
  echo "Local reward model access URL (for head node): ${REMOTE_RM_URL}"
  echo "Worker nodes will access reward model at: ${REMOTE_RM_URL_FOR_WORKER}"
  
  # Test local access to reward model
  echo "Testing local reward model access..."
  curl -s -o /dev/null -w "Local reward model connectivity: %{http_code}\n" "${REMOTE_RM_URL}" || \
    echo "Local connectivity test failed, but continuing"
  
  # Simplified network debugging
  echo "======= NETWORK INFO ======="
  echo "Network interfaces:"
  ip addr | grep "inet "
  echo "Listening ports:"
  netstat -tuln | grep -E ":${REWARD_MODEL_PORT}|:8265" || echo "No matching ports found"
  echo "==========================="
  
  # Wait a bit longer for the server to fully initialize
  echo "Waiting an additional 5 seconds for server to fully initialize..."
  sleep 5
fi

# =============== TRAINING JOB SUBMISSION ===============
echo "Starting training..."

# === GPU and Model Distribution Configuration ===
# We have 2 nodes with 2 GPUs each (4 GPUs total)
# --ref_num_nodes 2                  : Reference model distributed across 2 nodes
# --ref_num_gpus_per_node 2          : 2 GPUs per node for the reference model
# --actor_num_nodes 2                : Actor model distributed across 2 nodes
# --actor_num_gpus_per_node 2        : 2 GPUs per node for actor model
# --remote_rm_url                    : URL for remote reward model API (on head node)

# === vLLM Configuration for 4 GPUs across 2 nodes ===
# Using 4 vLLM engines with tensor_parallel_size=1
# --vllm_num_engines 4              : Create 4 separate vLLM engines (1 per GPU)
# --vllm_tensor_parallel_size 1     : Each engine uses 1 GPU

# Setup checkpoint and saving parameters based on node type
if [ "${HEAD_NODE:-0}" -eq "1" ]; then
  # Head node handles checkpointing
  CHECKPOINT_ARGS="--save_steps 5 --ckpt_path ${SAVE_PATH}/${MODEL_NAME}/ckpt --save_hf_ckpt --load_checkpoint"
  echo "Head node will save checkpoints to: ${SAVE_PATH}/${MODEL_NAME}/ckpt"
else
  # Worker nodes don't save checkpoints to avoid filesystem conflicts
  CHECKPOINT_ARGS="--save_steps -1 --load_checkpoint"
  echo "Worker node will not save checkpoints"
fi

# Submit Ray job only from the head node using the dashboard port
# Worker nodes don't need to submit jobs, they just join the cluster
if [ "${HEAD_NODE:-0}" -eq "1" ]; then
  echo "Using local reward model URL: ${REMOTE_RM_URL}"
  
  # Display Ray connectivity info
  echo "======= RAY DEBUG INFO ======="
  echo "Ray status:"
  ray status
  echo "Ray dashboard connectivity:"
  curl -s -o /dev/null -w "Ray dashboard (127.0.0.1): %{http_code}\n" "http://127.0.0.1:8265/api/jobs/"
  echo "=========================="


  # Submit the Ray job
  echo "Submitting Ray job with 127.0.0.1 for local network access..."
  
  echo "===== SUBMISSION DETAILS ====="
  echo "REMOTE_RM_URL: ${REMOTE_RM_URL}"
  echo "REMOTE_RM_URL_FOR_WORKER: ${REMOTE_RM_URL_FOR_WORKER}" 
  echo "TRAIN_SCRIPT: ${TRAIN_SCRIPT}"
  echo "CHECKPOINT_ARGS: ${CHECKPOINT_ARGS}"
  echo "============================="
  
  # Testing connection to Ray dashboard before job submission  
  echo "Testing Ray dashboard connection before submission:"
  curl -v http://127.0.0.1:8265/api/jobs/ -H "Content-Type: application/json" > "${CUR_LOG_DIR}/ray_test.log" 2>&1
  echo "Dashboard request result code: $?"
  
  # Add a longer delay to ensure dashboard is fully initialized
  echo "Waiting for Ray dashboard to be fully initialized (20 seconds)..."
  echo "This delay is crucial to ensure the job submission endpoint is ready."
  for i in {10..1}; do
    echo "  $i seconds remaining..."
    sleep 1
  done
  echo "Ready to submit test job now."
  
  # Create a simple test script
  TEST_SCRIPT="${CUR_LOG_DIR}/test_job.py"
  echo "Creating test script: ${TEST_SCRIPT}"
  cat > "${TEST_SCRIPT}" << 'EOF'
#!/usr/bin/env python3
import os
import socket
import sys
import time

print("=" * 50)
print("RAY TEST JOB - HEALTH CHECK")
print("=" * 50)
print(f"Hostname: {socket.gethostname()}")
print(f"Current working directory: {os.getcwd()}")
print(f"Python version: {sys.version}")
print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("Test job completed successfully!")
print("=" * 50)
EOF

  # First submit a test job to verify the job submission system works
  echo "Submitting a test job to verify the Ray job submission system..."
  
  ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{"env_vars": {"TEST_JOB": "1"}}' \
    -- python3 "${TEST_SCRIPT}" > "${CUR_LOG_DIR}/test_job_output.log" 2>&1
  
  TEST_JOB_RESULT=$?
  echo "Test job result code: ${TEST_JOB_RESULT}"
  
  if [ $TEST_JOB_RESULT -ne 0 ]; then
    echo "WARNING: Test job submission failed. This may indicate issues with the Ray job submission system."
    echo "Proceeding with main job submission anyway..."
    
    # Display test job output for debugging
    echo "Test job log contents:"
    cat "${CUR_LOG_DIR}/test_job_output.log"
    echo "--------------------"
    
    # Add another delay before main job submission 
    echo "Waiting an additional 10 seconds before main job submission..."
    sleep 10
  else
    echo "Test job submission succeeded! Proceeding with main job submission."
    sleep 5
  fi
  
  echo "Submitting main training job now..."
  
  ray job submit --address="http://127.0.0.1:8265" \
     --runtime-env-json='{"env_vars": {"REMOTE_RM_URL": "'"${REMOTE_RM_URL}"'", "REMOTE_RM_URL_FOR_WORKER": "'"${REMOTE_RM_URL_FOR_WORKER}"'", "RAY_HEAD_NODE": "1"}}' \
     -- python3 -m openrlhf.cli.train_ppo_ray \
     --worker_rm_url "${REMOTE_RM_URL_FOR_WORKER}" \
     --ref_num_nodes 2 \
     --ref_num_gpus_per_node 2 \
     --actor_num_nodes 2 \
     --actor_num_gpus_per_node 2 \
     --vllm_num_engines 4 \
     --vllm_tensor_parallel_size 1 \
     --colocate_all_models \
     --vllm_enable_sleep \
     --vllm_gpu_memory_utilization 0.5 \
     --vllm_sync_backend gloo \
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
     --prompt_max_len 4096 \
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
     ${CHECKPOINT_ARGS} \
     ${WANDB_ARGS} \
     --use_tensorboard ${LOG_DIR} > >(tee -a "${CUR_LOG_DIR}/train.log") 2>&1 &
  
  TRAIN_PID=$!
  echo "Job submitted with PID: $TRAIN_PID on head node"
  
  echo "Remote RM PID: $REMOTE_RM_PID" > "${CUR_LOG_DIR}/process_pids.txt"
  echo "Train PID: $TRAIN_PID" >> "${CUR_LOG_DIR}/process_pids.txt"
else
  echo "Worker node ${NODE_HOSTNAME} joined the cluster but did not submit a job"
  echo "Worker joined the cluster" > "${CUR_LOG_DIR}/worker_status.txt"
fi

echo "Training is running in the background. Check logs at ${CUR_LOG_DIR}/train.log"
if [ -n "$TRAIN_PID" ]; then
  echo "To attach to the training process: wait $TRAIN_PID"
fi
echo "When finished, clean up with:"
echo "pkill -f openrlhf"
echo "ray stop"
echo "All logs are available in ${CUR_LOG_DIR}"

# If using a dashboard, share the URL
if [ "${HEAD_NODE:-0}" -eq "1" ]; then
  echo "======= MONITORING ======="
  echo "Ray Dashboard URL (internal): http://${HEAD_NODE_IP}:8265"
  echo "You may need to use port forwarding to access this from outside:"
  echo "ssh -L 8265:${HEAD_NODE_IP}:8265 user@runpod-instance"
  echo "=========================="
fi 