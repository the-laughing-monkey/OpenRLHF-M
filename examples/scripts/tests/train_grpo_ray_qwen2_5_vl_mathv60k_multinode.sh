#!/bin/bash
#=============================================================================
# Simplified Multinode Training Script for OpenRLHF-M MathV60K (GRPO Version)
#=============================================================================
# This script simplifies multinode training by:
# 1) Removing the use of a shared network drive for checkpoints.
# 2) Using 127.0.0.1 (for head node local operations) and connecting workers via DNS from HEAD_POD_ID.
# 3) Using a single checkpoint directory (no worker-specific modifications).
# 4) Including optional Ray debugging support via DEBUG_RAY.
# 5) Detecting WandB API key to enable logging (but not halting if not set).
# 6) Pausing for reward model startup and verifying it is responding before proceeding.
# 7) Friendly output messages during execution.
#
# Usage:
#   For head node (default): Run without setting RAY_WORKER or set RAY_WORKER=0.
#   For worker nodes: Set RAY_WORKER=1 and also set HEAD_POD_ID.
#
#   Optionally, set DEBUG_RAY=1 to enable Ray debugging.
#   Optionally, set EXPECTED_WORKERS=N to wait for N worker nodes to join.
#
# Example (head node):
#   export HEAD_POD_ID=your-head-dns  # e.g., abc123 -> abc123.runpod.internal
#   export EXPECTED_WORKERS=2         # Wait for 2 worker nodes to join
#   ./train_grpo_ray_qwen2_5_vl_mathv60k_multinode.sh
#
# Example (worker node):
#   export RAY_WORKER=1
#   export HEAD_POD_ID=your-head-dns
#   ./train_grpo_ray_qwen2_5_vl_mathv60k_multinode.sh
#=============================================================================

# Friendly banner
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!"
echo "=== OpenRLHF-M MathV60K Multinode Training Script Start ==="
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

# Set NCCL environment variables for DNS resolution in RunPod Global Networking
echo "[INFO] Setting NCCL environment variables for DNS resolution in RunPod Global Networking."
export NCCL_SOCKET_IFNAME=lo,eth0,podnet1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_FAMILY=IPv4
export NCCL_LAUNCH_MODE=GROUP
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_P2P_DISABLE=1      # Disable peer-to-peer as it's causing issues in container environment
export NCCL_SHM_DISABLE=0      # Ensure shared memory is enabled
export DEEPSPEED_TIMEOUT=600   # Add timeout for DeepSpeed initialization

# Check and report shared memory size - critical for NCCL in containers
echo "[INFO] Checking shared memory size (should be at least 1GB for NCCL):"
df -h /dev/shm

# Verify network interfaces
echo "[INFO] Available network interfaces:"
ifconfig -a | grep -E "eth0|podnet1|lo" | grep -E "inet "

# Optional: Help user ensure proper Docker config if this is run in a container
echo "[INFO] Note: If running in a container, ensure it was started with '--shm-size=1g --ulimit memlock=-1'"

# Determine if this is a head or worker node.
if [ -z "${RAY_WORKER}" ] || [ "${RAY_WORKER}" = "0" ]; then
  IS_HEAD=1
  echo "[INFO] Running as HEAD NODE."
else
  IS_HEAD=0
  echo "[INFO] Running as WORKER NODE."
fi

# Set environment variables and paths.
NODE_HOSTNAME=$(hostname)
WORKSPACE_DIR="$(pwd)"
DATASET_PATH="/data/datasets/VerMulti/mathv60k_message.jsonl"
PRETRAIN_MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
SAVE_PATH="./checkpoints"
MODEL_NAME="qwen2.5-vl-3b-ins-mathvista-grpo"
WANDB_DIR="${WORKSPACE_DIR}"

# Check for WandB API key.
if [ -z "${WANDB_API_KEY}" ]; then
  echo "[INFO] WANDB_API_KEY not set. WandB logging will be disabled."
  WANDB_ARGS=""
else
  echo "[INFO] WANDB_API_KEY found. WandB logging enabled."
  WANDB_ARGS="--use_wandb ${WANDB_API_KEY} --wandb_run_name ${MODEL_NAME} --wandb_group \"openrlhf-m-training\""
fi

# Default to expecting at least 1 worker if not specified
if [ -z "${EXPECTED_WORKERS}" ]; then
  EXPECTED_WORKERS=1
  echo "[INFO] EXPECTED_WORKERS not set. Defaulting to expecting 1 worker node."
fi

# Logging directory.
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SAVE_PATH}/${MODEL_NAME}/logs"
CUR_LOG_DIR="${LOG_DIR}/${TIMESTAMP}"
mkdir -p "${CUR_LOG_DIR}"
echo "[INFO] Log directory: ${CUR_LOG_DIR}"

# Define ports.
RAY_PORT="6379"
DASHBOARD_PORT="8265"
REWARD_MODEL_PORT="5000"

# Create a PID file directory for worker tracking
WORKER_PID_DIR="/tmp/openrlhf_workers"
mkdir -p $WORKER_PID_DIR

# Determine head node IP: on head node, use HEAD_POD_ID DNS; workers use HEAD_POD_ID DNS.
if [ $IS_HEAD -eq 1 ]; then
  if [ -z "${HEAD_POD_ID}" ]; then
    echo "[ERROR] HEAD_POD_ID must be set for head node when using RunPod Global Networking. Exiting."
    exit 1
  else
    HEAD_NODE_IP="${HEAD_POD_ID}.runpod.internal"
    echo "[INFO] HEAD_POD_ID provided. Derived HEAD_NODE_IP: ${HEAD_NODE_IP}"
  fi
else
  if [ -z "${HEAD_POD_ID}" ]; then
    echo "[ERROR] For worker nodes, HEAD_POD_ID must be set. Exiting."
    exit 1
  fi
  HEAD_NODE_IP="${HEAD_POD_ID}.runpod.internal"
  echo "[INFO] Worker will connect to HEAD_NODE_IP: ${HEAD_NODE_IP}"
fi

# Function to cleanup worker processes
cleanup_worker() {
  echo "[WORKER NODE] Cleaning up worker processes..."
  ray stop
  rm -f $WORKER_PID_DIR/worker_${NODE_HOSTNAME}.pid
  echo "[WORKER NODE] Worker node stopped and cleaned up."
  exit 0
}

# Worker cleanup on exit
trap cleanup_worker SIGINT SIGTERM

# Head node operations.
if [ $IS_HEAD -eq 1 ]; then
  echo "-----------------------------------------------------------"
  echo "[HEAD NODE] Stopping any existing Ray instances..."
  ray stop

  echo "[HEAD NODE] Starting Ray head node..."
  ray start --head --node-ip-address 0.0.0.0 --port=${RAY_PORT} --dashboard-port=${DASHBOARD_PORT} --temp-dir ~/.cache/ray 
  
  echo "[HEAD NODE] Head node started. Waiting 5 seconds for cluster stabilization..."
  sleep 5

  # Start the remote reward model server and test connectivity.
  echo "[HEAD NODE] Starting remote reward model server..."
  python3 -m openrlhf.models.remote_rm.math_verifier \
      --dataset "${DATASET_PATH}" \
      --input_key message \
      --prompt-template chatml 2>&1 | tee -a "${CUR_LOG_DIR}/remote_rm.log" &
  REMOTE_RM_PID=$!
  
  echo "[HEAD NODE] Waiting for remote reward model server to initialize..."
  sleep 5
  
  # Test reward model connectivity with correctly formatted JSON
  RETRY_COUNT=0
  MAX_RETRIES=15
  REWARD_MODEL_URL="http://${HEAD_NODE_IP}:${REWARD_MODEL_PORT}/get_reward"
  until curl -s -X POST -H "Content-Type: application/json" -d '{"query": ["test query"], "prompts": ["test problem"]}' -o /dev/null -w "%{http_code}" "${REWARD_MODEL_URL}" | grep -q "200"; do
    echo "[HEAD NODE] Waiting for reward model server at ${REWARD_MODEL_URL}... (attempt $((RETRY_COUNT+1)))"
    sleep 3
    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
      echo "[ERROR] Reward model server did not respond after $MAX_RETRIES attempts. Exiting."
      kill $REMOTE_RM_PID
      exit 1
    fi
  done
  echo "[HEAD NODE] Reward model server is responding."

  # Wait for worker nodes to join
  echo "[HEAD NODE] Waiting for worker nodes to join the cluster..."
  WORKER_RETRY=0
  MAX_WORKER_RETRIES=20
  while true; do
    echo "[HEAD NODE] Running ray status to check for workers:"
    ray status
    
    # Count total active nodes from 'Active:' block until 'Pending:' is encountered and sum counts
    ACTIVE_NODES=$(ray status | sed -n '/Active:/,/Pending:/p' | grep -E '^[[:space:]]*[0-9]+' | awk '{sum += $1} END {print sum}')
    if [ -z "$ACTIVE_NODES" ]; then
      ACTIVE_NODES=1  # Default to 1 if unable to parse (just the head node)
    fi
    WORKER_COUNT=$((ACTIVE_NODES - 1))
    
    echo "[HEAD NODE] Detected active nodes: $ACTIVE_NODES (head + $WORKER_COUNT workers)"
    
    if [ $WORKER_COUNT -ge $EXPECTED_WORKERS ]; then
      echo "[HEAD NODE] All expected worker nodes ($EXPECTED_WORKERS) have joined the cluster."
      break
    fi
    echo "[HEAD NODE] Waiting for worker nodes to join... ($WORKER_COUNT/$EXPECTED_WORKERS joined) (attempt $((WORKER_RETRY+1)))"
    sleep 5
    WORKER_RETRY=$((WORKER_RETRY+1))
    if [ $WORKER_RETRY -ge $MAX_WORKER_RETRIES ]; then
      echo "[WARNING] Not all expected worker nodes joined after $MAX_WORKER_RETRIES attempts. Proceeding anyway with $WORKER_COUNT workers."
      break
    fi
  done

  # Calculate GPU parameters based on detected worker count
  # We need: actor_num_nodes * actor_num_gpus_per_node = vllm_num_engines * vllm_tensor_parallel_size
  TOTAL_NODES=$((WORKER_COUNT + 1)) # Including head node
  REF_NUM_NODES=$TOTAL_NODES
  REF_NUM_GPUS_PER_NODE=2
  ACTOR_NUM_NODES=$TOTAL_NODES
  ACTOR_NUM_GPUS_PER_NODE=2
  VLLM_NUM_ENGINES=$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))
  VLLM_TENSOR_PARALLEL_SIZE=1
  
  echo "[HEAD NODE] Detected $TOTAL_NODES total nodes (head + $WORKER_COUNT workers)"
  echo "[HEAD NODE] Setting up job with: actor_num_nodes=$ACTOR_NUM_NODES, actor_num_gpus_per_node=$ACTOR_NUM_GPUS_PER_NODE"
  echo "[HEAD NODE] vllm_num_engines=$VLLM_NUM_ENGINES, vllm_tensor_parallel_size=$VLLM_TENSOR_PARALLEL_SIZE"

  # Submit the training job.
  echo "[HEAD NODE] Submitting training job via Ray job submit..."
  ray job submit --address="http://127.0.0.1:${DASHBOARD_PORT}" \
     --runtime-env-json="{\"working_dir\": \"${WORKSPACE_DIR}\"}" \
     -- python3 -m openrlhf.cli.train_ppo_ray \
         --ref_num_nodes 2 \
         --ref_num_gpus_per_node 1 \
         --remote_rm_url "${REWARD_MODEL_URL}" \
         --actor_num_nodes 2 \
         --actor_num_gpus_per_node 1 \
         --vllm_num_engines 2 \
         --vllm_tensor_parallel_size 1 \
         --colocate_all_models \
         --vllm_enable_sleep \
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
         --save_steps 5 \
         --ckpt_path ${SAVE_PATH}/${MODEL_NAME}/ckpt \
         --save_hf_ckpt \
         --load_checkpoint \
         ${WANDB_ARGS} \
         --use_tensorboard ${LOG_DIR} > >(tee -a "${CUR_LOG_DIR}/train.log") 2>&1 &
  TRAIN_PID=$!
  
  echo "[HEAD NODE] Training job submitted with PID: $TRAIN_PID"
  echo "[HEAD NODE] Remote Reward Model running with PID: $REMOTE_RM_PID"
  echo "[HEAD NODE] Logs available in ${CUR_LOG_DIR}"
  
  echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  echo "=== OpenRLHF-M Head Node Script Completed ==="
  echo "=== Training job is now running. See logs for progress. ==="
  echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

# Worker node operations.
else
  echo "-----------------------------------------------------------"
  echo "[WORKER NODE] Stopping any existing Ray instances..."
  ray stop
  
  echo "[WORKER NODE] Waiting for head node ${HEAD_NODE_IP}:${RAY_PORT} to be reachable..."
  while ! (echo > /dev/tcp/${HEAD_NODE_IP}/${RAY_PORT}) 2>/dev/null; do
    echo "[WORKER NODE] Still waiting for head node..."
    sleep 5
  done
  echo "[WORKER NODE] Head node is reachable. Joining the Ray cluster..."
  ray start --address=${HEAD_NODE_IP}:${RAY_PORT} 
  echo "[WORKER NODE] Successfully joined the Ray cluster."
  
  # Save worker PID for tracking
  echo $$ > $WORKER_PID_DIR/worker_${NODE_HOSTNAME}.pid
  
  # Create a log file for worker status
  WORKER_LOG="${CUR_LOG_DIR}/worker_${NODE_HOSTNAME}.log"
  mkdir -p $(dirname "$WORKER_LOG")
  
  echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  echo "=== OpenRLHF-M Worker Node Ready ==="
  echo "=== Worker is now connected to the Ray cluster. ==="
  echo "=== The worker process is running in the background. ==="
  echo "=== Worker log: $WORKER_LOG ==="
  echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  
  # Print worker management instructions
  echo "To check the status of this worker node:"
  echo "  ray status"
  echo ""
  echo "To stop this worker node:"
  echo "  ray stop"
  echo "  # or kill the worker process:"
  echo "  kill $(cat $WORKER_PID_DIR/worker_${NODE_HOSTNAME}.pid)"
  echo ""
  
  # Fully detach the worker process by creating a background job monitoring Ray
  # and redirecting all output to the log file
  (
    # Print startup message to the log
    echo "[WORKER NODE] Ray worker started in background at $(date)" > "$WORKER_LOG"
    echo "[WORKER NODE] Worker PID: $$" >> "$WORKER_LOG"
    
    # Monitor Ray status in a loop
    while ray status >> "$WORKER_LOG" 2>&1; do
      echo "[WORKER NODE] Still active at $(date). Will check again in 60 seconds." >> "$WORKER_LOG"
      sleep 60
    done
    
    echo "[WORKER NODE] Ray process has stopped at $(date). Worker node process will exit." >> "$WORKER_LOG"
    
    # Cleanup
    ray stop >> "$WORKER_LOG" 2>&1
    rm -f $WORKER_PID_DIR/worker_${NODE_HOSTNAME}.pid
    echo "[WORKER NODE] Worker node stopped and cleaned up." >> "$WORKER_LOG"
  ) &
  
  disown
  
  # Exit the foreground script immediately
  exit 0
fi 