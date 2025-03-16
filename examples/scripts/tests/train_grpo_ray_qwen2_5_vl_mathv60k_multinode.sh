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
#
# Example (head node):
#   export HEAD_POD_ID=your-head-dns  # e.g., abc123 -> abc123.runpod.internal
#   ./train_grpo_ray_qwen2_5_vl_mathv60k_multinode.sh
#
# Example (worker node):
#   export RAY_WORKER=1
#   export HEAD_POD_ID=your-head-dns
#   ./train_grpo_ray_qwen2_5_vl_mathv60k_multinode.sh
#=============================================================================


# Friendly banner
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "=== OpenRLHF-M MathV60K Multinode Training Script Start ==="
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

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

# Logging directory.
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SAVE_PATH}/${MODEL_NAME}/logs"
CUR_LOG_DIR="${LOG_DIR}/${TIMESTAMP}"
mkdir -p "${CUR_LOG_DIR}"
echo "[INFO] Log directory: ${CUR_LOG_DIR}"

# Ray debugging options.
if [ -n "${DEBUG_RAY}" ]; then
  echo "[INFO] Ray debugging enabled."
  RAY_DEBUG_ARGS="RAY_DEBUG=legacy"
  RAY_DEBUG_OPTIONS="--ray-debugger-external"
else
  RAY_DEBUG_ARGS=""
  RAY_DEBUG_OPTIONS=""
fi

# Define ports.
RAY_PORT="6379"
DASHBOARD_PORT="8265"
REWARD_MODEL_PORT="5000"

# Determine head node IP: on head node, use local interface; workers use HEAD_POD_ID DNS.
if [ $IS_HEAD -eq 1 ]; then
  if [ -n "${HEAD_POD_ID}" ]; then
    HEAD_NODE_IP="${HEAD_POD_ID}.runpod.internal"
    echo "[INFO] HEAD_POD_ID provided. Derived HEAD_NODE_IP: ${HEAD_NODE_IP}"
  else
    HEAD_NODE_IP="127.0.0.1"
    echo "[INFO] HEAD_POD_ID not set. Using HEAD_NODE_IP: ${HEAD_NODE_IP}"
  fi
else
  if [ -z "${HEAD_POD_ID}" ]; then
    echo "[ERROR] For worker nodes, HEAD_POD_ID must be set. Exiting."
    exit 1
  fi
  HEAD_NODE_IP="${HEAD_POD_ID}.runpod.internal"
  echo "[INFO] Worker will connect to HEAD_NODE_IP: ${HEAD_NODE_IP}"
fi

# Head node operations.
if [ $IS_HEAD -eq 1 ]; then
  echo "-----------------------------------------------------------"
  echo "[HEAD NODE] Stopping any existing Ray instances..."
  ${RAY_DEBUG_ARGS} ray stop

  echo "[HEAD NODE] Starting Ray head node..."
  ${RAY_DEBUG_ARGS} ray start --head --node-ip-address 0.0.0.0 --port=${RAY_PORT} --dashboard-port=${DASHBOARD_PORT} --temp-dir ~/.cache/ray ${RAY_DEBUG_OPTIONS}
  
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
  
  # Test reward model connectivity.
  RETRY_COUNT=0
  MAX_RETRIES=10
  REWARD_MODEL_URL="http://127.0.0.1:${REWARD_MODEL_PORT}/get_reward"
  until curl -s -o /dev/null -w "%{http_code}" "${REWARD_MODEL_URL}" | grep -q "200"; do
    echo "[HEAD NODE] Waiting for reward model server at ${REWARD_MODEL_URL}... (attempt $((RETRY_COUNT+1)))"
    sleep 3
    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
      echo "[ERROR] Reward model server did not respond after $MAX_RETRIES attempts. Exiting."
      exit 1
    fi
  done
  echo "[HEAD NODE] Reward model server is responding."

  # Submit the training job.
  echo "[HEAD NODE] Submitting training job via Ray job submit..."
  ray job submit --address="http://127.0.0.1:${DASHBOARD_PORT}" \
     --runtime-env-json="{\"working_dir\": \"${WORKSPACE_DIR}\"}" \
     -- python3 -m openrlhf.cli.train_ppo_ray \
         --ref_num_nodes 2 \
         --ref_num_gpus_per_node 2 \
         --remote_rm_url "${REWARD_MODEL_URL}" \
         --actor_num_nodes 2 \
         --actor_num_gpus_per_node 2 \
         --vllm_num_engines 2 \
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
  ray start --address=${HEAD_NODE_IP}:${RAY_PORT} --temp-dir ~/.cache/ray
  echo "[WORKER NODE] Successfully joined the Ray cluster."
fi

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "=== OpenRLHF-M Multinode Training Script Completed ==="
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" 