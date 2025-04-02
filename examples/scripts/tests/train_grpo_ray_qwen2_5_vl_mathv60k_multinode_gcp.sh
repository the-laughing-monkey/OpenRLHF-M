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
export NCCL_SOCKET_IFNAME=ens7

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