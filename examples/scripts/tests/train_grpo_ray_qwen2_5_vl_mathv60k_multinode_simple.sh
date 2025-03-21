set -x

# Set environment variables and paths.
NODE_HOSTNAME=$(hostname)
WORKSPACE_DIR="$(pwd)"
DATASET_PATH="/data/datasets/VerMulti/mathv60k_message.jsonl"
PRETRAIN_MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
SAVE_PATH="./checkpoints"
MODEL_NAME="qwen2.5-vl-3b-ins-mathvista-grpo"
WANDB_DIR="${WORKSPACE_DIR}"

# NCCL Commands
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0

# Suppress unhandled errors in Ray
export RAY_IGNORE_UNHANDLED_ERRORS=1

# Check for WandB API key.
if [ -z "${WANDB_API_KEY}" ]; then
  echo "[INFO] WANDB_API_KEY not set. WandB logging will be disabled."
  WANDB_ARGS=""
else
  echo "[INFO] WANDB_API_KEY found. WandB logging enabled."
  WANDB_ARGS="--use_wandb ${WANDB_API_KEY} --wandb_run_name ${MODEL_NAME} --wandb_group \"openrlhf-m-training\""
fi

# Get the IP address of eth1 interface
ETH1_IP=$(ip addr show eth1 | grep -oP 'inet \K[\d.]+')
echo "Using eth1 IP address: ${ETH1_IP}"

 # Start the remote reward model server and test connectivity.
  echo "[HEAD NODE] Starting remote reward model server..."
  python3 -m openrlhf.models.remote_rm.math_verifier \
      --dataset "${DATASET_PATH}" \
      --input_key message \
      --prompt-template chatml 2>&1 | tee -a "${CUR_LOG_DIR}/remote_rm.log" &
  REMOTE_RM_PID=$!

  # Submit the training job.
  echo "[HEAD NODE] Submitting training job via Ray job submit..."
  ray job submit --address="http://127.0.0.1:8265" \
     --runtime-env-json="{\"working_dir\": \"${WORKSPACE_DIR}\"}" \
     -- python3 -m openrlhf.cli.train_ppo_ray \
         --ref_num_nodes 1 \
         --ref_num_gpus_per_node 4 \
         --remote_rm_url http://${ETH1_IP}:5000/get_reward \
         --actor_num_nodes 1 \
         --actor_num_gpus_per_node 4 \
         --vllm_num_engines 4 \
         --vllm_tensor_parallel_size 2 \
         --vllm_enable_sleep \
         --vllm_sync_backend nccl \
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
         --save_steps 5 \
         --ckpt_path ${SAVE_PATH}/${MODEL_NAME}/ckpt \
         --save_hf_ckpt \
         --load_checkpoint \
         ${WANDB_ARGS}
