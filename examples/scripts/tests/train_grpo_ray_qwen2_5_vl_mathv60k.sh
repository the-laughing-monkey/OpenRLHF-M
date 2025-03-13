#!/bin/bash
#=============================================================================
# OpenRLHF-M MathV60K Training Script - GRPO Version
#=============================================================================
# This script trains the Qwen2.5-VL-3B-Instruct model on the MathV60K dataset
# using Group Reward Policy Optimization (GRPO) with Ray for distributed training.
#
# The script:
# 1. Sets up environment variables and directories
# 2. Starts a Ray cluster for distributed training
# 3. Launches a remote reward model verifier
# 4. Submits a GRPO training job with the appropriate configuration
#
# The training uses a multimodal training approach with:
# - Qwen2.5-VL-3B-Instruct as the foundation model
# - Ray for distributed training across 4 GPUs
# - vLLM for efficient inference
# - GRPO algorithm (no critic network needed)
#=============================================================================

# Set environment variables for paths and model configuration
export WORKSPACE_DIR="$(pwd)"
export DATASET_PATH="/data/datasets/VerMulti/mathv60k_message.jsonl"
export PRETRAIN_MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
export SAVE_PATH="./checkpoints" 
export MODEL_NAME="qwen2.5-vl-3b-ins-mgt-percereas-grpo"
export WANDB_DIR="${WORKSPACE_DIR}"  

# Check if WANDB_API_KEY exists in environment
# If it doesn't exist, disable WandB logging
if [ -z "${WANDB_API_KEY}" ]; then
  echo "WANDB_API_KEY environment variable not set. WandB logging will be disabled."
  WANDB_ARGS=""
else
  echo "WANDB_API_KEY found. WandB logging will be enabled."
  WANDB_ARGS="--use_wandb ${WANDB_API_KEY} --wandb_run_name ${MODEL_NAME} --wandb_group \"openrlhf-m-training\""
fi

# Get process ID for later termination and set up log directories
SCRIPT_PID=$$
export TIMESTAMP=$(date +%Y%m%d_%H%M%S)
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
echo "OpenRLHF-M Math Vista Training with GRPO"
echo "================================================================"
echo "Model name: ${MODEL_NAME}"
echo "Dataset: ${DATASET_PATH}"
echo "Pretrained model: ${PRETRAIN_MODEL_PATH}"
echo "Logs will be saved to: ${CUR_LOG_DIR}"
echo
echo "To monitor logs:"
echo "  tail -f ${CUR_LOG_DIR}/train.log"
echo
echo "To stop training:"
echo "  kill ${SCRIPT_PID}"
echo "  or: pkill -f openrlhf"
echo "  followed by: ray stop"
echo "================================================================"

# Start Ray cluster for distributed training
# Using 4 GPUs with metrics export for monitoring
echo "Starting ray..."
ray start --head --node-ip-address 0.0.0.0 --num-gpus 4 --temp-dir ~/.cache/ray

# Launch the remote reward model verifier
# This will evaluate responses from the model against mathematical correctness
echo "Starting remote reward model server..."
python -m openrlhf.models.remote_rm.math_verifier \
    --dataset "${DATASET_PATH}" \
    --input_key message \
    --prompt-template chatml 2>&1 | tee -a "${CUR_LOG_DIR}/remote_rm.log" &
REMOTE_RM_PID=$!

# Submit the GRPO training job to Ray
echo "Starting training..."

# === GPU and Model Distribution Configuration ===
# Each model (ref, actor) has access to all 4 GPUs
# Note: No critic needed for GRPO/REINFORCE++
# --ref_num_nodes 1                  : Number of nodes for reference model
# --ref_num_gpus_per_node 4          : All 4 GPUs allocated to reference model
# --actor_num_nodes 1                : Number of nodes for actor model
# --actor_num_gpus_per_node 4        : All 4 GPUs allocated to actor model
# --remote_rm_url                    : URL for remote reward model API

# === vLLM Configuration for 4 GPUs ===
# Using 4 vLLM engines (one per GPU) with tensor_parallel_size=1
# This maximizes throughput while fitting the model in GPU memory
# --vllm_num_engines 4              : Create 4 separate vLLM engines (1 per GPU)
# --vllm_tensor_parallel_size 1     : Each engine uses 1 GPU
# --colocate_all_models             : Share GPUs between all model types
# --vllm_enable_sleep               : Enable sleep mode when not in use
# --vllm_gpu_memory_utilization 0.5 : Allow 50% GPU memory for vLLM
# --vllm_sync_backend gloo          : Use gloo backend for weight sync
# --enable_prefix_caching           : Cache prefixes for faster generation

# === GRPO Specific Configuration ===
# --advantage_estimator group_norm  : GRPO method for advantage estimation
# --use_kl_loss                     : Enable KL loss (required for GRPO)
# --kl_estimator k3                 : KL estimation method
# --init_kl_coef 1e-3               : Initial KL penalty coefficient

# === Model and Path Configuration ===  
# --pretrain                        : Path to pretrained model
# --save_path                       : Path to save checkpoints

# === Batch Sizes ===
# Controls memory usage and throughput
# --micro_train_batch_size 1        : Batch size per GPU for training
# --train_batch_size 256            : Global batch size for training
# --micro_rollout_batch_size 1      : Batch size per GPU for generation
# --rollout_batch_size 256          : Global batch size for generation

# === Sampling and Generation Parameters ===
# --temperature 1.0                 : Sampling temperature
# --n_samples_per_prompt 16         : Number of responses to generate per prompt
# --max_epochs 1                    : Number of training epochs per batch
# --num_episodes 2                  : Number of data passes (episodes)
# --prompt_max_len 4096             : Maximum input length
# --max_samples 100000              : Maximum samples to process
# --generate_max_len 8000           : Maximum length of generated text

# === Data and Input Configuration ===
# --prompt_data                     : Path to dataset
# --input_key message               : JSON key for input data
# --normalize_reward                : Normalize rewards for stable training

# === Performance Optimizations ===
# --flash_attn                      : Use Flash Attention for faster training
# --gradient_checkpointing          : Save memory by recomputing gradients
 
# === Checkpointing and Logging ===
# --save_steps 20                   : Save checkpoint every 20 steps
# --ckpt_path                       : Path to save checkpoints
# --save_hf_ckpt                    : Save HuggingFace format checkpoint
# --load_checkpoint                 : Load from checkpoint if available
# --use_tensorboard                 : Enable TensorBoard logging

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="{\"working_dir\": \"${WORKSPACE_DIR}\"}" \
   -- python -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 4 \
   --remote_rm_url http://127.0.0.1:5000/get_reward \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
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
   --train_batch_size 256 \
   --micro_rollout_batch_size 1 \
   --rollout_batch_size 256 \
   --temperature 1.0 \
   --n_samples_per_prompt 16 \
   --max_epochs 1 \
   --num_episodes 2 \
   --prompt_max_len 4096 \
   --max_samples 100000 \
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
   --save_steps 20 \
   --ckpt_path ${SAVE_PATH}/${MODEL_NAME}/ckpt \
   --save_hf_ckpt \
   --load_checkpoint \
   ${WANDB_ARGS} \
   --use_tensorboard ${LOG_DIR} > >(tee -a "${CUR_LOG_DIR}/train.log") 2>&1 &

# Store process IDs for later management
TRAIN_PID=$!

echo "Remote RM PID: $REMOTE_RM_PID" > "${CUR_LOG_DIR}/process_pids.txt"
echo "Train PID: $TRAIN_PID" >> "${CUR_LOG_DIR}/process_pids.txt"
echo "Training is running in the background. Check logs at ${CUR_LOG_DIR}/train.log"
echo "To attach to the training process: wait $TRAIN_PID"
echo "When finished, clean up with:"
echo "pkill -f openrlhf"
echo "ray stop"
echo "All logs are available in ${CUR_LOG_DIR}" 