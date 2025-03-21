set -x

# Set environment variables and paths.
NODE_HOSTNAME=$(hostname)
WORKSPACE_DIR="$(pwd)"
DATASET_PATH="/data/datasets/VerMulti/mathv60k_message.jsonl"
PRETRAIN_MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
SAVE_PATH="./checkpoints"
MODEL_NAME="qwen2.5-vl-3b-ins-mathvista-grpo"

  # Submit the training job.
  echo "[HEAD NODE] Submitting training job via Ray job submit..."
  ray job submit --address="http://127.0.0.1:8265" \
     --runtime-env-json="{\"working_dir\": \"${WORKSPACE_DIR}\"}" \
     -- python3 -m openrlhf.cli.train_ppo_ray \
         --ref_num_nodes 1 \
         --ref_num_gpus_per_node 8 \
         --remote_rm_url http://0.0.0.0:5000/get_reward \
         --actor_num_nodes 1 \
         --actor_num_gpus_per_node 8 \
         --vllm_num_engines 4 \
         --vllm_tensor_parallel_size 2 \
         --vllm_enable_sleep \
         --vllm_sync_backend nccl \
         --vllm_sync_with_ray \
         --colocate_actor_ref \
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
         --load_checkpoint