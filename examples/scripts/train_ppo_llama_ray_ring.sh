export ROOT_PATH="/mnt/private_berlinni/zgr/project/lmm-r1-dev"
export DATASET="$ROOT_PATH/data/deepscaler/deepscaler_chatml.jsonl"
#export HF_ENDPOINT="https://hf-mirror.com"
export WANDB_MODE="offline"
export PYTHONPATH=$(pwd):$PYTHONPATH
MODEL_CPK_NAME="test"
PRETRAIN_MODEL="$ROOT_PATH/ckpts/Qwen2.5-Math-1.5B"
SAVE_PATH="$ROOT_PATH/ckpts"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "./"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 2 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 2 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain $PRETRAIN_MODEL\
   --reward_pretrain $PRETRAIN_MODEL \
   --save_path $SAVE_PATH \
   --micro_train_batch_size 16 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 32 \
   --rollout_batch_size 1024 \
   --max_samples 100000 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --normalize_reward \
   --packing_samples \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --load_checkpoint \
   --ring_attn_size 2 \
   --ring_head_stride 2

# --runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}' [Install deps]
# --ref_reward_offload [Offload to CPU]
# --remote_rm_url http://localhost:5000/get_reward

