export ROOT_PATH="/mnt/private_berlinni/zgr/project/lmm-r1-dev"
export DATASET="$ROOT_PATH/data/deepscaler/deepscaler_chatml.jsonl"
#export HF_ENDPOINT="https://hf-mirror.com"
export WANDB_MODE="online"
export PYTHONPATH=$(pwd):$PYTHONPATH
MODEL_CPK_NAME="test"
PRETRAIN_MODEL="$ROOT_PATH/ckpts/Qwen2.5-Math-1.5B"
SAVE_PATH="$ROOT_PATH/ckpts"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}"

python -m openrlhf.models.remote_rm.math_verifier --dataset $DATASET --prompt-template chatml --input_key prompt --log_file "${SAVE_PATH}/${MODEL_CPK_NAME}/remote_rm.log" > /dev/null 2>&1  &
#childpid=$!

ray start --head --node-ip-address 0.0.0.0 --num-gpus 8 --temp-dir /tmp/ray --disable-usage-stats

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "./","env_vars":{"NCCL_SOCKET_IFNAME":"bond1"}}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --remote_rm_url http://29.127.80.111:5000/get_reward \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 2 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 1 \
   --colocate_actor_ref \
   --vllm_sync_backend nccl \
   --enable_prefix_caching \
   --pretrain $PRETRAIN_MODEL \
   --save_path $SAVE_PATH/$MODEL_CPK_NAME \
   --load_checkpoint \
   --micro_train_batch_size 4 \
   --train_batch_size 64 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 64 \
   --temperature 1 \
   --n_samples_per_prompt 1 \
   --max_epochs 1 \
   --num_episodes 2 \
   --prompt_max_len 1000 \
   --max_samples 100000 \
   --generate_max_len 3000 \
   --advantage_estimator gae \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 1e-6 \
   --init_kl_coef 0.0 \
   --lambd 1 \
   --gamma 1 \
   --prompt_data $DATASET \
   --input_key prompt \
   --normalize_reward \
   --flash_attn \
   --gradient_checkpointing \
   --save_steps 20 \
   --ckpt_path $SAVE_PATH/$MODEL_CPK_NAME/ckpt \
   --save_hf_ckpt \
   --packing_samples \
   --ring_attn_size 2 \
   --ring_head_stride 2 \
   --use_wandb 9aeddea3b60542704fd5cd44d4c4a1d1d911ce54 \
   --wandb_run_name $MODEL_CPK_NAME \
   --wandb_group deepscaler

# also supports --advantage_estimator rloo
#kill $childpid
ray stop