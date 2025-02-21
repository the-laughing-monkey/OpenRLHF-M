export ROOT_PATH="/seu_share/home/gengxin/230249020/projects/lmm-r1-dev"
export DATASET="$ROOT_PATH/data/deepscaler/deepscaler_img.jsonl"
export HF_ENDPOINT="https://hf-mirror.com"
export WANDB_MODE="offline"
export WANDB_DIR="/seu_share/home/gengxin/230249020/projects/lmm-r1-dev/wandb"
MODEL_CPK_NAME="qwenvl25_7B_ins_ppo_deepscaler"
PRETRAIN_MODEL="$ROOT_PATH/ckpts/Qwen2.5-VL-7B-Instruct"
SAVE_PATH="$ROOT_PATH/ckpts"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}"

python -m openrlhf.models.remote_rm.math_verifier --dataset $DATASET --prompt-template chatml --input_key message > "${SAVE_PATH}/${MODEL_CPK_NAME}/remote_rm.log" 2>&1 &
childpid=$!

ray start --head --node-ip-address 0.0.0.0 --num-gpus 8 --temp-dir /tmp/ray

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "./"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --remote_rm_url http://127.0.0.1:5000/get_reward \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 8 \
   --vllm_tensor_parallel_size 1 \
   --colocate_all_models \
   --vllm_enable_sleep \
   --vllm_gpu_memory_utilization 0.5 \
   --vllm_sync_backend gloo \
   --enable_prefix_caching \
   --pretrain $PRETRAIN_MODEL \
   --save_path $SAVE_PATH/$MODEL_CPK_NAME \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 2 \
   --rollout_batch_size 128 \
   --temperature 1 \
   --n_samples_per_prompt 16 \
   --max_epochs 1 \
   --num_episodes 30 \
   --prompt_max_len 4096 \
   --max_samples 100000 \
   --generate_max_len 4096 \
   --advantage_estimator gae \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 4e-7 \
   --init_kl_coef 0.001 \
   --lambd 1 \
   --gamma 1 \
   --prompt_data $DATASET \
   --input_key message \
   --normalize_reward \
   --flash_attn \
   --gradient_checkpointing \
   --save_steps 20 \
   --ckpt_path $SAVE_PATH/$MODEL_CPK_NAME/ckpt \
   --save_hf_ckpt \
   --use_wandb 9aeddea3b60542704fd5cd44d4c4a1d1d911ce54 \
   --wandb_run_name $MODEL_CPK_NAME \
   --freeze_prefix visual \
   --train_vlm


# also supports --advantage_estimator rloo
kill $childpid
ray stop
