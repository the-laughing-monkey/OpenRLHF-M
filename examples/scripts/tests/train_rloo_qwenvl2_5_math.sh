set -x

# Use the math12k dataset
export DATASET="hiyouga/math12k"

MODEL_CPK_NAME="qwenvl25_3B_ins_rloo_math"
PRETRAIN_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
SAVE_PATH="./ckpts"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}"

# Start the remote reward verifier with explicit port and wait to ensure it's running
python3 -m openrlhf.models.remote_rm.math_verifier \
    --dataset $DATASET \
    --input_key problem \
    --prompt-template "Question: {}\nAnswer:" \
    --port 5000 > "${SAVE_PATH}/${MODEL_CPK_NAME}/remote_rm.log" 2>&1 &
childpid=$!

# Wait for the reward model server to start
echo "Waiting for reward model server to start..."
sleep 10

# Check if reward model is running
if ! curl -s http://127.0.0.1:5000 > /dev/null; then
    echo "Warning: Reward model server doesn't seem to be running. Check logs at ${SAVE_PATH}/${MODEL_CPK_NAME}/remote_rm.log"
    # Continue anyway, but print warning
fi

# Start Ray with 2 GPUs
ray start --head --node-ip-address=0.0.0.0 --num-gpus 2

# Submit job with correct working directory
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/data/OpenRLHF-M"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --remote_rm_url http://127.0.0.1:5000/get_reward \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
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
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 256 \
   --temperature 1 \
   --n_samples_per_prompt 16 \
   --max_epochs 1 \
   --num_episodes 30 \
   --prompt_max_len 1024 \
   --max_samples 100000 \
   --generate_max_len 3000 \
   --advantage_estimator rloo \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 1e-6 \
   --init_kl_coef 0.0 \
   --prompt_data $DATASET \
   --input_key problem \
   --input_template "Question: {}\nAnswer:" \
   --normalize_reward \
   --flash_attn \
   --gradient_checkpointing \
   --save_steps 10 \
   --ckpt_path $SAVE_PATH/$MODEL_CPK_NAME/ckpt \
   --save_hf_ckpt \
   --use_tensorboard $SAVE_PATH/$MODEL_CPK_NAME/logs

ray stop