#!/bin/bash
# Use the math12k dataset from HF and the Qwen2.5‑VL–3B model with a formatted prompt.
export DATASET="hiyouga/math12k"

MODEL_CPK_NAME="qwenvl25_3B_ins_grpo_math"
PRETRAIN_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
SAVE_PATH="./ckpts"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}"

# Start the remote reward verifier 
python3 -m openrlhf.models.remote_rm.math_verifier \
    --dataset $DATASET \
    --input_key problem \
    --prompt-template "Question: {}\nAnswer:" \
    > "${SAVE_PATH}/${MODEL_CPK_NAME}/remote_rm.log" 2>&1 &
childpid=$!

# Start Ray on the head node with 2 GPUs.
 --num-gpus 2 --temp-dir ~/.cache/ray

# Submit the job using a runtime working directory of /data/lmm-r1 
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/data/OpenRLHF-M"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 2 \
   --vllm_sync_backend nccl \
   --vllm_enable_sleep \
   --vllm_gpu_memory_utilization 0.5 \
   --vllm_sync_backend gloo \
   --colocate_all_models \
   --enable_prefix_caching \
   --pretrain $PRETRAIN_MODEL \
   --reward_pretrain $PRETRAIN_MODEL \
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
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 1e-3 \
   --gamma 1.0 \
   --use_kl_loss \
   --kl_estimator k3 \
   --advantage_estimator group_norm \
   --prompt_data $DATASET \
   --input_key problem \
   --input_template "Question: {}\nAnswer:" \
   --normalize_reward \
   --flash_attn \
   --gradient_checkpointing \
   --packing_samples \
   --enforce_eager \
   --deepspeed_enable_sleep \
   --save_steps 10 \
   --ckpt_path $SAVE_PATH/$MODEL_CPK_NAME/ckpt \
   --save_hf_ckpt \
   --use_tensorboard $SAVE_PATH/$MODEL_CPK_NAME/logs

# You could also try
#   --kl_estimator k2 \
ray stop
