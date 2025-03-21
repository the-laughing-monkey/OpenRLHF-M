set -x

  # Submit the training job.
  echo "[HEAD NODE] Submitting training job via Ray job submit..."
  ray job submit --address="http://127.0.0.1:${DASHBOARD_PORT}" \
     --runtime-env-json="{\"working_dir\": \"${WORKSPACE_DIR}\"}" \
     -- python3 -m openrlhf.cli.train_ppo_ray \
         --ref_num_nodes 1 \
         --ref_num_gpus_per_node 8 \
         --remote_rm_url "${REWARD_MODEL_URL}" \
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
         --load_checkpoint \
         ${WANDB_ARGS} \
         --use_tensorboard ${LOG_DIR} > >(tee -a "${CUR_LOG_DIR}/train.log") 2>&1 &
  TRAIN_PID=$!
  
  echo "[HEAD NODE] Training job submitted with PID: $TRAIN_PID"
  echo "[HEAD NODE] Remote Reward Model running with PID: $REMOTE_RM_PID"
  echo "[HEAD NODE] Logs available in ${CUR_LOG_DIR}"
  
  echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  echo "=== OpenRLHF-M Head Node Script Completed ==="
  echo "=== Training job is now running. See logs for progress. ==="
  echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

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
  ray start --address=${HEAD_NODE_IP}:${RAY_PORT} 
  echo "[WORKER NODE] Successfully joined the Ray cluster."
  
  # Save worker PID for tracking
  echo $$ > $WORKER_PID_DIR/worker_${NODE_HOSTNAME}.pid
  
  # Create a log file for worker status
  WORKER_LOG="${CUR_LOG_DIR}/worker_${NODE_HOSTNAME}.log"
  mkdir -p $(dirname "$WORKER_LOG")
  
  echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  echo "=== OpenRLHF-M Worker Node Ready ==="
  echo "=== Worker is now connected to the Ray cluster. ==="
  echo "=== The worker process is running in the background. ==="
  echo "=== Worker log: $WORKER_LOG ==="
  echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  
  # Print worker management instructions
  echo "To check the status of this worker node:"
  echo "  ray status"
  echo ""
  echo "To stop this worker node:"
  echo "  ray stop"
  echo "  # or kill the worker process:"
  echo "  kill $(cat $WORKER_PID_DIR/worker_${NODE_HOSTNAME}.pid)"
  echo ""
  
  # Fully detach the worker process by creating a background job monitoring Ray
  # and redirecting all output to the log file
  (
    # Print startup message to the log
    echo "[WORKER NODE] Ray worker started in background at $(date)" > "$WORKER_LOG"
    echo "[WORKER NODE] Worker PID: $$" >> "$WORKER_LOG"
    
    # Monitor Ray status in a loop
    while ray status >> "$WORKER_LOG" 2>&1; do
      echo "[WORKER NODE] Still active at $(date). Will check again in 60 seconds." >> "$WORKER_LOG"
      sleep 60
    done
    
    echo "[WORKER NODE] Ray process has stopped at $(date). Worker node process will exit." >> "$WORKER_LOG"
    
    # Cleanup
    ray stop >> "$WORKER_LOG" 2>&1
    rm -f $WORKER_PID_DIR/worker_${NODE_HOSTNAME}.pid
    echo "[WORKER NODE] Worker node stopped and cleaned up." >> "$WORKER_LOG"
  ) &
  
  disown
  
  # Exit the foreground script immediately
  exit 0
fi 