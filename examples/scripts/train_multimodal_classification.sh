#!/bin/bash
set -x

MODEL_TYPE=${MODEL_TYPE:-qwen2vl}  # 默认使用qwen2vl，可以通过环境变量覆盖

read -r -d '' training_commands <<EOF
openrlhf.cli.train_multimodal_classification \
   --save_path ./checkpoint/${MODEL_TYPE}-classification \
   --save_steps 1000 \
   --logging_steps 10 \
   --eval_steps 500 \
   --train_batch_size 32 \
   --micro_train_batch_size 4 \
   --pretrain Qwen/Qwen2-VL-7B \
   --model_type ${MODEL_TYPE} \
   --bf16 \
   --max_len 2048 \
   --train_data multimodal_classification_dataset \
   --input_key image_text \
   --label_key label \
   --num_classes 10 \
   --max_epochs 3 \
   --learning_rate 5e-6 \
   --flash_attn \
   --gradient_checkpointing \
   --adam_offload
EOF
   # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
   # --load_in_4bit (use QLoRA)
   # --lora_rank 8 --target_modules q_proj,k_proj,v_proj,o_proj (use LoRA)

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi 