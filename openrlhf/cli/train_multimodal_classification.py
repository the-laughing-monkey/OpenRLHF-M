import argparse
import math
import os
import time
import logging
from datetime import datetime

import torch
import torch.distributed as dist
from transformers import get_scheduler, set_seed
from sklearn.metrics import classification_report, confusion_matrix

from openrlhf.datasets import MultimodalClassificationDataset
from openrlhf.models import load_multimodal_model_for_classification
from openrlhf.trainer import MultimodalClassificationTrainer
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer

logger = logging.getLogger(__name__)

def setup_logging(args):
    """Set up logging"""
    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=log_level,
    )
    
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setLevel(log_level)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)

def validate_args(args):
    """Validate command line arguments"""
    if not args.pretrain:
        raise ValueError("Must provide pretrained model path (--pretrain)")
    
    if not args.train_data:
        raise ValueError("Must provide training data path (--train_data)")
    
    if args.fp16 and args.bf16:
        raise ValueError("Cannot enable both fp16 and bf16")
    
    if args.lora_rank > 0 and args.target_modules == "all-linear" and args.vision_tower_lora:
        logger.warning("Applying LoRA to all linear layers and vision tower may cause OOM")
    
    if args.train_batch_size < args.micro_train_batch_size:
        logger.warning(f"train_batch_size ({args.train_batch_size}) is less than micro_train_batch_size ({args.micro_train_batch_size})")
        args.train_batch_size = args.micro_train_batch_size
        logger.warning(f"Setting train_batch_size to {args.train_batch_size}")

def train(args):
    """Train Multimodal classification model"""
    # Set up logging
    setup_logging(args)
    
    # Validate arguments
    validate_args(args)
    
    # Set random seed
    if args.seed is not None:
        set_seed(args.seed)
        logger.info(f"Random seed set: {args.seed}")
    
    # Configure distributed training strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()
    
    logger.info(f"Using device: {strategy.device}")
    logger.info(f"Distributed training: {dist.is_initialized()}")
    if dist.is_initialized():
        logger.info(f"World size: {dist.get_world_size()}, local rank: {dist.get_rank()}")
    
    # Configure model
    start_time = time.time()
    logger.info(f"Loading model: {args.pretrain}")
    
    try:
        model, tokenizer = load_multimodal_model_for_classification(
            args.pretrain,
            model_type=args.model_type,
            num_classes=args.num_classes,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            vision_tower_lora=args.vision_tower_lora,
        )
        logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    if args.verbose:
        strategy.print(model)
    
    # Configure dataset
    logger.info("Preparing training dataset...")
    try:
        train_data = blending_datasets(
            args.train_data,
            args.train_data_probs,
            strategy,
            return_eval=False,
            train_split=args.train_split,
        )
        logger.info(f"Raw training dataset size: {len(train_data)}")
        
        # Limit sample count
        max_samples = min(args.max_samples, len(train_data))
        train_data = train_data.select(range(max_samples))
        logger.info(f"Using {len(train_data)} training samples")
        
        # Create training dataset
        train_dataset = MultimodalClassificationDataset(
            train_data,
            tokenizer,
            args.max_len,
            strategy,
            image_key=args.image_key,
            text_key=args.text_key,
            label_key=args.label_key,
            image_folder=args.image_folder,
            image_size=args.image_size,
            use_augmentation=args.use_augmentation,
        )
        logger.info("Training dataset prepared")
    except Exception as e:
        logger.error(f"Failed to prepare training dataset: {e}")
        raise
    
    # Prepare evaluation dataset
    if args.eval_data:
        logger.info("Preparing evaluation dataset...")
        try:
            eval_data = blending_datasets(
                args.eval_data,
                args.eval_data_probs,
                strategy,
                return_eval=True,
                eval_split=args.eval_split,
            )
            logger.info(f"Raw evaluation dataset size: {len(eval_data)}")
            
            # Limit sample count
            max_eval_samples = min(args.max_eval_samples, len(eval_data))
            eval_data = eval_data.select(range(max_eval_samples))
            logger.info(f"Using {len(eval_data)} evaluation samples")
            
            # Create evaluation dataset
            eval_dataset = MultimodalClassificationDataset(
                eval_data,
                tokenizer,
                args.max_len,
                strategy,
                image_key=args.image_key,
                text_key=args.text_key,
                label_key=args.label_key,
                image_folder=args.image_folder,
                image_size=args.image_size,
                use_augmentation=False,  # No augmentation for evaluation
            )
            logger.info("Evaluation dataset prepared")
        except Exception as e:
            logger.error(f"Failed to prepare evaluation dataset: {e}")
            raise
    else:
        eval_dataset = None
        logger.info("No evaluation dataset provided")
    
    # Configure data loaders
    logger.info("Preparing data loaders...")
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=train_dataset.collate_fn,
    )
    logger.info(f"Training dataloader batches: {len(train_dataloader)}")
    
    if eval_dataset:
        eval_dataloader = strategy.setup_dataloader(
            eval_dataset,
            args.micro_eval_batch_size or args.micro_train_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=eval_dataset.collate_fn,
        )
        logger.info(f"Evaluation dataloader batches: {len(eval_dataloader)}")
    else:
        eval_dataloader = None
    
    # Calculate training steps
    num_update_steps_per_epoch = len(train_dataset) // args.train_batch_size
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)
    logger.info(f"Updates per epoch: {num_update_steps_per_epoch}, total steps: {max_steps}")
    
    # Configure optimizer
    logger.info("Configuring optimizer...")
    try:
        optim = strategy.create_optimizer(
            model,
            lr=args.learning_rate,
            betas=args.adam_betas,
            weight_decay=args.l2,
            fused=args.fused,
        )
        
        # Configure learning rate scheduler
        warmup_steps = math.ceil(max_steps * args.lr_warmup_ratio)
        logger.info(f"Learning rate warmup steps: {warmup_steps}")
        
        scheduler = get_scheduler(
            args.lr_scheduler,
            optim,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
        
        logger.info(f"Using scheduler: {args.lr_scheduler}")
    except Exception as e:
        logger.error(f"Failed to configure optimizer: {e}")
        raise
    
    model, optim, scheduler = strategy.prepare((model, optim, scheduler))
    
    # Configure trainer
    logger.info("Initializing trainer...")
    try:
        trainer = MultimodalClassificationTrainer(
            model=model,
            strategy=strategy,
            optim=optim,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            scheduler=scheduler,
            max_norm=args.max_norm,
            max_steps=max_steps,
            num_epochs=args.max_epochs,
            tokenizer=tokenizer,
            ckpt_path=args.ckpt_path,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            logging_steps=args.logging_steps,
            aux_loss_coef=args.aux_loss_coef,
            save_hf_ckpt=args.save_hf_ckpt,
            disable_ds_ckpt=args.disable_ds_ckpt,
            label_names=args.label_names,
            num_classes=args.num_classes,
        )
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}")
        raise
    
    # Start training
    logger.info("Starting training...")
    consumed_samples = 0
    
    try:
        trainer.fit(args, consumed_samples, num_update_steps_per_epoch)
        logger.info("Training completed")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise
    
    # Save model
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(f"Saving model to {args.save_path}")
        try:
            os.makedirs(args.save_path, exist_ok=True)
            strategy.save_model(model, tokenizer, args.save_path)
            logger.info("Model saved")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    # Final evaluation
    if eval_dataloader and (not dist.is_initialized() or dist.get_rank() == 0):
        logger.info("Performing final evaluation...")
        try:
            eval_results = trainer.evaluate(detailed=True)
            logger.info(f"Final evaluation results: {eval_results}")
            
            if args.save_eval_results:
                report_path = os.path.join(args.save_path, "evaluation_report.txt")
                with open(report_path, "w") as f:
                    f.write(f"Classification Report:\n{eval_results['classification_report']}\n\n")
                    f.write(f"Confusion Matrix:\n{eval_results['confusion_matrix']}\n")
                logger.info(f"Evaluation report saved to {report_path}")
        except Exception as e:
            logger.error(f"Final evaluation failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Qwen2VL multimodal classification model")
    
    # Checkpoint related
    parser.add_argument("--save_path", type=str, default="./ckpt", help="Path to save model")
    parser.add_argument("--save_steps", type=int, default=-1, help="Save checkpoint every X steps, -1 means no saving")
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False, help="Save HuggingFace format checkpoint")
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False, help="Disable DeepSpeed checkpoint")
    parser.add_argument("--logging_steps", type=int, default=1, help="Log every X steps")
    parser.add_argument("--eval_steps", type=int, default=-1, help="Evaluate every X steps, -1 means evaluate at end of epoch")
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_qwen2vl", help="Path to save checkpoints")
    parser.add_argument("--save_eval_results", action="store_true", default=False, help="Save detailed evaluation results")

    # DeepSpeed related
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--zero_stage", type=int, default=0, help="DeepSpeed ZeRO optimization stage")
    parser.add_argument("--bf16", action="store_true", default=False, help="Use bfloat16 precision")
    parser.add_argument("--fp16", action="store_true", default=False, help="Use float16 precision")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False, help="Use gradient checkpointing")
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False, help="Use reentrant gradient checkpointing")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False, help="Disable fast tokenizer")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")

    # LoRA related
    parser.add_argument("--load_in_4bit", action="store_true", default=False, help="Load model in 4-bit quantization")
    parser.add_argument("--lora_rank", type=int, default=0, help="LoRA rank, 0 means no LoRA")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0, help="LoRA dropout probability")
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear", help="LoRA target modules")
    parser.add_argument("--vision_tower_lora", action="store_true", default=False, help="Apply LoRA to vision tower")

    # Qwen2VL classification training
    parser.add_argument("--pretrain", type=str, required=True, help="Pretrained model path or name")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classification classes")
    parser.add_argument("--label_names", type=str, nargs="*", default=None, help="List of label names")
    parser.add_argument("--max_epochs", type=int, default=2, help="Maximum number of training epochs")
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="Auxiliary loss coefficient for MoE balance loss")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03, help="Learning rate warmup ratio")
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr", help="Learning rate scheduler type")
    parser.add_argument("--l2", type=float, default=0, help="Weight decay coefficient")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Adam optimizer beta parameters")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Maximum norm for gradient clipping")
    parser.add_argument("--fused", action="store_true", default=False, help="Use fused optimizer")
    parser.add_argument("--micro_train_batch_size", type=int, default=1, help="Training batch size per GPU")
    parser.add_argument("--micro_eval_batch_size", type=int, default=None, help="Evaluation batch size per GPU, defaults to same as training")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")

    # Dataset related
    parser.add_argument("--train_data", type=str, required=True, help="Training dataset name or path")
    parser.add_argument("--train_data_probs", type=str, default="1.0", help="Sampling probabilities for multiple datasets")
    parser.add_argument("--eval_data", type=str, default=None, help="Evaluation dataset name or path")
    parser.add_argument("--eval_data_probs", type=str, default="1.0", help="Sampling probabilities for multiple evaluation datasets")
    parser.add_argument("--train_split", type=str, default="train", help="Training dataset split")
    parser.add_argument("--eval_split", type=str, default="test", help="Evaluation dataset split")
    parser.add_argument("--image_key", type=str, default="image", help="JSON key for image path")
    parser.add_argument("--text_key", type=str, default="text", help="JSON key for text")
    parser.add_argument("--label_key", type=str, default="label", help="JSON key for label")
    parser.add_argument("--image_folder", type=str, default=None, help="Path to image folder")
    parser.add_argument("--image_size", type=int, default=448, help="Image size")
    parser.add_argument("--max_samples", type=int, default=int(1e8), help="Maximum number of training samples")
    parser.add_argument("--max_eval_samples", type=int, default=int(1e8), help="Maximum number of evaluation samples")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum token length")
    parser.add_argument("--use_augmentation", action="store_true", default=False, help="Use data augmentation")

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None, help="Whether to use wandb")
    parser.add_argument("--wandb_org", type=str, default=None, help="wandb organization")
    parser.add_argument("--wandb_group", type=str, default=None, help="wandb group")
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_qwen2vl", help="wandb project name")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="qwen2vl_%s" % datetime.now().strftime("%m%dT%H:%M"),
        help="wandb run name",
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard log path")

    # Other parameters
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--verbose", action="store_true", default=False, help="Output verbose logs")
    parser.add_argument("--log_file", type=str, default=None, help="Log file path")
    parser.add_argument("--use_ms", action="store_true", default=False, help="Use ModelScope")

    # New parameter for model type
    parser.add_argument("--model_type", type=str, default="qwen2vl", help="模型类型，支持qwen2vl、llava、blip2等")

    args = parser.parse_args()

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub

        # Patch hub to download models from modelscope for acceleration
        patch_hub()

    train(args)