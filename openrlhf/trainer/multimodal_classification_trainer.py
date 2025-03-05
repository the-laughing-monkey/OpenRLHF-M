import os
import time
from typing import Dict, List, Optional, Union, Any
import logging

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from abc import ABC
logger = logging.getLogger(__name__)

class MultimodalClassificationTrainer(ABC):
    """
    Multimodal Classification Trainer
    
    Args:
        model: Model to be trained
        strategy: Training strategy
        optim: Optimizer
        train_dataloader: Training data loader
        eval_dataloader: Evaluation data loader
        scheduler: Learning rate scheduler
        max_norm: Maximum norm for gradient clipping
        max_steps: Maximum training steps
        num_epochs: Number of training epochs
        tokenizer: Tokenizer
        ckpt_path: Checkpoint save path
        save_steps: Save checkpoint every certain steps
        eval_steps: Evaluate every certain steps
        logging_steps: Log every certain steps
        save_hf_ckpt: Whether to save HuggingFace format checkpoint
        disable_ds_ckpt: Whether to disable DeepSpeed checkpoint
        label_names: List of label names
        num_classes: Number of classification categories
    """
    
    def __init__(
        self,
        model,
        strategy,
        optim,
        train_dataloader,
        eval_dataloader=None,
        scheduler=None,
        max_norm=1.0,
        max_steps=None,
        num_epochs=None,
        tokenizer=None,
        ckpt_path=None,
        save_steps=None,
        eval_steps=None,
        logging_steps=None,
        save_hf_ckpt=False,
        disable_ds_ckpt=False,
        label_names=None,
        num_classes=2,
    ):
        self.model = model
        self.strategy = strategy
        self.optim = optim
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.max_norm = max_norm
        self.max_steps = max_steps
        self.num_epochs = num_epochs
        self.tokenizer = tokenizer
        self.ckpt_path = ckpt_path
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        self.save_hf_ckpt = save_hf_ckpt
        self.disable_ds_ckpt = disable_ds_ckpt
        
        if label_names is not None:
            self.label_names = label_names
        else:
            self.label_names = [str(i) for i in range(num_classes)]
        
        if hasattr(model, "processor"):
            self.processor = model.processor
        
        self.best_accuracy = 0.0
        self.best_step = 0
    
    def validate_args(self, args):
        """
            Validate the batch size
        Args:
            args: Training parameters
        """
        if hasattr(args, 'n_rollout') and hasattr(args, 'ring_head') and hasattr(args, 'world_size'):
            expected_micro_batch = args.batch_size * args.ring_head * args.n_rollout // args.world_size
            if hasattr(args, 'micro_batch_size') and args.micro_batch_size != expected_micro_batch:
                logger.warning(
                    f"micro_batch_size should be equal to batch_size * ring_head * n_rollout // world_size, "
                    f"current value is {args.micro_batch_size}, expected value is {expected_micro_batch}"
                )
                args.micro_batch_size = expected_micro_batch
                logger.info(f"micro_batch_size has been automatically corrected to {expected_micro_batch}")
        
        return args
    
    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        """
        Train the model
        
        Args:
            args: Training parameters
            consumed_samples: Number of consumed samples
            num_update_steps_per_epoch: Number of update steps per epoch
        """
        args = self.validate_args(args)
        
        if args.use_wandb and (not dist.is_initialized() or dist.get_rank() == 0):
            import wandb
            if args.use_wandb.lower() == "true":
                wandb.init(project=args.wandb_project, name=args.wandb_run_name)
            else:
                wandb.init(
                    project=args.wandb_project,
                    name=args.wandb_run_name,
                    entity=args.wandb_org,
                    group=args.wandb_group,
                )
            
            wandb.config.update(vars(args))
        
        if args.use_tensorboard and (not dist.is_initialized() or dist.get_rank() == 0):
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=args.use_tensorboard)
        
        global_step = consumed_samples // args.train_batch_size
        global_time = time.time()
        
        self.strategy.print("Starting training...")
        
        for epoch in range(self.num_epochs):
            self.strategy.print(f"Epoch {epoch+1}/{self.num_epochs}")
            
            self.model.train()
            train_loss = 0.0
            train_acc = 0.0
            train_samples = 0
            epoch_start_time = time.time()
            
            progress_bar = tqdm(
                total=len(self.train_dataloader),
                desc=f"Epoch {epoch+1}",
                disable=dist.is_initialized() and dist.get_rank() != 0,
            )
            
            for step, batch in enumerate(self.train_dataloader):
                step_start_time = time.time()
                
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                pixel_values = batch["pixel_values"]
                labels = batch["labels"]
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                )
                
                loss = outputs.loss
                
                self.strategy.backward(loss, self.model, self.optim)
                
                if self.max_norm > 0:
                    self.strategy.clip_grad_norm(self.model.parameters(), self.max_norm)
                
                self.strategy.optimizer_step(self.optim)
                self.optim.zero_grad()
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                correct = (preds == labels).sum().item()
                
                batch_size = labels.size(0)
                train_loss += loss.item() * batch_size
                train_acc += correct
                train_samples += batch_size
                
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{correct / batch_size:.4f}",
                    "lr": f"{self.optim.param_groups[0]['lr']:.2e}",
                    "time": f"{time.time() - step_start_time:.2f}s",
                })
                
                if self.logging_steps > 0 and global_step % self.logging_steps == 0:
                    avg_loss = train_loss / train_samples
                    avg_acc = train_acc / train_samples
                    
                    self.strategy.print(
                        f"Step {global_step} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Accuracy: {avg_acc:.4f} | "
                        f"Learning Rate: {self.optim.param_groups[0]['lr']:.2e} | "
                        f"Time per step: {(time.time() - step_start_time):.2f}s"
                    )
                    
                    if args.use_wandb and (not dist.is_initialized() or dist.get_rank() == 0):
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/accuracy": avg_acc,
                            "train/learning_rate": self.optim.param_groups[0]["lr"],
                            "train/global_step": global_step,
                        })
                    
                    if args.use_tensorboard and (not dist.is_initialized() or dist.get_rank() == 0):
                        writer.add_scalar("train/loss", avg_loss, global_step)
                        writer.add_scalar("train/accuracy", avg_acc, global_step)
                        writer.add_scalar("train/learning_rate", self.optim.param_groups[0]["lr"], global_step)
                
                if self.eval_dataloader is not None and self.eval_steps > 0 and global_step % self.eval_steps == 0:
                    eval_results = self.evaluate()
                    
                    self.strategy.print(
                        f"Evaluation | Step {global_step} | "
                        f"Loss: {eval_results['loss']:.4f} | "
                        f"Accuracy: {eval_results['accuracy']:.4f}"
                    )
                    
                    if args.use_wandb and (not dist.is_initialized() or dist.get_rank() == 0):
                        wandb.log({
                            "eval/loss": eval_results["loss"],
                            "eval/accuracy": eval_results["accuracy"],
                            "eval/global_step": global_step,
                        })
                    
                    if args.use_tensorboard and (not dist.is_initialized() or dist.get_rank() == 0):
                        writer.add_scalar("eval/loss", eval_results["loss"], global_step)
                        writer.add_scalar("eval/accuracy", eval_results["accuracy"], global_step)
                    
                    if eval_results["accuracy"] > self.best_accuracy:
                        self.best_accuracy = eval_results["accuracy"]
                        self.best_step = global_step
                        
                        if not dist.is_initialized() or dist.get_rank() == 0:
                            best_ckpt_dir = os.path.join(args.save_path, "best_model")
                            os.makedirs(best_ckpt_dir, exist_ok=True)
                            
                            if self.save_hf_ckpt:
                                self.strategy.save_model(self.model, self.tokenizer, best_ckpt_dir)
                                logger.info(f"Saved best model to {best_ckpt_dir}")
                    
                    self.model.train()
                
                if self.save_steps > 0 and global_step % self.save_steps == 0:
                    ckpt_dir = os.path.join(self.ckpt_path, f"checkpoint-{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    
                    if not self.disable_ds_ckpt:
                        self.strategy.save_ckpt(
                            self.model,
                            os.path.join(ckpt_dir, "ds_model.pt"),
                            {"consumed_samples": (global_step + 1) * args.train_batch_size},
                        )
                    
                    if self.save_hf_ckpt and (not dist.is_initialized() or dist.get_rank() == 0):
                        self.strategy.save_model(self.model, self.tokenizer, ckpt_dir)
                
                global_step += 1
            
            progress_bar.close()
            
            avg_loss = train_loss / train_samples
            avg_acc = train_acc / train_samples
            epoch_time = time.time() - epoch_start_time
            
            self.strategy.print(
                f"Epoch {epoch+1} completed | "
                f"Loss: {avg_loss:.4f} | "
                f"Accuracy: {avg_acc:.4f} | "
                f"Time: {epoch_time:.2f}s"
            )
            
            if args.use_wandb and (not dist.is_initialized() or dist.get_rank() == 0):
                wandb.log({
                    "train/epoch_loss": avg_loss,
                    "train/epoch_accuracy": avg_acc,
                    "train/epoch": epoch,
                    "train/epoch_time": epoch_time,
                })
            
            if args.use_tensorboard and (not dist.is_initialized() or dist.get_rank() == 0):
                writer.add_scalar("train/epoch_loss", avg_loss, epoch)
                writer.add_scalar("train/epoch_accuracy", avg_acc, epoch)
                writer.add_scalar("train/epoch_time", epoch_time, epoch)
            
            if self.eval_dataloader is not None:
                eval_results = self.evaluate()
                
                self.strategy.print(
                    f"Epoch {epoch+1} evaluation | "
                    f"Loss: {eval_results['loss']:.4f} | "
                    f"Accuracy: {eval_results['accuracy']:.4f}"
                )
                
                if args.use_wandb and (not dist.is_initialized() or dist.get_rank() == 0):
                    wandb.log({
                        "eval/epoch_loss": eval_results["loss"],
                        "eval/epoch_accuracy": eval_results["accuracy"],
                        "eval/epoch": epoch,
                    })
                
                if args.use_tensorboard and (not dist.is_initialized() or dist.get_rank() == 0):
                    writer.add_scalar("eval/epoch_loss", eval_results["loss"], epoch)
                    writer.add_scalar("eval/epoch_accuracy", eval_results["accuracy"], epoch)
                
                if eval_results["accuracy"] > self.best_accuracy:
                    self.best_accuracy = eval_results["accuracy"]
                    self.best_step = global_step
                    
                    if not dist.is_initialized() or dist.get_rank() == 0:
                        best_ckpt_dir = os.path.join(args.save_path, "best_model")
                        os.makedirs(best_ckpt_dir, exist_ok=True)
                        
                        if self.save_hf_ckpt:
                            self.strategy.save_model(self.model, self.tokenizer, best_ckpt_dir)
                            logger.info(f"Saved best model to {best_ckpt_dir}")
        
        total_time = time.time() - global_time
        self.strategy.print(f"Training completed, total time: {total_time:.2f}s")
        self.strategy.print(f"Best accuracy: {self.best_accuracy:.4f}, at step {self.best_step}")
        
        if args.use_wandb and (not dist.is_initialized() or dist.get_rank() == 0):
            wandb.finish()
        
        if args.use_tensorboard and (not dist.is_initialized() or dist.get_rank() == 0):
            writer.close()
    
    def evaluate(self, detailed=False):
        """
        Evaluate the model
        
        Args:
            detailed: Whether to return detailed evaluation results
            
        Returns:
            dict: Evaluation results
        """
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating", disable=dist.is_initialized() and dist.get_rank() != 0):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                pixel_values = batch["pixel_values"]
                labels = batch["labels"]
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                preds = torch.argmax(logits, dim=1)
                correct = (preds == labels).sum().item()
                
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_correct += correct
                total_samples += batch_size
                
                if detailed:
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        results = {
            "loss": avg_loss,
            "accuracy": accuracy,
        }
        
        if detailed and len(all_preds) > 0:
            try:
                report = classification_report(
                    all_labels,
                    all_preds,
                    target_names=self.label_names,
                    digits=4,
                    zero_division=0,
                )
                results["classification_report"] = report
            except Exception as e:
                logger.error(f"Failed to compute classification report: {e}")
            
            try:
                cm = confusion_matrix(all_labels, all_preds)
                results["confusion_matrix"] = cm
            except Exception as e:
                logger.error(f"Failed to compute confusion matrix: {e}")
        
        self.model.train()
        
        return results
    
    def predict(self, dataloader):
        """
        Make predictions using the model
        
        Args:
            dataloader: Data loader
            
        Returns:
            dict: Prediction results
        """
        self.model.eval()
        
        all_preds = []
        all_probs = []
        all_labels = []
        all_texts = []
        all_image_paths = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting", disable=dist.is_initialized() and dist.get_rank() != 0):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                pixel_values = batch["pixel_values"]
                
                if "labels" in batch:
                    labels = batch["labels"]
                    all_labels.extend(labels.cpu().numpy())
                
                if "texts" in batch:
                    all_texts.extend(batch["texts"])
                
                if "image_paths" in batch:
                    all_image_paths.extend(batch["image_paths"])
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                )
                
                logits = outputs.logits
                
                preds = torch.argmax(logits, dim=1)
                probs = torch.softmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        results = {
            "predictions": np.array(all_preds),
            "probabilities": np.array(all_probs),
        }
        
        if all_labels:
            results["labels"] = np.array(all_labels)
        
        if all_texts:
            results["texts"] = all_texts
        
        if all_image_paths:
            results["image_paths"] = all_image_paths
        
        self.model.train()
        
        return results
    
    def save_checkpoint(self, path, metadata=None):
        """
        Save checkpoint
        
        Args:
            path: Save path
            metadata: Metadata
        """
        if not dist.is_initialized() or dist.get_rank() == 0:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optim.state_dict(),
                "best_accuracy": self.best_accuracy,
                "best_step": self.best_step,
            }
            
            if self.scheduler is not None:
                checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
            
            if metadata is not None:
                checkpoint["metadata"] = metadata
            
            torch.save(checkpoint, path)
            logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """
        Load checkpoint
        
        Args:
            path: Checkpoint path
            
        Returns:
            dict: Metadata
        """
        if not os.path.exists(path):
            logger.error(f"Checkpoint {path} does not exist")
            return None
        
        checkpoint = torch.load(path, map_location=self.strategy.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optim.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if "best_accuracy" in checkpoint:
            self.best_accuracy = checkpoint["best_accuracy"]
        
        if "best_step" in checkpoint:
            self.best_step = checkpoint["best_step"]
        
        logger.info(f"Checkpoint loaded from {path}")
        
        return checkpoint.get("metadata", None) 