import os
from typing import Dict, List, Optional, Union, Any

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import logging

from transformers import AutoProcessor

logger = logging.getLogger(__name__)

class MultimodalClassificationDataset(Dataset):
    """
    Multimodal Classification Dataset
    
    Args:
        dataset: The original dataset
        tokenizer: The tokenizer
        max_len: Maximum sequence length
        strategy: Training strategy
        image_key: Key for image path
        text_key: Key for text
        label_key: Key for label
        image_folder: Path to the image folder, if provided, will be concatenated with the image path
        image_size: Size of the image
        use_augmentation: Whether to use data augmentation
        model_type: Type of the model, e.g., "qwen2vl", "generic"
    """
    
    def __init__(
        self,
        dataset,
        tokenizer,
        max_len,
        strategy,
        image_key="image",
        text_key="text",
        label_key="label",
        image_folder=None,
        image_size=448,
        use_augmentation=False,
        model_type="generic",
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.strategy = strategy
        self.image_key = image_key
        self.text_key = text_key
        self.label_key = label_key
        self.image_folder = image_folder
        self.image_size = image_size
        self.use_augmentation = use_augmentation
        self.model_type = model_type.lower()
        
        if hasattr(tokenizer, "processor"):
            self.processor = tokenizer.processor
        else:
            try:
                self.processor = AutoProcessor.from_pretrained(tokenizer.name_or_path)
            except Exception as e:
                logger.warning(f"Unable to load processor: {e}, using default image processing")
                self.processor = None
        
        if use_augmentation:
            logger.info("Using data augmentation")
            self.transform = self._get_augmentation_transform()
        else:
            self.transform = self._get_eval_transform()
    
    def _get_augmentation_transform(self):
        """Get image transformations with data augmentation"""
        from torchvision import transforms
        
        if self.model_type == "qwen2vl" and self.processor is not None:
            return None
        
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _get_eval_transform(self):
        """Get image transformations for evaluation"""
        from torchvision import transforms
        
        if self.model_type == "qwen2vl" and self.processor is not None:
            return None
        
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get dataset item"""
        item = self.dataset[idx]
        
        text = item[self.text_key]
        
        image_path = item[self.image_key]
        if self.image_folder is not None:
            image_path = os.path.join(self.image_folder, image_path)
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Cannot load image {image_path}: {e}")
            image = Image.new("RGB", (self.image_size, self.image_size), color=(128, 128, 128))
        
        if self.model_type == "qwen2vl" and self.processor is not None:
            try:
                encoding = self.processor(
                    text=text,
                    images=image,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors="pt"
                )
                
                input_ids = encoding.input_ids[0]
                attention_mask = encoding.attention_mask[0]
                pixel_values = encoding.pixel_values[0]
                
                label = item[self.label_key]
                if isinstance(label, str):
                    try:
                        label = int(label)
                    except ValueError:
                        logger.warning(f"Unable to convert label '{label}' to integer, using 0 as default")
                        label = 0
                
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "pixel_values": pixel_values,
                    "labels": torch.tensor(label, dtype=torch.long),
                    "text": text,
                    "image_path": image_path,
                }
            except Exception as e:
                logger.error(f"Qwen2VL processing failed: {e}, falling back to default processing")
        
        if self.transform is not None:
            try:
                image_tensor = self.transform(image)
            except Exception as e:
                logger.error(f"Image transformation failed: {e}")
                image_tensor = torch.zeros(3, self.image_size, self.image_size)
        else:
            if self.processor is not None:
                try:
                    image_tensor = self.processor(images=image, return_tensors="pt").pixel_values[0]
                except Exception as e:
                    logger.error(f"Processor failed to process image: {e}")
                    image_tensor = torch.zeros(3, self.image_size, self.image_size)
            else:
                from torchvision import transforms
                simple_transform = transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                ])
                image_tensor = simple_transform(image)
        
        label = item[self.label_key]
        if isinstance(label, str):
            try:
                label = int(label)
            except ValueError:
                logger.warning(f"Unable to convert label '{label}' to integer, using 0 as default")
                label = 0
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding.input_ids[0],
            "attention_mask": encoding.attention_mask[0],
            "pixel_values": image_tensor,
            "labels": torch.tensor(label, dtype=torch.long),
            "text": text,
            "image_path": image_path,
        }
    
    def collate_fn(self, batch):
        """Batch collate function"""
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        
        input_ids = input_ids.to(self.strategy.device)
        attention_mask = attention_mask.to(self.strategy.device)
        pixel_values = pixel_values.to(self.strategy.device)
        labels = labels.to(self.strategy.device)
        
        texts = [item["text"] for item in batch]
        image_paths = [item["image_path"] for item in batch]
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
            "texts": texts,
            "image_paths": image_paths,
        }
        
        if self.model_type == "qwen2vl":
            if "image_grid_thw" in batch[0]:
                image_grid_thw = torch.stack([item["image_grid_thw"] for item in batch])
                image_grid_thw = image_grid_thw.to(self.strategy.device)
                result["image_grid_thw"] = image_grid_thw
            
            if "video_grid_thw" in batch[0]:
                video_grid_thw = torch.stack([item["video_grid_thw"] for item in batch])
                video_grid_thw = video_grid_thw.to(self.strategy.device)
                result["video_grid_thw"] = video_grid_thw
        
        return result