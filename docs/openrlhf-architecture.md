# OpenRLHF-M Architecture Documentation

## Overview

OpenRLHF-M is a high-performance framework for Reinforcement Learning from Human Feedback (RLHF), with extended capabilities for multimodal models. Built on Ray, DeepSpeed, and Hugging Face Transformers, it provides a comprehensive solution for training large language models (LLMs) and multimodal models using various reinforcement learning techniques.

The framework is designed to be:
- **Efficient**: leveraging distributed computing to handle large models (70B+ parameters)
- **Comprehensive**: supporting multiple RL algorithms, from PPO to REINFORCE++ and more
- **Flexible**: allowing for different configurations and deployment scenarios
- **Scalable**: capable of running on multiple GPUs and nodes
- **Multimodal**: extending RLHF techniques to vision-language models

## Code Organization

The codebase follows a modular structure, organized into several key components:

```
openrlhf/
├── cli/               # Command-line interfaces for training and serving models
├── datasets/          # Dataset classes and data processing utilities
├── models/            # Model definitions and implementations
│   ├── lmm_kits/      # Multimodal model integration components
│   └── remote_rm/     # Remote reward model implementations
├── trainer/           # Training algorithm implementations
│   ├── ppo_utils/     # Utilities for PPO algorithm
│   └── ray/           # Ray-based distributed training components
└── utils/             # General utility functions
```

### Core Components

1. **CLI Layer**: Entry points for training and inference
   - `train_sft.py` - Supervised fine-tuning
   - `train_rm.py` - Reward model training
   - `train_ppo.py` - PPO training (single node)
   - `train_ppo_ray.py` - Distributed PPO training using Ray
   - `train_dpo.py` - Direct Preference Optimization
   - `train_kto.py` - Kahneman-Tversky Optimization
   - `train_prm.py` - Process Reward Model
   - Various utility scripts (lora_combiner, interactive_chat, etc.)

2. **Model Layer**: Core model implementations
   - `actor.py` - Actor model for PPO
   - `model.py` - Base model implementations and utilities
   - `loss.py` - Various loss functions (DPO, KTO, PolicyLoss, etc.)
   - `lmm_kits/` - Multimodal model integration

3. **Trainer Layer**: Training algorithm implementations
   - `ppo_trainer.py` - PPO training logic
   - `dpo_trainer.py` - DPO training logic
   - `rm_trainer.py` - Reward model training
   - `sft_trainer.py` - Supervised fine-tuning

4. **Ray Components**: Distributed training infrastructure
   - `ppo_actor.py` - Ray actor for PPO algorithm
   - `ppo_critic.py` - Ray critic for PPO algorithm
   - `vllm_engine.py` - vLLM integration for efficient inference
   - `launcher.py` - Orchestration of distributed training

## Reinforcement Learning Methods

OpenRLHF-M supports a comprehensive set of reinforcement learning methods for LLM alignment:

1. **Proximal Policy Optimization (PPO)**: 
   - Full implementation with various optimizations and tricks
   - Supports distributed training using Ray
   - Includes KL divergence control, gradient clipping, etc.

2. **REINFORCE++ and RLOO**:
   - Simpler and faster alternatives to PPO
   - More stable training in some cases

3. **Direct Preference Optimization (DPO)**:
   - Implementation of the offline RLHF algorithm
   - Variants include IPO and cDPO

4. **Kahneman-Tversky Optimization (KTO)**:
   - Alternative preference optimization technique
   - Based on prospect theory

5. **Iterative DPO**:
   - Online iterative version of DPO

6. **Reinforced Fine-tuning**:
   - Custom reward functions for fine-tuning

7. **Rejection Sampling**:
   - Simpler alignment approach

8. **Group Relative Policy Optimization (GRPO)**:
   - Variant of PPO with group-based normalization
   - Improves training stability

## Multimodal Support

OpenRLHF-M extends RLHF techniques to multimodal models through its `lmm_kits` module:

1. **Qwen2.5-VL Support**:
   - `data_processor.py` - Handles multimodal data processing for Qwen2.5-VL
   - `patch.py` - Contains patches to make the model compatible with the framework

2. **Base Multimodal Support**:
   - Common abstractions and utilities for multimodal models
   - Can be extended to support other multimodal models

3. **Integration with Vision-Language Models**:
   - Processes images and video inputs alongside text
   - Uses `qwen_vl_utils` for handling vision information
   - Supports chat templates and multimodal contexts

4. **Training Flow for Multimodal Models**:
   - Custom data processors handle the complexity of multimodal inputs
   - Multimodal batching and tensor preparation
   - Integration with standard RLHF training loops

## Key Requirements

### Software Dependencies:
- **Core ML Frameworks**:
  - PyTorch (2.5.1)
  - Transformers (specific GitHub commit)
  - DeepSpeed (0.16.4)
  - Ray (for distributed training)

- **Multimodal Support**:
  - qwen_vl_utils
  - torchvision

- **Optimization and Quantization**:
  - bitsandbytes 
  - peft
  - accelerate

- **Data Processing**:
  - datasets
  - jsonlines

- **Utilities and Metrics**:
  - tensorboard/wandb (for logging)
  - flask (for serving reward models)
  - math-verify (for verification)
  - pynvml (for GPU monitoring)

### Hardware Requirements:
- Recommended GPUs: NVIDIA A100/A800 (80GB) for large models
- Alternative: Multiple consumer GPUs (e.g., RTX 4090) for smaller models
- High-speed interconnects for multi-node setups
- Sufficient CPU and RAM for data processing and optimization

## System Architecture

The system architecture follows a distributed pattern, especially for training large models:

### Single-Node Training:
For smaller models (up to 7B parameters), training can be performed on a single node with multiple GPUs using DeepSpeed for parallelism:

```
┌─ Node ────────────────────────────────────┐
│                                           │
│ ┌─ DeepSpeed Process ──────────────────┐  │
│ │                                      │  │
│ │  Actor Model   Critic Model   RM     │  │
│ │                                      │  │
│ └──────────────────────────────────────┘  │
│                                           │
└───────────────────────────────────────────┘
```

### Multi-Node Ray-Based Training:
For larger models (34B-70B+ parameters), Ray enables distributing components across multiple nodes:

```
┌─ Node 1 ─────────────┐ ┌─ Node 2 ─────────────┐
│                      │ │                      │
│  ┌─ Ray Actor ─────┐ │ │  ┌─ Ray Actor ─────┐ │
│  │  Actor Model    │ │ │  │  Critic Model   │ │
│  └─────────────────┘ │ │  └─────────────────┘ │
│                      │ │                      │
└──────────────────────┘ └──────────────────────┘

┌─ Node 3 ─────────────┐ ┌─ Node 4 ─────────────┐
│                      │ │                      │
│  ┌─ Ray Actor ─────┐ │ │  ┌─ Ray Actor ─────┐ │
│  │  Reward Model   │ │ │  │  Reference Model │ │
│  └─────────────────┘ │ │  └─────────────────┘ │
│                      │ │                      │
└──────────────────────┘ └──────────────────────┘
```

### Hybrid Engine Configuration:
For optimized resource utilization, all models can be co-located on the same GPUs:

```
┌─ Node ────────────────────────────────────┐
│                                           │
│ ┌─ GPU 0 ─────────┐ ┌─ GPU 1 ─────────┐   │
│ │                 │ │                 │   │
│ │  Actor + Ref    │ │  Critic + RM    │   │
│ │                 │ │                 │   │
│ └─────────────────┘ └─────────────────┘   │
│                                           │
└───────────────────────────────────────────┘
```

### vLLM Acceleration:
For inference efficiency, vLLM can be integrated:

```
┌─ Node ─────────────────────────────────────────────┐
│                                                    │
│ ┌─ GPU 0,1 ─────────────┐ ┌─ GPU 2,3 ─────────────┐│
│ │                       │ │                       ││
│ │  vLLM Engine          │ │  RL Training          ││
│ │  (tensor parallel)    │ │  (ZeRO-3)             ││
│ │                       │ │                       ││
│ └───────────────────────┘ └───────────────────────┘│
│                                                    │
└────────────────────────────────────────────────────┘
```

## Training Pipeline

The RLHF training pipeline in OpenRLHF-M typically follows this sequence:

1. **Supervised Fine-Tuning (SFT)**:
   - Initial training on high-quality data
   - Produces a model that can follow instructions

2. **Reward Model Training**:
   - Training on human preference data
   - Learns to predict which responses are preferred

3. **RL-Based Optimization**:
   - Using the reward model to guide policy optimization
   - Applying algorithms like PPO or REINFORCE++
   - Balancing reward maximization with KL divergence constraints

For multimodal models, the same pipeline applies with additional processing for handling images and other modalities.

## Performance Optimizations

The framework includes several optimization techniques:

1. **Packing Samples**: Efficiently batches training data
2. **Flash Attention**: Faster attention computation
3. **Gradient Checkpointing**: Reduces memory usage
4. **vLLM Integration**: Accelerates inference during PPO
5. **ZeRO Optimization**: Enables training larger models
6. **Hybrid Engine**: Co-locates models for better GPU utilization
7. **Parameter-Efficient Fine-Tuning (PEFT)**: LoRA, QLoRA support
8. **Adam Offloading**: Moves optimizer states to CPU

## Conclusion

OpenRLHF-M represents a comprehensive framework for applying RLHF techniques to both language and multimodal models. Its modular design, support for various RL algorithms, and distributed training capabilities make it suitable for both research and production applications. The integration of multimodal support extends these capabilities to the next generation of AI models that combine language understanding with visual perception.
