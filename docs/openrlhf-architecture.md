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

OpenRLHF-M extends RLHF techniques to multimodal models through its `lmm_kits` (Language-Multimodal Kits) module, providing an abstracted architecture for handling different types of vision-language models:

### Architecture for Multimodal Support

1. **Abstracted Base Classes**:
   - `BaseDataProcessor`: An abstract base class that defines the interface for processing multimodal data
   - `BasePatch`: Provides a foundation for model-specific patches that extend model functionality

2. **Model-Specific Implementations**:
   - Dedicated modules for each supported model (e.g., `qwen2_5_vl`) that implement the base interfaces
   - Each model module contains:
     - `data_processor.py`: Handles model-specific multimodal data processing
     - `patch.py`: Contains model-specific patches for embedding and position handling

3. **Key Multimodal Processing Features**:
   - **Unified Input Format**: Processes text along with images and videos using consistent abstractions
   - **Vision Information Extraction**: Uses `process_vision_info` to extract image and video inputs from messages
   - **Pixel Bound Management**: Handles image resolution limits with min/max pixel settings
   - **Chat Template Integration**: Seamlessly works with model-specific chat templates

4. **Qwen2.5-VL Integration**:
   The implementation for Qwen2.5-VL showcases the framework's multimodal capabilities:
   
   - **Custom Data Processor**: `Qwen2_5_VLDataProcessor` handles the specific requirements of Qwen2.5-VL
     - Processes both text and vision inputs (images and videos)
     - Manages batching and tensor preparation specifically for multimodal inputs
     - Handles the complex task of splitting batches with mixed vision and text content
   
   - **Model Patching**: `Qwen2_5_VLPatch` extends the model with functions needed for RLHF:
     - `get_inputs_embeds`: Manages embedding of text tokens and substitution with visual embeddings
     - `get_position_ids`: Handles position encoding for mixed text-image sequences
     - `offset_split_position_ids`: Manages position IDs across packed sequences with visual content

5. **Integration with Actor Model**:
   - The `Actor` class accepts `visual_inputs` in its forward method
   - Visual inputs are processed alongside text inputs to generate appropriate embeddings
   - Special handling for positional encodings in multimodal sequences

### Multimodal Data Format and Processing

1. **Data Format Requirements**:
   - Multimodal data must follow a specific messaging format that enables the model to distinguish between text and visual content
   - Messages are structured as JSON objects in a conversation format:
   ```json
   [
     {
       "role": "user",
       "content": [
         {"type": "text", "text": "What's in this image?"},
         {
           "type": "image",
           "image_url": {"url": "path/to/image.jpg"},
           "min_pixels": 4*28*28,  // Optional image resolution parameters
           "max_pixels": 640*28*28 // Default values are provided if not specified
         }
       ]
     },
     {
       "role": "assistant",
       "content": "The image shows a mountain landscape."
     }
   ]
   ```
   - This format supports both simple text messages and messages with mixed content (text, images, videos)
   - Images can be referenced via URLs or local file paths

2. **Data Preparation Pipeline**:
   - Messages are first formatted through the `_format_messages` method in `BaseDataProcessor`
   - The `_add_pixel_bounds` method ensures all visual content has appropriate resolution constraints
   - The `apply_chat_template` method applies model-specific chat templates before tokenization
   - The `get_images_from_messages` method extracts and prepares visual content for processing

3. **Tokenization Process for Multimodal Content**:
   - Special tokens like `<|vision_start|>` and `<|vision_end|>` mark positions for visual content in the token sequence
   - These markers act as placeholders for the visual embeddings that will replace them
   - The number of special tokens must match the number of visual features processed

4. **Batch Processing**:
   - The `make_input_batch` method combines individual inputs into a batched format for efficient processing
   - For text tokens, standard stacking is used (`torch.stack`)
   - For visual content, pixels are concatenated in the first dimension (`torch.cat`)
   - The `split_input_batch` method reverses this process, carefully tracking image locations

### Command Line Configuration for Multimodal Training

To enable multimodal training, several command-line arguments must be configured:

1. **Basic Multimodal Parameters**:
   - The model must be a supported multimodal model (currently Qwen2.5-VL)
   - Required dependencies must be installed (e.g., `qwen_vl_utils`, `torchvision`)

2. **Resolution Control**:
   - `--min_pixels`: Sets the minimum pixel count for images (default: 4*28*28)
   - `--max_pixels`: Sets the maximum pixel count for images (default: 640*28*28)

3. **Data Formatting**:
   - `--apply_chat_template`: Must be enabled to properly process multimodal messages
   - `--input_key`: Specifies the JSON field containing the message content (default: "input")

4. **Example Command for Multimodal SFT**:
   ```bash
   deepspeed --module openrlhf.cli.train_sft \
     --pretrain Qwen/Qwen2.5-VL \
     --dataset multimodal_dataset \
     --input_key messages \
     --apply_chat_template \
     --min_pixels 3136 \
     --max_pixels 503808 \
     --bf16 \
     --flash_attn
   ```

5. **For Multimodal PPO with Ray**:
   ```bash
   ray job submit --address="http://127.0.0.1:8265" \
     --runtime-env-json='{"working_dir": "/path/to/OpenRLHF-M"}' \
     -- python3 -m openrlhf.cli.train_ppo_ray \
     --pretrain Qwen/Qwen2.5-VL \
     --reward_pretrain reward_model_path \
     --apply_chat_template \
     --min_pixels 3136 \
     --max_pixels 503808 \
     --prompt_data vision_prompts_dataset \
     --input_key context_messages
   ```

### Integration with vLLM for Inference

For multimodal inference in vLLM-accelerated training:

1. **vLLM Input Format**:
   ```python
   vllm_inputs = [{
     "prompt": prompt_text,
     "multi_modal_data": {"image": image_list} if image_list else None,
     "mm_processor_kwargs": {
       "min_pixels": min_pixels_value,
       "max_pixels": max_pixels_value,
     },
   }]
   ```

2. **Experience Generation**:
   - The `RemoteExperienceMaker._generate_vllm` method processes multimodal inputs
   - Images are extracted from messages using `get_images_from_messages`
   - Chat templates are applied with `apply_chat_template`
   - The resulting prompts and images are formatted for vLLM inference

### Multimodal Data Flow

The data flow for multimodal models in OpenRLHF-M follows these steps:

1. **Data Preparation**:
   - Messages containing text and image/video references are processed
   - Images and videos are extracted and prepared for model input

2. **Tokenization and Embedding**:
   - Text is tokenized using model-specific tokenizers
   - Special tokens mark positions for visual content (e.g., `<|vision_start|>` and `<|vision_end|>`)
   - Visual content is processed through the model's vision encoder

3. **Mixed-Modal Processing**:
   - Visual embeddings replace corresponding special tokens in the text embedding sequence
   - Position IDs are adjusted to account for visual tokens
   - Attention masks are modified to handle the combined text-visual sequence

4. **Training with RLHF**:
   - The standard RLHF pipeline (SFT, reward model, RL optimization) is applied to this mixed-modal representation
   - Gradient updates affect both the language and vision components of the model

### Technical Requirements for Multimodal Support

- **Specific Dependencies**:
  - `qwen_vl_utils`: Provides utilities for handling Qwen VL models
  - `torchvision`: Necessary for image processing
  - Specific version of `transformers` (from a GitHub commit) that supports the multimodal models

- **Hardware Considerations**:
  - Higher GPU memory requirements due to vision processing components
  - Benefits from GPUs with larger VRAM for handling high-resolution images

### Supported Multimodal Models

Currently, OpenRLHF-M has a complete implementation for:
- **Qwen2.5-VL**: A powerful vision-language model that can process both images and videos

The abstracted architecture allows for easy extension to other multimodal models by:
1. Creating a new model-specific directory in `lmm_kits/`
2. Implementing the required `data_processor.py` and `patch.py` files
3. Adding any necessary model-specific utilities

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

OpenRLHF-M represents a comprehensive framework for applying RLHF techniques to both language and multimodal models. Its modular design, support for various RL algorithms, and distributed training capabilities make it suitable for both research and production applications. The integration of multimodal support extends these capabilities to the next generation of AI models that combine language understanding with visual perception, enabling RLHF training of vision-language models like Qwen2.5-VL with the same robust techniques used for text-only models.
