# RunPod Multi-Node Training Setup for OpenRLHF-M

This document explains how to set up and run multinode training for OpenRLHF-M on RunPod instances with Global Networking, using the simplified multinode training script.

## Overview

The updated multinode training script for OpenRLHF-M has been streamlined with the following features:

1. **Node Role Detection:**
   - The script distinguishes between head and worker nodes using the environment variable `RAY_WORKER` (head node if unset or set to `0`, worker node if set to `1`).

2. **Checkpoint Management:**
   - A single checkpoint directory is used for all nodes (no node-specific modifications), ensuring simplicity and avoiding filesystem conflicts.

3. **WandB Integration:**
   - The script checks for the `WANDB_API_KEY` and enables WandB logging if the key is found. If not, it logs an informational message and continues without halting.

4. **Ray Debugging Option:**
   - Optional Ray debugging can be enabled by setting `DEBUG_RAY=1`. This adds the necessary flags to start Ray with its legacy debugger support.

5. **Head Node Responsibility:**
   - The head node starts the Ray cluster (bound to `0.0.0.0`) and launches a remote reward model server. It pauses for a few seconds, then verifies that the reward model server is responding (using repeated curl checks) before submitting the training job via `ray job submit`.

6. **Worker Node Behavior:**
   - Worker nodes require the `HEAD_POD_ID` variable to be set. They use this to derive the head node's DNS (formatted as `<HEAD_POD_ID>.runpod.internal`) and join the Ray cluster.

7. **Friendly Output:**
   - The script includes detailed, friendly output messages that report its progress at each step, including when it stops existing Ray instances, waits for head accessibility, and submits jobs.

## Required Environment Variables

Here's a clear breakdown of the environment variables needed for running the multinode training script:

### For Head Node

```bash
# REQUIRED: Your RunPod ID (e.g., if your pod URL contains "abc123")
export HEAD_POD_ID=abc123

# OPTIONAL: Explicitly set this to 0 to indicate a head node (default behavior if unset)
export RAY_WORKER=0

# OPTIONAL: Number of worker nodes the head node should wait for (defaults to 1 if not set)
export EXPECTED_WORKERS=2

# OPTIONAL: Enable WandB logging by providing your API key
export WANDB_API_KEY=your_wandb_api_key_here

# OPTIONAL: Enable Ray debugging mode
export DEBUG_RAY=1
```

### For Worker Nodes

```bash
# REQUIRED: Set to 1 to indicate this is a worker node
export RAY_WORKER=1

# REQUIRED: The RunPod ID of the head node (must match the HEAD_POD_ID set on the head node)
export HEAD_POD_ID=abc123

# OPTIONAL: Enable Ray debugging mode (should match head node setting)
export DEBUG_RAY=1
```

Set these on both the head and worker nodes:

```bash
export NCCL_NET_SOCKET_FAMILY=IPv4
export NCCL_LAUNCH_MODE=GROUP
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=podnet1
export NCCL_SOCKET_NTHREADS=1
export NCCL_NSOCKS_PERTHREAD=1
export FORCE_MODEL_INIT_TO_CUDA=1
```


## Setting Up Your Instances

### On the Head Node

1. **Environment Variables:**
   - Set your pod's ID to `HEAD_POD_ID`. For example:
     ```bash
     export HEAD_POD_ID=abc123
     ```
   - Set the expected number of worker nodes (if more than 1):
     ```bash
     export EXPECTED_WORKERS=2  # If you're setting up a 3-node cluster (1 head + 2 workers)
     ```
   - Ensure the head node is designated (either by not setting `RAY_WORKER` or setting it to `0`).
   - Optionally, enable Ray debugging:
     ```bash
     export DEBUG_RAY=1
     ```

### On Worker Nodes

1. **Environment Variables:**
   - Set the head node's pod ID via `HEAD_POD_ID` (e.g., `export HEAD_POD_ID=abc123`).
   - Designate the node as a worker by setting:
     ```bash
     export RAY_WORKER=1
     ```

2. **Run the Script:**
   ```bash
   cd /data/OpenRLHF-M
   bash examples/scripts/tests/train_grpo_ray_qwen2_5_vl_mathv60k_multinode.sh
   ```

   Worker nodes will:
   - Stop any existing Ray instances.
   - Wait until the head node (derived from `HEAD_POD_ID`) is reachable on the required port.
   - Join the Ray cluster.
   - Return control to your terminal while the Ray worker process runs in the background.

## Preparing the Dataset

Before running the multinode training job, you need to prepare the dataset. This step should be done on the head node.

As an example, we will download and prepare the MathV60K dataset using the following script in our OpenRLHF-M repository: examples/scripts/data_downloaders/download_mathv60k.py

### Download and Prepare the MathV60K Dataset

1. Create the datasets directory:
```bash
mkdir -p /data/datasets
```

2. Download and prepare the MathV60K dataset:
```bash
cd /data/OpenRLHF-M
python examples/scripts/data_downloaders/download_mathv60k.py --root_dir /data/datasets/VerMulti
```

This script will:
- Download the dataset files from Hugging Face
- Extract images to the specified directory
- Process the JSONL file to update paths
- Make the dataset ready for training

The script provides detailed progress information and will tell you when the dataset is ready.

### Start the Training Job

1. **Run the Script on Head Node:**
   ```bash
   cd /data/OpenRLHF-M
   bash examples/scripts/tests/train_grpo_ray_qwen2_5_vl_mathv60k_multinode.sh
   ```

   The script will:
   - Stop any existing Ray instances.
   - Start the Ray head node on `0.0.0.0` and on the defined ports (default: 6379 for Ray, 8265 for the dashboard).
   - Launch the remote reward model server and wait until it responds.
   - Wait for the specified number of worker nodes to join (default: 1).
   - Submit the training job to the Ray cluster.

## Accessing the Cluster and Logs

- The Ray dashboard will be accessible internally at: `http://<head_pod_id>.runpod.internal:8265`.
- Training logs are stored in the script's checkpoint directory (e.g., in `./checkpoints/qwen2.5-vl-3b-ins-mathvista-grpo/logs/`).
- Worker node logs are stored in individual log files within the same log directory.
- Friendly output messages in the script assist in troubleshooting and monitoring the setup progress.

## Troubleshooting

* **DNS Resolution:** Ensure that `HEAD_POD_ID` is correctly set on all nodes.
* **Connection Issues:** Verify that firewall rules or network settings on your RunPod instances allow connections on ports 6379 and 8265.
* **Reward Model Startup:** If the reward model does not respond after the specified number of retries, check the logs in the remote reward model log file for further diagnosis.
* **NCCL Communication Errors:** If you encounter NCCL errors like "socketStartConnect: Connect failed" or "Software caused connection abort":
  1. Verify all nodes have the same NCCL environment variables set
  2. Try adjusting the `NCCL_SOCKET_IFNAME` value to match your actual network interface (run `ifconfig` to see available interfaces)
  3. Increase verbosity with `export NCCL_DEBUG=TRACE` to get more detailed logs
  4. Check network connectivity between nodes with standard tools:
     ```bash
     ping othernode.runpod.internal
     traceroute othernode.runpod.internal
     ```
  5. For persistent issues, test if your nodes can communicate on different ports using a simple test like netcat:
     ```bash
     # On one node
     nc -l 12345
     # On another node
     echo "test" | nc firstnode.runpod.internal 12345
     ```

For additional details on RunPod's Global Networking, please refer to the [RunPod Networking Documentation](https://docs.runpod.io/pods/networking). For more on Ray clusters, see [Ray's getting started guide](https://docs.ray.io/en/latest/cluster/getting-started.html).
