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

## Setting Up Your Instances

### On the Head Node

1. **Environment Variables:**
   - Set your pod's ID to `HEAD_POD_ID`. For example:
     ```bash
     export HEAD_POD_ID=abc123
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

### Start the Training Job

2. **Run the Script:**
   ```bash
   cd /data/OpenRLHF-M
   bash examples/scripts/tests/train_grpo_ray_qwen2_5_vl_mathv60k_multinode.sh
   ```

   The script will:
   - Stop any existing Ray instances.
   - Start the Ray head node on `0.0.0.0` and on the defined ports (default: 6379 for Ray, 8265 for the dashboard).
   - Launch the remote reward model server and wait until it responds.
   - Submit the training job to the Ray cluster.





## Accessing the Cluster and Logs

- The Ray dashboard will be accessible internally at: `http://<head_pod_id>.runpod.internal:8265`.
- Training logs are stored in the script's checkpoint directory (e.g., in `./checkpoints/qwen2.5-vl-3b-ins-mathvista-grpo/logs/`).
- Friendly output messages in the script assist in troubleshooting and monitoring the setup progress.

## Troubleshooting

* **DNS Resolution:** Ensure that `HEAD_POD_ID` is correctly set on all nodes.
* **Connection Issues:** Verify that firewall rules or network settings on your RunPod instances allow connections on ports 6379 and 8265.
* **Reward Model Startup:** If the reward model does not respond after the specified number of retries, check the logs in the remote reward model log file for further diagnosis.

For additional details on RunPod's Global Networking, please refer to the [RunPod Networking Documentation](https://docs.runpod.io/pods/networking). For more on Ray clusters, see [Ray's getting started guide](https://docs.ray.io/en/latest/cluster/getting-started.html).
