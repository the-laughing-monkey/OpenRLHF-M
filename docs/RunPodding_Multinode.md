# RunPod Multi-Node Training Setup for OpenRLHF-M

This document provides a guide for setting up multinode training using OpenRLHF-M on RunPod instances using RunPod's Global Networking feature.

## Understanding RunPod Global Networking

RunPod's [Global Networking](https://docs.runpod.io/pods/networking) is a private networking feature that enables secure communication between all Pods within your RunPod account. It creates a private, virtual network that connects your Pods, allowing them to communicate with each other as if they were on the same local network, regardless of their physical location.

Key features of Global Networking:

1. Each pod is assigned a private DNS name in the format: `$podid.runpod.internal`
2. Pods can communicate securely without needing to expose public ports
3. Services running on one pod are accessible to other pods in your account
4. Network communication happens directly through RunPod's private network

**Current Limitations**:
- Available only on NVIDIA GPU Pods (not available for CPU Pods)
- Only available in select datacenters:
  - CA-MTL-3
  - US-GA-1
  - US-GA-2
  - US-KS-2

## Setting Up RunPod Instances

### 1. Create Pods with Global Networking Enabled

1. Go to the RunPod dashboard and select **+ Deploy**
2. Select a template with the necessary software (CUDA, Python, etc.)
3. Under "Networking" section, toggle **Global Networking** to ON
4. Ensure you're deploying in a supported datacenter (CA-MTL-3, US-GA-1, US-GA-2, US-KS-2)
5. Deploy at least two pods with identical configurations

After the pods are created, note the pod IDs for each instance. These will be displayed on the pod cards and in the pod details. The pod ID is part of the URL and also visible in the pod information panel.

### 2. Configure the Head Node

On the first instance (designated as the head node):

1. Find your pod ID from the RunPod dashboard (e.g., "abc123")
2. Set up the necessary environment variables:

```bash
# Enable Global Networking mode
export USE_GLOBAL_NETWORKING=1

# Set your pod ID for this head node
export HEAD_POD_ID="your-pod-id"  # Example: abc123

# Indicate this is the head node (not a worker)
export RAY_WORKER=0
```

3. Run the training script on the head node:

```bash
cd /data/OpenRLHF-M
bash examples/scripts/tests/train_grpo_ray_qwen2_5_vl_mathv60k_multinode.sh
```

The script will initialize the Ray head node and wait for worker nodes to join the cluster.

### 3. Configure Worker Nodes

On each additional instance (worker nodes):

1. Set up environment variables pointing to the head node:

```bash
# Enable Global Networking mode
export USE_GLOBAL_NETWORKING=1

# Set the head node's pod ID (from the previous step)
export HEAD_POD_ID="head-node-pod-id"  # Example: abc123

# Indicate this is a worker node
export e=1
```

2. Run the same training script on each worker node:

```bash
cd /data/OpenRLHF-M
bash examples/scripts/tests/train_grpo_ray_qwen2_5_vl_mathv60k_multinode.sh
```

The worker node will join the Ray cluster and participate in the distributed training.

## Monitoring Your Cluster

Once your cluster is running:

1. The Ray dashboard will be available internally at: `http://<head_pod_id>.runpod.internal:8265`
2. To access it externally, you may need to set up port forwarding or use SSH tunneling
3. Training logs will be saved in the `/workspace/OpenRLHF-M/checkpoints/qwen2.5-vl-3b-ins-mathvista-grpo/logs/` directory
4. Training progress can be monitored via WandB (if configured) or TensorBoard

## Verifying Connectivity

To verify connectivity between nodes before starting training:

```bash
# From any node, test connection to other nodes
ping <pod_id>.runpod.internal  # Replace with the actual pod ID
```

You can also check that the standard Ray ports are accessible between nodes:

```bash
# Test Ray head port connectivity
nc -zv <head_pod_id>.runpod.internal 6379

# Test Ray dashboard connectivity
nc -zv <head_pod_id>.runpod.internal 8265
```

## Troubleshooting

### Common Issues

1. **DNS Resolution Failures**:
   - Ensure the pod ID is correct in your environment variables
   - Verify that Global Networking is enabled on all pods
   - Check that all pods are in supported datacenters

2. **Connection Refused**:
   - Ensure the Ray process is running on the head node
   - Verify that internal ports (6379, 8265, etc.) are being used correctly
   - Check Ray logs with `tail -f /tmp/ray/session_latest/logs/raylet.out`

3. **Shared Filesystem Conflicts**:
   - The training script creates node-specific directories to avoid conflicts
   - Only the head node saves final checkpoints by default

4. **Node Not Joining Cluster**:
   - Verify the head node pod ID is correct 
   - Ensure Ray is properly started on the head node first
   - Run `ray status` to check if the Ray cluster is running

## Cleaning Up

When you're done training:
1. Stop the Ray processes on all nodes: `ray stop`
2. Terminate your RunPod instances

## Further Help

If you encounter issues with your multinode setup, please check:
1. Ray documentation on [multinode clusters](https://docs.ray.io/en/latest/cluster/getting-started.html)
2. RunPod documentation on [Global Networking](https://docs.runpod.io/pods/networking)
