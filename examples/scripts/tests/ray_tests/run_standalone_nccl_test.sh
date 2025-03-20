#!/bin/bash
# Script to run the standalone NCCL test on Ray

# Define a simple Python script that executes our NCCL test
cat > run_nccl_test.py << 'EOF'
import ray
import os
import subprocess
import sys
import socket

@ray.remote(num_gpus=1)
def run_nccl_test():
    hostname = socket.gethostname()
    print(f"Running NCCL test on node: {hostname}")
    
    # Set NCCL environment variables
    nccl_socket_ifname = os.environ.get("NCCL_SOCKET_IFNAME", "lo,eth0,podnet1")
    os.environ["NCCL_SOCKET_IFNAME"] = nccl_socket_ifname
    
    # Run the standalone NCCL test
    cmd = "./standalone_nccl_test"
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)
        print(f"Output from {hostname}:")
        print(result.stdout)
        if result.stderr:
            print(f"Errors from {hostname}:")
            print(result.stderr)
        return {"status": "success", "hostname": hostname}
    except subprocess.CalledProcessError as e:
        print(f"Error running NCCL test on {hostname}:")
        print(f"Exit code: {e.returncode}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return {"status": "failed", "hostname": hostname, "error": str(e)}

if __name__ == "__main__":
    # Initialize Ray (will connect to the existing cluster)
    if not ray.is_initialized():
        ray.init(address="auto")
    
    # Get number of GPUs in the cluster
    resources = ray.cluster_resources()
    num_gpus = int(resources.get("GPU", 0))
    
    if num_gpus == 0:
        print("No GPUs found in the Ray cluster!")
        sys.exit(1)
    
    print(f"Found {num_gpus} GPUs in the Ray cluster")
    
    # Run the test on each GPU
    results_refs = [run_nccl_test.remote() for _ in range(min(num_gpus, 8))]  # Limit to 8 workers max
    
    # Get results
    results = ray.get(results_refs)
    
    # Print summary
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = len(results) - success_count
    
    print("\n=== NCCL Test Summary ===")
    print(f"Total tests: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    
    # Exit with error if any test failed
    if failed_count > 0:
        sys.exit(1)
EOF

# Check for Ray address
if [ -z "$RAY_ADDRESS" ]; then
    if [ -z "$HEAD_POD_ID" ]; then
        echo "Error: Either RAY_ADDRESS or HEAD_POD_ID must be set."
        exit 1
    fi
    RAY_ADDRESS="http://${HEAD_POD_ID}.runpod.internal:8265"
fi

# Ensure the standalone NCCL test is compiled
if [ ! -f "./standalone_nccl_test" ]; then
    echo "Standalone NCCL test binary not found. Compiling..."
    bash examples/scripts/tests/ray_tests/compile_standalone_nccl_test.sh
    if [ $? -ne 0 ]; then
        echo "Failed to compile standalone NCCL test."
        exit 1
    fi
fi

# Set NCCL_SOCKET_IFNAME from environment or use default
if [ -z "$NCCL_SOCKET_IFNAME" ]; then
    export NCCL_SOCKET_IFNAME="lo,eth0,podnet1"
fi
echo "Using NCCL network interfaces: $NCCL_SOCKET_IFNAME"

# Submit the job to Ray
echo "Submitting standalone NCCL test job to Ray..."
ray job submit --address="$RAY_ADDRESS" --no-wait -- python run_nccl_test.py

echo "Job submitted to Ray. Check Ray dashboard for results." 