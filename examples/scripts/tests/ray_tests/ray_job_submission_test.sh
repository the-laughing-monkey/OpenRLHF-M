#!/bin/bash
#=============================================================================
# Ray Job Submission Diagnostic Script
#=============================================================================
# This script runs a simple test to diagnose Ray job submission problems:
# 1. Starts a Ray head node
# 2. Tests job submission with multiple connection methods
# 3. Logs detailed information to help identify the issue
#=============================================================================

# Add HEAD_POD_ID check and set HEAD_NODE_IP at the top
if [ -z "${HEAD_POD_ID}" ]; then
  echo "ERROR: HEAD_POD_ID environment variable is not set."
  exit 1
fi
export HEAD_NODE_IP="${HEAD_POD_ID}.runpod.internal"
export RAY_PORT="6379"
export DASHBOARD_PORT="8265"
echo "Using head node internal DNS: ${HEAD_NODE_IP}"

# Get hostname and set up variables
SCRIPT_PID=$$
LOG_DIR="./ray_test_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/${TIMESTAMP}_ray_test.log"
RAY_TEMP_DIR="/data/cache-ray-test-${TIMESTAMP}"

# Create log directory
mkdir -p "${LOG_DIR}"

# Function to log with timestamp
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "${LOG_FILE}"
}

# Log system information
log "=== System Information ==="
log "Python version: $(python3 --version 2>&1)"
log "Python path: $(which python3 2>&1)"
log "Ray version: $(python3 -c 'import ray; print(ray.__version__)' 2>&1)"
log "OS Release: $(cat /etc/os-release | grep PRETTY_NAME 2>&1)"
log "Ray Path: $(python3 -c 'import ray; print(ray.__path__)' 2>&1)"
log "Ray Script Path: $(which ray 2>&1)"
log "Ray temp directory: ${RAY_TEMP_DIR}"

# Stop any existing Ray instances
log "Stopping any existing Ray processes..."
ray stop

# Create Ray temp directory
mkdir -p ${RAY_TEMP_DIR}
log "Created temporary directory: ${RAY_TEMP_DIR}"

# Start Ray head node
log "Starting Ray head node..."
ray start --head --node-ip-address="${HEAD_NODE_IP}" --port="${RAY_PORT}" --dashboard-port="${DASHBOARD_PORT}" --temp-dir ${RAY_TEMP_DIR} --dashboard-host="0.0.0.0" &> >(tee -a "${LOG_FILE}")
sleep 5

# Check if Ray is running
log "Checking Ray status..."
ray status &> >(tee -a "${LOG_FILE}")

# Check job server by making a direct HTTP request
log "Testing job server with curl..."
curl -v http://127.0.0.1:8265/api/jobs/ &> >(tee -a "${LOG_FILE}")
log "Testing job server with curl using hostname..."
curl -v http://${NODE_HOSTNAME}:8265/api/jobs/ &> >(tee -a "${LOG_FILE}")

# Create a simple test Python script
TEST_SCRIPT="${LOG_DIR}/test_job.py"
cat > "${TEST_SCRIPT}" << 'EOF'
import ray
import socket
import os
import sys

def get_node_info():
    info = {
        "hostname": socket.gethostname(),
        "ip": socket.gethostbyname(socket.gethostname()),
        "python_version": sys.version,
        "ray_version": ray.__version__,
        "env_vars": {k: v for k, v in os.environ.items() if k.startswith("RAY_") or k.startswith("PYTHON")}
    }
    return info

@ray.remote
def hello_world():
    node_info = get_node_info()
    return f"Hello World from Ray! Node info: {node_info}"

# Initialize Ray (will connect to the existing cluster)
if not ray.is_initialized():
    ray.init()

# Run the remote function
result = ray.get(hello_world.remote())
print(result)
EOF

log "Created test script at ${TEST_SCRIPT}"

# Try submitting the job using different methods
log "=== Testing job submission with HEAD_NODE_IP ==="
ray job submit --address="http://${HEAD_NODE_IP}:${DASHBOARD_PORT}" --no-wait -- python3 "${TEST_SCRIPT}" &> >(tee -a "${LOG_FILE}")

log "=== Testing job submission with HEAD_NODE_IP (alternative) ==="
ray job submit --address="http://${HEAD_NODE_IP}:${DASHBOARD_PORT}" --no-wait -- python3 "${TEST_SCRIPT}" &> >(tee -a "${LOG_FILE}")

# Test with a runtime environment and working directory
log "=== Testing job submission with runtime environment ==="
TEST_DIR="${LOG_DIR}/test_working_dir"
mkdir -p "${TEST_DIR}"
cp "${TEST_SCRIPT}" "${TEST_DIR}/"

log "=== Testing with smaller working directory ==="
ray job submit --address="http://${HEAD_NODE_IP}:${DASHBOARD_PORT}" \
    --runtime-env-json="{\"working_dir\": \"${TEST_DIR}\"}" \
    --no-wait -- python3 test_job.py &> >(tee -a "${LOG_FILE}")

# Check if Ray dashboard is accessible
log "=== Testing Ray dashboard access ==="
if command -v curl &> /dev/null; then
    log "Checking dashboard with curl..."
    curl -I http://${HEAD_NODE_IP}:${DASHBOARD_PORT} &>> "${LOG_FILE}"
else
    log "curl not found, skipping dashboard test"
fi

# Create a test job with fewer dependencies
log "=== Testing with a simpler job ==="
SIMPLE_SCRIPT="${LOG_DIR}/simple_job.py"
cat > "${SIMPLE_SCRIPT}" << 'EOF'
import sys
import platform
import socket

print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Hostname: {socket.gethostname()}")
print("Hello from a simple job!")
EOF

log "Created simple script at ${SIMPLE_SCRIPT}"
ray job submit --address="http://127.0.0.1:8265" --no-wait -- python3 "${SIMPLE_SCRIPT}" &> >(tee -a "${LOG_FILE}")

# Check if the Ray job API is accessible via Python
log "=== Testing Ray job API via Python ==="
PYTHON_API_TEST="${LOG_DIR}/python_api_test.py"
cat > "${PYTHON_API_TEST}" << 'EOF'
import sys
import requests
import socket

def test_job_api(host="127.0.0.1", port=8265):
    print(f"Testing Ray Job API at http://{host}:{port}/api/jobs/")
    try:
        response = requests.get(f"http://{host}:{port}/api/jobs/")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text[:500]}...")  # Print first 500 chars
        return response.status_code == 200
    except Exception as e:
        print(f"Error connecting to Ray Job API: {str(e)}")
        return False

print(f"Hostname: {socket.gethostname()}")
test_job_api("127.0.0.1")
test_job_api(socket.gethostname())
EOF

log "Created Python API test at ${PYTHON_API_TEST}"
python3 "${PYTHON_API_TEST}" &> >(tee -a "${LOG_FILE}")

# Check network connections
log "=== Checking network connections ==="
log "Checking ports 6379 and 8265..."
if command -v netstat &> /dev/null; then
    netstat -tuln | grep -E '6379|8265' &>> "${LOG_FILE}"
elif command -v ss &> /dev/null; then
    ss -tuln | grep -E '6379|8265' &>> "${LOG_FILE}"
else
    log "Neither netstat nor ss found, skipping port check"
fi

# Show summary
log "=== Test Complete ==="
log "Log file: ${LOG_FILE}"
log "To stop Ray: ray stop"
log "To check Ray dashboard: http://127.0.0.1:8265"

echo "Test complete! See log file: ${LOG_FILE}"
echo "To stop Ray: ray stop" 