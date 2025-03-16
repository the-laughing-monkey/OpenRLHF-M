#!/bin/bash
#=============================================================================
# RunPod Global Networking Diagnostic Script
#=============================================================================
# This script verifies connectivity to the head node and checks for any network issues
# that might be preventing Ray nodes from communicating effectively.
#=============================================================================

# Get current hostname
NODE_HOSTNAME=$(hostname)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="./networking_test_${TIMESTAMP}.log"
echo "RunPod Global Networking Test - $(date)" | tee -a "${LOG_FILE}"
echo "Node: ${NODE_HOSTNAME}" | tee -a "${LOG_FILE}"
echo "===========================================" | tee -a "${LOG_FILE}"

# Check for required environment variables
if [ -z "${HEAD_POD_ID}" ]; then
  echo "ERROR: HEAD_POD_ID environment variable is not set." | tee -a "${LOG_FILE}"
  echo "Please set it to the head node's pod ID from RunPod dashboard." | tee -a "${LOG_FILE}"
  echo "Example: export HEAD_POD_ID=abc123" | tee -a "${LOG_FILE}"
  exit 1
fi

HEAD_NODE_IP="${HEAD_POD_ID}.runpod.internal"
echo "Head node DNS: ${HEAD_NODE_IP}" | tee -a "${LOG_FILE}"

# Initialize test status counters
TESTS_PASSED=0
TESTS_FAILED=0

# Check DNS resolution
echo -e "\n1. Testing DNS resolution..." | tee -a "${LOG_FILE}"
DNS_RESULT=$(getent hosts ${HEAD_NODE_IP} 2>&1)
echo "$DNS_RESULT" | tee -a "${LOG_FILE}"
if [ $? -ne 0 ] || [ -z "$DNS_RESULT" ]; then
  echo "ERROR: Cannot resolve ${HEAD_NODE_IP}" | tee -a "${LOG_FILE}"
  echo "This indicates a DNS resolution problem." | tee -a "${LOG_FILE}"
  TESTS_FAILED=$((TESTS_FAILED+1))
else
  echo "DNS resolution successful!" | tee -a "${LOG_FILE}"
  TESTS_PASSED=$((TESTS_PASSED+1))
fi

# Check basic connectivity with ping
echo -e "\n2. Testing basic connectivity with ping..." | tee -a "${LOG_FILE}"
if ! command -v ping &> /dev/null; then
  echo "WARNING: 'ping' command not available. Skipping ping test." | tee -a "${LOG_FILE}"
else
  PING_OUTPUT=$(ping -c 3 ${HEAD_NODE_IP} 2>&1)
  echo "$PING_OUTPUT" | tee -a "${LOG_FILE}"
  if [ $? -ne 0 ] || [[ "$PING_OUTPUT" == *"100% packet loss"* ]]; then
    echo "ERROR: Cannot ping ${HEAD_NODE_IP}" | tee -a "${LOG_FILE}"
    echo "This indicates a network routing or firewall problem." | tee -a "${LOG_FILE}"
    TESTS_FAILED=$((TESTS_FAILED+1))
  else
    echo "Ping successful!" | tee -a "${LOG_FILE}"
    TESTS_PASSED=$((TESTS_PASSED+1))
  fi
fi

# Check Ray port connectivity
echo -e "\n3. Testing Ray port connectivity (6379)..." | tee -a "${LOG_FILE}"
if ! command -v nc &> /dev/null; then
  echo "WARNING: 'nc' (netcat) command not available. Skipping port connectivity tests." | tee -a "${LOG_FILE}"
else
  NC_OUTPUT=$(nc -zv ${HEAD_NODE_IP} 6379 2>&1)
  echo "$NC_OUTPUT" | tee -a "${LOG_FILE}"
  if [[ "$NC_OUTPUT" == *"refused"* ]] || [[ "$NC_OUTPUT" == *"timed out"* ]]; then
    echo "ERROR: Cannot connect to Ray port 6379 on ${HEAD_NODE_IP}" | tee -a "${LOG_FILE}"
    echo "This indicates Ray is not running or a firewall is blocking the port." | tee -a "${LOG_FILE}"
    TESTS_FAILED=$((TESTS_FAILED+1))
  else
    echo "Ray port connectivity successful!" | tee -a "${LOG_FILE}"
    TESTS_PASSED=$((TESTS_PASSED+1))
  fi

  # Check Dashboard port connectivity
  echo -e "\n4. Testing Dashboard port connectivity (8265)..." | tee -a "${LOG_FILE}"
  NC_OUTPUT=$(nc -zv ${HEAD_NODE_IP} 8265 2>&1)
  echo "$NC_OUTPUT" | tee -a "${LOG_FILE}"
  if [[ "$NC_OUTPUT" == *"refused"* ]] || [[ "$NC_OUTPUT" == *"timed out"* ]]; then
    echo "ERROR: Cannot connect to Dashboard port 8265 on ${HEAD_NODE_IP}" | tee -a "${LOG_FILE}"
    echo "This indicates the Ray dashboard is not running or a firewall is blocking the port." | tee -a "${LOG_FILE}"
    TESTS_FAILED=$((TESTS_FAILED+1))
  else
    echo "Dashboard port connectivity successful!" | tee -a "${LOG_FILE}"
    TESTS_PASSED=$((TESTS_PASSED+1))
  fi

  # Check Reward Model port connectivity
  echo -e "\n5. Testing Reward Model port connectivity (5000)..." | tee -a "${LOG_FILE}"
  NC_OUTPUT=$(nc -zv ${HEAD_NODE_IP} 5000 2>&1)
  echo "$NC_OUTPUT" | tee -a "${LOG_FILE}"
  if [[ "$NC_OUTPUT" == *"refused"* ]] || [[ "$NC_OUTPUT" == *"timed out"* ]]; then
    echo "ERROR: Cannot connect to Reward Model port 5000 on ${HEAD_NODE_IP}" | tee -a "${LOG_FILE}"
    echo "This indicates the reward model server is not running or a firewall is blocking the port." | tee -a "${LOG_FILE}"
    TESTS_FAILED=$((TESTS_FAILED+1))
  else
    echo "Reward Model port connectivity successful!" | tee -a "${LOG_FILE}"
    TESTS_PASSED=$((TESTS_PASSED+1))
  fi
fi

# Check HTTP connectivity to Ray dashboard
echo -e "\n6. Testing HTTP connectivity to Ray dashboard..." | tee -a "${LOG_FILE}"
if ! command -v curl &> /dev/null; then
  echo "WARNING: 'curl' command not available. Skipping HTTP connectivity test." | tee -a "${LOG_FILE}"
else
  HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://${HEAD_NODE_IP}:8265/api/jobs/ 2>&1)
  echo "$HTTP_STATUS" | tee -a "${LOG_FILE}"
  if [[ "$HTTP_STATUS" != "200" ]]; then
    echo "ERROR: Cannot make HTTP connection to Ray dashboard" | tee -a "${LOG_FILE}"
    echo "This indicates the Ray dashboard is not properly configured." | tee -a "${LOG_FILE}"
    TESTS_FAILED=$((TESTS_FAILED+1))
  else
    echo "HTTP connectivity to Ray dashboard successful!" | tee -a "${LOG_FILE}"
    TESTS_PASSED=$((TESTS_PASSED+1))
  fi
fi

# Get IP address information
echo -e "\n7. Current node IP configuration:" | tee -a "${LOG_FILE}"
ip addr show | grep inet | tee -a "${LOG_FILE}"

# Print summary and recommendations
echo -e "\n===========================================" | tee -a "${LOG_FILE}"
echo "Network Connectivity Test Summary" | tee -a "${LOG_FILE}"
echo "Tests passed: $TESTS_PASSED" | tee -a "${LOG_FILE}"
echo "Tests failed: $TESTS_FAILED" | tee -a "${LOG_FILE}"
echo "Results saved to: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "===========================================" | tee -a "${LOG_FILE}"

# Add overall status
if [ $TESTS_FAILED -gt 0 ]; then
  echo "⚠️  FAILED: Some connectivity tests failed. Ray cluster might not work properly." | tee -a "${LOG_FILE}"
  echo "You need to start Ray on the head node before worker nodes can connect." | tee -a "${LOG_FILE}"
  echo "Run the cleanup script first, then start the head node with:" | tee -a "${LOG_FILE}"
  echo "  bash examples/scripts/tests/cleanup_ray.sh" | tee -a "${LOG_FILE}"
  echo "  bash examples/scripts/tests/improved_multinode_training.sh" | tee -a "${LOG_FILE}"
else
  echo "✅ PASSED: All connectivity tests passed. Nodes should be able to communicate." | tee -a "${LOG_FILE}"
fi

echo
echo "RECOMMENDATION: If you're having issues with Ray connectivity:"
echo "1. Make sure all nodes are in supported datacenters (CA-MTL-3, US-GA-1, US-GA-2, US-KS-2)"
echo "2. Verify Global Networking is enabled on all pods"
echo "3. Ensure that you've stopped Ray on all nodes before restarting"
echo "4. When restarting, start the head node first, then worker nodes"
echo "5. Make sure the HEAD_POD_ID environment variable is correctly set on all nodes"

echo "Test completed. Check the log file for detailed results." 