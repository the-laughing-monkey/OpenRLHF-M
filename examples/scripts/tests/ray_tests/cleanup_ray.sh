#!/bin/bash
#=============================================================================
# Ray and OpenRLHF Cleanup Utility
#=============================================================================
# This script forcefully stops all Ray and OpenRLHF processes, cleans up
# temporary directories, and ensures ports are free for the next run.
#=============================================================================

echo "Ray and OpenRLHF Cleanup Utility"
echo "================================="
echo "This will forcefully stop all Ray and OpenRLHF processes."
echo

# Check for required tools
if ! command -v lsof &> /dev/null; then
  echo "WARNING: 'lsof' command not available. Installing it now..."
  apt-get update && apt-get install -y lsof netcat-openbsd iputils-ping curl
  if [ $? -ne 0 ]; then
    echo "ERROR: Could not install required tools. Some cleanup operations may fail."
    echo "Try running: apt-get update && apt-get install -y lsof netcat-openbsd iputils-ping curl"
  else
    echo "Tools installed successfully."
  fi
fi

# Define critical ports to check and cleanup
CRITICAL_PORTS=(6379 8265 8266 5000)

# Kill all Ray and OpenRLHF processes
echo "Step 1: Stopping Ray processes..."
ray stop --force || true
sleep 2

echo "Step 2: Forcefully killing Ray and OpenRLHF processes..."
pkill -9 -f ray || true
pkill -9 -f openrlhf || true
pkill -9 -f gcs_server || true
pkill -9 -f raylet || true
pkill -9 -f plasma_store || true
sleep 1

# Kill any processes using the critical ports
echo "Step 3: Checking for processes using critical ports..."
if command -v lsof &> /dev/null; then
  for PORT in "${CRITICAL_PORTS[@]}"; do
    PIDS=$(lsof -t -i:${PORT} 2>/dev/null)
    if [ ! -z "$PIDS" ]; then
      echo "  Found processes using port ${PORT}: $PIDS"
      echo "  Killing these processes..."
      for PID in $PIDS; do
        kill -9 $PID 2>/dev/null || true
      done
    else
      echo "  No processes found using port ${PORT}"
    fi
  done
else
  echo "  WARNING: 'lsof' command not available. Cannot check ports."
  echo "  Try installing with: apt-get install -y lsof"
fi

# Explicitly kill any remaining process on port 5000
echo "Step 4: Killing remaining processes on port 5000..."
lsof -ti:5000 | xargs -r kill -9

# Clean up Ray temporary directories
echo "Step 5: Cleaning up Ray temporary directories..."
HOSTNAME=$(hostname)
RAY_TEMP_DIR="/data/cache-ray-${HOSTNAME}"
if [ -d "$RAY_TEMP_DIR" ]; then
  echo "  Removing Ray temp directory: $RAY_TEMP_DIR"
  rm -rf "$RAY_TEMP_DIR" || true
fi

echo "  Removing Ray session directories..."
rm -rf /tmp/ray/* 2>/dev/null || true

# Verify ports are now free
echo "Step 6: Verifying ports are now free..."
if command -v lsof &> /dev/null; then
  PORT_ISSUES=0
  for PORT in "${CRITICAL_PORTS[@]}"; do
    if lsof -i:${PORT} > /dev/null 2>&1; then
      echo "  WARNING: Port ${PORT} is still in use!"
      lsof -i:${PORT}
      PORT_ISSUES=$((PORT_ISSUES+1))
    else
      echo "  Port ${PORT} is free"
    fi
  done
  
  if [ $PORT_ISSUES -gt 0 ]; then
    echo "WARNING: Some ports are still in use. This may cause problems when starting Ray."
    echo "You may need to manually kill the processes using these ports."
  else
    echo "All required ports are free and available."
  fi
else
  echo "  WARNING: 'lsof' command not available. Cannot verify ports."
fi

echo
echo "Cleanup completed."
echo "To start a new training run, use:"
echo "  bash examples/scripts/tests/improved_multinode_training.sh" 