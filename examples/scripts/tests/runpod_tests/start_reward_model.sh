#!/bin/bash
set -e

# Default values
DATASET_PATH="/data/datasets/VerMulti/mathv60k_message.jsonl"
INPUT_KEY="message"
PROMPT_TEMPLATE="chatml"
LOG_DIR="./logs"
LOG_FILE="${LOG_DIR}/remote_rm.log"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)
      DATASET_PATH="$2"
      shift 2
      ;;
    --input_key)
      INPUT_KEY="$2"
      shift 2
      ;;
    --prompt-template)
      PROMPT_TEMPLATE="$2"
      shift 2
      ;;
    --log_dir)
      LOG_DIR="$2"
      LOG_FILE="${LOG_DIR}/remote_rm.log"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--dataset path] [--input_key key] [--prompt-template template] [--log_dir dir]"
      exit 1
      ;;
  esac
done

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Get the IP address of eth1 interface
ETH1_IP=$(ip addr show eth1 2>/dev/null | grep -oP 'inet \K[\d.]+' || echo "")
if [ -z "${ETH1_IP}" ]; then
  echo "Warning: Could not find eth1 interface. Using localhost."
  ETH1_IP="127.0.0.1"
else
  echo "Using eth1 IP address: ${ETH1_IP}"
fi

# Check if dataset exists
if [ ! -f "${DATASET_PATH}" ]; then
  echo "Error: Dataset file '${DATASET_PATH}' does not exist."
  echo "Please provide a valid dataset path with --dataset"
  exit 1
fi

echo "Starting reward model server..."
echo "Dataset: ${DATASET_PATH}"
echo "Input key: ${INPUT_KEY}"
echo "Prompt template: ${PROMPT_TEMPLATE}"
echo "Listening on: ${ETH1_IP}:5000"
echo "Logs at: ${LOG_FILE}"

# Start the reward model server
python3 -m openrlhf.models.remote_rm.math_verifier \
    --dataset "${DATASET_PATH}" \
    --input_key "${INPUT_KEY}" \
    --host "${ETH1_IP}" \
    --prompt-template "${PROMPT_TEMPLATE}" \
    --log_file "${LOG_FILE}"

echo "Reward model server stopped."
