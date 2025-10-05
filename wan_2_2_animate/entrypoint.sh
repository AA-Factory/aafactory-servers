#!/bin/bash
set -e

# Define the model repository and local directory
MODEL_REPO="Wan-AI/Wan2.2-Animate-14B"
MODEL_DIR="/.weights/Wan2.2-Animate-14B"

# Download the model weights using uv + huggingface-cli
# This command will only download if the model directory doesn't exist
echo "Downloading model weights to $MODEL_DIR..."
uv run hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"

# Wait for all background downloads to finish
wait

# Check if download was successful
if [ $? -ne 0 ]; then
    echo "Failed to download model weights. Exiting."
    exit 1
fi

echo "Model weights downloaded. Starting Celery worker..."

# Start redis-server in the background
echo "Starting Celery worker listening on queues: animate, replace"
redis-server --protected-mode no &
uv run celery -A celery_worker.app worker --loglevel=info -Q animate,replace
