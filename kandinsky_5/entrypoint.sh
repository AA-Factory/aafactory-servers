#!/bin/bash
set -e

uv run python kandinsky/download_models.py &
./install_sage_attention.sh &
# Start redis-server in the background
redis-server --protected-mode no &

uv run celery -A celery_worker.app worker --loglevel=info -Q kandinsky -P solo
