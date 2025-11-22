#!/bin/bash
set -euo pipefail

./download_models.sh
uv pip install sageattention==2.2.0

echo "Starting Celery worker..."
redis-server --protected-mode no &
uv run celery -A celery_worker.app worker --loglevel=info -Q wan_animate -P solo
