#!/bin/bash
set -euo pipefail

./download_models.sh

echo "Starting Celery worker..."
redis-server --protected-mode no &
uv run celery -A celery_worker.app worker --loglevel=info -Q wan_animate,wan_replace -P solo
