#!/bin/bash
set -euo pipefail

./download_models.sh
./install_sage_attention.sh

echo "Starting Celery worker..."
redis-server --protected-mode no &
uv run celery -A celery_worker.app worker --loglevel=info -Q animate
