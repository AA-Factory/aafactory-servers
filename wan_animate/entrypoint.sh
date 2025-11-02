#!/bin/bash
set -euo pipefail

echo "Starting Celery worker..."
redis-server --protected-mode no &
uv run celery -A celery_worker.app worker --loglevel=info -Q animate
