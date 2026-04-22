#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$SCRIPT_DIR}"
VENV_PYTHON="${VENV_PYTHON:-$REPO_DIR/.venv/bin/python}"
LOG_FILE="${LOG_FILE:-$REPO_DIR/update.log}"
ENV_FILE="${ENV_FILE:-$REPO_DIR/.env}"

if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi
DEVICE_ID=${DEVICE_ID:-"pi-posture-001"}

update_db() {
    curl -X PATCH "${SUPABASE_URL}/rest/v1/device_configs?device_id=eq.${DEVICE_ID}" \
      -H "apikey: ${SUPABASE_KEY}" \
      -H "Authorization: Bearer ${SUPABASE_KEY}" \
      -H "Content-Type: application/json" \
      -d "{\"current_version\":\"$1\"}"
}

cd "$REPO_DIR" || exit

git fetch origin main
BEFORE=$(git rev-parse --short HEAD)
LATEST=$(git rev-parse --short origin/main)

if [ "$BEFORE" = "$LATEST" ]; then 
    echo "NO_UPDATE"
    exit 0
fi

git reset --hard origin/main
AFTER=$(git rev-parse --short HEAD)

if git diff --name-only "$BEFORE" "$AFTER" | grep -q "requirements.txt"; then
    $VENV_PYTHON -m pip install -r requirements.txt >> "$LOG_FILE" 2>&1
fi

update_db "$AFTER" >> "$LOG_FILE" 2>&1

sudo systemctl restart posture-bot

echo "SUCCESS:$AFTER"
