#!/bin/bash

REPO_DIR="/home/theo/smart-posture-assistant"
VENV_PYTHON="/home/theo/posture-env/bin/python"
LOG_FILE="$REPO_DIR/update.log"
ENV_FILE="$REPO_DIR/.env"

if [ -f "$ENV_FILE" ]; then export $(grep -v '^#' "$ENV_FILE" | xargs); fi
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
