#!/bin/bash

# ============================================
# OTA Update Script
# ============================================

REPO_DIR="/home/theo/smart-posture-assistant"
VENV_PYTHON="/home/theo/posture-env/bin/python"
MAIN_SCRIPT="$REPO_DIR/src/main.py"
LOG_FILE="$REPO_DIR/update.log"
PID_FILE="$REPO_DIR/bot.pid"

SUPABASE_URL="${SUPABASE_URL:-your_supabase_url}"
SUPABASE_KEY="${SUPABASE_KEY:-your_supabase_key}"
DEVICE_ID="pi-posture-001"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

update_version_in_db() {
    local version=$1
    log "Updating current_version to: $version"
    
    curl -X PATCH \
      "${SUPABASE_URL}/rest/v1/device_configs?device_id=eq.${DEVICE_ID}" \
      -H "apikey: ${SUPABASE_KEY}" \
      -H "Authorization: Bearer ${SUPABASE_KEY}" \
      -H "Content-Type: application/json" \
      -d "{\"current_version\":\"${version}\"}" \
      2>&1 | tee -a "$LOG_FILE"
}

cd "$REPO_DIR"

log "====== OTA Update Started ======"

BEFORE=$(git rev-parse --short HEAD)
log "Current commit: $BEFORE"

log "Fetching from GitHub..."
git fetch origin main 2>&1 | tee -a "$LOG_FILE"

LATEST=$(git rev-parse --short origin/main)

if [ "$BEFORE" = "$LATEST" ]; then
    log "Already up to date"
    echo "NO_UPDATE"
    exit 0
fi

log "New version found: $LATEST"

BACKUP_DIR="$REPO_DIR/backups/$(date '+%Y%m%d_%H%M%S')"
mkdir -p "$BACKUP_DIR"
cp -r src "$BACKUP_DIR/" 2>/dev/null
log "Backup created: $BACKUP_DIR"

log "Stopping bot..."
if [ -f "$PID_FILE" ]; then
    BOT_PID=$(cat "$PID_FILE")
    if ps -p "$BOT_PID" > /dev/null 2>&1; then
        sudo kill "$BOT_PID"
        log "Stopped bot (PID: $BOT_PID)"
    fi
fi

pkill -f "main.py"
sleep 2

log "Pulling new code..."
git reset --hard origin/main 2>&1 | tee -a "$LOG_FILE"

AFTER=$(git rev-parse --short HEAD)
log "Updated from $BEFORE to $AFTER"

if git diff --name-only "$BEFORE" "$AFTER" | grep -q "requirements.txt"; then
    log "Updating dependencies..."
    $VENV_PYTHON -m pip install -r requirements.txt 2>&1 | tee -a "$LOG_FILE"
fi

log "Restarting bot..."
nohup sudo $VENV_PYTHON $MAIN_SCRIPT > "$REPO_DIR/bot.log" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"

log "Bot restarted (PID: $NEW_PID)"

update_version_in_db "$AFTER"

log "====== Update Complete ======"

echo "SUCCESS:$AFTER"
