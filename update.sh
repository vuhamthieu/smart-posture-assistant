#!/bin/bash

# ============================================
# OTA Update Script
# ============================================

REPO_DIR="/home/theo/smart-posture-assistant"
VENV_PYTHON="/home/theo/posture-env/bin/python"
MAIN_SCRIPT="$REPO_DIR/src/main.py"
LOG_FILE="$REPO_DIR/update.log"
PID_FILE="$REPO_DIR/bot.pid"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

cd "$REPO_DIR"

log "====== OTA Update Started ======"

# 1. Get current commit
BEFORE=$(git rev-parse --short HEAD)
log "Current commit: $BEFORE"

# 2. Fetch latest from GitHub
log "Fetching from GitHub..."
git fetch origin main 2>&1 | tee -a "$LOG_FILE"

# 3. Check if update available
LATEST=$(git rev-parse --short origin/main)

if [ "$BEFORE" = "$LATEST" ]; then
    log "Already up to date"
    echo "NO_UPDATE"
    exit 0
fi

log "New version found: $LATEST"

# 4. Create backup
BACKUP_DIR="$REPO_DIR/backups/$(date '+%Y%m%d_%H%M%S')"
mkdir -p "$BACKUP_DIR"
cp -r src "$BACKUP_DIR/" 2>/dev/null
log "Backup created: $BACKUP_DIR"

# 5. Stop running bot
log "Stopping bot..."
if [ -f "$PID_FILE" ]; then
    BOT_PID=$(cat "$PID_FILE")
    if ps -p "$BOT_PID" > /dev/null 2>&1; then
        sudo kill "$BOT_PID"
        log "Stopped bot (PID: $BOT_PID)"
    fi
fi

# Fallback: kill by name
pkill -f "main.py"
sleep 2

# 6. Pull new code
log "Pulling new code..."
git reset --hard origin/main 2>&1 | tee -a "$LOG_FILE"

AFTER=$(git rev-parse --short HEAD)
log "Updated from $BEFORE to $AFTER"

# 7. Update dependencies if requirements.txt changed
if git diff --name-only "$BEFORE" "$AFTER" | grep -q "requirements.txt"; then
    log "Updating dependencies..."
    $VENV_PYTHON -m pip install -r requirements.txt 2>&1 | tee -a "$LOG_FILE"
fi

# 8. Restart bot
log "Restarting bot..."
nohup sudo $VENV_PYTHON $MAIN_SCRIPT > "$REPO_DIR/bot.log" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"

log "Bot restarted (PID: $NEW_PID)"
log "====== Update Complete ======"

echo "SUCCESS:$AFTER"
