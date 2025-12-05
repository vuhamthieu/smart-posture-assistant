# src/config_manager.py
import os
import time
import threading
import subprocess
from supabase import create_client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
DEVICE_ID = "pi-posture-001"
UPDATE_SCRIPT_PATH = "/home/theo/smart-posture-assistant/update.sh"

class ConfigManager:
    def __init__(self):
        self.client = None
        self.config = {
            "neck_threshold": 35,
            "smoothing_frames": 6,
            "ml_confidence_threshold": 0.75,
            "bad_vote_required": 5,
            "led_color_bad": [0, 255, 0],
            "led_color_good": [255, 200, 0]
        }
        self.lock = threading.Lock()
        
        if SUPABASE_URL and SUPABASE_KEY:
            try:
                self.client = create_client(SUPABASE_URL, SUPABASE_KEY)
                print("✓ [Config] Connected to Supabase Broker")
            except Exception as e:
                print(f"⚠ [Config] Connection failed: {e}")

    def get(self, key, default=None):
        with self.lock:
            return self.config.get(key, default)

    def start_polling(self):
        """Bắt đầu luồng chạy ngầm để check lệnh"""
        t = threading.Thread(target=_loop, args=(self,), daemon=True)
        t.start()

def _loop(mgr):
    while True:
        if mgr.client:
            try:
                res = mgr.client.table("device_commands")\
                    .select("*")\
                    .eq("device_id", DEVICE_ID)\
                    .eq("status", "PENDING")\
                    .execute()
                
                for cmd in res.data:
                    cid = cmd['id']
                    action = cmd['command']
                    print(f"🚀 [OTA] Executing: {action}")
                    
                    mgr.client.table("device_commands").update({"status": "EXECUTING"}).eq("id", cid).execute()
                    
                    if action == 'UPDATE':
                        subprocess.Popen(["/bin/bash", UPDATE_SCRIPT_PATH])
                    elif action == 'RESTART':
                        subprocess.run(["sudo", "systemctl", "restart", "posture-bot"])
                        
                    mgr.client.table("device_commands").update({"status": "COMPLETED"}).eq("id", cid).execute()
            except Exception as e:
                print(f"Error checking commands: {e}")

            try:
                res = mgr.client.table("device_configs").select("settings").eq("device_id", DEVICE_ID).execute()
                if res.data:
                    with mgr.lock:
                        mgr.config.update(res.data[0]['settings'])
            except Exception:
                pass

        time.sleep(5)

config_mgr = ConfigManager()
