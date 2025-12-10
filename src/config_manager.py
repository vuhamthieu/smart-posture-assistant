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
            "neck_threshold": 35.0,
            "ml_confidence_threshold": 0.75,
            "smoothing_frames": 6,
            "bad_vote_required": 5,
            "status_buffer_size": 8,
            "bad_duration_to_alert": 1.0,
            
            "led_color_good": [255, 200, 0],   
            "led_color_bad": [0, 255, 0],    
            "oled_icon_style": "A",
            "alert_language": "vi",

            "alert_messages_vi": [
                "Bạn đang cúi đầu quá thấp, hãy ngồi thẳng lại",
                "Tư thế ngồi của bạn không đúng, giữ thẳng lưng nhé",
                "Hãy giữ đầu thẳng với cột sống",
                "Bạn đang ngồi quá gần màn hình, hãy lùi ra xa một chút",
                "Ngồi thẳng dậy đi nào"
            ],
            "alert_messages_en": [
                "Your head is too low, please sit up straight",
                "Bad posture detected, keep your back straight",
                "Please align your head with your spine",
                "You are sitting too close to the screen",
                "Sit up straight!"
            ]
        }
        self.lock = threading.Lock()
        
        if SUPABASE_URL and SUPABASE_KEY:
            try:
                self.client = create_client(SUPABASE_URL, SUPABASE_KEY)
                print("✓ [Config] Connected to Supabase Broker")
                self._fetch_config_once()
            except Exception as e:
                print(f"⚠ [Config] Connection failed: {e}")

    def get(self, key, default=None):
        with self.lock:
            return self.config.get(key, default)

    def start_polling(self):
        t = threading.Thread(target=_loop, args=(self,), daemon=True)
        t.start()

    def _fetch_config_once(self):
        try:
            res = self.client.table("device_configs").select("settings").eq("device_id", DEVICE_ID).execute()
            if res.data:
                with self.lock:
                    self.config.update(res.data[0]['settings'])
                print("✓ [Config] Initial settings loaded")
        except Exception:
            pass

def _loop(mgr):
    while True:
        if mgr.client:
            # 1. Commands
            try:
                res = mgr.client.table("device_commands").select("*").eq("device_id", DEVICE_ID).eq("status", "PENDING").execute()
                for cmd in res.data:
                    cid = cmd['id']
                    action = cmd['command']
                    mgr.client.table("device_commands").update({"status": "EXECUTING"}).eq("id", cid).execute()
                    if action == 'UPDATE': subprocess.Popen(["/bin/bash", UPDATE_SCRIPT_PATH])
                    elif action == 'RESTART': subprocess.run(["systemctl", "restart", "posture-bot"])
                    mgr.client.table("device_commands").update({"status": "COMPLETED"}).eq("id", cid).execute()
            except Exception: pass

            # 2. Config
            try:
                res = mgr.client.table("device_configs").select("settings").eq("device_id", DEVICE_ID).execute()
                if res.data:
                    with mgr.lock:
                        mgr.config.update(res.data[0]['settings'])
            except Exception: pass
        time.sleep(5)

config_mgr = ConfigManager()
