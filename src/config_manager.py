import os
import time
import threading
import subprocess
from supabase import create_client
from dotenv import load_dotenv
from pathlib import Path 

env_path = Path('/home/theo/smart-posture-assistant/.env')
load_dotenv(dotenv_path=env_path)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

DEVICE_ID = os.environ.get("DEVICE_ID", "pi-posture-001")

UPDATE_SCRIPT_PATH = "/home/theo/smart-posture-assistant/update.sh"

class ConfigManager:
    def __init__(self):
        self.client = None
        self.current_user_id = None
        
        self.config = {
            "neck_threshold": 35.0,
            "ml_confidence_threshold": 0.75,
            "smoothing_frames": 6,
            "bad_vote_required": 5,
            "status_buffer_size": 8,
            "bad_duration_to_alert": 1.0,
            
            "led_color_good": [0, 255, 0],   
            "led_color_bad": [255, 0, 0],      
            "oled_icon_style": "A",
            "alert_language": "vi",
            "alert_messages_vi": [
                "Bạn đang cúi đầu quá thấp",
                "Đừng gù lưng",
                "Đừng nghiêng đầu",
                "Ngồi xa màn hình ra",
                "Tư thế xấu"
            ],
            "alert_messages_en": [
                "Head too low",
                "Don't slouch",
                "Don't tilt head",
                "Sit away",
                "Bad posture"
            ]
        }
        self.lock = threading.Lock()
        
        if SUPABASE_URL and SUPABASE_KEY:
            try:
                self.client = create_client(SUPABASE_URL, SUPABASE_KEY)
                print(f"Config Connected to Supabase (Device: {DEVICE_ID})")
                self._fetch_config_and_owner()
            except Exception as e:
                print(f"Config Connection failed: {e}")

    def upload_record(self, posture_type, confidence, metrics=None):
        if not self.client or not self.current_user_id:
            return

        type_mapping = {
            "Good": "good",
            "Lean": "leaning",
            "Hunch": "slouching",
            "Tilt": "tilt" 
        }
        
        db_posture_type = type_mapping.get(posture_type, posture_type.lower())
        if posture_type == "Good": db_posture_type = "good"

        try:
            data = {
                "user_id": self.current_user_id, 
                "posture_type": db_posture_type,
                "confidence": float(confidence),
                "metrics": metrics if metrics else {}
            }
            self.client.table("posture_records").insert(data).execute()
        except Exception as e:
            print(f"Error uploading: {e}")

    def get(self, key, default=None):
        with self.lock:
            return self.config.get(key, default)

    def start_polling(self):
        t = threading.Thread(target=_loop, args=(self,), daemon=True)
        t.start()

    def _fetch_config_and_owner(self):
        try:
            res = self.client.table("device_configs").select("settings, user_id").eq("device_id", DEVICE_ID).execute()
            
            if res.data:
                data = res.data[0]

                new_owner = data.get('user_id')
                if new_owner != self.current_user_id:
                    print(f">>> DEVICE PAIRED WITH NEW USER: {new_owner}")
                    self.current_user_id = new_owner
                
                if data.get('settings'):
                    clean_settings = {k: v for k, v in data['settings'].items() if v is not None}
                    with self.lock:
                        self.config.update(clean_settings)
                        
        except Exception as e:
            print(f"Fetch config error: {e}")

def _loop(mgr):
    while True:
        if mgr.client:
            try:
                res = mgr.client.table("device_commands").select("*").eq("device_id", DEVICE_ID).eq("status", "PENDING").execute()
                
                for cmd in res.data:
                    cid = cmd['id']
                    action = cmd['command']
                    
                    print(f">>> Received Command: {action}")

                    mgr.client.table("device_commands").update({"status": "EXECUTING"}).eq("id", cid).execute()
                    
                    if action == 'UPDATE_CODE' or action == 'UPDATE': 
                        if os.path.exists(UPDATE_SCRIPT_PATH):
                            subprocess.Popen(["/bin/bash", UPDATE_SCRIPT_PATH])
                        else:
                            print("Update script not found!")
                            
                    elif action == 'RESTART':
                        subprocess.run(["sudo", "systemctl", "restart", "posture-bot"])
                    
                    elif action == 'UPDATE_CONFIG':
                        mgr._fetch_config_and_owner()

                    mgr.client.table("device_commands").update({"status": "COMPLETED"}).eq("id", cid).execute()
            
            except Exception as e:
                print(f"Command loop error: {e}")

            mgr._fetch_config_and_owner()
            
        time.sleep(5)

config_mgr = ConfigManager()
