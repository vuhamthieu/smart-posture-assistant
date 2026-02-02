#!/usr/bin/env python3
import time
import csv
import os
import subprocess
import threading
import atexit
from collections import deque
from datetime import datetime
import numpy as np
import cv2
import joblib
from flask import Flask, Response, jsonify, request
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
from board import SCL, SDA
import busio
from adafruit_ssd1306 import SSD1306_I2C
from rpi_ws281x import PixelStrip, Color
from utils import extract_features_31, validate_keypoints
from config_manager import config_mgr
import voice_agent
from ctypes import *
from contextlib import contextmanager
import glob

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
def py_error_handler(filename, line, function, err, fmt): pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

@contextmanager
def no_alsa_error():
    try:
        asound = cdll.LoadLibrary('libasound.so')
        asound.snd_lib_error_set_handler(c_error_handler)
        yield
        asound.snd_lib_error_set_handler(None)
    except: yield

CALIBRATION_FRAMES = 60
LOG_FILE = "posture_log.csv"
last_upload_time = 0
last_upload_status = ""
HEARTBEAT_INTERVAL = 15.0

print("Initializing Camera...")
def auto_find_camera():
    dev_list = glob.glob('/dev/video*')
    dev_list.sort()
    
    for dev_path in dev_list:
        try:
            dev_id = int(dev_path.replace('/dev/video', ''))
        except: continue

        print(f"{YELLOW}Checking {dev_path}...{RESET}", end=" ")
        temp_cap = cv2.VideoCapture(dev_id)
        if temp_cap.isOpened():
            ret, _ = temp_cap.read()
            temp_cap.release()
            if ret:
                print(f"{GREEN}OK{RESET}")
                return dev_id
        print(f"{RED}Fail{RESET}")
    return None

def reset_camera_driver():
    print(f"{RED}Camera not found. Resetting driver...{RESET}")
    try:
        os.system("sudo modprobe -r uvcvideo")
        time.sleep(1)
        os.system("sudo modprobe uvcvideo")
        time.sleep(2)
    except: pass

print(f"{YELLOW}Initializing...{RESET}")
found_id = auto_find_camera()

if found_id is None:
    reset_camera_driver()
    found_id = auto_find_camera()

if found_id is None:
    print(f"{RED}CRITICAL ERROR: No camera hardware detected.{RESET}")
    exit(1)

print(f"{GREEN}Selected Camera ID: {found_id}{RESET}")
cap = cv2.VideoCapture(found_id)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, 120)
cap.set(cv2.CAP_PROP_FPS, 30)

def cleanup_camera():
    if 'cap' in globals() and cap.isOpened():
        print(f"\n{YELLOW}Releasing camera resource...{RESET}")
        cap.release()

atexit.register(cleanup_camera)

if not cap.isOpened():
    print(f"{RED}Error: Failed to open camera stream{RESET}")
else:
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"{GREEN}Camera Started: {int(w)}x{int(h)}{RESET}")
try: from tflite_runtime.interpreter import Interpreter
except: from tensorflow.lite.python.interpreter import Interpreter

MODEL_PATH = "/home/theo/4.tflite"
ENSEMBLE_MODEL = "/home/theo/smart-posture-assistant/models/posture_ensemble.pkl"
ENSEMBLE_SCALER = "/home/theo/smart-posture-assistant/models/posture_scaler.pkl"

use_ml_model = False
posture_model = None
posture_scaler = None

try:
    posture_model = joblib.load(ENSEMBLE_MODEL)
    posture_scaler = joblib.load(ENSEMBLE_SCALER)
    use_ml_model = True
    print("ML Model Loaded")
except Exception as e:
    print(f"Model Error: {e}")

interpreter = Interpreter(MODEL_PATH, num_threads=2)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
in_h, in_w = input_details[0]['shape'][1:3]
input_dtype = input_details[0]['dtype']

LED_COUNT = 12; LED_PIN = 12; LED_FREQ_HZ = 800000; LED_DMA = 10; LED_INVERT = False; LED_CHANNEL = 0
oled = None; oled_lock = threading.Lock()

try:
    import board
    i2c = board.I2C() 
    oled = SSD1306_I2C(128, 64, i2c, addr=0x3c)
    oled.fill(0); oled.show()
    print("OLED Connected")
except Exception as e:
    print(f"OLED Error: {e}")

strip = None
try:
    strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, 20, LED_CHANNEL)
    strip.begin()
    for i in range(LED_COUNT): strip.setPixelColor(i, Color(0,0,0))
    strip.show()
    print("LED Strip Connected")
except Exception as e:
    print(f"LED Error: {e}")

def cleanup():
    if oled:
        try:
            oled.fill(0); oled.show()
        except: pass
    if strip:
        try:
            for i in range(LED_COUNT): strip.setPixelColor(i, Color(0,0,0))
            strip.show()
        except: pass

atexit.register(cleanup)

def log_data(phys_lbl, ai_lbl, ai_conf, final_res, neck_val, nose_val, angle):
    global last_upload_time, last_upload_status
    
    upload_type = "Good"
    ai_clean = str(ai_lbl).lower().strip()
    phys_clean = str(phys_lbl).lower()

    if final_res == "Bad":
        if 'tilt' in ai_clean:
            upload_type = "Tilt"
        elif "lean" in phys_clean or "lean" in ai_clean:
            upload_type = "Lean"
        elif "hunch" in phys_clean or "hunch" in ai_clean:
            upload_type = "Hunch"
        else:
            upload_type = "Hunch"
    
    current_time = time.time()
    
    is_status_changed = (upload_type != last_upload_status)
    
    is_heartbeat_time = (current_time - last_upload_time > HEARTBEAT_INTERVAL)

    if is_status_changed or is_heartbeat_time:
        metrics_data = {
            "neck_angle": float(angle),
            "neck_ratio": float(neck_val),
            "nose_dist": float(nose_val)
        }
        threading.Thread(target=config_mgr.upload_record, 
                         args=(upload_type, ai_conf,metrics_data), 
                         daemon=True).start()
        
        last_upload_time = current_time
        last_upload_status = upload_type

def init_audio():
    try: subprocess.run(['amixer', 'set', 'Master', '100%'], check=False); return True
    except: return False
audio_available = init_audio()

class StudyTimer:
    def __init__(self):
        self.duration = 0; self.remaining = 0; self.is_running = False; self.lock = threading.Lock()
    def start(self, m):
        with self.lock: self.duration = m*60; self.remaining = self.duration; self.is_running = True; self.start_t = time.time()
    def stop(self):
        with self.lock: self.is_running = False; self.remaining = 0
    def update(self):
        with self.lock:
            if self.is_running:
                self.remaining = max(0, self.duration - (time.time() - self.start_t))
                if self.remaining == 0: self.is_running = False; return True
            return False
    def get_time_str(self):
        with self.lock: return f"{int(self.remaining//60):02d}:{int(self.remaining%60):02d}" if self.is_running else ""

study_timer = StudyTimer()

class LEDController:
    def __init__(self, strip): self.strip = strip; self.lock = threading.Lock()
    
    def set_color_array(self, color_array):
        if not self.strip: return
        try:
            r, g, b = color_array if len(color_array) == 3 else [0,0,0]
            with self.lock:
                for i in range(LED_COUNT): self.strip.setPixelColor(i, Color(r, g, b))
                self.strip.show()
        except: pass

    def pulse_while_speaking(self, color_array, duration=3):
        if not self.strip: return
        start = time.time()
        r, g, b = color_array if len(color_array) == 3 else [255, 0, 0]
        dim_r, dim_g, dim_b = int(r*0.2), int(g*0.2), int(b*0.2)
        
        while time.time() - start < duration:
            self.set_color_array([r, g, b])
            time.sleep(0.2)
            self.set_color_array([dim_r, dim_g, dim_b])
            time.sleep(0.2)

led_controller = LEDController(strip)

class FaceDisplay:
    def __init__(self, oled, lock):
        self.oled = oled; self.lock = lock
    
    def _draw(self, func):
        if not self.oled: return
        with self.lock:
            try:
                img = Image.new("1", (128, 64))
                func(ImageDraw.Draw(img))
                self.oled.image(img); self.oled.show()
            except: pass

    def draw_normal(self):
        style = config_mgr.get('oled_icon_style', 'A')
        if style == 'B':
            self._draw(lambda d: (
                d.ellipse((32, 20, 52, 40), fill=255), d.ellipse((76, 20, 96, 40), fill=255),
                d.ellipse((42, 24, 46, 28), fill=0), d.ellipse((86, 24, 90, 28), fill=0),
                d.polygon([(60, 42), (68, 42), (64, 47)], fill=255),
                d.arc((58, 47, 64, 52), 0, 180, fill=255), d.arc((64, 47, 70, 52), 0, 180, fill=255),
                d.line((15, 32, 30, 35), fill=255), d.line((15, 40, 30, 40), fill=255), d.line((15, 48, 30, 45), fill=255),
                d.line((113, 32, 98, 35), fill=255), d.line((113, 40, 98, 40), fill=255), d.line((113, 48, 98, 45), fill=255)
            ))
        elif style == 'C':
            self._draw(lambda d: (
                d.rectangle((20, 10, 108, 54), outline=255),
                d.rectangle((35, 20, 55, 35), fill=255), d.rectangle((73, 20, 93, 35), fill=255),
                d.line((45, 45, 83, 45), fill=255, width=3),
                d.rectangle((15, 25, 20, 40), fill=255), d.rectangle((108, 25, 113, 40), fill=255)
            ))
        else:
            self._draw(lambda d: (
                d.ellipse((30, 15, 50, 35), outline=255), d.ellipse((78, 15, 98, 35), outline=255),
                d.arc((40, 35, 88, 55), 0, 180, fill=255, width=2)
            ))

    def draw_angry(self):
        style = config_mgr.get('oled_icon_style', 'A')
        if style == 'B':
             self._draw(lambda d: (
                d.line((32, 25, 52, 35), fill=255, width=3), d.line((32, 35, 52, 25), fill=255, width=3),
                d.line((76, 25, 96, 35), fill=255, width=3), d.line((76, 35, 96, 25), fill=255, width=3),
                d.polygon([(60, 42), (68, 42), (64, 47)], fill=255),
                d.ellipse((58, 48, 70, 58), outline=255, width=2),
                d.line((15, 25, 30, 32), fill=255), d.line((15, 40, 30, 40), fill=255), d.line((15, 55, 30, 48), fill=255),
                d.line((113, 25, 98, 32), fill=255), d.line((113, 40, 98, 40), fill=255), d.line((113, 55, 98, 48), fill=255)
             ))
        elif style == 'C':
             self._draw(lambda d: (
                d.rectangle((20, 10, 108, 54), outline=255),
                d.line((35, 35, 55, 20), fill=255, width=3), d.line((73, 20, 93, 35), fill=255, width=3),
                d.line((35, 45, 45, 50), fill=255, width=2), d.line((45, 50, 55, 45), fill=255, width=2),
                d.line((55, 45, 65, 50), fill=255, width=2), d.line((65, 50, 75, 45), fill=255, width=2),
                d.line((75, 45, 85, 50), fill=255, width=2), d.line((85, 50, 93, 45), fill=255, width=2)
            ))
        else:
             self._draw(lambda d: (
                d.line((25, 20, 45, 30), fill=255, width=2), d.line((83, 30, 103, 20), fill=255, width=2),
                d.ellipse((30, 30, 40, 40), fill=255), d.ellipse((88, 30, 98, 40), fill=255),
                d.arc((45, 45, 83, 60), 180, 360, fill=255, width=2)
            ))

    def draw_timer(self, txt): 
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 25)
            self._draw(lambda d: d.text((25, 18), txt, font=font, fill=255))
            
        except IOError:
            print("Use default font")
            self._draw(lambda d: d.text((45, 25), txt, fill=255))

face_display = FaceDisplay(oled, oled_lock)
if oled: 
    face_display.draw_normal()

TTS_CACHE_DIR = "/tmp/tts_cache"; os.makedirs(TTS_CACHE_DIR, exist_ok=True)

def get_alert_message(key):
    lang = config_mgr.get('alert_language', 'vi')
    messages = config_mgr.get(f'alert_messages_{lang}', [])
    idx_map = {'lean': 0, 'hunch': 1, 'tilt': 2, 'close': 3}
    if messages and key in idx_map and len(messages) > idx_map[key]:
        return messages[idx_map[key]], lang
    fallback = {'lean': "Bạn đang cúi đầu quá thấp", 'hunch': "Đừng gù lưng", 'tilt': "Đừng nghiêng đầu", 'close': "Ngồi xa ra"}
    return fallback.get(key, "Cảnh báo"), 'vi'

def speak_alert(key):
    if not audio_available: return
    text, lang = get_alert_message(key)
    filename = f"{hash(text+lang)}.mp3"
    filepath = os.path.join(TTS_CACHE_DIR, filename)
    if not os.path.exists(filepath):
        try: gTTS(text=text, lang=lang).save(filepath)
        except: return
    
    bad_color = config_mgr.get('led_color_bad', [255, 0, 0])
    
    def play_sound():
        voice_agent.IS_BOT_SPEAKING = True
        subprocess.run(['mpg123', '-q', filepath], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.0)
        voice_agent.IS_BOT_SPEAKING = False

    threading.Thread(target=play_sound, daemon=True).start()
    threading.Thread(target=lambda: led_controller.pulse_while_speaking(bad_color), daemon=True).start()

def timer_updater():
    last = 0
    while True:
        if study_timer.update(): speak_alert('close')
        if study_timer.is_running and time.time()-last>=1: face_display.draw_timer(study_timer.get_time_str()); last=time.time()
        time.sleep(0.5)

app = Flask(__name__)
outputFrame = None
lock = threading.Lock()
stats = {'posture_status': 'Init', 'fps': 0}

base_neck_ratio = 0.0
base_nose_ear_diff = 0.0
is_calibrated = False
calib_neck = []
calib_nose = []

def detect_posture():
    global outputFrame, cap, base_neck_ratio, base_nose_ear_diff, is_calibrated, calib_neck, calib_nose
    
    status_buf = deque(maxlen=5); last_bad=0; last_alert=0; fps_frame_cnt=0; fps_start_time=time.time()
    last_final_status = None 
    
    while True:
        if not cap.isOpened(): time.sleep(1); continue
        ret, frame = cap.read()
        if not ret: continue
        
        stream_frame = cv2.resize(frame, (640, 480))

        img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (in_w, in_h))
        input_data = np.expand_dims(img, axis=0) if input_dtype == np.uint8 else np.expand_dims((img.astype(np.float32)-127.5)/127.5, axis=0)
        
        interpreter.set_tensor(input_details[0]['index'], input_data); interpreter.invoke()
        kpts = interpreter.get_tensor(output_details[0]['index'])[0][0]
        
        h, w, _ = frame.shape; raw_status="Good"; method="Init"; conf=0.0; angle=0.0; dist_stat="OK"; debug_msg=""
        
        NECK_SHRINK_TOLERANCE = config_mgr.get('neck_threshold', 35.0) / 100.0 * 2.0
        if NECK_SHRINK_TOLERANCE > 1.0 or NECK_SHRINK_TOLERANCE < 0.5: NECK_SHRINK_TOLERANCE = 0.70
        NOSE_DROP_THRESHOLD = config_mgr.get('nose_drop_threshold', 0.25)

        if not is_calibrated:
            if validate_keypoints(kpts):
                face_w = np.linalg.norm(np.array([kpts[3][1], kpts[3][0]]) - np.array([kpts[4][1], kpts[4][0]]))
                neck_h = (kpts[5][0] + kpts[6][0]) / 2 - kpts[0][0]
                ear_y = (kpts[3][0] + kpts[4][0]) / 2; nose_y = kpts[0][0]
                nose_ear_val = nose_y - ear_y
                if face_w > 0:
                    calib_neck.append(neck_h / face_w)
                    calib_nose.append(nose_ear_val / face_w)
                msg = f"CALIB... {int(len(calib_neck)/CALIBRATION_FRAMES*100)}%"
                
                cv2.putText(stream_frame, "SIT STRAIGHT", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                cv2.putText(stream_frame, msg, (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                
                if len(calib_neck) >= CALIBRATION_FRAMES:
                    base_neck_ratio = np.mean(calib_neck)
                    base_nose_ear_diff = np.mean(calib_nose)
                    is_calibrated = True
                    speak_alert("close")
            else:
                cv2.putText(stream_frame, "NO PERSON", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            
            with lock: outputFrame = stream_frame.copy(); continue

        if validate_keypoints(kpts):
            try:
                feats = extract_features_31(kpts, w, h); angle = feats[21]
                
                face_w = np.linalg.norm(np.array([kpts[3][1], kpts[3][0]]) - np.array([kpts[4][1], kpts[4][0]]))
                neck_h = (kpts[5][0] + kpts[6][0]) / 2 - kpts[0][0]
                ear_y = (kpts[3][0] + kpts[4][0]) / 2; nose_y = kpts[0][0]
                
                current_nose_diff = (nose_y - ear_y) / (face_w + 1e-6)
                curr_neck_ratio = neck_h / (face_w + 1e-6)
                neck_change = curr_neck_ratio / (base_neck_ratio + 1e-6)
                nose_drop_amount = current_nose_diff - base_nose_ear_diff
                
                phys_label = "Good"; is_bad_posture = False; detected_type = ""
                
                if neck_change < NECK_SHRINK_TOLERANCE:
                    if nose_drop_amount > NOSE_DROP_THRESHOLD: 
                        phys_label = "Lean(Nose)"; is_bad_posture = True; detected_type = "lean"
                    else:
                        phys_label = "Hunch(Neck)"; is_bad_posture = True; detected_type = "hunch"
                
                ai_label = "None"; ai_conf = 0.0
                if use_ml_model:
                    X_in = posture_scaler.transform([feats])
                    ai_label = posture_model.predict(X_in)[0]
                    ai_conf = 0.9 
                
                if use_ml_model:
                    if ai_label == 'tilt' and ai_conf > 0.7:
                        raw_status = "Bad"; detected_type = "tilt"; method = f"AI_Tilt({ai_conf:.2f})"
                    elif is_bad_posture:
                        raw_status = "Bad"; method = phys_label
                    elif ai_conf > 0.7 and ai_label != 'good':
                        raw_status = "Bad"; detected_type = ai_label; method = f"AI_{ai_label}"
                    elif angle <= 15.0:
                        raw_status = "Good"; method = f"Safe({angle:.1f})"
                    else:
                        raw_status = "Good"
                else:
                    if is_bad_posture: raw_status = "Bad"; method = phys_label
                
                debug_msg = f"N:{neck_change:.2f} D:{nose_drop_amount:.2f}"
                log_data(phys_label, ai_label, ai_conf, raw_status, neck_change, nose_drop_amount, angle)

            except: method="Err"
        else: method="NoPerson"
        
        status_buf.append(raw_status)
        final_status = "Bad" if status_buf.count("Bad") >= 3 else "Good"
        
        if final_status != last_final_status:
            if final_status == "Bad":
                bad_color = config_mgr.get('led_color_bad', [255, 0, 0])
                if not study_timer.is_running: 
                    led_controller.set_color_array(bad_color)
                    face_display.draw_angry() 
            else:
                good_color = config_mgr.get('led_color_good', [0, 255, 0])
                if not study_timer.is_running:
                    led_controller.set_color_array(good_color)
                    face_display.draw_normal()
            last_final_status = final_status

        if final_status == "Bad":
            if time.time()-last_bad>1.5:
                if time.time()-last_alert>8:
                    speak_alert(detected_type if detected_type else 'lean')
                    last_alert=time.time()
            else:
                if last_bad==0: last_bad=time.time()
        else:
            last_bad=0
            
        if fps_frame_cnt%2==0:
            cv2.putText(stream_frame, f"Stat:{final_status}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) if final_status=="Bad" else (0,255,0), 2)
            cv2.putText(stream_frame, f"{method}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
            cv2.putText(stream_frame, debug_msg, (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
            
        fps_frame_cnt += 1
        if time.time() - fps_start_time >= 1.0:
            stats['fps'] = fps_frame_cnt
            fps_frame_cnt = 0
            fps_start_time = time.time()
        
        stats['posture_status'] = final_status
        with lock: outputFrame = stream_frame.copy()

def generate():
    global outputFrame
    while True:
        frame_to_encode = None
        with lock:
            if outputFrame is None:
                time.sleep(0.01)
                continue
            frame_to_encode = outputFrame
        
        if frame_to_encode is not None:
            flag, buf = cv2.imencode('.jpg', frame_to_encode)
            if flag:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + bytearray(buf) + b'\r\n')
            else:
                time.sleep(0.01)
        else:
            time.sleep(0.01)

@app.route("/video_feed")
def vf(): return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route("/stats")
def st(): return jsonify(stats)
@app.route("/command", methods=['POST'])
def cmd():
    c=request.json.get('command')
    if c=='recalib': global is_calibrated, calib_neck; is_calibrated=False; calib_neck=[]
    return jsonify({'success':True})
@app.route("/")
def idx(): 
    return """<html><head><title>Posture Monitor</title>
    <script>setInterval(()=>{fetch('/stats').then(r=>r.json()).then(d=>{
        document.getElementById('s').innerText=d.posture_status+" | "+d.fps+" FPS";
    })},1000)</script></head>
    <body style="background:#000;color:#fff;text-align:center">
    <h2>Posture Monitor</h2><img src="/video_feed"><h3 id="s">...</h3></body></html>"""

if __name__ == '__main__':
    print("Started")
    config_mgr.start_polling()
    threading.Thread(target=detect_posture, daemon=True).start()
    threading.Thread(target=timer_updater, daemon=True).start()
    
    try:
        with no_alsa_error():
            threading.Thread(target=voice_agent.voice_listener_loop, 
                            args=(led_controller, face_display, stats, study_timer), 
                            daemon=True).start()
        print("Voice Thread Started")
    except Exception as e:
        print(f"Voice Error: {e}")

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)