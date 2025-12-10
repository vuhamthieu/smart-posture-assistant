#!/usr/bin/env python3
import time
from collections import deque
import numpy as np
import cv2
import threading
import csv
from datetime import datetime
from flask import Flask, Response, jsonify, request
from gtts import gTTS
import os
import subprocess
from PIL import Image, ImageDraw, ImageFont
from board import SCL, SDA
import busio
from adafruit_ssd1306 import SSD1306_I2C
from rpi_ws281x import PixelStrip, Color
import joblib
from utils import extract_features_31, validate_keypoints
from config_manager import config_mgr

# ==========================================
# 1. CONFIGURATION
# ==========================================

CALIBRATION_FRAMES = 60
NECK_SHRINK_TOLERANCE = 0.80 
NOSE_DROP_THRESHOLD = 0.15 
LOG_FILE = "posture_log.csv"

CAMERA_ID = 0
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print("Error: Camera not found!")
else:
    cap.set(3, 640); cap.set(4, 480); cap.set(5, 30)

# --- AI SETUP ---
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
    print("✓ ML Model Loaded (Hybrid Mode)")
except Exception as e: print(f"Model Error: {e}")

interpreter = Interpreter(MODEL_PATH, num_threads=2)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
in_h, in_w = input_details[0]['shape'][1:3]
input_dtype = input_details[0]['dtype']

# --- HARDWARE & DISPLAY ---
LED_COUNT = 12; LED_PIN = 12; LED_FREQ_HZ = 800000; LED_DMA = 10; LED_INVERT = False; LED_CHANNEL = 0
oled = None; oled_lock = threading.Lock()
try:
    i2c = busio.I2C(SCL, SDA); oled = SSD1306_I2C(128, 64, i2c); oled.fill(0); oled.show()
except: pass
strip = None
try:
    strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, 20, LED_CHANNEL)
    strip.begin(); 
    for i in range(LED_COUNT): strip.setPixelColor(i, Color(0,0,0))
    strip.show()
except: pass

# --- LOGGER HELPER ---
def init_logger():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Phys_Label', 'AI_Label', 'AI_Conf', 'Final_Result', 'Neck_Change', 'Nose_Drop'])

def log_data(phys_lbl, ai_lbl, ai_conf, final_res, neck_val, nose_val):
    try:
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%H:%M:%S.%f")[:-3],
                phys_lbl, ai_lbl, f"{ai_conf:.2f}", final_res, 
                f"{neck_val:.2f}", f"{nose_val:.2f}"
            ])
    except: pass

init_logger()

# ==========================================
# 2. HELPER CLASSES
# ==========================================
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
    def set_color(self, r, g, b):
        if not self.strip: return
        with self.lock:
            for i in range(LED_COUNT): self.strip.setPixelColor(i, Color(r, g, b))
            self.strip.show()
    def pulse_while_speaking(self, duration=3):
        if not self.strip: return
        start = time.time()
        while time.time() - start < duration:
            self.set_color(100,100,0); time.sleep(0.2); self.set_color(20,20,0); time.sleep(0.2)

led_controller = LEDController(strip)

class FaceDisplay:
    def __init__(self, oled, lock): self.oled = oled; self.lock = lock
    def _draw(self, func):
        if not self.oled: return
        if self.lock.acquire(blocking=False):
            try: img = Image.new("1", (128, 64)); func(ImageDraw.Draw(img)); self.oled.image(img); self.oled.show()
            except: pass
            finally: self.lock.release()
    def draw_normal(self): self._draw(lambda d: (d.ellipse((30,15,50,35),outline=255), d.ellipse((78,15,98,35),outline=255), d.arc((40,35,88,55),0,180,fill=255)))
    def draw_angry(self): self._draw(lambda d: (d.line((25,15,45,20),fill=255,width=2), d.line((83,20,103,15),fill=255,width=2), d.arc((40,45,88,58),180,360,fill=255,width=2)))
    def draw_timer(self, txt): self._draw(lambda d: d.text((10, 25), txt, fill=255))

face_display = FaceDisplay(oled, oled_lock)
if oled: face_display.draw_normal()

ALERT_MESSAGES = {'lean': "Bạn đang cúi đầu", 'tilt': "Đừng nghiêng đầu", 'hunch': "Đừng gù lưng", 'close': "Ngồi quá gần"}
TTS_CACHE_DIR = "/tmp/tts_cache"; os.makedirs(TTS_CACHE_DIR, exist_ok=True)
def get_cached_tts(msg, lang='vi'):
    path = f"{TTS_CACHE_DIR}/{hash(msg)}.mp3"
    if not os.path.exists(path):
        try: gTTS(text=msg, lang=lang).save(path)
        except: return None
    return path
for m in ALERT_MESSAGES.values(): get_cached_tts(m)

def speak_alert(key):
    if audio_available and key in ALERT_MESSAGES:
        threading.Thread(target=lambda: subprocess.run(['mpg123', get_cached_tts(ALERT_MESSAGES[key])], capture_output=True), daemon=True).start()
        threading.Thread(target=lambda: led_controller.pulse_while_speaking(), daemon=True).start()

def timer_updater():
    last = 0
    while True:
        if study_timer.update(): speak_alert('close')
        if study_timer.is_running and time.time()-last>=1: face_display.draw_timer(study_timer.get_time_str()); last=time.time()
        time.sleep(0.5)

# ==========================================
# 3. MAIN LOGIC (HYBRID + LOGGING)
# ==========================================

app = Flask(__name__)
outputFrame = None
lock = threading.Lock()
stats = {'posture_status': 'Init', 'fps': 0}

# Global Calibration
base_neck_ratio = 0.0
base_nose_ear_diff = 0.0
face_size_ref = 0.0
is_calibrated = False
calib_neck = []
calib_nose = []
calib_face = []

def detect_posture():
    global outputFrame, cap, base_neck_ratio, base_nose_ear_diff, face_size_ref, is_calibrated, calib_neck, calib_nose, calib_face
    
    status_buf = deque(maxlen=5); last_bad=0; last_alert=0; frame_cnt=0; start_t=time.time()
    log_counter = 0 
    
    while True:
        if not cap.isOpened(): time.sleep(1); continue
        ret, frame = cap.read()
        if not ret: continue
        frame_cnt += 1
        
        img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (in_w, in_h))
        input_data = np.expand_dims(img, axis=0) if input_dtype == np.uint8 else np.expand_dims((img.astype(np.float32)-127.5)/127.5, axis=0)
        
        interpreter.set_tensor(input_details[0]['index'], input_data); interpreter.invoke()
        kpts = interpreter.get_tensor(output_details[0]['index'])[0][0]
        
        h, w, _ = frame.shape; raw_status="Good"; method="Init"; conf=0.0; angle=0.0; dist_stat="OK"; debug_msg=""
        
        # --- CALIBRATION ---
        if not is_calibrated:
            if validate_keypoints(kpts):
                face_w = np.linalg.norm(np.array([kpts[3][1], kpts[3][0]]) - np.array([kpts[4][1], kpts[4][0]]))
                neck_h = (kpts[5][0] + kpts[6][0]) / 2 - kpts[0][0]
                ear_y = (kpts[3][0] + kpts[4][0]) / 2; nose_y = kpts[0][0]
                nose_ear_val = nose_y - ear_y
                
                if face_w > 0:
                    calib_neck.append(neck_h / face_w)
                    calib_nose.append(nose_ear_val / face_w)
                    calib_face.append(face_w)
                
                msg = f"CALIBRATING... {int(len(calib_neck)/CALIBRATION_FRAMES*100)}%"
                cv2.putText(frame, "NGOI THANG...", (20, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                cv2.putText(frame, msg, (20, h//2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                
                if len(calib_neck) >= CALIBRATION_FRAMES:
                    base_neck_ratio = np.mean(calib_neck)
                    base_nose_ear_diff = np.mean(calib_nose)
                    is_calibrated = True
                    speak_alert("close") 
            else:
                cv2.putText(frame, "NO PERSON", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            with lock: outputFrame = frame.copy(); continue

        # --- MONITORING ---
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
                
                phys_label = "Good"
                is_bad_posture = False
                detected_type = ""
                
                if neck_change < NECK_SHRINK_TOLERANCE:
                    if nose_drop_amount > NOSE_DROP_THRESHOLD: 
                        phys_label = "Lean(Nose)"; is_bad_posture = True; detected_type = "lean"
                    else:
                        phys_label = "Hunch(Neck)"; is_bad_posture = True; detected_type = "hunch"
                
                # --- 2. AI OPINION (ML) ---
                ai_label = "None"
                ai_conf = 0.0
                if use_ml_model:
                    X_in = posture_scaler.transform([feats])
                    pred_class = posture_model.predict(X_in)[0]
                    ai_conf = np.max(posture_model.predict_proba(X_in)[0])
                    ai_label = pred_class
                
                # --- 3. FINAL DECISION (HYBRID) ---
                if use_ml_model:
                    if angle <= 15.0 and not is_bad_posture:
                        raw_status = "Good"; method = f"Safe({angle:.1f})"
                    elif is_bad_posture:
                        raw_status = "Bad"; method = phys_label
                    elif ai_conf > 0.7:
                        if ai_label != 'good':
                            raw_status = "Bad"; detected_type = ai_label; method = f"AI_{ai_label}"
                        else:
                            raw_status = "Good"; method = "AI_Good"
                    else:
                        raw_status = "Good"
                else:
                    if is_bad_posture: raw_status = "Bad"; method = phys_label
                
                debug_msg = f"N:{neck_change:.2f} D:{nose_drop_amount:.2f}"
                
                log_counter += 1
                if log_counter % 10 == 0:
                    log_data(phys_label, ai_label, ai_conf, raw_status, neck_change, nose_drop_amount)

            except Exception as e: method="Err"
        else: method="NoPerson"
        
        status_buf.append(raw_status)
        final_status = "Bad" if status_buf.count("Bad") >= 3 else "Good"
        
        # --- ALERT ---
        if final_status == "Bad":
            if not study_timer.is_running: led_controller.set_color(255,0,0)
            if face_display: face_display.draw_angry()
            if time.time()-last_bad>1.5:
                if time.time()-last_alert>8:
                    speak_alert(detected_type if detected_type else 'lean')
                    last_alert=time.time()
            else:
                if last_bad==0: last_bad=time.time()
        else:
            if not study_timer.is_running: led_controller.set_color(0,255,0)
            if face_display: face_display.draw_normal(); last_bad=0
            
        if frame_cnt%2==0:
            cv2.putText(frame, f"Stat:{final_status}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) if final_status=="Bad" else (0,255,0), 2)
            cv2.putText(frame, f"{method}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
            cv2.putText(frame, debug_msg, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
            
        elapsed = time.time()-start_t; fps = frame_cnt/elapsed if elapsed>0 else 0
        stats.update({'posture_status': final_status, 'fps': round(fps,1)})
        with lock: outputFrame = frame.copy()

def generate():
    global outputFrame
    while True:
        with lock:
            if outputFrame is None: time.sleep(0.01); continue
            _, buf = cv2.imencode('.jpg', outputFrame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + bytearray(buf) + b'\r\n')

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
    return """<html><head><title>Posture Detect</title>
    <script>setInterval(()=>{fetch('/stats').then(r=>r.json()).then(d=>{
        document.getElementById('s').innerText=d.posture_status+" | "+d.fps+" FPS";
    })},1000)</script></head>
    <body style="background:#000;color:#fff;text-align:center">
    <h2>Hybrid AI Monitor</h2><img src="/video_feed"><h3 id="s">...</h3></body></html>"""

if __name__ == '__main__':
    print("Started (Hybrid + Logging).")
    config_mgr.start_polling()
    threading.Thread(target=detect_posture, daemon=True).start()
    threading.Thread(target=timer_updater, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
