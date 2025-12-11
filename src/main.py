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

CALIBRATION_FRAMES = 60
LOG_FILE = "posture_log.csv"
CAMERA_ID = 0

cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print("Error: Camera not found!")
else:
    cap.set(3, 640); cap.set(4, 480); cap.set(5, 30)

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
    print("✓ ML Model Loaded")
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
    print("✓ OLED Connected")
except Exception as e:
    print(f"⚠ OLED Error: {e}")

strip = None
try:
    strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, 20, LED_CHANNEL)
    strip.begin()
    for i in range(LED_COUNT): strip.setPixelColor(i, Color(0,0,0))
    strip.show()
    print("✓ LED Strip Connected")
except Exception as e:
    print(f"⚠ LED Error: {e}")

def cleanup():
    if oled:
        oled.fill(0); oled.show()
    if strip:
        for i in range(LED_COUNT): strip.setPixelColor(i, Color(0,0,0))
        strip.show()

atexit.register(cleanup)

def log_data(phys_lbl, ai_lbl, ai_conf, final_res, neck_val, nose_val):
    try:
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'w', newline='') as f:
                csv.writer(f).writerow(['Timestamp', 'Phys_Label', 'AI_Label', 'AI_Conf', 'Final_Result', 'Neck_Change', 'Nose_Drop'])
        with open(LOG_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([
                datetime.now().strftime("%H:%M:%S.%f")[:-3],
                phys_lbl, ai_lbl, f"{ai_conf:.2f}", final_res,
                f"{neck_val:.2f}", f"{nose_val:.2f}"
            ])
    except: pass

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
        self.oled = oled
        self.lock = lock
        self.current_style = config_mgr.get('oled_icon_style', 'A')
    
    def _draw(self, func):
        if not self.oled: return
        with self.lock:
            try:
                img = Image.new("1", (128, 64))
                func(ImageDraw.Draw(img))
                self.oled.image(img); self.oled.show()
            except: pass

    # STYLE A: Human
    def draw_normal_A(self):
        def draw_func(d):
            d.ellipse((24, 8, 104, 56), outline=255, width=2)
            d.arc((35, 20, 55, 35), 0, 180, fill=255, width=3)
            d.arc((73, 20, 93, 35), 0, 180, fill=255, width=3)
            d.arc((44, 28, 84, 50), 0, 180, fill=255, width=3)
            d.ellipse((20, 35, 28, 40), outline=255, width=1)
            d.ellipse((100, 35, 108, 40), outline=255, width=1)
        self._draw(draw_func)

    def draw_angry_A(self):
        def draw_func(d):
            d.ellipse((24, 8, 104, 56), outline=255, width=2)
            d.line((32, 18, 50, 23), fill=255, width=3)
            d.line((78, 23, 96, 18), fill=255, width=3)
            d.ellipse((38, 26, 48, 36), fill=255)
            d.ellipse((80, 26, 90, 36), fill=255)
            d.arc((44, 42, 84, 58), 180, 360, fill=255, width=3)
        self._draw(draw_func)

    # STYLE B: Cat
    def draw_normal_B(self):
        def draw_func(d):
            # Ears
            d.polygon([(30, 10), (40, 2), (45, 15)], outline=255, fill=0)
            d.polygon([(83, 15), (88, 2), (98, 10)], outline=255, fill=0)
            d.polygon([(34, 10), (40, 5), (42, 12)], fill=255)
            d.polygon([(86, 12), (88, 5), (94, 10)], fill=255)
            # Face
            d.ellipse((32, 12, 96, 58), outline=255, width=2)
            # Eyes
            d.arc((42, 24, 58, 36), 0, 180, fill=255, width=2)
            d.arc((70, 24, 86, 36), 0, 180, fill=255, width=2)
            # Nose
            d.polygon([(62, 38), (66, 38), (64, 42)], fill=255)
            # Mouth
            d.arc((48, 40, 64, 52), 0, 180, fill=255, width=2)
            d.arc((64, 40, 80, 52), 0, 180, fill=255, width=2)
            # Whiskers
            d.line((20, 35, 32, 33), fill=255, width=1)
            d.line((20, 40, 32, 40), fill=255, width=1)
            d.line((20, 45, 32, 47), fill=255, width=1)
            d.line((96, 33, 108, 35), fill=255, width=1)
            d.line((96, 40, 108, 40), fill=255, width=1)
            d.line((96, 47, 108, 45), fill=255, width=1)
        self._draw(draw_func)

    def draw_angry_B(self):
        def draw_func(d):
            # Flat ears
            d.polygon([(25, 15), (35, 8), (42, 18)], outline=255, fill=0)
            d.polygon([(86, 18), (93, 8), (103, 15)], outline=255, fill=0)
            d.ellipse((32, 12, 96, 58), outline=255, width=2)
            # Angry eyes
            d.line((42, 30, 58, 26), fill=255, width=3)
            d.line((70, 26, 86, 30), fill=255, width=3)
            d.ellipse((48, 28, 52, 32), fill=255)
            d.ellipse((76, 28, 80, 32), fill=255)
            # Nose
            d.polygon([(62, 38), (66, 38), (64, 42)], fill=255)
            # Hissing mouth
            d.ellipse((52, 42, 76, 54), outline=255, width=2)
            d.polygon([(56, 44), (58, 44), (57, 50)], fill=255)
            d.polygon([(70, 44), (72, 44), (71, 50)], fill=255)
            # Whiskers
            d.line((18, 32, 32, 36), fill=255, width=1)
            d.line((18, 38, 32, 40), fill=255, width=1)
            d.line((96, 36, 110, 32), fill=255, width=1)
            d.line((96, 40, 110, 38), fill=255, width=1)
        self._draw(draw_func)

    # STYLE C: Robot
    def draw_normal_C(self):
        def draw_func(d):
            d.rounded_rectangle((20, 8, 108, 56), radius=8, outline=255, width=2)
            d.line((64, 8, 64, 2), fill=255, width=2)
            d.ellipse((61, 0, 67, 4), fill=255)
            # Eyes
            d.rectangle((32, 20, 52, 32), outline=255, width=2)
            d.rectangle((36, 24, 48, 28), fill=255)
            d.rectangle((76, 20, 96, 32), outline=255, width=2)
            d.rectangle((80, 24, 92, 28), fill=255)
            # Smile
            d.line((38, 44, 48, 44), fill=255, width=2)
            d.line((48, 44, 52, 46), fill=255, width=2)
            d.line((52, 46, 76, 46), fill=255, width=2)
            d.line((76, 46, 80, 44), fill=255, width=2)
            d.line((80, 44, 90, 44), fill=255, width=2)
            # Side panels
            d.rectangle((12, 24, 18, 40), outline=255, width=1)
            d.rectangle((110, 24, 116, 40), outline=255, width=1)
            d.ellipse((14, 28, 16, 30), fill=255)
            d.ellipse((14, 34, 16, 36), fill=255)
            d.ellipse((112, 28, 114, 30), fill=255)
            d.ellipse((112, 34, 114, 36), fill=255)
        self._draw(draw_func)

    def draw_angry_C(self):
        def draw_func(d):
            d.rounded_rectangle((20, 8, 108, 56), radius=8, outline=255, width=2)
            d.line((64, 8, 64, 2), fill=255, width=2)
            d.polygon([(61, 2), (67, 2), (64, -2)], outline=255)
            # X eyes
            d.line((32, 20, 52, 32), fill=255, width=2)
            d.line((52, 20, 32, 32), fill=255, width=2)
            d.rectangle((32, 20, 52, 32), outline=255, width=2)
            d.line((76, 20, 96, 32), fill=255, width=2)
            d.line((96, 20, 76, 32), fill=255, width=2)
            d.rectangle((76, 20, 96, 32), outline=255, width=2)
            # Frown
            d.line((38, 46, 48, 46), fill=255, width=2)
            d.line((48, 46, 52, 44), fill=255, width=2)
            d.line((52, 44, 76, 44), fill=255, width=2)
            d.line((76, 44, 80, 46), fill=255, width=2)
            d.line((80, 46, 90, 46), fill=255, width=2)
            # Alert panels
            d.rectangle((12, 24, 18, 40), outline=255, width=2)
            d.rectangle((110, 24, 116, 40), outline=255, width=2)
            d.rectangle((13, 27, 17, 31), fill=255)
            d.rectangle((13, 35, 17, 39), fill=255)
            d.rectangle((111, 27, 115, 31), fill=255)
            d.rectangle((111, 35, 115, 39), fill=255)
        self._draw(draw_func)

    def draw_timer(self, time_str):
        def draw_func(d):
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
            except:
                font = ImageFont.load_default()
            bbox = d.textbbox((0, 0), time_str, font=font)
            x = (128 - (bbox[2] - bbox[0])) // 2
            y = (64 - (bbox[3] - bbox[1])) // 2
            d.text((x, y), time_str, font=font, fill=255)
            d.rectangle((10, 10, 20, 20), outline=255, width=1)
            d.line((15, 10, 15, 15), fill=255, width=1)
            d.line((15, 15, 18, 18), fill=255, width=1)
        self._draw(draw_func)

    def draw_normal(self):
        if self.current_style == 'B':
            self.draw_normal_B()
        elif self.current_style == 'C':
            self.draw_normal_C()
        else:
            self.draw_normal_A()
    
    def draw_angry(self):
        if self.current_style == 'B':
            self.draw_angry_B()
        elif self.current_style == 'C':
            self.draw_angry_C()
        else:
            self.draw_angry_A()
    
    def update_style(self, new_style):
        if new_style in ['A', 'B', 'C']:
            self.current_style = new_style

face_display = FaceDisplay(oled, oled_lock)
if oled: face_display.draw_normal()

TTS_CACHE_DIR = "/tmp/tts_cache"; os.makedirs(TTS_CACHE_DIR, exist_ok=True)

def get_alert_message(key):
    lang = config_mgr.get('alert_language', 'vi')
    messages = config_mgr.get(f'alert_messages_{lang}', [])
    idx_map = {'lean': 0, 'hunch': 1, 'tilt': 2, 'close': 3}
    if messages and key in idx_map and len(messages) > idx_map[key]:
        return messages[idx_map[key]], lang
    fallback = {'lean': "Ngồi thẳng lại", 'hunch': "Đừng gù lưng", 'tilt': "Đừng nghiêng đầu", 'close': "Ngồi xa ra"}
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
    threading.Thread(target=lambda: subprocess.run(['mpg123', filepath], capture_output=True), daemon=True).start()
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
        
        NECK_SHRINK_TOLERANCE = config_mgr.get('neck_threshold', 35.0) / 100.0 * 2.5 
        if NECK_SHRINK_TOLERANCE > 1.0 or NECK_SHRINK_TOLERANCE < 0.5: NECK_SHRINK_TOLERANCE = 0.80
        NOSE_DROP_THRESHOLD = config_mgr.get('nose_drop_threshold', 0.15)

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
                cv2.putText(frame, "SIT STRAIGHT", (20, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                cv2.putText(frame, msg, (20, h//2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                
                if len(calib_neck) >= CALIBRATION_FRAMES:
                    base_neck_ratio = np.mean(calib_neck)
                    base_nose_ear_diff = np.mean(calib_nose)
                    is_calibrated = True
                    speak_alert("close")
            else:
                cv2.putText(frame, "NO PERSON", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            with lock: outputFrame = frame.copy(); continue

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
                    ai_conf = np.max(posture_model.predict_proba(X_in)[0])
                
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
                log_counter += 1
                if log_counter % 10 == 0: log_data(phys_label, ai_label, ai_conf, raw_status, neck_change, nose_drop_amount)

            except: method="Err"
        else: method="NoPerson"
        
        status_buf.append(raw_status)
        final_status = "Bad" if status_buf.count("Bad") >= 3 else "Good"
        
        if dist_stat == "TOO CLOSE":
             if time.time()-last_alert>10: speak_alert('close'); last_alert=time.time()
        elif final_status == "Bad":
            bad_color = config_mgr.get('led_color_bad', [255, 0, 0])
            if not study_timer.is_running: led_controller.set_color_array(bad_color)
            
            face_display.draw_angry()
            
            if time.time()-last_bad>1.5:
                if time.time()-last_alert>8:
                    speak_alert(detected_type if detected_type else 'lean')
                    last_alert=time.time()
            else:
                if last_bad==0: last_bad=time.time()
        else:
            good_color = config_mgr.get('led_color_good', [0, 255, 0])
            if not study_timer.is_running: led_controller.set_color_array(good_color)
            
            face_display.draw_normal()
            last_bad=0
            
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
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
