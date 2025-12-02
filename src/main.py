#!/usr/bin/env python3
import time
from collections import deque
import numpy as np
import cv2
import math
import sys
import threading
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
from utils import extract_features

def init_audio():
    try:
        subprocess.run(['amixer', 'set', 'Master', '100%'], check=False, capture_output=True)
        subprocess.run(['amixer', 'set', 'PCM', '100%'], check=False, capture_output=True)
        result = subprocess.run(['which', 'mpg123'], capture_output=True)
        if result.returncode != 0:
            return False
        return True
    except:
        return False

audio_available = init_audio()

try:
    from tflite_runtime.interpreter import Interpreter
except:
    from tensorflow.lite.python.interpreter import Interpreter

MODEL_PATH = "/home/theo/4.tflite"
CAMERA_ID = 0
NECK_THRESHOLD = 35
SMOOTHING_FRAMES = 7
BAD_DURATION_TO_ALERT = 2.0
INFER_THREADS = 2
FACE_TOO_CLOSE_RATIO = 0.35
FACE_TOO_FAR_RATIO = 0.08
LED_COUNT = 12
LED_PIN = 12
LED_BRIGHTNESS = 20
LED_FREQ_HZ = 800000
LED_DMA = 10
LED_INVERT = False
LED_CHANNEL = 0

# Load ML Model 
try:
    posture_model = joblib.load("/home/theo/smart-posture-assistant/models/posture_model.pkl")
    posture_scaler = joblib.load("/home/theo/smart-posture-assistant/models/scaler.pkl")
    use_ml_model = True
    print("✓ ML Model loaded successfully")
except Exception as e:
    print(f"⚠ ML Model not found: {e}")
    print("Using angle-based detection only")
    posture_model = None
    posture_scaler = None
    use_ml_model = False

interpreter = Interpreter(MODEL_PATH, num_threads=INFER_THREADS)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_dtype = input_details[0]['dtype']
in_h = input_details[0]['shape'][1]
in_w = input_details[0]['shape'][2]

try:
    i2c = busio.I2C(SCL, SDA)
    oled = SSD1306_I2C(128, 64, i2c)
    oled.fill(0)
    oled.show()
except Exception as e:
    oled = None

try:
    strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
    strip.begin()
    for i in range(LED_COUNT):
        strip.setPixelColor(i, Color(0, 0, 0))
    strip.show()
except Exception as e:
    strip = None

class StudyTimer:
    def __init__(self):
        self.duration = 0
        self.remaining = 0
        self.is_running = False
        self.is_paused = False
        self.start_time = 0
        self.lock = threading.Lock()
        
    def start(self, minutes):
        with self.lock:
            self.duration = minutes * 60
            self.remaining = self.duration
            self.is_running = True
            self.is_paused = False
            self.start_time = time.time()
            
    def pause(self):
        with self.lock:
            if self.is_running and not self.is_paused:
                self.is_paused = True
                
    def resume(self):
        with self.lock:
            if self.is_running and self.is_paused:
                self.is_paused = False
                self.start_time = time.time()
                
    def stop(self):
        with self.lock:
            self.is_running = False
            self.is_paused = False
            self.remaining = 0
            
    def update(self):
        with self.lock:
            if self.is_running and not self.is_paused:
                elapsed = time.time() - self.start_time
                self.remaining = max(0, self.duration - elapsed)
                if self.remaining == 0:
                    self.is_running = False
                    return True
            return False
            
    def get_time_str(self):
        with self.lock:
            if not self.is_running:
                return ""
            mins = int(self.remaining // 60)
            secs = int(self.remaining % 60)
            return f"{mins:02d}:{secs:02d}"

study_timer = StudyTimer()

class FaceDisplay:
    def __init__(self, oled_display):
        self.oled = oled_display
        self.current_state = "normal"
        
    def draw_normal_face(self):
        if not self.oled:
            return
        image = Image.new("1", (128, 64))
        draw = ImageDraw.Draw(image)
        draw.ellipse((30, 15, 50, 35), outline=255, fill=0)
        draw.ellipse((35, 20, 45, 30), outline=255, fill=255)
        draw.ellipse((78, 15, 98, 35), outline=255, fill=0)
        draw.ellipse((83, 20, 93, 30), outline=255, fill=255)
        draw.arc((40, 35, 88, 55), 0, 180, fill=255, width=2)
        self.oled.image(image)
        self.oled.show()
        self.current_state = "normal"
    
    def draw_angry_face(self):
        if not self.oled:
            return
        image = Image.new("1", (128, 64))
        draw = ImageDraw.Draw(image)
        draw.line((25, 15, 45, 20), fill=255, width=2)
        draw.ellipse((30, 18, 50, 38), outline=255, fill=0)
        draw.ellipse((35, 23, 45, 33), outline=255, fill=255)
        draw.line((83, 20, 103, 15), fill=255, width=2)
        draw.ellipse((78, 18, 98, 38), outline=255, fill=0)
        draw.ellipse((83, 23, 93, 33), outline=255, fill=255)
        draw.arc((40, 45, 88, 58), 180, 360, fill=255, width=2)
        self.oled.image(image)
        self.oled.show()
        self.current_state = "angry"
    
    def draw_timer(self, time_str):
        if not self.oled:
            return
        image = Image.new("1", (128, 64))
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        except:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), time_str, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (128 - text_width) // 2
        y = (64 - text_height) // 2
        draw.text((x, y), time_str, font=font, fill=255)
        self.oled.image(image)
        self.oled.show()
    
    def blink(self):
        if not self.oled:
            return
        image = Image.new("1", (128, 64))
        draw = ImageDraw.Draw(image)
        draw.line((30, 25, 50, 25), fill=255, width=2)
        draw.line((78, 25, 98, 25), fill=255, width=2)
        if self.current_state == "normal":
            draw.arc((40, 35, 88, 55), 0, 180, fill=255, width=2)
        else:
            draw.arc((40, 45, 88, 58), 180, 360, fill=255, width=2)
        self.oled.image(image)
        self.oled.show()

class LEDController:
    def __init__(self, led_strip):
        self.strip = led_strip
        self.current_color = (255, 200, 0)
        self.is_blinking = False
        self.lock = threading.Lock()
        
    def set_color(self, r, g, b):
        if not self.strip:
            return
        with self.lock:
            for i in range(LED_COUNT):
                self.strip.setPixelColor(i, Color(g, r, b))
            self.strip.show()
            self.current_color = (r, g, b)
    
    def set_yellow(self):
        self.set_color(255, 200, 0)
    
    def set_red(self):
        self.set_color(255, 0, 0)
    
    def pulse_while_speaking(self, duration=4):
        if not self.strip:
            return
        self.is_blinking = True
        start_time = time.time()
        base_color = self.current_color
        while time.time() - start_time < duration and self.is_blinking:
            for brightness in range(30, 100, 15):
                if not self.is_blinking:
                    break
                with self.lock:
                    for i in range(LED_COUNT):
                        r, g, b = base_color
                        dimmed_r = int(r * brightness / 100)
                        dimmed_g = int(g * brightness / 100)
                        dimmed_b = int(b * brightness / 100)
                        self.strip.setPixelColor(i, Color(dimmed_g, dimmed_r, dimmed_b))
                    self.strip.show()
                time.sleep(0.05)
            for brightness in range(100, 30, -15):
                if not self.is_blinking:
                    break
                with self.lock:
                    for i in range(LED_COUNT):
                        r, g, b = base_color
                        dimmed_r = int(r * brightness / 100)
                        dimmed_g = int(g * brightness / 100)
                        dimmed_b = int(b * brightness / 100)
                        self.strip.setPixelColor(i, Color(dimmed_g, dimmed_r, dimmed_b))
                    self.strip.show()
                time.sleep(0.05)
        self.set_color(*base_color)
        self.is_blinking = False

face_display = FaceDisplay(oled)
led_controller = LEDController(strip)

if oled:
    face_display.draw_normal_face()
if strip:
    led_controller.set_yellow()

def calculate_neck_angle(left_shoulder, right_shoulder, mouth):
    shoulder_vector = right_shoulder - left_shoulder
    mid_shoulder = (left_shoulder + right_shoulder) / 2
    neck_vector = mouth - mid_shoulder
    dot_product = np.dot(shoulder_vector, neck_vector)
    norm_product = np.linalg.norm(shoulder_vector) * np.linalg.norm(neck_vector)
    if norm_product == 0:
        return 0
    cosine_angle = np.clip(dot_product / norm_product, -1.0, 1.0)
    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)
    neck_angle = abs(90 - angle_deg)
    return neck_angle

TTS_CACHE_DIR = "/tmp/tts_cache"
os.makedirs(TTS_CACHE_DIR, exist_ok=True)

def get_cached_tts(message):
    import hashlib
    msg_hash = hashlib.md5(message.encode()).hexdigest()
    cache_file = os.path.join(TTS_CACHE_DIR, f"{msg_hash}.mp3")
    if not os.path.exists(cache_file):
        tts = gTTS(text=message, lang='vi', slow=False)
        tts.save(cache_file)
    return cache_file

def speak_alert(message):
    if not audio_available:
        return
    def _speak():
        try:
            tts_file = get_cached_tts(message)
            led_thread = threading.Thread(target=led_controller.pulse_while_speaking, args=(5,), daemon=True)
            led_thread.start()
            blink_thread = threading.Thread(target=blink_while_speaking, args=(5,), daemon=True)
            blink_thread.start()
            subprocess.run(['mpg123', tts_file], capture_output=True)
            time.sleep(0.5)
            led_controller.is_blinking = False
        except:
            pass
    threading.Thread(target=_speak, daemon=True).start()

def blink_while_speaking(duration):
    start_time = time.time()
    while time.time() - start_time < duration:
        face_display.blink()
        time.sleep(0.15)
        if face_display.current_state == "normal":
            face_display.draw_normal_face()
        else:
            face_display.draw_angry_face()
        time.sleep(0.3)

def timer_updater():
    last_update = 0
    while True:
        if study_timer.update():
            speak_alert("Hết giờ học rồi, hãy nghỉ giải lao nhé")
            if oled:
                face_display.draw_normal_face()
        current_time = time.time()
        if study_timer.is_running and oled and current_time - last_update >= 1:
            time_str = study_timer.get_time_str()
            if time_str and face_display.current_state == "normal":
                face_display.draw_timer(time_str)
            last_update = current_time
        time.sleep(0.5)

angle_buf = deque(maxlen=SMOOTHING_FRAMES)
last_bad_time = None
alert_playing = False
last_alert_time = 0
ALERT_MESSAGES = [
    "Bạn đang cúi đầu quá thấp, hãy ngồi thẳng lại",
    "Tư thế ngồi của bạn không đúng, giữ thẳng lưng nhé",
    "Hãy giữ đầu thẳng với cột sống"
]
alert_index = 0

cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    sys.exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

app = Flask(__name__)
outputFrame = None
lock = threading.Lock()
stats = {'posture_status': 'Good', 'neck_angle': 0, 'distance_status': 'OK', 'fps': 0, 'ml_confidence': 0}
last_display_text = ""
last_distance_status = "OK"
last_fps_text = ""

for msg in ALERT_MESSAGES:
    try:
        get_cached_tts(msg)
    except:
        pass
get_cached_tts("Bạn đang ngồi quá gần màn hình, hãy lùi ra xa một chút")
get_cached_tts("Hết giờ học rồi, hãy nghỉ giải lao nhé")
get_cached_tts("Đã bắt đầu học 15 phút")
get_cached_tts("Đã bắt đầu học 30 phút")
get_cached_tts("Đã bắt đầu học 45 phút")
get_cached_tts("Đã tạm dừng")
get_cached_tts("Tiếp tục nào")
get_cached_tts("Đã dừng")

def detect_posture():
    global outputFrame, last_bad_time, alert_playing
    global last_display_text, last_distance_status, last_fps_text
    global last_alert_time, alert_index
    frame_count = 0
    start_time = time.time()
    
    # Thêm buffer cho dự đoán ML
    ml_prediction_buf = deque(maxlen=5)  # Làm mượt dự đoán ML
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_count += 1
        h, w, _ = frame.shape
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (in_w, in_h), interpolation=cv2.INTER_NEAREST)
        if input_dtype == np.float32:
            inp = img_resized.astype(np.float32)
            inp = (inp - 127.5) / 127.5
            inp = np.expand_dims(inp, axis=0)
        else:
            inp = np.expand_dims(img_resized.astype(input_dtype), axis=0)
        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        kpts = output_data[0][0] if output_data.ndim == 4 else output_data[0]
        nose = np.array([kpts[0][1] * w, kpts[0][0] * h])
        s0 = kpts[0][2]
        left_eye = np.array([kpts[1][1] * w, kpts[1][0] * h])
        s_le = kpts[1][2]
        right_eye = np.array([kpts[2][1] * w, kpts[2][0] * h])
        s_re = kpts[2][2]
        lear = np.array([kpts[3][1] * w, kpts[3][0] * h])
        s_lear = kpts[3][2]
        rear = np.array([kpts[4][1] * w, kpts[4][0] * h])
        s_rear = kpts[4][2]
        ls = np.array([kpts[5][1] * w, kpts[5][0] * h])
        s1 = kpts[5][2]
        rs = np.array([kpts[6][1] * w, kpts[6][0] * h])
        s2 = kpts[6][2]
        min_conf = 0.25
        key_scores = [s0, s1, s2, s_le, s_re]
        distance_status = "OK"
        if s0 > min_conf:
            if s_lear > min_conf and s_rear > min_conf:
                face_width = np.linalg.norm(lear - rear)
            else:
                face_width = w * 0.15
            face_ratio = face_width / w
            if face_ratio > FACE_TOO_CLOSE_RATIO:
                distance_status = "TOO CLOSE"
                current_time = time.time()
                if current_time - last_alert_time > 15:
                    speak_alert("Bạn đang ngồi quá gần màn hình, hãy lùi ra xa một chút")
                    last_alert_time = current_time
            elif face_ratio < FACE_TOO_FAR_RATIO:
                distance_status = "Too far"
        if sum(s > min_conf for s in key_scores) < 3:
            posture_status = "Unknown"
            smoothed = 0
            display_text = "No person"
            ml_confidence = 0
        else:
            if s_le > min_conf and s_re > min_conf:
                eye_distance = np.linalg.norm(left_eye - right_eye)
                mouth_offset_y = eye_distance * 0.6
                mid_mouth = nose + np.array([0, mouth_offset_y])
            elif s_lear > min_conf and s_rear > min_conf:
                ear_distance = np.linalg.norm(lear - rear)
                mouth_offset_y = ear_distance * 0.4
                mid_mouth = nose + np.array([0, mouth_offset_y])
            else:
                mid_mouth = nose + np.array([0, np.linalg.norm(ls - rs) * 0.4])
            angle_val = calculate_neck_angle(ls, rs, mid_mouth)
            angle_buf.append(angle_val)
            smoothed = float(np.mean(angle_buf))
            
            ml_confidence = 0
            if use_ml_model and posture_model is not None:
                try:
                    features = extract_features(kpts, w, h)
                    features_array = np.array(features).reshape(1, -1)
                    features_scaled = posture_scaler.transform(features_array)
                    ml_prediction = posture_model.predict(features_scaled)[0]
                    ml_confidence = posture_model.predict_proba(features_scaled)[0].max()
                    
            
                    ml_prediction_buf.append(ml_prediction)
                    
                   
                    if len(ml_prediction_buf) >= 3:
                        from collections import Counter
                        final_prediction = Counter(ml_prediction_buf).most_common(1)[0][0]
                    else:
                        final_prediction = ml_prediction               
                    if ml_confidence > 0.7:  
                        posture_status = "Good" if final_prediction == 'good' else "Bad"
                        method = "ML"
                    else:
                        posture_status = "Good" if smoothed <= NECK_THRESHOLD else "Bad"
                        method = "Angle"
                        
                except Exception as e:
                    print(f"ML prediction error: {e}")
                    posture_status = "Good" if smoothed <= NECK_THRESHOLD else "Bad"
                    method = "Angle (fallback)"
            else:
                posture_status = "Good" if smoothed <= NECK_THRESHOLD else "Bad"
                method = "Angle"
            
            mid_shoulder = (ls + rs) / 2
            if frame_count % 2 == 0:
                cv2.line(frame, tuple(ls.astype(int)), tuple(rs.astype(int)), (255, 0, 255), 2)
                cv2.line(frame, tuple(mid_shoulder.astype(int)), tuple(mid_mouth.astype(int)), (0, 255, 0), 3)
                cv2.circle(frame, tuple(mid_shoulder.astype(int)), 5, (0, 255, 255), -1)
                cv2.circle(frame, tuple(mid_mouth.astype(int)), 5, (255, 0, 255), -1)
                cv2.circle(frame, tuple(ls.astype(int)), 6, (255, 255, 0), -1)
                cv2.circle(frame, tuple(rs.astype(int)), 6, (255, 255, 0), -1)
            
            method_text = f"Method: {method}"
            cv2.putText(frame, method_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            if posture_status == "Bad":
                display_text = f"BAD: {smoothed:.1f}deg"
                if face_display.current_state != "angry":
                    face_display.draw_angry_face()
                if not study_timer.is_running:
                    led_controller.set_red()
                if last_bad_time is None:
                    last_bad_time = time.time()
                elapsed = time.time() - last_bad_time
                if elapsed >= BAD_DURATION_TO_ALERT:
                    current_time = time.time()
                    if current_time - last_alert_time > 10:
                        speak_alert(ALERT_MESSAGES[alert_index])
                        alert_index = (alert_index + 1) % len(ALERT_MESSAGES)
                        last_alert_time = current_time
                        alert_playing = True
            else:
                display_text = f"GOOD: {smoothed:.1f}deg"
                led_controller.is_blinking = False
                if face_display.current_state != "normal":
                    face_display.draw_normal_face()
                if not study_timer.is_running:
                    time.sleep(0.1)
                    led_controller.set_yellow()
                last_bad_time = None
                alert_playing = False
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        if display_text != last_display_text:
            last_display_text = display_text
        if distance_status != last_distance_status:
            last_distance_status = distance_status
        if frame_count % 5 == 0:
            last_fps_text = f"FPS: {fps:.1f}"
        if posture_status == "Bad":
            cv2.putText(frame, last_display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif posture_status == "Good":
            cv2.putText(frame, last_display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, last_display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Distance: {last_distance_status}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, last_fps_text, (w-120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if use_ml_model:
            cv2.putText(frame, f"ML Conf: {ml_confidence:.2f}", (w-120, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if study_timer.is_running:
            time_str = study_timer.get_time_str()
            cv2.putText(frame, f"Timer: {time_str}", (w//2-80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        stats.update({
            'posture_status': posture_status, 
            'neck_angle': f"{smoothed:.1f}", 
            'distance_status': distance_status, 
            'fps': f"{fps:.1f}",
            'ml_confidence': f"{ml_confidence:.2f}"
        })
        with lock:
            outputFrame = frame

def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                time.sleep(0.01)
                continue
            frame = outputFrame
        (flag, encodedImage) = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 55])
        if not flag:
            continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stats")
def get_stats():
    return jsonify(stats)

@app.route("/command", methods=['POST'])
def command():
    data = request.json
    cmd = data.get('command', '')
    if cmd == 'start_15':
        study_timer.start(15)
        speak_alert("Đã bắt đầu học 15 phút")
    elif cmd == 'start_30':
        study_timer.start(30)
        speak_alert("Đã bắt đầu học 30 phút")
    elif cmd == 'start_45':
        study_timer.start(45)
        speak_alert("Đã bắt đầu học 45 phút")
    elif cmd == 'pause':
        study_timer.pause()
        speak_alert("Đã tạm dừng")
    elif cmd == 'resume':
        study_timer.resume()
        speak_alert("Tiếp tục nào")
    elif cmd == 'stop':
        study_timer.stop()
        speak_alert("Đã dừng")
        if oled:
            face_display.draw_normal_face()
    return jsonify({'success': True})

@app.route("/")
def index():
    return """
    <html>
      <head>
        <title>Smart Posture Assistant</title>
        <style>
          body {
            background-color: #1a1a1a;
            color: #ffffff;
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
          }
          .container {
            max-width: 1400px;
            margin: 0 auto;
          }
          h1 {
            text-align: center;
            color: #00ff88;
            margin-bottom: 30px;
          }
          .content {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
          }
          .video-container {
            flex: 1;
            min-width: 640px;
          }
          .video-container img {
            width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,255,136,0.3);
          }
          .side-panel {
            flex: 0 0 350px;
          }
          .stats-panel {
            background: #2a2a2a;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            margin-bottom: 20px;
          }
          .timer-panel {
            background: #2a2a2a;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
          }
          .stat-item {
            padding: 15px;
            margin: 10px 0;
            background: #333;
            border-radius: 5px;
            border-left: 4px solid #00ff88;
          }
          .stat-label {
            font-size: 14px;
            color: #aaa;
            margin-bottom: 5px;
          }
          .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #00ff88;
          }
          .timer-btn {
            width: 100%;
            padding: 15px;
            margin: 10px 0;
            font-size: 16px;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
          }
          .btn-start {
            background: #00ff88;
            color: #000;
          }
          .btn-start:hover {
            background: #00dd77;
            transform: scale(1.02);
          }
          .btn-control {
            background: #3a3a3a;
            color: #fff;
          }
          .btn-control:hover {
            background: #4a4a4a;
          }
          .btn-stop {
            background: #ff4444;
            color: #fff;
          }
          .btn-stop:hover {
            background: #dd3333;
          }
          .info {
            margin-top: 15px;
            padding: 15px;
            background: #2a2a2a;
            border-radius: 5px;
            font-size: 13px;
            color: #aaa;
          }
          .good { color: #00ff88 !important; }
          .bad { color: #ff4444 !important; }
          .warning { color: #ff8800 !important; }
          h2 {
            margin-top: 0;
            color: #00ff88;
            font-size: 20px;
          }
        </style>
        <script>
          function updateStats() {
            fetch('/stats')
              .then(r => r.json())
              .then(data => {
                document.getElementById('posture').textContent = data.posture_status;
                document.getElementById('angle').textContent = data.neck_angle + '°';
                document.getElementById('distance').textContent = data.distance_status;
                document.getElementById('fps').textContent = data.fps;
                document.getElementById('ml_conf').textContent = data.ml_confidence;
                const postureEl = document.getElementById('posture');
                postureEl.className = 'stat-value ' + (data.posture_status === 'Good' ? 'good' : 'bad');
                const distanceEl = document.getElementById('distance');
                distanceEl.className = 'stat-value ' + (data.distance_status === 'OK' ? 'good' : 'warning');
              });
          }
          function sendCommand(cmd) {
            fetch('/command', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({command: cmd})
            });
          }
          setInterval(updateStats, 1000);
          window.onload = updateStats;
        </script>
      </head>
      <body>
        <div class="container">
          <h1>Smart Posture Assistant</h1>
          <div class="content">
            <div class="video-container">
              <img src="/video_feed" />
            </div>
            <div class="side-panel">
              <div class="stats-panel">
                <h2>Status</h2>
                <div class="stat-item">
                  <div class="stat-label">Posture</div>
                  <div class="stat-value" id="posture">-</div>
                </div>
                <div class="stat-item">
                  <div class="stat-label">Neck Angle</div>
                  <div class="stat-value" id="angle">-</div>
                </div>
                <div class="stat-item">
                  <div class="stat-label">Distance</div>
                  <div class="stat-value" id="distance">-</div>
                </div>
                <div class="stat-item">
                  <div class="stat-label">FPS</div>
                  <div class="stat-value" id="fps">-</div>
                </div>
                <div class="stat-item">
                  <div class="stat-label">ML Confidence</div>
                  <div class="stat-value" id="ml_conf">-</div>
                </div>
              </div>
              <div class="timer-panel">
                <h2>Study Timer</h2>
                <button class="timer-btn btn-start" onclick="sendCommand('start_15')">Start 15 min</button>
                <button class="timer-btn btn-start" onclick="sendCommand('start_30')">Start 30 min</button>
                <button class="timer-btn btn-start" onclick="sendCommand('start_45')">Start 45 min</button>
                <button class="timer-btn btn-control" onclick="sendCommand('pause')">Pause</button>
                <button class="timer-btn btn-control" onclick="sendCommand('resume')">Resume</button>
                <button class="timer-btn btn-stop" onclick="sendCommand('stop')">Stop</button>
                <div class="info">
                  <strong>Instructions:</strong><br>
                  - Neck angle ≤ 35°: Good posture<br>
                  - OLED: Happy/Angry face<br>
                  - LED: Yellow (good) / Red (bad)<br>
                  - LED blinks during alert<br>
                  - Take breaks every 30 minutes
                </div>
              </div>
            </div>
          </div>
        </div>
      </body>
    </html>
    """

if __name__ == '__main__':
    print("Starting Smart Posture Assistant...")
    print("Access at: http://localhost:5000")
    t1 = threading.Thread(target=detect_posture, daemon=True)
    t1.start()
    t2 = threading.Thread(target=timer_updater, daemon=True)
    t2.start()
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)
    finally:
        if strip:
            for i in range(LED_COUNT):
                strip.setPixelColor(i, Color(0, 0, 0))
            strip.show()
        if oled:
            oled.fill(0)
            oled.show()
