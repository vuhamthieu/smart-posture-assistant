#!/usr/bin/env python3
import time
from collections import deque
import numpy as np
import cv2
import math
import sys
import threading
from flask import Flask, Response, jsonify
from gtts import gTTS
import tempfile
import os
import subprocess
from PIL import Image, ImageDraw
from board import SCL, SDA
import busio
from adafruit_ssd1306 import SSD1306_I2C
import RPi.GPIO as GPIO
from rpi_ws281x import PixelStrip, Color

def init_audio():
    """Initialize I2S audio for MAX98357A"""
    try:
        # Check available audio devices
        result = subprocess.run(['aplay', '-L'], capture_output=True, text=True)
        print("Available audio devices:")
        print(result.stdout)
        
        # Set volume
        subprocess.run(['amixer', 'set', 'Master', '100%'], check=False, capture_output=True)
        subprocess.run(['amixer', 'set', 'PCM', '100%'], check=False, capture_output=True)
        
        # Check mpg123
        result = subprocess.run(['which', 'mpg123'], capture_output=True)
        if result.returncode != 0:
            print("mpg123 not found! Install: sudo apt-get install mpg123")
            return False
        
        print("Audio initialized for MAX98357A")
        return True
    except Exception as e:
        print(f"Audio setup failed: {e}")
        return False

audio_available = init_audio()

try:
    from tflite_runtime.interpreter import Interpreter
except Exception:
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
LED_BRIGHTNESS = 80
LED_FREQ_HZ = 800000
LED_DMA = 10
LED_INVERT = False
LED_CHANNEL = 0

# MAX98357A I2S Audio Amplifier:
# LRC (WS)   -> GPIO 19 (Pin 35)
# BCLK       -> GPIO 18 (Pin 12)
# DIN (Data) -> GPIO 21 (Pin 40)
# VIN -> 5V, GND -> GND

# LED Ring WS2812B:
# DIN -> GPIO 10 (Pin 19 - SPI0 MOSI) *** MOST COMPATIBLE ***
# VCC -> 5V external, GND -> GND (common with Pi)
# Note: GPIO10 uses SPI, most stable for WS2812B with I2S audio

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
    print("OLED initialized")
except Exception as e:
    print(f"OLED init failed: {e}")
    oled = None

try:
    strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
    strip.begin()
    for i in range(LED_COUNT):
        strip.setPixelColor(i, Color(0, 0, 0))
    strip.show()
    print("LED Ring initialized")
except Exception as e:
    print(f"LED init failed: {e}")
    strip = None

class FaceDisplay:
    def __init__(self, oled_display):
        self.oled = oled_display
        self.current_state = "normal"
        
    def draw_normal_face(self):
        """Draw normal/happy face"""
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
        """Draw angry face"""
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
    
    def blink(self):
        """Blink eyes (close briefly)"""
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
        """Set all LEDs to one color"""
        if not self.strip:
            return
        with self.lock:
            # Test both RGB and GRB order
            # WS2812B typically uses GRB
            for i in range(LED_COUNT):
                self.strip.setPixelColor(i, Color(g, r, b))  # GRB order
            self.strip.show()
            self.current_color = (r, g, b)
            print(f"LED set to RGB({r},{g},{b}) -> GRB({g},{r},{b})")
    
    def set_yellow(self):
        """Good posture - Yellow"""
        print("Setting LED to YELLOW (good posture)")
        self.set_color(255, 200, 0)  # RGB: Red+Green = Yellow
    
    def set_red(self):
        """Bad posture - Red"""
        print("Setting LED to RED (bad posture)")
        self.set_color(0, 255, 0)  # RGB: Full Red
    
    def pulse_while_speaking(self, duration=4):
        """Pulse LEDs while speaking"""
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

def neck_angle(hip, shoulder, head, shoulder_ref):
    """Calculate neck angle similar to MediaPipe method"""
    vec1 = hip - shoulder_ref
    vec2 = head - shoulder_ref
    denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if denom == 0:
        return 0.0
    cosang = np.dot(vec1, vec2) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    angle = math.degrees(math.acos(cosang))
    return abs(angle - 180.0)

def speak_alert(message):
    """Speak alert in Vietnamese using cached TTS with LED/OLED effects"""
    if not audio_available:
        print(f"Audio not available, message: {message}")
        return
    
    def _speak():
        try:
            print(f"TTS: {message[:30]}...")
            
            # Use cached TTS file (instant!)
            tts_file = get_cached_tts(message)
            
            led_thread = threading.Thread(target=led_controller.pulse_while_speaking, args=(5,), daemon=True)
            led_thread.start()
            
            blink_thread = threading.Thread(target=blink_while_speaking, args=(5,), daemon=True)
            blink_thread.start()
            
            # Play cached file
            result = subprocess.run(['mpg123', tts_file], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"mpg123 stderr: {result.stderr}")
            
            time.sleep(0.5)
            led_controller.is_blinking = False
            
        except Exception as e:
            print(f"TTS error: {e}")
            import traceback
            traceback.print_exc()
    
    thread = threading.Thread(target=_speak, daemon=True)
    thread.start()

def blink_while_speaking(duration):
    """Blink OLED eyes while speaking"""
    start_time = time.time()
    while time.time() - start_time < duration:
        face_display.blink()
        time.sleep(0.15)
        if face_display.current_state == "normal":
            face_display.draw_normal_face()
        else:
            face_display.draw_angry_face()
        time.sleep(0.3)

angle_buf = deque(maxlen=SMOOTHING_FRAMES)
last_bad_time = None
alert_playing = False
last_alert_time = 0

cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print("Cannot open camera")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

app = Flask(__name__)
outputFrame = None
lock = threading.Lock()

stats = {
    'posture_status': 'Good',
    'neck_angle': 0,
    'distance_status': 'OK',
    'fps': 0
}

last_display_text = ""
last_distance_status = "OK"
last_fps_text = ""

ALERT_MESSAGES = [
    "Bạn đang cúi đầu quá thấp, hãy ngồi thẳng lại",
    "Tư thế ngồi của bạn không đúng, giữ thẳng lưng nhé",
    "Hãy giữ đầu thẳng với cột sống"
]
alert_index = 0

# Cache TTS files to reduce delay
TTS_CACHE_DIR = "/tmp/tts_cache"
os.makedirs(TTS_CACHE_DIR, exist_ok=True)

def get_cached_tts(message):
    """Get or create cached TTS file"""
    import hashlib
    msg_hash = hashlib.md5(message.encode()).hexdigest()
    cache_file = os.path.join(TTS_CACHE_DIR, f"{msg_hash}.mp3")
    
    if not os.path.exists(cache_file):
        print(f"Creating TTS cache for: {message[:30]}...")
        tts = gTTS(text=message, lang='vi', slow=False)
        tts.save(cache_file)
    
    return cache_file

# Pre-cache alert messages at startup
print("Pre-caching TTS messages...")
for msg in ALERT_MESSAGES:
    try:
        get_cached_tts(msg)
    except:
        pass
get_cached_tts("Bạn đang ngồi quá gần màn hình, hãy lùi ra xa một chút")
print("TTS cache ready!")

def detect_posture():
    global outputFrame, last_bad_time, alert_playing, stats
    global last_display_text, last_distance_status, last_fps_text
    global last_alert_time, alert_index
    
    frame_count = 0
    start_time = time.time()
    
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
        lh = np.array([kpts[11][1] * w, kpts[11][0] * h])
        s3 = kpts[11][2]
        rh = np.array([kpts[12][1] * w, kpts[12][0] * h])
        s4 = kpts[12][2]

        min_conf = 0.25
        key_scores = [s0, s1, s2, s3, s4]
        
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
        else:
            mid_shoulder = (ls + rs) * 0.5
            mid_hip = (lh + rh) * 0.5
            
            # Use eyes to estimate mouth position (like MediaPipe)
            # Mouth is approximately below nose at distance = eye_distance * 0.6
            if s_le > min_conf and s_re > min_conf:
                eye_distance = np.linalg.norm(left_eye - right_eye)
                # Estimate mouth position: below nose
                mouth_offset_y = eye_distance * 0.6
                mid_mouth = nose + np.array([0, mouth_offset_y])
            else:
                # Fallback: use ears
                if s_lear > min_conf and s_rear > min_conf:
                    ear_distance = np.linalg.norm(lear - rear)
                    mouth_offset_y = ear_distance * 0.4
                    mid_mouth = nose + np.array([0, mouth_offset_y])
                else:
                    # Last resort: estimate from shoulder width
                    mid_mouth = nose + np.array([0, np.linalg.norm(ls - rs) * 0.4])
            
            # Calculate angle like MediaPipe: hip->shoulder, mouth->shoulder
            angle_val = neck_angle(mid_hip, mid_shoulder, mid_mouth, mid_shoulder)
            angle_buf.append(angle_val)
            smoothed = float(np.mean(angle_buf))
            
            if frame_count % 2 == 0:
                cv2.line(frame, tuple(mid_hip.astype(int)), tuple(mid_shoulder.astype(int)), (0, 255, 0), 2)
                cv2.line(frame, tuple(mid_shoulder.astype(int)), tuple(mid_mouth.astype(int)), (0, 255, 0), 2)
                cv2.circle(frame, tuple(mid_mouth.astype(int)), 5, (255, 0, 255), -1)
            
            if smoothed > NECK_THRESHOLD:
                posture_status = "Bad"
                display_text = f"BAD POSTURE: {smoothed:.1f}deg"
                
                if face_display.current_state != "angry":
                    face_display.draw_angry_face()
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
                posture_status = "Good"
                display_text = f"GOOD: {smoothed:.1f}deg"
                
                if face_display.current_state != "normal":
                    face_display.draw_normal_face()
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
            cv2.putText(frame, last_display_text, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif posture_status == "Good":
            cv2.putText(frame, last_display_text, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, last_display_text, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Distance: {last_distance_status}", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.putText(frame, last_fps_text, (w-120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if last_distance_status == "TOO CLOSE":
            cv2.putText(frame, "Move back from screen!", (w//2-140, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif last_distance_status == "Too far":
            cv2.putText(frame, "Move closer to screen", (w//2-130, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        stats.update({
            'posture_status': posture_status,
            'neck_angle': f"{smoothed:.1f}",
            'distance_status': distance_status,
            'fps': f"{fps:.1f}"
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
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stats")
def get_stats():
    return jsonify(stats)

@app.route("/")
def index():
    return """
    <html>
      <head>
        <title>Posture Assistant</title>
        <style>
          body {
            background-color: #1a1a1a;
            color: #ffffff;
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
          }
          .container {
            max-width: 1200px;
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
          .stats-panel {
            flex: 0 0 300px;
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
          .info {
            margin-top: 20px;
            padding: 15px;
            background: #2a2a2a;
            border-radius: 5px;
            font-size: 14px;
            color: #aaa;
          }
          .good { color: #00ff88 !important; }
          .bad { color: #ff4444 !important; }
          .warning { color: #ff8800 !important; }
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
                
                const postureEl = document.getElementById('posture');
                postureEl.className = 'stat-value ' + (data.posture_status === 'Good' ? 'good' : 'bad');
                
                const distanceEl = document.getElementById('distance');
                distanceEl.className = 'stat-value ' + (data.distance_status === 'OK' ? 'good' : 'warning');
              });
          }
          setInterval(updateStats, 1000);
          window.onload = updateStats;
        </script>
      </head>
      <body>
        <div class="container">
          <h1>Smart Posture Assistant with OLED & LED</h1>
          <div class="content">
            <div class="video-container">
              <img src="/video_feed" />
            </div>
            <div class="stats-panel">
              <h2 style="margin-top:0; color:#00ff88;">Live Status</h2>
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
              <div class="info">
                <strong>Instructions:</strong><br>
                • Good: neck angle under 35 degrees<br>
                • OLED: Happy/Angry face<br>
                • LED: Yellow (good) / Red (bad)<br>
                • LED blinks while speaking<br>
                • Rest every 20 minutes
              </div>
            </div>
          </div>
        </div>
      </body>
    </html>
    """

if __name__ == '__main__':
    print("Starting Smart Posture Assistant with OLED & LED...")
    print("Access at: http://localhost:5000")
    
    t = threading.Thread(target=detect_posture)
    t.daemon = True
    t.start()
    
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
