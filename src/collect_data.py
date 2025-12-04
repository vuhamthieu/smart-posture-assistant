#!/usr/bin/env python3
"""
Posture Data Collection Tool - Web Interface
Access at http://raspberrypi.local:5001
Press buttons on web page to label: Good / Bad / Skip
"""
import cv2
import numpy as np
import csv
import os
import time
import threading
from flask import Flask, Response, render_template_string
from utils import extract_features, validate_keypoints, get_feature_names

try:
    from tflite_runtime.interpreter import Interpreter
    print("Using tflite_runtime")
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
        print("Using tensorflow.lite")
    except ImportError:
        print("Error: Neither tflite_runtime nor tensorflow found!")
        print("\nInstall one of these:")
        print("  pip3 install tflite-runtime")
        print("  pip3 install tensorflow")
        exit(1)

# Configuration
MODEL_PATH = "/home/theo/4.tflite"
CAMERA_ID = 1
DATA_FILE = "../data/posture_data.csv"  # Use parent directory
MIN_CONFIDENCE = 0.2

# Create data directory
os.makedirs("../data", exist_ok=True)

# Initialize TFLite model
print("Loading MoveNet model...")
interpreter = Interpreter(MODEL_PATH, num_threads=2)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_dtype = input_details[0]['dtype']
in_h = input_details[0]['shape'][1]
in_w = input_details[0]['shape'][2]

# Initialize camera
print("Opening camera...")
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print("Cannot open camera")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Check if CSV file exists, if not create with header
file_exists = os.path.exists(DATA_FILE)
csv_lock = threading.Lock()

if not file_exists:
    with open(DATA_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        header = get_feature_names() + ['label']
        writer.writerow(header)
    print(f"Created new data file: {DATA_FILE}")
else:
    print(f"Appending to existing file: {DATA_FILE}")

# Statistics
stats = {
    'good_count': 0,
    'bad_count': 0,
    'last_action': 'Ready to collect data',
    'total': 0
}

# Flask app
app = Flask(__name__)
outputFrame = None
current_kpts = None
frame_lock = threading.Lock()
label_command = None

def save_sample(label):
    """Save current keypoints with label"""
    global current_kpts, stats
    
    if current_kpts is None:
        return False, "No keypoints available"
    
    if not validate_keypoints(current_kpts, MIN_CONFIDENCE):
        return False, "Low confidence keypoints"
    
    try:
        features = extract_features(current_kpts, 640, 480)
        row = features + [label]
        
        with csv_lock:
            with open(DATA_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        
        if label == 'good':
            stats['good_count'] += 1
        else:
            stats['bad_count'] += 1
        
        stats['total'] = stats['good_count'] + stats['bad_count']
        stats['last_action'] = f"Saved {label.upper()} sample"
        
        return True, f"Saved {label} sample"
    except Exception as e:
        return False, str(e)

def process_frames():
    """Process camera frames in background thread"""
    global outputFrame, current_kpts, label_command
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        h, w, _ = frame.shape
        
        # Prepare input for MoveNet
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        
        if input_dtype == np.float32:
            inp = img_resized.astype(np.float32)
            inp = (inp - 127.5) / 127.5
            inp = np.expand_dims(inp, axis=0)
        else:
            inp = np.expand_dims(img_resized.astype(input_dtype), axis=0)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        kpts = output_data[0][0] if output_data.ndim == 4 else output_data[0]
        current_kpts = kpts
        
        # Draw keypoints
        for i, (y, x, score) in enumerate(kpts):
            if score > MIN_CONFIDENCE:
                color = (0, 255, 0) if i < 7 else (0, 0, 255)
                cv2.circle(frame, (int(x * w), int(y * h)), 4, color, -1)
        
        # Draw skeleton
        connections = [
            (5, 6), (5, 11), (6, 12), (11, 12), (0, 5), (0, 6),
        ]
        
        for (idx1, idx2) in connections:
            y1, x1, s1 = kpts[idx1]
            y2, x2, s2 = kpts[idx2]
            if s1 > MIN_CONFIDENCE and s2 > MIN_CONFIDENCE:
                pt1 = (int(x1 * w), int(y1 * h))
                pt2 = (int(x2 * w), int(y2 * h))
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
        
        # Display statistics
        cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"Good: {stats['good_count']}  Bad: {stats['bad_count']}  Total: {stats['total']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, stats['last_action'], (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, "Use web interface to label samples", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        with frame_lock:
            outputFrame = frame.copy()

def generate():
    """Generate MJPEG stream"""
    global outputFrame, frame_lock
    
    while True:
        with frame_lock:
            if outputFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        
        if not flag:
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/save/<label>")
def save_label(label):
    if label not in ['good', 'bad']:
        return {'success': False, 'message': 'Invalid label'}
    
    success, message = save_sample(label)
    return {
        'success': success,
        'message': message,
        'stats': stats
    }

@app.route("/stats")
def get_stats():
    return stats

@app.route("/")
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Posture Data Collection</title>
        <style>
            body { 
                background: #1a1a1a; 
                color: #fff; 
                font-family: Arial; 
                padding: 20px;
                text-align: center;
            }
            .container { max-width: 1000px; margin: 0 auto; }
            h1 { color: #00ff88; }
            .video { 
                width: 100%; 
                max-width: 640px; 
                border-radius: 10px; 
                margin: 20px auto;
                box-shadow: 0 4px 20px rgba(0,255,136,0.3);
            }
            .controls { margin: 30px 0; }
            button {
                padding: 20px 40px;
                font-size: 18px;
                font-weight: bold;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                margin: 10px;
                transition: transform 0.1s;
            }
            button:active { transform: scale(0.95); }
            .btn-good { background: #00ff88; color: #000; }
            .btn-bad { background: #ff4444; color: #fff; }
            .btn-good:hover { background: #00dd77; }
            .btn-bad:hover { background: #dd3333; }
            .stats {
                background: #2a2a2a;
                padding: 20px;
                border-radius: 10px;
                margin: 20px auto;
                max-width: 500px;
            }
            .stat-row { 
                display: flex; 
                justify-content: space-between; 
                margin: 10px 0;
                font-size: 18px;
            }
            .message {
                padding: 15px;
                border-radius: 5px;
                margin: 20px auto;
                max-width: 500px;
                display: none;
            }
            .success { background: #00ff8833; border: 1px solid #00ff88; }
            .error { background: #ff444433; border: 1px solid #ff4444; }
            .instructions {
                background: #2a2a2a;
                padding: 20px;
                border-radius: 10px;
                margin: 20px auto;
                max-width: 600px;
                text-align: left;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Posture Data Collection</h1>
            
            <div class="instructions">
                <h3>Instructions:</h3>
                <ul>
                    <li>Click <strong>GOOD</strong> when sitting with proper posture</li>
                    <li>Click <strong>BAD</strong> when slouching or poor posture</li>
                    <li>Collect at least <strong>50 samples per label</strong></li>
                    <li>Try different angles and distances</li>
                    <li>When done, run: <code>python train_model.py</code></li>
                </ul>
            </div>
            
            <img src="/video_feed" class="video">
            
            <div class="controls">
                <button class="btn-good" onclick="saveLabel('good')">
                    GOOD POSTURE
                </button>
                <button class="btn-bad" onclick="saveLabel('bad')">
                    BAD POSTURE
                </button>
            </div>
            
            <div id="message" class="message"></div>
            
            <div class="stats">
                <h3>Statistics</h3>
                <div class="stat-row">
                    <span>Good samples:</span>
                    <strong id="good-count">0</strong>
                </div>
                <div class="stat-row">
                    <span>Bad samples:</span>
                    <strong id="bad-count">0</strong>
                </div>
                <div class="stat-row">
                    <span>Total:</span>
                    <strong id="total-count">0</strong>
                </div>
            </div>
        </div>
        
        <script>
            function saveLabel(label) {
                fetch('/save/' + label)
                    .then(r => r.json())
                    .then(data => {
                        const msg = document.getElementById('message');
                        msg.textContent = data.message;
                        msg.className = 'message ' + (data.success ? 'success' : 'error');
                        msg.style.display = 'block';
                        
                        if (data.success) {
                            updateStats(data.stats);
                            setTimeout(() => {
                                msg.style.display = 'none';
                            }, 2000);
                        }
                    });
            }
            
            function updateStats(stats) {
                document.getElementById('good-count').textContent = stats.good_count;
                document.getElementById('bad-count').textContent = stats.bad_count;
                document.getElementById('total-count').textContent = stats.total;
            }
            
            setInterval(() => {
                fetch('/stats')
                    .then(r => r.json())
                    .then(updateStats);
            }, 2000);
        </script>
    </body>
    </html>
    """)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("POSTURE DATA COLLECTION - WEB INTERFACE")
    print("="*60)
    print(f"\nAccess at: http://raspberrypi.local:5001")
    print(f"Or: http://192.168.x.x:5001")
    print(f"Data will be saved to: {DATA_FILE}")
    print("\n" + "="*60 + "\n")
    
    # Start frame processing thread
    t = threading.Thread(target=process_frames, daemon=True)
    t.start()
    
    # Start Flask app
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
