#!/usr/bin/env python3
import cv2
import numpy as np
import csv
import os
import time
import threading
from flask import Flask, Response, request, jsonify, render_template_string

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

MODEL_PATH = "/home/theo/4.tflite"
DATA_FILE = "posture_dataset_v2.csv"
CAMERA_ID = 0
CONFIDENCE_THRESHOLD = 0.3
SAVE_DELAY = 0.1 

interpreter = Interpreter(MODEL_PATH, num_threads=2)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
in_h, in_w = input_details[0]['shape'][1:3]


HEADERS = ['label', 'person_id']
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]
for kp in KEYPOINT_NAMES:
    HEADERS.extend([f"{kp}_y", f"{kp}_x", f"{kp}_conf"])

if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'w', newline='') as f:
        csv.writer(f).writerow(HEADERS)
    print(f"Created new dataset file: {DATA_FILE}")

app = Flask(__name__)
outputFrame = None
current_kpts = None
lock = threading.Lock()
stats = {'good': 0, 'lean': 0, 'hunch': 0, 'tilt': 0, 'total': 0}

def process_camera():
    global outputFrame, current_kpts
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(3, 640); cap.set(4, 480); cap.set(5, 30)
    
    while True:
        ret, frame = cap.read()
        if not ret: continue
        
        img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (in_w, in_h))
        input_data = np.expand_dims(img, axis=0)
        if input_details[0]['dtype'] == np.float32:
            input_data = (input_data.astype(np.float32) - 127.5) / 127.5
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        kpts = interpreter.get_tensor(output_details[0]['index'])[0][0]
        
        with lock: current_kpts = kpts
        
        h, w, _ = frame.shape
        indices_to_draw = [0, 1, 2, 3, 4, 5, 6]
        for idx in indices_to_draw:
            y, x, s = kpts[idx]
            if s > CONFIDENCE_THRESHOLD:
                cv2.circle(frame, (int(x*w), int(y*h)), 5, (0, 255, 0), -1)
        
        # Vẽ đường vai
        if kpts[5][2] > 0.3 and kpts[6][2] > 0.3:
            ls = (int(kpts[5][1]*w), int(kpts[5][0]*h))
            rs = (int(kpts[6][1]*w), int(kpts[6][0]*h))
            cv2.line(frame, ls, rs, (255, 0, 255), 2)

        with lock: outputFrame = frame.copy()

def generate():
    while True:
        with lock:
            if outputFrame is None: time.sleep(0.01); continue
            _, buf = cv2.imencode('.jpg', outputFrame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + bytearray(buf) + b'\r\n')

# ================= WEB SERVER =================
@app.route("/")
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Data collector</title>
        <style>
            body { background: #222; color: #fff; font-family: sans-serif; text-align: center; padding: 10px; }
            img { width: 100%; max-width: 640px; border: 2px solid #555; border-radius: 8px; }
            .controls { margin-top: 20px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
            button { 
                padding: 20px; font-size: 18px; border: none;
                color: white; font-weight: bold; cursor: pointer; user-select: none;
            }
            button:active { opacity: 0.7; transform: scale(0.98); }
            .btn-good { background: #28a745; }
            .btn-lean { background: #dc3545; }
            .btn-hunch { background: #ffc107; color: #000; }
            .btn-tilt { background: #17a2b8; }
            
            input { padding: 10px; font-size: 16px; width: 80px; text-align: center; margin-left: 10px; }
            #stats { margin-top: 20px; font-size: 16px; color: #ccc; }
        </style>
    </head>
    <body>
        <h3>DATA COLLECTOR</h3>
        <img src="/video_feed">
        <div style="margin-top:10px">
            <label>Person ID:</label> <input type="number" id="pid" value="1">
        </div>
        
        <div class="controls">
            <button class="btn-good" ontouchstart="start('good')" ontouchend="stop()" onmousedown="start('good')" onmouseup="stop()">
                GOOD
            </button>
            <button class="btn-lean" ontouchstart="start('lean')" ontouchend="stop()" onmousedown="start('lean')" onmouseup="stop()">
                LEAN
            </button>
            <button class="btn-hunch" ontouchstart="start('hunch')" ontouchend="stop()" onmousedown="start('hunch')" onmouseup="stop()">
                HUNCH
            </button>
            <button class="btn-tilt" ontouchstart="start('tilt')" ontouchend="stop()" onmousedown="start('tilt')" onmouseup="stop()">
                TILT
            </button>
        </div>
        
        <div id="stats">
            Total: 0 | Good: 0 | Lean: 0 | Hunch: 0 | Tilt: 0
        </div>

        <script>
            let interval;
            
            function start(label) {
                if (interval) clearInterval(interval);
                let pid = document.getElementById('pid').value;
                save(label, pid); // Lưu ngay 1 cái
                interval = setInterval(() => save(label, pid), 150); // Lưu liên tục mỗi 150ms
            }
            
            function stop() {
                if (interval) clearInterval(interval);
            }
            
            function save(label, pid) {
                fetch(`/save?label=${label}&pid=${pid}`)
                .then(r => r.json())
                .then(d => {
                    document.getElementById('stats').innerHTML = 
                        `Total: ${d.total} | Good: ${d.good} | Lean: ${d.lean} | Hunch: ${d.hunch} | Tilt: ${d.tilt}`;
                });
            }
        </script>
    </body>
    </html>
    """)

@app.route("/video_feed")
def video_feed(): return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/save")
def save_data():
    label = request.args.get('label')
    pid = request.args.get('pid')
    
    with lock: kpts = current_kpts
    
    if kpts is not None:
        if kpts[0][2] > CONFIDENCE_THRESHOLD and kpts[5][2] > CONFIDENCE_THRESHOLD and kpts[6][2] > CONFIDENCE_THRESHOLD:
            row = [label, pid]
            for kp in kpts: row.extend([kp[0], kp[1], kp[2]])
            
            with open(DATA_FILE, 'a', newline='') as f:
                csv.writer(f).writerow(row)
            
            stats[label] += 1
            stats['total'] += 1
    
    return jsonify(stats)

@app.route("/stats")
def get_stats():
    return jsonify(stats)

if __name__ == '__main__':
    t = threading.Thread(target=process_camera, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
