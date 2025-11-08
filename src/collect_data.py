#!/usr/bin/env python3
"""
Posture Data Collection Tool
Press 'g' for Good posture, 'b' for Bad posture, 'q' to quit
"""
import cv2
import numpy as np
import csv
import os
import time
from utils import extract_features, validate_keypoints, get_feature_names

try:
    from tflite_runtime.interpreter import Interpreter
except Exception:
    from tensorflow.lite.python.interpreter import Interpreter

# Configuration
MODEL_PATH = "/home/theo/4.tflite"
CAMERA_ID = 0
DATA_FILE = "data/posture_data.csv"
MIN_CONFIDENCE = 0.2

# Create data directory
os.makedirs("data", exist_ok=True)

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
csv_file = open(DATA_FILE, 'a', newline='')
csv_writer = csv.writer(csv_file)

if not file_exists:
    header = get_feature_names() + ['label']
    csv_writer.writerow(header)
    print(f"Created new data file: {DATA_FILE}")
else:
    print(f"Appending to existing file: {DATA_FILE}")

# Statistics
good_count = 0
bad_count = 0
last_save_time = 0
SAVE_COOLDOWN = 0.5  # Minimum time between saves (seconds)

print("\n" + "="*60)
print("POSTURE DATA COLLECTION TOOL")
print("="*60)
print("\nControls:")
print("  [G] - Save as GOOD posture")
print("  [B] - Save as BAD posture")
print("  [Q] - Quit and save")
print("\nTips:")
print("  - Sit in different positions for each label")
print("  - Collect at least 50 samples per label")
print("  - Try different angles and distances")
print("="*60 + "\n")

try:
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
        
        # Draw keypoints
        for i, (y, x, score) in enumerate(kpts):
            if score > MIN_CONFIDENCE:
                color = (0, 255, 0) if i < 7 else (0, 0, 255)
                cv2.circle(frame, (int(x * w), int(y * h)), 4, color, -1)
        
        # Draw skeleton
        connections = [
            (5, 6),   # shoulders
            (5, 11),  # left torso
            (6, 12),  # right torso
            (11, 12), # hips
            (0, 5),   # left neck
            (0, 6),   # right neck
        ]
        
        for (idx1, idx2) in connections:
            y1, x1, s1 = kpts[idx1]
            y2, x2, s2 = kpts[idx2]
            if s1 > MIN_CONFIDENCE and s2 > MIN_CONFIDENCE:
                pt1 = (int(x1 * w), int(y1 * h))
                pt2 = (int(x2 * w), int(y2 * h))
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
        
        # Display statistics
        cv2.rectangle(frame, (0, 0), (300, 120), (0, 0, 0), -1)
        cv2.putText(frame, f"Good samples: {good_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Bad samples: {bad_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Total: {good_count + bad_count}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        cv2.rectangle(frame, (0, h-80), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, "Press: [G] Good | [B] Bad | [Q] Quit", (10, h-50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Collect at least 50 samples per label!", (10, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.imshow('Posture Data Collection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        current_time = time.time()
        
        # Check cooldown
        can_save = (current_time - last_save_time) > SAVE_COOLDOWN
        
        if key == ord('g') and can_save:
            # Save as good posture
            if validate_keypoints(kpts, MIN_CONFIDENCE):
                features = extract_features(kpts, w, h)
                row = features + ['good']
                csv_writer.writerow(row)
                csv_file.flush()
                good_count += 1
                last_save_time = current_time
                print(f"✓ Saved GOOD sample #{good_count}")
            else:
                print("⚠ Skipped: Low confidence keypoints")
        
        elif key == ord('b') and can_save:
            # Save as bad posture
            if validate_keypoints(kpts, MIN_CONFIDENCE):
                features = extract_features(kpts, w, h)
                row = features + ['bad']
                csv_writer.writerow(row)
                csv_file.flush()
                bad_count += 1
                last_save_time = current_time
                print(f"Saved BAD sample #{bad_count}")
            else:
                print("Skipped: Low confidence keypoints")
        
        elif key == ord('q'):
            print("\nQuitting...")
            break

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("DATA COLLECTION SUMMARY")
    print("="*60)
    print(f"Good samples: {good_count}")
    print(f"Bad samples: {bad_count}")
    print(f"Total samples: {good_count + bad_count}")
    print(f"Data saved to: {DATA_FILE}")
    print("\nNext step: Run train_model.py to train the classifier")
    print("="*60)
