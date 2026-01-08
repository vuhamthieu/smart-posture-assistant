


#!/usr/bin/env python3
import numpy as np
import math

IMPORTANT_KEYPOINTS = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6
}

def calculate_neck_angle(left_shoulder, right_shoulder, mouth):
    shoulder_vector = right_shoulder - left_shoulder
    mid_shoulder = (left_shoulder + right_shoulder) / 2
    neck_vector = mouth - mid_shoulder
    dot = np.dot(shoulder_vector, neck_vector)
    norm = np.linalg.norm(shoulder_vector) * np.linalg.norm(neck_vector)
    if norm == 0: return 0.0
    cosine_angle = np.clip(dot/norm, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return abs(90.0 - angle)

def calculate_head_tilt(left_eye, right_eye):
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    if abs(dx) < 1e-6: return 0.0
    angle = math.degrees(math.atan2(dy, dx))
    angle = abs(angle)
    if angle > 90: angle = 180 - angle
    return angle

def validate_keypoints(kpts, min_confidence=0.25):
    count = sum(1 for k in kpts[:7] if k[2] > min_confidence)
    return count >= 5
def extract_features_31(kpts, w, h):
    features = []
    
    nose_y_origin = kpts[0][0] * h
    nose_x_origin = kpts[0][1] * w
    
    confs = []
    for idx in range(7):
        y, x, s = kpts[idx]
        
        x_px = x * w
        y_px = y * h
        
        x_rel = x_px - nose_x_origin
        y_rel = y_px - nose_y_origin
        
        features.extend([x_rel, y_rel, s])
        confs.append(s)
        
    nose = np.array([kpts[0][1]*w, kpts[0][0]*h])
    le = np.array([kpts[1][1]*w, kpts[1][0]*h])
    re = np.array([kpts[2][1]*w, kpts[2][0]*h])
    lear = np.array([kpts[3][1]*w, kpts[3][0]*h])
    rear = np.array([kpts[4][1]*w, kpts[4][0]*h])
    ls = np.array([kpts[5][1]*w, kpts[5][0]*h])
    rs = np.array([kpts[6][1]*w, kpts[6][0]*h])
    
    if confs[1] > 0.25 and confs[2] > 0.25:
        mid_mouth = nose + np.array([0.0, np.linalg.norm(le - re) * 0.6])
    else:
        mid_mouth = nose + np.array([0.0, np.linalg.norm(ls - rs) * 0.4])
    
    neck_angle = calculate_neck_angle(ls, rs, mid_mouth)
    features.append(neck_angle)
    
    mid_x = (ls[0] + rs[0]) / 2
    features.append(abs(mid_x - (w/2)) / w)
    
    eye_dist = np.linalg.norm(le - re)
    neck_len = np.linalg.norm(nose - (ls + rs)/2)
    features.append(neck_len / eye_dist if eye_dist > 0 else 0)
    
    face_w = np.linalg.norm(lear - rear) if (confs[3]>0.2 and confs[4]>0.2) else w*0.15
    face_ratio = face_w / w
    features.append(face_ratio)
    
    features.append(abs(neck_angle))
    features.append(abs(ls[1] - rs[1]))
    
    ear_y_avg = (lear[1] + rear[1]) / 2
    nose_ear_diff = (nose[1] - ear_y_avg) / (face_w + 1e-6)
    features.append(nose_ear_diff)
    
    features.append(np.mean(confs))
    features.append(neck_angle * face_ratio)
    features.append(calculate_head_tilt(le, re))
    
    return features
