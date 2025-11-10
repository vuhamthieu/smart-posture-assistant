#!/usr/bin/env python3
"""
Utility functions shared between collect_data.py, train_model.py, and main.py
"""
import numpy as np

IMPORTANT_KEYPOINTS = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6
}

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

def extract_features(kpts, frame_width, frame_height):
    features = []
    
    indices = list(IMPORTANT_KEYPOINTS.values())
    
    for idx in indices:
        y, x, score = kpts[idx]
        features.extend([x, y, score])
    
    try:
        nose = np.array([kpts[0][1] * frame_width, kpts[0][0] * frame_height])
        left_eye = np.array([kpts[1][1] * frame_width, kpts[1][0] * frame_height])
        right_eye = np.array([kpts[2][1] * frame_width, kpts[2][0] * frame_height])
        left_ear = np.array([kpts[3][1] * frame_width, kpts[3][0] * frame_height])
        right_ear = np.array([kpts[4][1] * frame_width, kpts[4][0] * frame_height])
        left_shoulder = np.array([kpts[5][1] * frame_width, kpts[5][0] * frame_height])
        right_shoulder = np.array([kpts[6][1] * frame_width, kpts[6][0] * frame_height])
        
        s_le = kpts[1][2]
        s_re = kpts[2][2]
        s_lear = kpts[3][2]
        s_rear = kpts[4][2]
        
        if s_le > 0.25 and s_re > 0.25:
            eye_distance = np.linalg.norm(left_eye - right_eye)
            mouth_offset_y = eye_distance * 0.6
            mid_mouth = nose + np.array([0, mouth_offset_y])
        elif s_lear > 0.25 and s_rear > 0.25:
            ear_distance = np.linalg.norm(left_ear - right_ear)
            mouth_offset_y = ear_distance * 0.4
            mid_mouth = nose + np.array([0, mouth_offset_y])
        else:
            mid_mouth = nose + np.array([0, np.linalg.norm(left_shoulder - right_shoulder) * 0.4])
        
        neck_angle_deg = calculate_neck_angle(left_shoulder, right_shoulder, mid_mouth)
        features.append(neck_angle_deg)
        
        shoulder_line = right_shoulder - left_shoulder
        eye_line = right_eye - left_eye
        
        shoulder_angle = np.arctan2(shoulder_line[1], shoulder_line[0])
        eye_angle = np.arctan2(eye_line[1], eye_line[0])
        head_tilt = np.degrees(eye_angle - shoulder_angle)
        features.append(head_tilt)
        
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        if shoulder_width > 0:
            mid_shoulder = (left_shoulder + right_shoulder) / 2
            shoulder_center_offset = np.abs(mid_shoulder[0] - frame_width/2) / frame_width
            features.append(shoulder_center_offset)
        else:
            features.append(0)
        
        eye_distance = np.linalg.norm(left_eye - right_eye)
        mid_shoulder = (left_shoulder + right_shoulder) / 2
        neck_length = np.linalg.norm(nose - mid_shoulder)
        if eye_distance > 0:
            neck_ratio = neck_length / eye_distance
            features.append(neck_ratio)
        else:
            features.append(1.0)
        
        if s_lear > 0.25 and s_rear > 0.25:
            face_width = np.linalg.norm(left_ear - right_ear)
            face_ratio = face_width / frame_width
            features.append(face_ratio)
        else:
            features.append(0.15)
        
    except Exception as e:
        features.extend([0, 0, 0, 1.0, 0.15])
    
    return features

def get_feature_names():
    names = []
    
    for name in IMPORTANT_KEYPOINTS.keys():
        names.extend([f"{name}_x", f"{name}_y", f"{name}_conf"])
    
    names.extend([
        "neck_angle_deg",
        "head_tilt_deg",
        "shoulder_asymmetry",
        "neck_length_ratio",
        "face_size_ratio"
    ])
    
    return names

def validate_keypoints(kpts, min_confidence=0.2):
    important_indices = list(IMPORTANT_KEYPOINTS.values())
    good_count = sum(1 for idx in important_indices if kpts[idx][2] > min_confidence)
    return good_count >= 5

def get_posture_from_angle(neck_angle, threshold=35):
    return 'good' if neck_angle <= threshold else 'bad'

if __name__ == '__main__':
    print("Utils module for posture detection")
    print(f"Using {len(IMPORTANT_KEYPOINTS)} keypoints")
    print(f"Feature names ({len(get_feature_names())} total):")
    for i, name in enumerate(get_feature_names(), 1):
        print(f"  {i:2d}. {name}")
    
    print("\nTest feature extraction:")
    dummy_kpts = np.zeros((17, 3))
    dummy_kpts[:7, 2] = 0.9
    dummy_kpts[:7, 0] = np.linspace(0.3, 0.7, 7)
    dummy_kpts[:7, 1] = np.linspace(0.4, 0.6, 7)
    
    features = extract_features(dummy_kpts, 640, 480)
    print(f"Extracted {len(features)} features")
    print(f"Neck angle: {features[21]:.2f}°")
