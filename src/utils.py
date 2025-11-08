#!/usr/bin/env python3
"""
Utility functions shared between collect_data.py, train_model.py, and main.py
"""
import numpy as np

# Important keypoint indices for posture detection
IMPORTANT_KEYPOINTS = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_hip': 11,
    'right_hip': 12
}

def extract_features(kpts, frame_width, frame_height):
    """
    Extract normalized features from keypoints
    
    Args:
        kpts: Keypoints array from MoveNet [17 x 3] (y, x, confidence)
        frame_width: Frame width for normalization
        frame_height: Frame height for normalization
    
    Returns:
        List of features (normalized coordinates + confidence scores)
    """
    features = []
    
    # Extract only important keypoints
    indices = list(IMPORTANT_KEYPOINTS.values())
    
    for idx in indices:
        y, x, score = kpts[idx]
        
        # Add confidence score as feature
        features.extend([x, y, score])
    
    # Add derived features: angles and ratios
    try:
        # Neck angle approximation
        nose = np.array([kpts[0][1], kpts[0][0]])
        left_shoulder = np.array([kpts[5][1], kpts[5][0]])
        right_shoulder = np.array([kpts[6][1], kpts[6][0]])
        left_hip = np.array([kpts[11][1], kpts[11][0]])
        right_hip = np.array([kpts[12][1], kpts[12][0]])
        
        mid_shoulder = (left_shoulder + right_shoulder) / 2
        mid_hip = (left_hip + right_hip) / 2
        
        # Vector from hip to shoulder
        v1 = mid_shoulder - mid_hip
        # Vector from shoulder to nose
        v2 = nose - mid_shoulder
        
        # Calculate angle
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm > 0 and v2_norm > 0:
            cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            angle_deg = np.degrees(angle)
            features.append(abs(angle_deg - 180))
        else:
            features.append(0)
        
        # Shoulder-hip ratio
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        hip_width = np.linalg.norm(left_hip - right_hip)
        
        if hip_width > 0:
            features.append(shoulder_width / hip_width)
        else:
            features.append(1.0)
        
    except Exception as e:
        # If derived features fail, append zeros
        features.extend([0, 1.0])
    
    return features

def get_feature_names():
    """Get feature names for the extracted features"""
    names = []
    
    for name in IMPORTANT_KEYPOINTS.keys():
        names.extend([f"{name}_x", f"{name}_y", f"{name}_conf"])
    
    names.extend(["neck_angle", "shoulder_hip_ratio"])
    
    return names

def validate_keypoints(kpts, min_confidence=0.2):
    """
    Check if keypoints have sufficient confidence
    
    Args:
        kpts: Keypoints array
        min_confidence: Minimum confidence threshold
    
    Returns:
        True if valid, False otherwise
    """
    important_indices = list(IMPORTANT_KEYPOINTS.values())
    
    # Count how many important keypoints have good confidence
    good_count = sum(1 for idx in important_indices if kpts[idx][2] > min_confidence)
    
    # Require at least 6 out of 9 important keypoints
    return good_count >= 6
