#!/usr/bin/env python3

import numpy as np
from collections import deque

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
        return 0.0
    cosine_angle = np.clip(dot_product / norm_product, -1.0, 1.0)
    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)
    neck_angle = abs(90.0 - angle_deg)
    return float(neck_angle)


def get_feature_names_41():
    names = []
    for name in IMPORTANT_KEYPOINTS.keys():
        names.extend([f"{name}_x", f"{name}_y", f"{name}_conf"])
    names.extend([
        "neck_angle_deg",
        "head_tilt_deg",
        "shoulder_asymmetry",
        "neck_length_ratio",
        "face_size_ratio",
        "neck_angle_abs",
        "head_tilt_abs",
        "eye_mid_x",
        "eye_mid_y",
        "shoulder_mid_x",
        "shoulder_mid_y",
        "vertical_alignment",
        "vertical_alignment_norm",
        "shoulder_x_diff",
        "shoulder_y_diff",
        "nose_y_norm",
        "avg_kpt_conf",
        "neck_angle_by_face",
        "neck_angle_roll_std",
        "head_tilt_roll_std"
    ])
    return names


def validate_keypoints(kpts, min_confidence=0.2):
    important_indices = list(IMPORTANT_KEYPOINTS.values())
    good_count = sum(1 for idx in important_indices if len(kpts) > idx and kpts[idx][2] > min_confidence)
    return good_count >= 5


def _safe_kpt_xy(kpts, idx, frame_w, frame_h):
    """Return (x_px, y_px, conf) for a keypoint index if available, else zeros."""
    if idx < 0 or idx >= len(kpts):
        return 0.0, 0.0, 0.0
    y, x, conf = kpts[idx]
    return float(x * frame_w), float(y * frame_h), float(conf)


def extract_features_41(kpts, frame_w, frame_h, buffers=None, window=5):
    """
    Returns list of 41 features in exactly the same order as `all_features`.
    buffers: optional dict with keys 'neck_angle' and 'head_tilt' each a deque used to compute rolling std.
             If not provided, temporary buffers will be used (so roll_std=0).
    window: window size for rolling std (default 5 to match training)
    """
    features = []

    for idx in IMPORTANT_KEYPOINTS.values():
        x_px, y_px, conf = _safe_kpt_xy(kpts, idx, frame_w, frame_h)
        features.extend([x_px, y_px, conf])

    nose = np.array([_safe_kpt_xy(kpts, 0, frame_w, frame_h)[0], _safe_kpt_xy(kpts, 0, frame_w, frame_h)[1]])
    left_eye = np.array([_safe_kpt_xy(kpts, 1, frame_w, frame_h)[0], _safe_kpt_xy(kpts, 1, frame_w, frame_h)[1]])
    right_eye = np.array([_safe_kpt_xy(kpts, 2, frame_w, frame_h)[0], _safe_kpt_xy(kpts, 2, frame_w, frame_h)[1]])
    left_ear = np.array([_safe_kpt_xy(kpts, 3, frame_w, frame_h)[0], _safe_kpt_xy(kpts, 3, frame_w, frame_h)[1]])
    right_ear = np.array([_safe_kpt_xy(kpts, 4, frame_w, frame_h)[0], _safe_kpt_xy(kpts, 4, frame_w, frame_h)[1]])
    left_shoulder = np.array([_safe_kpt_xy(kpts, 5, frame_w, frame_h)[0], _safe_kpt_xy(kpts, 5, frame_w, frame_h)[1]])
    right_shoulder = np.array([_safe_kpt_xy(kpts, 6, frame_w, frame_h)[0], _safe_kpt_xy(kpts, 6, frame_w, frame_h)[1]])

    # confidences
    s_le = float(kpts[1][2]) if len(kpts) > 1 else 0.0
    s_re = float(kpts[2][2]) if len(kpts) > 2 else 0.0
    s_lear = float(kpts[3][2]) if len(kpts) > 3 else 0.0
    s_rear = float(kpts[4][2]) if len(kpts) > 4 else 0.0
    conf_cols = []
    for idx in IMPORTANT_KEYPOINTS.values():
        _, _, c = _safe_kpt_xy(kpts, idx, frame_w, frame_h)
        conf_cols.append(c)

    # mid_mouth estimation
    if s_le > 0.25 and s_re > 0.25:
        eye_distance = np.linalg.norm(left_eye - right_eye)
        mouth_offset_y = eye_distance * 0.6
        mid_mouth = nose + np.array([0.0, mouth_offset_y])
    elif s_lear > 0.25 and s_rear > 0.25:
        ear_distance = np.linalg.norm(left_ear - right_ear)
        mouth_offset_y = ear_distance * 0.4
        mid_mouth = nose + np.array([0.0, mouth_offset_y])
    else:
        # fallback using shoulder width
        mouth_offset_y = np.linalg.norm(left_shoulder - right_shoulder) * 0.4
        mid_mouth = nose + np.array([0.0, mouth_offset_y])

    neck_angle_deg = float(calculate_neck_angle(left_shoulder, right_shoulder, mid_mouth))
    features.append(neck_angle_deg)

    # head tilt: eye_line angle minus shoulder_line angle (deg)
    shoulder_line = right_shoulder - left_shoulder
    eye_line = right_eye - left_eye
    shoulder_angle = float(np.arctan2(shoulder_line[1], shoulder_line[0]) if np.linalg.norm(shoulder_line) > 1e-6 else 0.0)
    eye_angle = float(np.arctan2(eye_line[1], eye_line[0]) if np.linalg.norm(eye_line) > 1e-6 else 0.0)
    head_tilt_deg = float(np.degrees(eye_angle - shoulder_angle))
    features.append(head_tilt_deg)

    # shoulder_asymmetry (center offset normalized)
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    if shoulder_width > 1e-6:
        mid_shoulder = (left_shoulder + right_shoulder) / 2.0
        shoulder_asymmetry = float(np.abs(mid_shoulder[0] - (frame_w / 2.0)) / frame_w)
    else:
        shoulder_asymmetry = 0.0
    features.append(shoulder_asymmetry)

    # neck_length_ratio = neck_length / eye_distance
    eye_distance = np.linalg.norm(left_eye - right_eye) if (np.linalg.norm(left_eye - right_eye) > 1e-6) else 0.0
    mid_shoulder = (left_shoulder + right_shoulder) / 2.0
    neck_length = np.linalg.norm(nose - mid_shoulder)
    neck_length_ratio = float((neck_length / eye_distance) if eye_distance > 1e-6 else 1.0)
    features.append(neck_length_ratio)

    if s_lear > 0.25 and s_rear > 0.25:
        face_width = np.linalg.norm(left_ear - right_ear)
        face_size_ratio = float(face_width / frame_w)
    else:
        face_size_ratio = 0.15
    features.append(face_size_ratio)

    features.append(abs(neck_angle_deg))                 # neck_angle_abs
    features.append(abs(head_tilt_deg))                  # head_tilt_abs

    # eye midpoint
    eye_mid_x = float((left_eye[0] + right_eye[0]) / 2.0)
    eye_mid_y = float((left_eye[1] + right_eye[1]) / 2.0)
    features.append(eye_mid_x)
    features.append(eye_mid_y)

    # shoulder midpoint
    shoulder_mid_x = float(((left_shoulder[0] + right_shoulder[0]) / 2.0))
    shoulder_mid_y = float(((left_shoulder[1] + right_shoulder[1]) / 2.0))
    features.append(shoulder_mid_x)
    features.append(shoulder_mid_y)

    # vertical_alignment: abs(eye_mid_x - shoulder_mid_x)
    vertical_alignment = float(abs(eye_mid_x - shoulder_mid_x))
    features.append(vertical_alignment)

    vertical_alignment_norm = float(vertical_alignment / (face_size_ratio + 1e-6))
    features.append(vertical_alignment_norm)

    # shoulder diffs
    features.append(float(abs(left_shoulder[0] - right_shoulder[0])))  # shoulder_x_diff
    features.append(float(abs(left_shoulder[1] - right_shoulder[1])))  # shoulder_y_diff

    # nose_y_norm
    nose_y_norm = float(nose[1] / (face_size_ratio + 1e-6)) if face_size_ratio > 1e-6 else 0.0
    features.append(nose_y_norm)

    # avg_kpt_conf
    avg_kpt_conf = float(np.mean(conf_cols)) if len(conf_cols) > 0 else 0.0
    features.append(avg_kpt_conf)

    # neck_angle_by_face
    features.append(float(neck_angle_deg * face_size_ratio))

    if buffers is None:
        neck_roll_std = 0.0
        head_roll_std = 0.0
    else:
        if 'neck_angle' not in buffers:
            buffers['neck_angle'] = deque(maxlen=window)
        if 'head_tilt' not in buffers:
            buffers['head_tilt'] = deque(maxlen=window)

        buffers['neck_angle'].append(neck_angle_deg)
        buffers['head_tilt'].append(head_tilt_deg)

        # compute std (ddof=0)
        neck_roll_std = float(np.std(np.array(buffers['neck_angle']), ddof=0))
        head_roll_std = float(np.std(np.array(buffers['head_tilt']), ddof=0))

    features.append(neck_roll_std)
    features.append(head_roll_std)

    if len(features) != len(get_feature_names_41()):
        desired = len(get_feature_names_41())
        if len(features) < desired:
            features.extend([0.0] * (desired - len(features)))
        else:
            features = features[:desired]

    return features
