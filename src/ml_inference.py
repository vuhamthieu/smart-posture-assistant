import os
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from utils import extract_features_31, validate_keypoints


CALIBRATION_FRAMES_DEFAULT = 60


def _load_interpreter(model_path: str):
    try:
        from tflite_runtime.interpreter import Interpreter
    except Exception:
        from tensorflow.lite.python.interpreter import Interpreter
    interpreter = Interpreter(model_path, num_threads=2)
    interpreter.allocate_tensors()
    return interpreter


def _load_ensemble(model_path: str, scaler_path: str):
    import joblib

    posture_model = joblib.load(model_path)
    posture_scaler = joblib.load(scaler_path)
    return posture_model, posture_scaler


@dataclass
class DetectorConfig:
    neck_threshold_pct: float = 35.0
    nose_drop_threshold: float = 0.25
    calibration_frames: int = CALIBRATION_FRAMES_DEFAULT


class PostureModels:
    def __init__(
        self,
        tflite_model_path: str,
        ensemble_model_path: str | None = None,
        ensemble_scaler_path: str | None = None,
    ):
        self.interpreter = _load_interpreter(tflite_model_path)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.in_h, self.in_w = self.input_details[0]['shape'][1:3]
        self.input_dtype = self.input_details[0]['dtype']

        self.use_ml_model = False
        self.posture_model = None
        self.posture_scaler = None

        if ensemble_model_path and ensemble_scaler_path:
            try:
                self.posture_model, self.posture_scaler = _load_ensemble(
                    ensemble_model_path,
                    ensemble_scaler_path,
                )
                self.use_ml_model = True
                print("ML Model Loaded")
            except Exception as e:
                print(f"Model Error: {e}")

    def infer_keypoints(self, frame_bgr):
        import cv2

        img = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), (self.in_w, self.in_h))
        if self.input_dtype == np.uint8:
            input_data = np.expand_dims(img, axis=0)
        else:
            input_data = np.expand_dims((img.astype(np.float32) - 127.5) / 127.5, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        kpts = self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]
        return kpts

    def infer_ai_label(self, feats, frame_idx: int, last_known_conf: float):
        if not self.use_ml_model:
            return "None", 0.0, last_known_conf

        X_in = self.posture_scaler.transform([feats])
        ai_label = self.posture_model.predict(X_in)[0]

        if frame_idx % 30 == 0:
            ai_conf = float(np.max(self.posture_model.predict_proba(X_in)[0]))
            last_known_conf = ai_conf
        else:
            ai_conf = float(last_known_conf)

        return ai_label, ai_conf, last_known_conf


class PostureDetector:
    def __init__(self, models: PostureModels, config: DetectorConfig | None = None):
        self.models = models
        self.config = config or DetectorConfig()

        self.status_buf = deque(maxlen=15)
        self.last_kpts = None
        self.last_known_conf = 0.95

        self.base_neck_ratio = 0.0
        self.base_nose_ear_diff = 0.0
        self.is_calibrated = False
        self.calib_neck = []
        self.calib_nose = []

        self.last_raw_status = "Init"
        self.last_method = "Init"
        self.last_debug_msg = ""
        self.last_detected_type = ""

    def reset_calibration(self):
        self.base_neck_ratio = 0.0
        self.base_nose_ear_diff = 0.0
        self.is_calibrated = False
        self.calib_neck = []
        self.calib_nose = []

    def step(self, frame_bgr, frame_idx: int, neck_threshold_pct: float | None = None, nose_drop_threshold: float | None = None):
        import cv2

        h, w, _ = frame_bgr.shape
        stream_frame = cv2.resize(frame_bgr, (640, 480))

        run_inference = (frame_idx % 2 == 0)
        if run_inference:
            kpts = self.models.infer_keypoints(frame_bgr)
            self.last_kpts = kpts
        else:
            kpts = self.last_kpts

        raw_status = self.last_raw_status
        method = self.last_method
        debug_msg = self.last_debug_msg
        detected_type = self.last_detected_type

        neck_threshold_pct = neck_threshold_pct if neck_threshold_pct is not None else self.config.neck_threshold_pct
        neck_shrink_tolerance = neck_threshold_pct / 100.0 * 2.0
        if neck_shrink_tolerance > 1.0 or neck_shrink_tolerance < 0.5:
            neck_shrink_tolerance = 0.70

        nose_drop_threshold = nose_drop_threshold if nose_drop_threshold is not None else self.config.nose_drop_threshold

        metrics = None
        ai_label = "None"
        ai_conf = 0.0
        phys_label = "Good"
        angle = 0.0

        if kpts is not None:
            if not self.is_calibrated:
                if validate_keypoints(kpts):
                    if run_inference:
                        face_w = np.linalg.norm(
                            np.array([kpts[3][1], kpts[3][0]]) - np.array([kpts[4][1], kpts[4][0]])
                        )
                        neck_h = (kpts[5][0] + kpts[6][0]) / 2 - kpts[0][0]
                        ear_y = (kpts[3][0] + kpts[4][0]) / 2
                        nose_y = kpts[0][0]
                        nose_ear_val = nose_y - ear_y
                        if face_w > 0:
                            self.calib_neck.append(neck_h / face_w)
                            self.calib_nose.append(nose_ear_val / face_w)

                        if len(self.calib_neck) >= self.config.calibration_frames:
                            self.base_neck_ratio = float(np.mean(self.calib_neck))
                            self.base_nose_ear_diff = float(np.mean(self.calib_nose))
                            self.is_calibrated = True

                    msg = f"CALIB... {int(len(self.calib_neck) / self.config.calibration_frames * 100)}%"
                    cv2.putText(stream_frame, "SIT STRAIGHT", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(stream_frame, msg, (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(stream_frame, "NO PERSON", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                return {
                    'stream_frame': stream_frame,
                    'calibrated': False,
                    'raw_status': 'Init',
                    'final_status': 'Init',
                    'method': 'CALIB',
                    'debug': '',
                    'detected_type': '',
                    'metrics': None,
                    'ai_label': 'None',
                    'ai_conf': 0.0,
                    'phys_label': 'Init',
                    'keypoints': kpts,
                }

            if validate_keypoints(kpts) and run_inference:
                try:
                    feats = extract_features_31(kpts, w, h)
                    angle = float(feats[21])

                    face_w = np.linalg.norm(
                        np.array([kpts[3][1], kpts[3][0]]) - np.array([kpts[4][1], kpts[4][0]])
                    )
                    neck_h = (kpts[5][0] + kpts[6][0]) / 2 - kpts[0][0]
                    ear_y = (kpts[3][0] + kpts[4][0]) / 2
                    nose_y = kpts[0][0]
                    current_nose_diff = (nose_y - ear_y) / (face_w + 1e-6)
                    curr_neck_ratio = neck_h / (face_w + 1e-6)
                    neck_change = curr_neck_ratio / (self.base_neck_ratio + 1e-6)
                    nose_drop_amount = current_nose_diff - self.base_nose_ear_diff

                    phys_label = "Good"
                    is_bad_posture = False
                    detected_type = ""

                    if neck_change < neck_shrink_tolerance:
                        if nose_drop_amount > nose_drop_threshold:
                            phys_label = "Lean(Nose)"
                            is_bad_posture = True
                            detected_type = "lean"
                        else:
                            phys_label = "Hunch(Neck)"
                            is_bad_posture = True
                            detected_type = "hunch"

                    ai_label, ai_conf, self.last_known_conf = self.models.infer_ai_label(
                        feats,
                        frame_idx,
                        self.last_known_conf,
                    )

                    if self.models.use_ml_model:
                        if ai_label == 'tilt' and ai_conf > 0.7:
                            raw_status = "Bad"
                            detected_type = "tilt"
                            method = f"AI_Tilt({ai_conf:.2f})"
                        elif is_bad_posture:
                            raw_status = "Bad"
                            method = phys_label
                        elif ai_conf > 0.7 and ai_label != 'good':
                            raw_status = "Bad"
                            detected_type = str(ai_label)
                            method = f"AI_{ai_label}"
                        elif angle <= 15.0:
                            raw_status = "Good"
                            method = f"Safe({angle:.1f})"
                        else:
                            raw_status = "Good"
                    else:
                        if is_bad_posture:
                            raw_status = "Bad"
                            method = phys_label

                    debug_msg = f"N:{neck_change:.2f} D:{nose_drop_amount:.2f}"
                    self.last_raw_status = raw_status
                    self.last_method = method
                    self.last_debug_msg = debug_msg
                    self.last_detected_type = detected_type

                    metrics = {
                        'neck_angle': float(angle),
                        'neck_ratio': float(neck_change),
                        'nose_dist': float(nose_drop_amount),
                    }
                except Exception:
                    method = "Err"

        if run_inference:
            self.status_buf.append(raw_status)

        final_status = "Bad" if self.status_buf.count("Bad") >= 10 else "Good"

        return {
            'stream_frame': stream_frame,
            'calibrated': True,
            'raw_status': raw_status,
            'final_status': final_status,
            'method': method,
            'debug': debug_msg,
            'detected_type': detected_type,
            'metrics': metrics,
            'ai_label': ai_label,
            'ai_conf': float(ai_conf),
            'phys_label': phys_label,
            'keypoints': kpts,
        }


def default_model_paths(project_root):
    tflite = os.environ.get('TFLITE_MODEL_PATH', str(project_root / '4.tflite'))
    ensemble_model = os.environ.get('ENSEMBLE_MODEL_PATH', str(project_root / 'models' / 'posture_ensemble.pkl'))
    ensemble_scaler = os.environ.get('ENSEMBLE_SCALER_PATH', str(project_root / 'models' / 'posture_scaler.pkl'))
    return tflite, ensemble_model, ensemble_scaler
