# Smart AIoT Posture Assistant

![Status](https://img.shields.io/badge/status-active-success)
![Platform](https://img.shields.io/badge/platform-Raspberry_Pi_4-lightgrey)
![Python](https://img.shields.io/badge/python-3.9-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Smart AIoT Posture Assistant is a real-time, edge-first posture monitoring system for seated users. It detects and classifies sitting postures using TensorFlow Lite MoveNet for pose estimation and a machine-learning classifier for posture recognition, designed for strong generalization to new, unseen users.

## Overview

**Problem:** Poor sitting posture contributes to back pain and reduced productivity. Many solutions are intrusive, cloud-dependent, or expensive.

**Solution:** A non-invasive, camera-based assistant that runs entirely on edge hardware (Raspberry Pi 4). It performs local pose estimation, engineered-feature extraction, and posture classification, then provides immediate feedback via LEDs, OLED display, voice alerts, and a local web interface.

## Table of Contents
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [AI & Data Processing](#ai--data-processing)
- [Dataset & Data Collection](#dataset--data-collection)
- [Getting Started](#getting-started)
- [Results and Performance](#results-and-performance)
- [Performance & Resource Usage](#performance--resource-usage)
- [Project Structure](#project-structure)


## Key Features

- Real-time classification of four posture types: Good, Hunch, Lean, Head Tilt.
- Two-stage pipeline:
  - Stage 1: Pose estimation with TensorFlow Lite MoveNet (CPU).
  - Stage 2: Posture classification with scikit-learn (Random Forest in the current training script), using engineered geometric features.
- 31 hand-crafted geometric features from keypoints to improve accuracy vs raw coordinates.
- Person-independent evaluation using Leave-One-Group-Out (LOGO) Cross-Validation.
- Hardware feedback: WS281x NeoPixel LEDs and SSD1306 OLED display.
- Local Flask web interface for monitoring and live MJPEG streaming.

## System Architecture

1. Camera captures frames in real time.
2. MoveNet (TFLite) extracts 17 skeletal keypoints.
3. Feature engineering transforms keypoints into 31 biomechanical features.
4. Classifier predicts posture using those features.
5. Feedback is delivered via LEDs, OLED, voice alerts, and the web UI.

### System Block Diagram

<img width="627" height="564" alt="image" src="https://github.com/user-attachments/assets/187ef250-d16d-4616-963f-8da5da830eac" />

End-to-end edge pipeline from camera capture, pose estimation,
feature engineering, posture classification to multi-modal feedback.

## Technology Stack

**AI / ML**
- TensorFlow Lite (MoveNet SinglePose Lightning)
- Scikit-learn (current training uses Random Forest)
- Engineered geometric features (31)

**Software & Backend**
- Python, OpenCV, Flask (web streaming and REST)
- Multithreading for real-time processing
- Cloud storage: Supabase 

**Embedded & Hardware**
- Raspberry Pi 4 Model B (4GB RAM)
- USB Camera OV3660 (3MP)
- SSD1306 OLED Display (0.96" I2C)
- WS2812/WS281x RGB LED Ring (12 LEDs)
- USB Microphone (e.g., MI-305)
- MAX98357 audio amplifier + 8Ω 3W speaker
- Li-ion 18650 battery + TP4056 charger + MT3608 boost (3.7V→5V) (only for LED Ring)

**Evaluation**
- Leave-One-Group-Out (LOGO) Cross-Validation
- Balanced Accuracy, Weighted F1-Score

## AI & Data Processing

- Pose Estimation: Primarily MoveNet (TensorFlow Lite) for single-person pose; optional MediaPipe support where 33 landmarks are required.
- Posture Classification: Ensemble (soft voting) of Random Forest, Gradient Boosting, and XGBoost (trained externally on Kaggle; see Model section).
- NLP (voice intent): Speech-to-text via Google Speech-to-Text API (optional), tokenization with pyvi (ViTokenizer), and intent classification using TF-IDF + SGDClassifier (SVM).

## Dataset & Data Collection

Data collection is handled by the web tool in (src/collect_data.py). Frames are processed on-device and keypoints are saved directly to CSV; images/videos are not stored.

- Output file: `posture_dataset_v2.csv`
- Columns: `label`, `person_id`, then 17×3 values per keypoint (y, x, confidence)
- Suggested protocol: collect balanced samples per class from multiple people; use `person_id` as the group key for LOGO evaluation.

## Getting Started

### Prerequisites
- Raspberry Pi OS Lite with Python 3.8+
- Camera connected and recognized by OpenCV
- Optional: I2C enabled for OLED; PWM-capable GPIO for LEDs

### Installation

```bash
git clone https://github.com/vuhamthieu/smart-posture-assistant.git
cd smart-posture-assistant

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Download a MoveNet TFLite model and set its path in `src/main.py` and `src/collect_data.py` (see `MODEL_PATH`).

### Environment Variables (optional)
Create `.env` for cloud sync or device settings (example keys):

- `SUPABASE_URL`, `SUPABASE_KEY`
- `CAMERA_ID`, `LED_COUNT`, `LED_PIN`
- `ENABLE_AUDIO`

### Run

Collect data:
```bash
cd src
python3 collect_data.py
# Open http://<device-ip>:5000 to label samples in real time
```

Model (trained externally on Kaggle):

- The posture classifier is trained in a Kaggle notebook and exported as artifacts (e.g., `posture_ensemble.pkl`, `posture_scaler.pkl`).
- Place the downloaded files under `models/` and ensure paths in `src/main.py` match your setup.
- Recommended paths:
  - `models/posture_ensemble.pkl`
  - `models/posture_scaler.pkl`
- If your paths differ, update `ENSEMBLE_MODEL` and `ENSEMBLE_SCALER` in `src/main.py` accordingly.

### Model Provenance (Kaggle)

Document the Kaggle notebook and run used to produce the artifacts:
- Kaggle notebook: https://www.kaggle.com/code/thieuvu/smart-posture-assistant-v4
- Feature set: 31 engineered features from MoveNet keypoints
- Evaluation: LOGO CV; Balanced Accuracy ~86%, Weighted F1 ~0.83


Run real-time monitoring:
```bash
cd src
python3 main.py
# Open http://<device-ip>:5000 for live stream and stats
```

## Systemd Auto Start (Raspberry Pi)

To run the assistant on boot via systemd:

1) Create a service file `/etc/systemd/system/posture-assistant.service`:

```
[Unit]
Description=Smart Posture Assistant
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/smart-posture-assistant/src
Environment=PYTHONUNBUFFERED=1
ExecStart=/home/pi/smart-posture-assistant/venv/bin/python3 /home/pi/smart-posture-assistant/src/main.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

2) Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable posture-assistant.service
sudo systemctl start posture-assistant.service
sudo systemctl status posture-assistant.service
```

Logs:
```bash
journalctl -u posture-assistant.service -f
```

## Results and Performance

- Balanced Accuracy (LOGO CV): ~86%
- Weighted F1-Score: ~0.83
- Feature engineering improves Balanced Accuracy by ~19% vs raw keypoints.
- Raspberry Pi 4: near real-time CPU inference (`num_threads=2`).

### Ablation Study

| Configuration | Balanced Accuracy |
|--------------|-------------------|
| Raw keypoints only | 66.7% |
| Engineered features (31, single model) | 84.6% |
| Remove top-5 features | 78.1% |
| Ensemble (full system) | **85.9%** |

**Key Insights:**
- Feature engineering contributes **~19% absolute improvement**.
- Top-5 geometric features account for **~7.8% accuracy gain**.
- Ensemble improves robustness across unseen users compared to single models.

## Performance & Resource Usage

Typical on Raspberry Pi 4 (CPU-only, MoveNet `num_threads=2`):
- **End-to-end FPS:** ~10–14 FPS
- **Inference latency (MoveNet):** ~30–40 ms/frame
- **Pipeline latency:** ~100–150 ms (capture → pose → classify → render)
- **CPU Load:** ~120% (≈1.2 cores utilized, ~67% idle)
- **Memory footprint:** ~**~424 MiB**  Out of 4GB RAM
- **Device temperature:** ~50–58°C under sustained load

## Project Structure

```
smart-posture-assistant/
├── src/
│   ├── main.py                  # Real-time inference and feedback
│   ├── collect_data.py          # Web-based data collection (CSV only)
│   ├── utils.py                 # 31-feature engineering and validation
│   ├── voice_agent.py           # Voice interaction
│   ├── config_manager.py        # Config & optional cloud sync
│   ├── train_nlp.py             # NLP model training (optional)
│   └── nlp_data.json            # NLP training data
├── data/
│   └── posture_data.csv         # Sample/legacy data
├── models/                      # Model artifacts (downloaded from Kaggle; large files not committed)
├── api.py                       # OTA/remote management API
├── requirements.txt             # Python dependencies
└── README.md
```

## Lessons Learned

- Person-independent generalization benefits from LOGO CV and engineered features.
- Relative positions (nose-centered) and biomechanical angles reduce user variance.
- Edge performance improves by limiting resolution and using TFLite with multiple threads.
- Camera angle changes can mimic leaning; add features to distinguish frame-center offset.
- Balanced metrics (Balanced Accuracy, Weighted F1) better reflect real-world performance.

## Future Enhancements

- Detect user emotional states and cognitive status through multimodal analysis.
- Multi-user profiles and automatic recognition (Classroom, Office use,...)
- Temporal modeling (sequence analysis) for fatigue detection.
- Edge TPU/CUDA acceleration where available.
- Smart Home Integration

## Demo Media

- Demo Videos: https://youtu.be/O5sq4FfQQvA

## Web Frontend 

- Web repo: https://github.com/vuhamthieu/posture-dashboard

## Author

This project was designed, implemented, and evaluated by:

**Vũ Hạm Thiều** 
