# Smart Posture Assistant (SPA)

**SPA** is a real-time bio-feedback system targeted at embedded Linux platforms (specifically Raspberry Pi 4). It utilizes a hybrid Computer Vision architecture to monitor spinal posture and provides multi-modal feedback (Visual, Audio) to correct "tech neck" syndrome.

The system is designed with a decoupled architecture, separating the AI inference engine from the IoT control layer, allowing for fail-safe Over-The-Air (OTA) updates and remote management.

## Introduction

Unlike standard motion detection systems, SPA implements a **Hybrid Classification Engine**. It combines deterministic geometric heuristics (Vector Algebra) for obvious posture deviations with a stochastic Ensemble Learning model (RandomForest + XGBoost) for subtle spinal tilts.

The software stack is optimized for ARM64 architecture, running the Movenet SinglePose Lightning model at ~10 FPS on a Raspberry Pi 4 (4GB) without an external accelerator.

## Usage

 The main entry point is `main.py`. The system requires root privileges to access GPIO and hardware PWM for the LED ring.

sudo python3 main.py [options]

### Voice Interaction (REPL-like)

SPA operates as a voice-driven agent. Instead of command-line arguments, it accepts verbal commands processed via the `voice_agent` module.

| Voice Command (Vietnamese) | Action |
| --- | --- |
| `"Bắt đầu tính giờ"` | **Start Timer**: Initiates a 25-minute Pomodoro focus session. |
| `"Kiểm tra tư thế"` | **Status Check**: Forces an immediate inference and reports the current posture state verbally. |
| `"Dừng lại"` | **Stop**: Halts the current timer or alarm. |

### Remote Management (CLI)

For remote administration, the device exposes a local REST API on port `8080`. You can interact with it using `curl`:

**Trigger OTA Update:**
curl -X POST http://<DEVICE_IP>:8080/update \
     -H "Authorization: Bearer <YOUR_SECRET_TOKEN>"

**Restart Service:**

curl -X POST http://<DEVICE_IP>:8080/control \
     -H "Content-Type: application/json" \
     -d '{"action": "restart"}'

## System Internals

SPA differs from typical Python scripts by implementing a state-machine based workflow with a dedicated calibration phase.

### 1. Hybrid Detection Algorithm

The posture determination logic relies on two concurrent pipelines:

* **Geometric Heuristics (Hard Constraints):**
Calculates the dot product of the shoulder vector and the neck vector.

θ = arccos((v_shoulder . v_neck) / (|v_shoulder| * |v_neck|))
It also computes the *Normalized Nose-Ear Vertical Difference* to detect forward leaning ("Nose Drop").
* **Ensemble Classifier (Soft Voting):**
If heuristics are inconclusive, the 31-dimensional feature vector is passed to a Voting Classifier (RandomForest + GradientBoosting).
* *Input:* Normalized keypoints (x, y, confidence).
* *Output:* Probability distribution `P(class | features)`.

### 2. Adaptive Calibration

Upon initialization, the system enters `CALIB_MODE`. It collects 60 frames of "ideal" posture data to compute a user-specific baseline.

* **Base Neck Ratio:** 
* **Thresholding:** The runtime trigger is dynamic: .

### 3. IoT Architecture & OTA

The system uses a sidecar pattern. The `api.py` process runs independently of the main AI loop.

* **Fail-safety:** If the main CV loop crashes due to memory overflow or camera I/O errors, the API server remains active to accept `restart` or `update` commands.
* **Update Mechanism:** The `update.sh` script performs an atomic `git pull` followed by a `systemctl` service restart.

## Hardware Support

SPA is tested on the following hardware configuration:

* **SoC:** Broadcom BCM2711 (Raspberry Pi 4 Model B).
* **Camera:** Sony IMX219 (Pi Camera Module V2) or USB Webcam.
* **Display:** SSD1306 0.96" OLED (I2C interface).
* **Indicator:** WS2812B LED Ring (12-bit PWM driven).
* **Audio:** Generic USB Microphone + 3.5mm Speaker + MAX98357A.

## Benchmarks

Performance metrics measured on Raspberry Pi 4 (Stock Clock):

| Metric | Value | Note |
| --- | --- | --- |
| **Inference Time** | 35ms - 45ms | Movenet TFLite (Int8) |
| **Total Latency** | ~70ms | Capture + Inference + Logic |
| **CPU Usage** | 45% - 60% | Across 4 cores |
| **RAM Usage** | ~350 MB | Python process overhead |

## Installation

### Prerequisites
* Raspberry Pi OS Lite (Bullseye/Bookworm) 64-bit.
* Python 3.9+.
* Enabled interfaces: I2C, SPI, Camera.

### Build Steps

1. **Clone the repository:**
git clone [https://github.com/username/smart-posture-assistant.git](https://github.com/username/smart-posture-assistant.git)
cd smart-posture-assistant

2. **Install system dependencies:**
sudo apt-get install libatlas-base-dev libopenjp2-7 portaudio19-dev

3. **Install Python requirements:**
pip3 install -r requirements.txt

4. **Configure Environment:**
Create a `.env` file for Cloud connectivity:

SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_key
UPDATE_SECRET=your_secure_tokenyoururl

