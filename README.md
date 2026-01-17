# Smart AIoT Posture Assistant

Smart AIoT Posture Assistant is a real-time posture monitoring and feedback system running entirely on edge devices. The system is designed to detect and classify sitting postures using a hybrid AI architecture that combines deep learning–based pose estimation with an ensemble machine learning classifier, achieving high accuracy while maintaining strong generalization to unseen users.

## Key Features

- Real-time detection and classification of sitting postures (Good Posture, Hunched Back, Leaning, Head Tilt).
- Two-stage hybrid AI pipeline:
  - Stage 1: Human pose estimation using TensorFlow Lite MoveNet on CPU.
  - Stage 2: Posture classification using an ensemble of machine learning models with soft voting.
- Advanced feature engineering with 31 hand-crafted geometric features derived from skeletal keypoints, significantly improving classification performance compared to raw keypoint coordinates.
- Robust model evaluation using Leave-One-Group-Out (LOGO) Cross-Validation to ensure person-independent generalization.
- Embedded hardware integration including camera input, WS281x NeoPixel LEDs, and SSD1306 OLED display for real-time visual feedback.
- Local web-based interface (Flask) for live video streaming and system statistics monitoring.

## System Architecture

The system follows an edge-computing architecture, where all inference and decision-making processes are executed locally on the device.

1. The camera captures video frames of the user in real time.
2. MoveNet (TensorFlow Lite) performs pose estimation and extracts skeletal keypoints.
3. A feature engineering module transforms keypoints into 31 biomechanically meaningful geometric features.
4. An ensemble classifier predicts the current sitting posture.
5. The prediction result is used to:
   - Trigger visual feedback via LEDs and OLED display.
   - Provide data to the web interface and cloud synchronization module.

## Technology Stack

### AI / Machine Learning
- TensorFlow Lite (MoveNet SinglePose Lightning)
- Scikit-learn (Random Forest, Gradient Boosting, Voting Classifier)
- XGBoost
- Hand-crafted geometric feature engineering

### Software & Backend
- Python
- OpenCV
- Flask (Web streaming and REST APIs)
- Multithreading for real-time processing
- SQLite / Supabase (PostgreSQL) for data storage

### Embedded & Hardware
- Raspberry Pi 4
- USB / CSI Camera
- SSD1306 OLED Display (I2C)
- WS281x NeoPixel LED Strip
- ALSA-based audio alerts

### Model Evaluation
- Leave-One-Group-Out (LOGO) Cross-Validation
- Balanced Accuracy
- Weighted F1-Score

## Results and Performance

- The model achieves approximately **86% Balanced Accuracy** under Leave-One-Group-Out Cross-Validation across multiple users.
- The weighted average **F1-Score reaches around 0.83**, indicating stable performance across posture classes.
- The proposed feature engineering approach improves Balanced Accuracy by approximately **19%** compared to using raw keypoint coordinates alone.
- The system runs reliably on Raspberry Pi 4 with near real-time performance using CPU inference (num_threads = 2).
