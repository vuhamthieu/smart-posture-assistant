#!/usr/bin/env python3
import os
from pathlib import Path

from api_client import PostureIngestClient
from config_manager import get_config_mgr
from hardware_io import (
    init_audio,
    init_camera,
    init_led_strip,
    init_oled,
    register_camera_cleanup,
    register_display_cleanup,
)
from ml_inference import PostureDetector, PostureModels, default_model_paths
from runtime import PostureRuntime


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    print("Started")

    config_mgr = get_config_mgr()
    config_mgr.start_polling()

    audio_available = init_audio()
    cap = init_camera()
    register_camera_cleanup(cap)

    led_strip, led_controller = init_led_strip()
    oled, _, face_display = init_oled()
    register_display_cleanup(oled, led_strip, led_count=12)

    if oled:
        face_display.draw_normal(style=config_mgr.get("oled_icon_style", "A"))

    tflite_path, ensemble_model, ensemble_scaler = default_model_paths(PROJECT_ROOT)
    models = PostureModels(tflite_path, ensemble_model, ensemble_scaler)
    detector = PostureDetector(models)

    ingest_client = PostureIngestClient()
    runtime = PostureRuntime(
        cap=cap,
        detector=detector,
        config_mgr=config_mgr,
        led_controller=led_controller,
        face_display=face_display,
        audio_available=audio_available,
        ingest_client=ingest_client,
    )

    host = os.environ.get("POSTURE_HOST", "0.0.0.0")
    port = int(os.environ.get("POSTURE_PORT", "5000"))
    runtime.run(host=host, port=port)


if __name__ == "__main__":
    main()
