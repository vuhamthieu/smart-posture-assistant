from __future__ import annotations

import os
import threading
import time

import cv2
from flask import Flask, Response, jsonify, request

import voice_agent
from api_client import PostureIngestClient
from hardware_io import ensure_tts_cache_dir, no_alsa_error, play_mp3, synthesize_tts


class StudyTimer:
    def __init__(self):
        self.duration = 0
        self.remaining = 0
        self.is_running = False
        self.lock = threading.Lock()
        self.start_t = 0.0

    def start(self, minutes: int):
        with self.lock:
            self.duration = int(minutes) * 60
            self.remaining = self.duration
            self.is_running = True
            self.start_t = time.time()

    def stop(self):
        with self.lock:
            self.is_running = False
            self.remaining = 0

    def update(self) -> bool:
        with self.lock:
            if not self.is_running:
                return False

            self.remaining = max(0, self.duration - (time.time() - self.start_t))
            if self.remaining == 0:
                self.is_running = False
                return True
            return False

    def get_time_str(self) -> str:
        with self.lock:
            if not self.is_running:
                return ""
            return f"{int(self.remaining // 60):02d}:{int(self.remaining % 60):02d}"


def decide_upload_type(final_status: str, phys_label: str, ai_label: str, detected_type: str) -> str:
    if final_status != "Bad":
        return "Good"

    ai_clean = str(ai_label).lower().strip()
    phys_clean = str(phys_label).lower().strip()
    dt = (detected_type or "").lower().strip()

    if "tilt" in dt or "tilt" in ai_clean:
        return "Tilt"
    if "lean" in dt or "lean" in phys_clean or "lean" in ai_clean:
        return "Lean"
    if "hunch" in dt or "hunch" in phys_clean or "hunch" in ai_clean:
        return "Hunch"
    return "Hunch"


def keypoints_to_jsonable(kpts):
    if kpts is None:
        return {}
    try:
        return [[float(v) for v in row] for row in kpts.tolist()]
    except Exception:
        try:
            return [[float(v) for v in row] for row in kpts]
        except Exception:
            return {}


def _get_alert_message(config_mgr, key: str):
    lang = config_mgr.get("alert_language", "vi")
    messages = config_mgr.get(f"alert_messages_{lang}", [])
    idx_map = {"lean": 0, "hunch": 1, "tilt": 2, "close": 3}
    if messages and key in idx_map and len(messages) > idx_map[key]:
        return messages[idx_map[key]], lang

    fallback = {
        "lean": "Bạn đang cúi đầu quá thấp",
        "hunch": "Đừng gù lưng",
        "tilt": "Đừng nghiêng đầu",
        "close": "Ngồi xa ra",
    }
    return fallback.get(key, "Cảnh báo"), "vi"


class AlertSpeaker:
    def __init__(self, config_mgr, led_controller, audio_available: bool):
        self.config_mgr = config_mgr
        self.led_controller = led_controller
        self.audio_available = bool(audio_available)
        self.tts_cache_dir = ensure_tts_cache_dir(os.environ.get("TTS_CACHE_DIR", "/tmp/tts_cache"))

    def speak(self, key: str):
        if not self.audio_available:
            return

        text, lang = _get_alert_message(self.config_mgr, key)
        filename = f"{hash(text + lang)}.mp3"
        filepath = os.path.join(self.tts_cache_dir, filename)

        if not os.path.exists(filepath):
            try:
                synthesize_tts(text=text, lang=lang, filepath=filepath)
            except Exception:
                return

        bad_color = self.config_mgr.get("led_color_bad", [255, 0, 0])

        def play_sound():
            voice_agent.IS_BOT_SPEAKING = True
            play_mp3(filepath)
            time.sleep(1.0)
            voice_agent.IS_BOT_SPEAKING = False

        threading.Thread(target=play_sound, daemon=True).start()
        threading.Thread(
            target=lambda: self.led_controller.pulse_while_speaking(bad_color),
            daemon=True,
        ).start()


class PostureRuntime:
    def __init__(
        self,
        *,
        cap,
        detector,
        config_mgr,
        led_controller,
        face_display,
        audio_available: bool,
        ingest_client: PostureIngestClient | None = None,
        heartbeat_interval_s: float | None = None,
    ):
        self.cap = cap
        self.detector = detector
        self.config_mgr = config_mgr
        self.led_controller = led_controller
        self.face_display = face_display
        self.audio_available = bool(audio_available)

        self.ingest_client = ingest_client or PostureIngestClient()
        self.heartbeat_interval_s = float(
            heartbeat_interval_s
            if heartbeat_interval_s is not None
            else os.environ.get("HEARTBEAT_INTERVAL", "15.0")
        )

        self.timer = StudyTimer()
        self.alerts = AlertSpeaker(config_mgr, led_controller, self.audio_available)

        self._frame_lock = threading.Lock()
        self._output_frame = None
        # Shared, mutable stats dict (used by Flask + voice agent)
        self.stats = {"posture_status": "Init", "fps": 0}

        self._last_upload_time = 0.0
        self._last_upload_status = ""

    # --- stats + frame accessors ---

    def get_stats(self) -> dict:
        # return a copy for safe JSON serialization
        return dict(self.stats)

    def _set_status(self, posture_status: str):
        self.stats["posture_status"] = posture_status

    def _set_fps(self, fps: int):
        self.stats["fps"] = int(fps)

    def _set_output_frame(self, frame_bgr):
        with self._frame_lock:
            self._output_frame = frame_bgr

    def _get_output_frame(self):
        with self._frame_lock:
            return self._output_frame

    # --- Flask server ---

    def create_app(self) -> Flask:
        app = Flask(__name__)

        def generate():
            while True:
                frame_to_encode = self._get_output_frame()
                if frame_to_encode is None:
                    time.sleep(0.01)
                    continue

                ok, buf = cv2.imencode(".jpg", frame_to_encode, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                if ok:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + bytearray(buf) + b"\r\n"
                else:
                    time.sleep(0.01)

        @app.get("/video_feed")
        def video_feed():
            return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

        @app.get("/stats")
        def stats():
            return jsonify(self.get_stats())

        @app.post("/command")
        def command():
            payload = request.get_json(silent=True) or {}
            c = payload.get("command")
            if c == "recalib":
                try:
                    self.detector.reset_calibration()
                except Exception:
                    pass
            return jsonify({"success": True})

        @app.get("/")
        def index():
            return (
                """<html><head><title>Posture Monitor</title>
<script>setInterval(()=>{fetch('/stats').then(r=>r.json()).then(d=>{
  document.getElementById('s').innerText=d.posture_status+" | "+d.fps+" FPS";
})},1000)</script></head>
<body style=\"background:#000;color:#fff;text-align:center\">
<h2>Posture Monitor</h2><img src=\"/video_feed\"><h3 id=\"s\">...</h3></body></html>"""
            )

        return app

    # --- background loops ---

    def _timer_updater_loop(self):
        last = 0.0
        while True:
            if self.timer.is_running:
                if time.time() - last >= 1.0:
                    try:
                        self.face_display.draw_timer(self.timer.get_time_str())
                    except Exception:
                        pass
                    last = time.time()

                if self.timer.update():
                    self.alerts.speak("close")
            time.sleep(0.5)

    def _detect_posture_loop(self):
        last_bad = 0.0
        last_alert = 0.0
        fps_frame_cnt = 0
        fps_start_time = time.time()
        frame_idx = 0
        last_final_status = None
        current_led_color = None
        last_icon_style = None

        while True:
            if not self.cap.isOpened():
                time.sleep(1.0)
                continue

            ret, frame = self.cap.read()
            if not ret:
                continue

            neck_threshold = float(self.config_mgr.get("neck_threshold", 35.0))
            nose_drop_threshold = float(self.config_mgr.get("nose_drop_threshold", 0.25))

            result = self.detector.step(
                frame,
                frame_idx=frame_idx,
                neck_threshold_pct=neck_threshold,
                nose_drop_threshold=nose_drop_threshold,
            )

            stream_frame = result["stream_frame"]
            final_status = result["final_status"]
            method = result["method"]
            debug_msg = result["debug"]
            detected_type = result["detected_type"]

            self._set_status(final_status)

            # Upload posture record (HTTP -> Next.js ingest)
            upload_type = decide_upload_type(
                final_status,
                phys_label=result.get("phys_label", ""),
                ai_label=result.get("ai_label", ""),
                detected_type=detected_type,
            )

            now = time.time()
            is_status_changed = upload_type != self._last_upload_status
            is_heartbeat_time = (now - self._last_upload_time) > self.heartbeat_interval_s
            if result.get("calibrated") and (is_status_changed or is_heartbeat_time):
                metrics = result.get("metrics") or {}
                kp_json = keypoints_to_jsonable(result.get("keypoints"))
                threading.Thread(
                    target=self.ingest_client.send_posture_record,
                    args=(upload_type, float(result.get("ai_conf", 0.0)), metrics, kp_json),
                    daemon=True,
                ).start()
                self._last_upload_time = now
                self._last_upload_status = upload_type

            # LED/OLED updates (when timer is not running)
            if not self.timer.is_running and final_status in ("Good", "Bad"):
                if final_status == "Bad":
                    target_color_list = self.config_mgr.get("led_color_bad", [255, 0, 0])
                else:
                    target_color_list = self.config_mgr.get("led_color_good", [0, 255, 0])

                target_color_tuple = tuple(target_color_list)
                target_icon_style = self.config_mgr.get("oled_icon_style", "A")

                is_led_changed = target_color_tuple != current_led_color
                is_status_changed_ui = final_status != last_final_status
                is_style_changed = target_icon_style != last_icon_style

                if is_status_changed_ui or is_led_changed:
                    if final_status == "Good":
                        self.led_controller.stop_pulsing()
                    self.led_controller.set_color_array(target_color_list)
                    current_led_color = target_color_tuple

                if is_status_changed_ui or is_style_changed:
                    if final_status == "Bad":
                        self.face_display.draw_angry(style=target_icon_style)
                    else:
                        self.face_display.draw_normal(style=target_icon_style)
                    last_final_status = final_status
                    last_icon_style = target_icon_style

            # Alert logic
            if final_status == "Bad":
                if time.time() - last_bad > 1.5:
                    if time.time() - last_alert > 8.0:
                        self.alerts.speak(detected_type if detected_type else "lean")
                        last_alert = time.time()
                else:
                    if last_bad == 0.0:
                        last_bad = time.time()
            else:
                last_bad = 0.0

            # draw overlay sometimes
            if fps_frame_cnt % 2 == 0:
                color = (0, 0, 255) if final_status == "Bad" else (0, 255, 0)
                cv2.putText(stream_frame, f"Stat:{final_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(stream_frame, f"{method}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                if debug_msg:
                    cv2.putText(stream_frame, debug_msg, (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            fps_frame_cnt += 1
            if time.time() - fps_start_time >= 1.0:
                self._set_fps(fps_frame_cnt)
                fps_frame_cnt = 0
                fps_start_time = time.time()

            self._set_output_frame(stream_frame.copy())
            frame_idx += 1

    def start_background_threads(self):
        threading.Thread(target=self._detect_posture_loop, daemon=True).start()
        threading.Thread(target=self._timer_updater_loop, daemon=True).start()

        try:
            with no_alsa_error():
                threading.Thread(
                    target=voice_agent.voice_listener_loop,
                    args=(self.led_controller, self.face_display, self.stats, self.timer),
                    daemon=True,
                ).start()
        except Exception as e:
            print(f"Voice Error: {e}")

    def run(self, *, host: str = "0.0.0.0", port: int = 5000):
        self.start_background_threads()
        app = self.create_app()
        app.run(host=host, port=int(port), debug=False, threaded=True)
