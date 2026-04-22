import atexit
import glob
import os
import subprocess
import threading
import time
from contextlib import contextmanager


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def auto_find_camera():
    import cv2

    dev_list = glob.glob('/dev/video*')
    dev_list.sort()
    for dev_path in dev_list:
        try:
            dev_id = int(dev_path.replace('/dev/video', ''))
        except Exception:
            continue

        print(f"{YELLOW}Checking {dev_path}...{RESET}", end=" ")
        temp_cap = cv2.VideoCapture(dev_id)
        if temp_cap.isOpened():
            ret, _ = temp_cap.read()
            temp_cap.release()
            if ret:
                print(f"{GREEN}OK{RESET}")
                return dev_id
        print(f"{RED}Fail{RESET}")
    return None


def reset_camera_driver():
    print(f"{RED}Camera not found. Resetting driver...{RESET}")
    try:
        os.system("sudo modprobe -r uvcvideo")
        time.sleep(1)
        os.system("sudo modprobe uvcvideo")
        time.sleep(2)
    except Exception as e:
        print(f"{RED}Driver reset failed: {e}{RESET}")


def init_camera(width=1024, height=768, fps=30):
    import cv2

    print("Initializing Camera...")
    found_id = auto_find_camera()
    if found_id is None:
        reset_camera_driver()
        found_id = auto_find_camera()
    if found_id is None:
        raise RuntimeError("No camera hardware detected")

    print(f"{GREEN}Selected Camera ID: {found_id}{RESET}")
    cap = cv2.VideoCapture(found_id)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, 120)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        raise RuntimeError("Failed to open camera stream")

    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"{GREEN}Camera Started: {int(w)}x{int(h)}{RESET}")
    return cap


def register_camera_cleanup(cap):
    def cleanup_camera():
        try:
            if cap is not None and cap.isOpened():
                print(f"\n{YELLOW}Releasing camera resource...{RESET}")
                cap.release()
        except Exception:
            pass

    atexit.register(cleanup_camera)


class LEDController:
    def __init__(self, strip, led_count: int):
        self.strip = strip
        self.led_count = led_count
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

    def set_color_array(self, color_array):
        if not self.strip:
            return
        try:
            from rpi_ws281x import Color

            r, g, b = color_array if len(color_array) == 3 else [0, 0, 0]
            with self.lock:
                for i in range(self.led_count):
                    self.strip.setPixelColor(i, Color(r, g, b))
                self.strip.show()
        except Exception:
            pass

    def stop_pulsing(self):
        self.stop_event.set()

    def pulse_while_speaking(self, color_array, duration=3):
        if not self.strip:
            return
        self.stop_event.clear()
        start = time.time()
        r, g, b = color_array if len(color_array) == 3 else [255, 0, 0]
        dim_r, dim_g, dim_b = int(r * 0.2), int(g * 0.2), int(b * 0.2)

        while time.time() - start < duration:
            if self.stop_event.is_set():
                break
            self.set_color_array([r, g, b])
            for _ in range(2):
                if self.stop_event.is_set():
                    break
                time.sleep(0.1)

            if self.stop_event.is_set():
                break
            self.set_color_array([dim_r, dim_g, dim_b])
            for _ in range(2):
                if self.stop_event.is_set():
                    break
                time.sleep(0.1)


def init_led_strip(
    led_count=12,
    led_pin=12,
    led_freq_hz=800000,
    led_dma=10,
    led_invert=False,
    led_channel=0,
):
    try:
        from rpi_ws281x import PixelStrip, Color

        strip = PixelStrip(led_count, led_pin, led_freq_hz, led_dma, led_invert, 20, led_channel)
        strip.begin()
        for i in range(led_count):
            strip.setPixelColor(i, Color(0, 0, 0))
        strip.show()
        print("LED Strip Connected")
        return strip, LEDController(strip, led_count)
    except Exception as e:
        print(f"LED Error: {e}")
        return None, LEDController(None, led_count)


class FaceDisplay:
    def __init__(self, oled, lock):
        self.oled = oled
        self.lock = lock

    def _draw(self, func):
        if not self.oled:
            return
        from PIL import Image, ImageDraw

        with self.lock:
            try:
                img = Image.new("1", (128, 64))
                func(ImageDraw.Draw(img))
                self.oled.image(img)
                self.oled.show()
            except Exception:
                pass

    def draw_normal(self, style='A'):
        if style == 'B':
            self._draw(
                lambda d: (
                    d.ellipse((32, 20, 52, 40), fill=255),
                    d.ellipse((76, 20, 96, 40), fill=255),
                    d.ellipse((42, 24, 46, 28), fill=0),
                    d.ellipse((86, 24, 90, 28), fill=0),
                    d.polygon([(60, 42), (68, 42), (64, 47)], fill=255),
                    d.arc((58, 47, 64, 52), 0, 180, fill=255),
                    d.arc((64, 47, 70, 52), 0, 180, fill=255),
                    d.line((15, 32, 30, 35), fill=255),
                    d.line((15, 40, 30, 40), fill=255),
                    d.line((15, 48, 30, 45), fill=255),
                    d.line((113, 32, 98, 35), fill=255),
                    d.line((113, 40, 98, 40), fill=255),
                    d.line((113, 48, 98, 45), fill=255),
                )
            )
        elif style == 'C':
            self._draw(
                lambda d: (
                    d.rectangle((20, 10, 108, 54), outline=255),
                    d.rectangle((35, 20, 55, 35), fill=255),
                    d.rectangle((73, 20, 93, 35), fill=255),
                    d.line((45, 45, 83, 45), fill=255, width=3),
                    d.rectangle((15, 25, 20, 40), fill=255),
                    d.rectangle((108, 25, 113, 40), fill=255),
                )
            )
        else:
            self._draw(
                lambda d: (
                    d.ellipse((30, 15, 50, 35), outline=255),
                    d.ellipse((78, 15, 98, 35), outline=255),
                    d.arc((40, 35, 88, 55), 0, 180, fill=255, width=2),
                )
            )

    def draw_angry(self, style='A'):
        if style == 'B':
            self._draw(
                lambda d: (
                    d.line((32, 25, 52, 35), fill=255, width=3),
                    d.line((32, 35, 52, 25), fill=255, width=3),
                    d.line((76, 25, 96, 35), fill=255, width=3),
                    d.line((76, 35, 96, 25), fill=255, width=3),
                    d.polygon([(60, 42), (68, 42), (64, 47)], fill=255),
                    d.ellipse((58, 48, 70, 58), outline=255, width=2),
                    d.line((15, 25, 30, 32), fill=255),
                    d.line((15, 40, 30, 40), fill=255),
                    d.line((15, 55, 30, 48), fill=255),
                    d.line((113, 25, 98, 32), fill=255),
                    d.line((113, 40, 98, 40), fill=255),
                    d.line((113, 55, 98, 48), fill=255),
                )
            )
        elif style == 'C':
            self._draw(
                lambda d: (
                    d.rectangle((20, 10, 108, 54), outline=255),
                    d.line((35, 35, 55, 20), fill=255, width=3),
                    d.line((73, 20, 93, 35), fill=255, width=3),
                    d.line((35, 45, 45, 50), fill=255, width=2),
                    d.line((45, 50, 55, 45), fill=255, width=2),
                    d.line((55, 45, 65, 50), fill=255, width=2),
                    d.line((65, 50, 75, 45), fill=255, width=2),
                    d.line((75, 45, 85, 50), fill=255, width=2),
                    d.line((85, 50, 93, 45), fill=255, width=2),
                )
            )
        else:
            self._draw(
                lambda d: (
                    d.line((25, 20, 45, 30), fill=255, width=2),
                    d.line((83, 30, 103, 20), fill=255, width=2),
                    d.ellipse((30, 30, 40, 40), fill=255),
                    d.ellipse((88, 30, 98, 40), fill=255),
                    d.arc((45, 45, 83, 60), 180, 360, fill=255, width=2),
                )
            )

    def draw_timer(self, txt):
        from PIL import ImageFont

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                28,
            )
            self._draw(lambda d: d.text((20, 15), txt, font=font, fill=255))
        except Exception:
            self._draw(lambda d: d.text((45, 25), txt, fill=255))


def init_oled():
    try:
        from adafruit_ssd1306 import SSD1306_I2C
        import board

        oled_lock = threading.Lock()
        i2c = board.I2C()
        oled = SSD1306_I2C(128, 64, i2c, addr=0x3C)
        oled.fill(0)
        oled.show()
        print("OLED Connected")
        return oled, oled_lock, FaceDisplay(oled, oled_lock)
    except Exception as e:
        print(f"OLED Error: {e}")
        oled_lock = threading.Lock()
        return None, oled_lock, FaceDisplay(None, oled_lock)


def init_audio():
    try:
        subprocess.run(['amixer', 'set', 'Master', '100%'], check=False)
        return True
    except Exception:
        return False


def ensure_tts_cache_dir(cache_dir: str = "/tmp/tts_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def play_mp3(filepath: str):
    subprocess.run(['mpg123', '-q', filepath], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def synthesize_tts(text: str, lang: str, filepath: str):
    from gtts import gTTS

    gTTS(text=text, lang=lang).save(filepath)


@contextmanager
def no_alsa_error():
    try:
        from ctypes import CFUNCTYPE, c_char_p, c_int, cdll

        ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)

        def py_error_handler(filename, line, function, err, fmt):
            pass

        c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

        asound = cdll.LoadLibrary('libasound.so')
        asound.snd_lib_error_set_handler(c_error_handler)
        yield
        asound.snd_lib_error_set_handler(None)
    except Exception:
        yield


def register_display_cleanup(oled, strip, led_count: int):
    def cleanup():
        if oled:
            try:
                oled.fill(0)
                oled.show()
            except Exception:
                pass
        if strip:
            try:
                from rpi_ws281x import Color

                for i in range(led_count):
                    strip.setPixelColor(i, Color(0, 0, 0))
                strip.show()
            except Exception:
                pass

    atexit.register(cleanup)
