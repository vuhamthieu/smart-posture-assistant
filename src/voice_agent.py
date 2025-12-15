import threading
import speech_recognition as sr
import os
import sys
import subprocess
import time
from gtts import gTTS
import random
import pickle
import numpy as np
from contextlib import contextmanager
from pyvi import ViTokenizer

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"

IS_BOT_SPEAKING = False

@contextmanager
def ignore_stderr():
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        sys.stderr.flush()
        os.dup2(devnull, 2)
        os.close(devnull)
        yield
    except Exception:
        yield
    finally:
        try:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
        except: pass

MODEL_PATH = "/home/theo/smart-posture-assistant/src/nlp_model.pkl"
nlp_model = None

try:
    with open(MODEL_PATH, 'rb') as f:
        nlp_model = pickle.load(f)
    print(f"{GREEN}[SYSTEM] NLP Model Loaded{RESET}")
except:
    print(f"{RED}[ERROR] NLP Model not found{RESET}")

with ignore_stderr():
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300 
    recognizer.dynamic_energy_threshold = True

def get_target_mic_index():
    with ignore_stderr():
        try:
            mics = sr.Microphone.list_microphone_names()
            print(f"\n{BLUE}[SYSTEM] Scanning Microphones...{RESET}")
            for i, name in enumerate(mics):
                if ("Texas" in name or "PCM2902" in name or "USB PnP" in name) and "sysdefault" not in name:
                    print(f"{GREEN}[HARDWARE] Found Target Mic at Index {i}: {name}{RESET}")
                    return i
            for i, name in enumerate(mics):
                if "USB" in name and "sysdefault" not in name:
                    print(f"{YELLOW}[HARDWARE] Fallback USB Mic at Index {i}: {name}{RESET}")
                    return i
        except: pass
    return None 

RESPONSES = {
    'check_status': ["De toi xem nao...", "Dang kiem tra tu the."],
    'start_timer': ["Ok, bat dau tinh gio 25 phut.", "Da kich hoat che do tap trung."],
    'stop_timer': ["Da dung dong ho.", "Ok, dung tinh gio."],
    'check_time': ["Thoi gian con lai la...", "Ban muon xem gio a?"],
    'greeting': ["Xin chao! Can toi giup gi khong?", "Chao ban!"],
    'unknown': ["Xin loi, toi chua hieu.", "Ban noi lai duoc khong?"]
}

def speak_text(text, lang='vi'):
    global IS_BOT_SPEAKING
    print(f"{CYAN}[BOT] {text}{RESET}")
    IS_BOT_SPEAKING = True
    try:
        filename = f"/tmp/voice_{hash(text)}.mp3"
        if not os.path.exists(filename):
            tts = gTTS(text=text, lang=lang)
            tts.save(filename)
        subprocess.run(['mpg123', '-q', filename], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    time.sleep(1.0)
    IS_BOT_SPEAKING = False

def predict_intent(text):
    if not nlp_model: return "unknown"
    try:
        processed_text = ViTokenizer.tokenize(text)
        intent = nlp_model.predict([processed_text])[0]
        print(f"{BLUE}[ANALYSIS] Input: '{text}' -> Tokenized: '{processed_text}' -> Intent: {intent}{RESET}")
        return intent
    except: return "unknown"

def execute_command(intent, current_stats, timer_obj):
    if intent in RESPONSES: speak_text(random.choice(RESPONSES[intent]))
    
    if intent == 'check_status':
        status = current_stats.get('posture_status', 'Init')
        if status == 'Good': speak_text("Tu the chuan, 10 diem!")
        elif status == 'Bad': speak_text("Ngoi thang lung len ban oi!")
        else: speak_text("He thong dang khoi dong.")
        
    elif intent == 'start_timer':
        if timer_obj: timer_obj.start(25)
        
    elif intent == 'stop_timer':
        if timer_obj: timer_obj.stop()
        
    elif intent == 'check_time':
        if timer_obj and timer_obj.is_running:
            rem = timer_obj.remaining
            speak_text(f"Con {int(rem//60)} phut {int(rem%60)} giay.")
        else:
            speak_text("Dong ho dang tat.")

def voice_listener_loop(led_ctrl, face_ctrl, stats_ref, timer_ref):
    global IS_BOT_SPEAKING
    mic_idx = get_target_mic_index()
    
    try:
        with ignore_stderr():
            microphone = sr.Microphone(device_index=mic_idx)
            
        print(f"{GREEN}[SYSTEM] Voice Agent Listening on Device {mic_idx}...{RESET}")
        print(f"{BLUE}[INFO] Adjusting for noise...{RESET}")
        
        with ignore_stderr():
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source, duration=2)
        
        print(f"{GREEN}[INFO] Ready! Speak now.{RESET}")
            
        while True:
            if IS_BOT_SPEAKING:
                time.sleep(0.1)
                continue

            try:
                with ignore_stderr():
                    with microphone as source:
                        try:
                            audio = recognizer.listen(source, timeout=1, phrase_time_limit=10)
                        except sr.WaitTimeoutError:
                            continue
                
                if IS_BOT_SPEAKING: continue

                led_ctrl.set_color_array([0, 0, 255]) 
                
                try:
                    text = recognizer.recognize_google(audio, language="vi-VN").lower()
                    print(f"{YELLOW}[USER] {text}{RESET}")
                    intent = predict_intent(text)
                    led_ctrl.set_color_array([0, 255, 255])
                    execute_command(intent, stats_ref, timer_ref)
                except sr.UnknownValueError: 
                    pass
                except sr.RequestError: 
                    speak_text("Loi mang.")
                
                led_ctrl.set_color_array([0, 255, 0])
            except Exception:
                pass
                
    except Exception as init_err:
        print(f"{RED}[ERROR] Cannot init microphone: {init_err}{RESET}")
