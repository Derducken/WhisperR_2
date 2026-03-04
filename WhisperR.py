import sys
import os
import json
import time
import threading
import queue
import subprocess
import shutil
import wave
import zipfile
import urllib.request
import traceback
import logging
from pathlib import Path
from datetime import datetime

# Application version
__version__ = "2.0.0"
APP_NAME = "WhisperR"

# --- 1. GLOBAL CRASH LOGGING ---
def crash_logger(etype, value, tb):
    try:
        with open("CRASH_LOG.txt", "w") as f:
            f.write(f"--- CRASH REPORT {datetime.now()} ---\n")
            f.write(f"{APP_NAME} v{__version__}\n\n")
            traceback.print_exception(etype, value, tb, file=f)
    except: pass
sys.excepthook = crash_logger

# --- 2. DLL & ENVIRONMENT HARDENING ---
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
    LIB_DIR = os.path.join(BASE_DIR, "_internal")
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LIB_DIR = BASE_DIR

# Critical: Prevent torch from loading shared memory DLL in frozen mode
# This fixes the "Invalid access to memory location" error
os.environ["PYTORCH_JIT"] = "0"
os.environ["PYTORCH_JIT_USE_NNC_NOT_NVFUSER"] = "1"
os.environ["QT_PA_PLATFORM"] = "windows:dpiawareness=0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_NVFUSER_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

if os.name == 'nt':
    # Ensure dependencies like zlibwapi.dll and torch/ctranslate libs are found
    dll_search_path = [BASE_DIR, LIB_DIR]
    try:
        import site
        for sp in site.getsitepackages():
            for lib in ["cudnn", "cublas", "cuda_runtime", "ctranslate2"]:
                p = os.path.join(sp, "nvidia", lib, "bin")
                if os.path.exists(p): dll_search_path.append(p)
                p_internal = os.path.join(LIB_DIR, lib)
                if os.path.exists(p_internal): dll_search_path.append(p_internal)
    except: pass

    for p in set(dll_search_path):
        if os.path.exists(p):
            try:
                os.add_dll_directory(p)
                os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
            except: pass

import pyaudio
import numpy as np
import pyautogui
import pyperclip
from pynput import keyboard
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, 
    QPushButton, QComboBox, QLabel, QFileDialog, QTabWidget, QCheckBox, 
    QDoubleSpinBox, QProgressBar, QFormLayout, QLineEdit, QGroupBox, QSpinBox, 
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea, QDialog, QMessageBox,
    QSystemTrayIcon, QMenu
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect, QPoint, QObject
from PyQt6.QtGui import QPainter, QColor, QFont, QIcon, QAction, QKeyEvent

# --- 3. CONSTANTS ---
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]
LANG_MAP = {"Auto": None, "English": "en", "Greek": "el", "German": "de", "French": "fr", "Spanish": "es"}
HALLUCINATIONS = ["thank you.", "thanks for watching.", "god bless.", "god bless you.", "subtitles by", "Thank you for watching, and I'll see you in the next video"]

DARK_STYLE = """
QMainWindow, QDialog, QScrollArea, QTabWidget { background-color: #121212; }
QWidget { color: #e0e0e0; font-family: 'Segoe UI'; font-size: 9pt; }
QTextEdit { background-color: #1e1e1e; border: 1px solid #333; color: #fff; border-radius: 4px; }
QPushButton { background-color: #2a2a2a; border: 1px solid #444; padding: 6px; border-radius: 4px; }
QPushButton:hover { background-color: #353535; border: 1px solid #0078d7; }
QGroupBox { border: 1px solid #333; margin-top: 12px; font-weight: bold; padding: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
QProgressBar { background: #1e1e1e; border: 1px solid #333; text-align: center; height: 12px; border-radius: 6px; }
QProgressBar::chunk { background-color: #0078d7; border-radius: 6px; }
QHeaderView::section { background-color: #252525; color: white; padding: 4px; border: 1px solid #333; }
QTableWidget { background-color: #1e1e1e; gridline-color: #333; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { background-color: #2a2a2a; border: 1px solid #444; padding: 4px; }
"""

# --- 4. LOGGING SETUP ---
class AppLogger:
    def __init__(self):
        self.log_path = os.path.join(BASE_DIR, "app_log.txt")
        self.level = logging.INFO
        self.logger = logging.getLogger(APP_NAME)
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler(self.log_path, mode='w', encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        self.logger.addHandler(fh)
        
        # Console handler for debugging
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        self.logger.addHandler(ch)
    
    def set_level(self, level_name):
        levels = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}
        self.level = levels.get(level_name, logging.INFO)
        self.logger.setLevel(self.level)
    
    def debug(self, msg): self.logger.debug(msg)
    def info(self, msg): self.logger.info(msg)
    def warning(self, msg): self.logger.warning(msg)
    def error(self, msg): self.logger.error(msg)

app_logger = AppLogger()

# --- 5. CONFIGURATION ---
class AppConfig:
    def __init__(self):
        self.path = os.path.join(BASE_DIR, "config.json")
        self.settings = {
            "model": "base", "lang_name": "English", "lang_code": "en", 
            "translate": False, "timestamps": False, "initial_prompt": "",
            "audio_folder": str(Path.home() / "WhisperR_Recordings"),
            "mon_folder": str(Path.home() / "WhisperR_Watch"),
            "clear_exit": False, "save_to_disk": False, "auto_space": True,
            "min_to_tray": False, "input_device_name": "", "paste_delay": 0.5, 
            "hotkey": "ctrl+alt+r", "ptt_key": "f8", 
            "visibility_hotkey": "ctrl+shift+w", "live_mode": "Simple", 
            "dict_mode": "Continuous", "auto_pause_sec": 1.5, "noise_floor": 200, 
            "speech_vol": 1500, "commands": {"Launch Notepad": "notepad.exe"},
            "ind_show": True, "ind_type": "Both", "ind_pos": "Top-Right", 
            "ind_size": 32, "ind_off": 20, "bar_edge": "Top", "bar_size": 5,
            "log_level": "INFO"
        }
        self.load()
        app_logger.set_level(self.settings["log_level"])

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    self.settings.update(loaded)
                app_logger.info("Configuration loaded successfully")
            except Exception as e:
                app_logger.error(f"Failed to load config: {e}")
        for k in ["audio_folder", "mon_folder"]: 
            Path(self.settings[k]).mkdir(parents=True, exist_ok=True)

    def save(self):
        try:
            with open(self.path, 'w', encoding='utf-8') as f: 
                json.dump(self.settings, f, indent=4)
            app_logger.info("Configuration saved successfully")
        except Exception as e:
            app_logger.error(f"Failed to save config: {e}")

# --- 6. WORKERS ---
class CalibrationWorker(QThread):
    progress = pyqtSignal(int)
    status_msg = pyqtSignal(str)
    finished = pyqtSignal(int, int)
    
    def __init__(self, dev_idx): 
        super().__init__()
        self.dev_idx = dev_idx
        app_logger.info(f"Calibration worker initialized for device {dev_idx}")
    
    def run(self):
        p = pyaudio.PyAudio()
        try:
            dev_info = p.get_device_info_by_index(self.dev_idx)
            rate = int(dev_info['defaultSampleRate'])
            app_logger.info(f"Calibration starting: device={dev_info['name']}, rate={rate}")
            
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, 
                          input_device_index=self.dev_idx, frames_per_buffer=1024)
            
            n, s = [], []
            self.status_msg.emit("Stay SILENT (Noise detection)...")
            for i in range(100):
                d = stream.read(1024, exception_on_overflow=False)
                n.append(np.sqrt(np.mean(np.frombuffer(d, dtype=np.int16).astype(np.float64)**2)))
                self.progress.emit(i+1)
                time.sleep(0.04)
            
            noise_level = int(np.percentile(n, 90))
            app_logger.info(f"Noise level calibrated: {noise_level}")
            
            self.status_msg.emit("SPEAK normally (Voice level)...")
            for i in range(100):
                d = stream.read(1024, exception_on_overflow=False)
                s.append(np.sqrt(np.mean(np.frombuffer(d, dtype=np.int16).astype(np.float64)**2)))
                self.progress.emit(i+101)
                time.sleep(0.04)
            
            speech_level = int(np.percentile(s, 90))
            app_logger.info(f"Speech level calibrated: {speech_level}")
            
            self.finished.emit(noise_level, speech_level)
            stream.stop_stream()
            stream.close()
        except Exception as e:
            app_logger.error(f"Calibration error: {e}")
        finally:
            p.terminate()

class TranscriberWorker(QThread):
    finished_text = pyqtSignal(str, str)
    status_changed = pyqtSignal(bool)
    log_msg = pyqtSignal(str)
    
    def __init__(self, config): 
        super().__init__()
        self.config = config
        self.queue = queue.Queue()
        self.running = True
        self.model = None
        app_logger.info("Transcriber worker initialized")
    
    def run(self):
        try:
            # Import in thread to avoid main thread DLL issues
            from faster_whisper import WhisperModel
            app_logger.info("faster_whisper imported successfully")
        except Exception as e:
            err_msg = f"AI Import Error: {e}"
            app_logger.error(err_msg)
            self.log_msg.emit(err_msg)
            return
        
        while self.running:
            try:
                task = self.queue.get(timeout=1)
                audio_data, src = task
                app_logger.debug(f"Processing audio from source: {src}")
                
                if not self.model:
                    model_name = self.config.settings['model']
                    self.log_msg.emit(f"Loading {model_name} (trying GPU)...")
                    app_logger.info(f"Loading Whisper model: {model_name}")
                    
                    try:
                        # Try CUDA first
                        self.model = WhisperModel(
                            model_name, 
                            device="cuda", 
                            compute_type="float16",
                            # Critical: Disable features that cause DLL issues in frozen apps
                            download_root=None,
                            local_files_only=False
                        )
                        app_logger.info("Model loaded successfully on GPU")
                        self.log_msg.emit("✓ GPU acceleration active")
                    except Exception as e:
                        app_logger.warning(f"GPU loading failed: {e}. Trying CPU...")
                        self.log_msg.emit(f"GPU unavailable. Using CPU...")
                        try:
                            self.model = WhisperModel(
                                model_name, 
                                device="cpu", 
                                compute_type="int8",
                                download_root=None,
                                local_files_only=False
                            )
                            app_logger.info("Model loaded successfully on CPU")
                            self.log_msg.emit("✓ CPU mode active (slower but stable)")
                        except Exception as e2:
                            app_logger.error(f"CPU loading also failed: {e2}")
                            self.log_msg.emit(f"Model loading failed: {e2}")
                            self.queue.task_done()
                            continue
                
                self.status_changed.emit(True)
                
                lang_code = self.config.settings["lang_code"]
                task_type = "translate" if self.config.settings["translate"] else "transcribe"
                
                app_logger.debug(f"Transcribing: lang={lang_code}, task={task_type}")
                
                segs, _ = self.model.transcribe(
                    audio_data, 
                    language=lang_code, 
                    vad_filter=True, 
                    initial_prompt=self.config.settings["initial_prompt"],
                    task=task_type
                )
                
                text = " ".join([s.text.strip() for s in segs if s.no_speech_prob < 0.8]).strip()
                
                if text and text.lower() not in HALLUCINATIONS:
                    app_logger.info(f"Transcription completed: '{text[:50]}...'")
                    self.finished_text.emit(text, src)
                else:
                    app_logger.debug("No valid speech detected or hallucination filtered")
                
                self.status_changed.emit(False)
                self.queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                err_msg = f"AI Error: {e}"
                app_logger.error(err_msg)
                self.log_msg.emit(err_msg)
                self.status_changed.emit(False)

class AudioRecorder(QThread):
    data_ready = pyqtSignal(object)
    speech_active = pyqtSignal(bool)
    volume_out = pyqtSignal(int)
    
    def __init__(self, config): 
        super().__init__()
        self.config = config
        self.active = False
        self.ptt_pressed = False
        app_logger.info("Audio recorder initialized")
    
    def run(self):
        p = pyaudio.PyAudio()
        idx = 0
        rate = 16000
        
        # Find the selected device
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev["name"] == self.config.settings["input_device_name"]:
                idx = i
                rate = int(dev['defaultSampleRate'])
                app_logger.info(f"Using audio device: {dev['name']} at {rate}Hz")
                break
        
        try:
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, 
                          input_device_index=idx, frames_per_buffer=2048)
            app_logger.info("Audio stream opened successfully")
        except Exception as e:
            app_logger.warning(f"Failed to open selected device: {e}. Using default.")
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True)
            rate = 16000
        
        frames = []
        last_speech = time.time()
        threshold = (self.config.settings["noise_floor"] + self.config.settings["speech_vol"]) / 2
        app_logger.info(f"Recording started with threshold: {threshold}")
        
        self.active = True
        
        while self.active:
            if self.config.settings["live_mode"] == "Push-To-Talk" and not self.ptt_pressed:
                time.sleep(0.05)
                continue
            
            try:
                data = stream.read(1024, exception_on_overflow=False)
                rms = int(np.sqrt(np.mean(np.frombuffer(data, dtype=np.int16).astype(np.float64)**2)))
                self.volume_out.emit(rms)
                
                if rms > threshold:
                    if not frames:
                        self.speech_active.emit(True)
                        app_logger.debug("Speech detected")
                    frames.append(data)
                    last_speech = time.time()
                elif frames:
                    frames.append(data)
                
                if self.config.settings["dict_mode"] == "Auto-Pause" and (time.time() - last_speech) > self.config.settings["auto_pause_sec"]:
                    if len(frames) > 20:
                        app_logger.debug(f"Auto-pause triggered, dispatching {len(frames)} frames")
                        self.speech_active.emit(False)
                        self.dispatch(frames, rate)
                    frames = []
                    last_speech = time.time()
            except Exception as e:
                app_logger.error(f"Audio recording error: {e}")
                break
        
        if len(frames) > 20:
            self.dispatch(frames, rate)
        
        self.speech_active.emit(False)
        stream.stop_stream()
        stream.close()
        p.terminate()
        app_logger.info("Audio recording stopped")
    
    def dispatch(self, frames, rate):
        raw_np = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0
        
        if rate != 16000:
            audio_16k = np.interp(
                np.linspace(0, 1, int(len(raw_np)*16000/rate)), 
                np.linspace(0, 1, len(raw_np)), 
                raw_np
            ).astype(np.float32)
        else:
            audio_16k = raw_np
        
        if self.config.settings["save_to_disk"]:
            path = os.path.join(self.config.settings["audio_folder"], f"rec_{int(time.time()*1000)}.wav")
            try:
                with wave.open(path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes((audio_16k * 32767).astype(np.int16).tobytes())
                app_logger.debug(f"Audio saved to: {path}")
                self.data_ready.emit(os.path.abspath(path))
            except Exception as e:
                app_logger.error(f"Failed to save audio: {e}")
        else:
            self.data_ready.emit(audio_16k)

# --- 7. UI UTILS ---
class HotkeyCaptureDialog(QDialog):
    key_captured = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hotkey Recorder")
        self.setFixedSize(350, 180)
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        
        info_label = QLabel(
            "Press your desired key combination.\n\n"
            "Supported modifiers: Ctrl, Shift, Alt\n"
            "Supported keys: A-Z, 0-9, F1-F12, and more\n\n"
            "Press Escape to cancel."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        self.status_label = QLabel("Waiting for input...")
        self.status_label.setStyleSheet("color: #0078d7; font-weight: bold;")
        layout.addWidget(self.status_label)
        
        self.pressed_keys = set()
        self.main_key = None
        
        app_logger.debug("Hotkey capture dialog opened")
    
    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            app_logger.debug("Hotkey capture cancelled")
            self.reject()
            return
        
        # Handle modifiers
        if e.key() in [Qt.Key.Key_Control, Qt.Key.Key_Shift, Qt.Key.Key_Alt]:
            mod_name = {
                Qt.Key.Key_Control: "ctrl",
                Qt.Key.Key_Shift: "shift",
                Qt.Key.Key_Alt: "alt"
            }.get(e.key())
            self.pressed_keys.add(mod_name)
            self.update_status()
        else:
            # Handle main key
            key_text = self.get_key_name(e.key())
            if key_text:
                self.main_key = key_text
                self.update_status()
                # Auto-accept after main key is pressed
                QTimer.singleShot(200, self.accept_hotkey)
    
    def get_key_name(self, key):
        """Convert Qt key code to readable key name"""
        # Function keys
        if Qt.Key.Key_F1 <= key <= Qt.Key.Key_F12:
            return f"f{key - Qt.Key.Key_F1 + 1}"
        
        # Number keys
        if Qt.Key.Key_0 <= key <= Qt.Key.Key_9:
            return chr(key)
        
        # Letter keys
        if Qt.Key.Key_A <= key <= Qt.Key.Key_Z:
            return chr(key).lower()
        
        # Special keys
        special_keys = {
            Qt.Key.Key_Space: "space",
            Qt.Key.Key_Return: "return",
            Qt.Key.Key_Enter: "enter",
            Qt.Key.Key_Tab: "tab",
            Qt.Key.Key_Backspace: "backspace",
            Qt.Key.Key_Delete: "delete",
            Qt.Key.Key_Insert: "insert",
            Qt.Key.Key_Home: "home",
            Qt.Key.Key_End: "end",
            Qt.Key.Key_PageUp: "page_up",
            Qt.Key.Key_PageDown: "page_down",
            Qt.Key.Key_Up: "up",
            Qt.Key.Key_Down: "down",
            Qt.Key.Key_Left: "left",
            Qt.Key.Key_Right: "right",
        }
        
        return special_keys.get(key, None)
    
    def update_status(self):
        parts = []
        if "ctrl" in self.pressed_keys:
            parts.append("Ctrl")
        if "shift" in self.pressed_keys:
            parts.append("Shift")
        if "alt" in self.pressed_keys:
            parts.append("Alt")
        if self.main_key:
            parts.append(self.main_key.upper())
        
        if parts:
            self.status_label.setText(" + ".join(parts))
    
    def accept_hotkey(self):
        if self.main_key:
            parts = []
            if "ctrl" in self.pressed_keys:
                parts.append("ctrl")
            if "shift" in self.pressed_keys:
                parts.append("shift")
            if "alt" in self.pressed_keys:
                parts.append("alt")
            parts.append(self.main_key)
            
            hotkey_str = "+".join(parts)
            app_logger.info(f"Hotkey captured: {hotkey_str}")
            self.key_captured.emit(hotkey_str)
            self.accept()

# --- 8. MAIN APP ---
class WhisperRApp(QMainWindow):
    sig_toggle_vis = pyqtSignal()
    sig_toggle_rec = pyqtSignal()

    def __init__(self):
        super().__init__()
        app_logger.info(f"=== {APP_NAME} Application Starting ===")
        app_logger.info(f"Base directory: {BASE_DIR}")
        app_logger.info(f"Frozen: {getattr(sys, 'frozen', False)}")
        
        self.config = AppConfig()
        self.recorder = None
        
        self.transcriber = TranscriberWorker(self.config)
        self.transcriber.finished_text.connect(self.on_text)
        self.transcriber.status_changed.connect(self.on_trans_status)
        self.transcriber.log_msg.connect(lambda m: self.scratchpad.append(f"[System] {m}"))
        self.transcriber.start()
        
        self.indicator = StatusOverlay(self.config)
        
        self.sig_toggle_vis.connect(self.toggle_visibility_safe)
        self.sig_toggle_rec.connect(self.toggle_rec)
        
        self.tray = QSystemTrayIcon(self)
        self.tray.setIcon(QIcon.fromTheme("audio-input-microphone"))
        tm = QMenu()
        tm.addAction("Show/Restore", self.toggle_visibility_safe)
        tm.addAction("Quit", QApplication.instance().quit)
        self.tray.setContextMenu(tm)
        self.tray.show()
        
        self.setup_ui()
        self.setup_logic()
        
        self.m_timer = QTimer()
        self.m_timer.timeout.connect(self.monitor_dirs)
        self.m_timer.start(5000)
        
        self.pa_sys = pyaudio.PyAudio()
        self.meter_stream = None
        self.meter_timer = QTimer()
        self.meter_timer.timeout.connect(self.update_meter)
        self.meter_timer.start(100)
        
        app_logger.info("Application initialized successfully")

    def toggle_visibility_safe(self):
        if self.isVisible() and self.windowState() != Qt.WindowState.WindowMinimized:
            if self.config.settings["min_to_tray"]:
                self.hide()
                app_logger.debug("Window hidden to tray")
            else:
                self.setWindowState(Qt.WindowState.WindowMinimized)
        else:
            self.show()
            self.setWindowState(Qt.WindowState.WindowNoState)
            self.raise_()
            self.activateWindow()
            app_logger.debug("Window restored")

    def setup_ui(self):
        self.setWindowTitle(f"{APP_NAME} v{__version__}")
        self.resize(600, 500)
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # ===== MAIN TAB =====
        t1 = QWidget()
        l1 = QVBoxLayout(t1)
        
        l1.addWidget(QLabel("Logs & Results:"))
        self.scratchpad = QTextEdit()
        self.scratchpad.setFont(QFont("Consolas", 9))
        self.scratchpad.setMaximumHeight(300)
        l1.addWidget(self.scratchpad)
        
        hb = QHBoxLayout()
        self.btn_toggle = QPushButton("Start Dictation")
        self.btn_toggle.setFixedHeight(40)
        self.btn_toggle.clicked.connect(self.toggle_rec)
        
        self.btn_import = QPushButton("Import Audio Files")
        self.btn_import.setFixedHeight(40)
        self.btn_import.clicked.connect(self.import_files)
        
        hb.addWidget(self.btn_toggle)
        hb.addWidget(self.btn_import)
        l1.addLayout(hb)
        
        self.tabs.addTab(t1, "Main")
        
        # ===== PROMPT TAB =====
        tp = QWidget()
        lp = QVBoxLayout(tp)
        
        lp.addWidget(QLabel("Whisper Steering Prompt (helps guide transcription):"))
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setText(self.config.settings["initial_prompt"])
        self.prompt_edit.setMaximumHeight(250)
        lp.addWidget(self.prompt_edit)
        
        hbp = QHBoxLayout()
        bi = QPushButton("Import .txt")
        bi.clicked.connect(self.import_p)
        be = QPushButton("Export .txt")
        be.clicked.connect(self.export_p)
        hbp.addWidget(bi)
        hbp.addWidget(be)
        lp.addLayout(hbp)
        lp.addStretch()
        
        self.tabs.addTab(tp, "AI Prompt")
        
        # ===== COMMANDS TAB =====
        t2 = QWidget()
        l2 = QVBoxLayout(t2)
        
        l2.addWidget(QLabel("Voice Commands (phrase detection → action):"))
        
        self.cmd_table = QTableWidget(0, 2)
        self.cmd_table.setHorizontalHeaderLabels(["Phrase to Detect", "Command to Execute"])
        self.cmd_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        for k, v in self.config.settings["commands"].items():
            r = self.cmd_table.rowCount()
            self.cmd_table.insertRow(r)
            self.cmd_table.setItem(r, 0, QTableWidgetItem(k))
            self.cmd_table.setItem(r, 1, QTableWidgetItem(v))
        
        l2.addWidget(self.cmd_table)
        
        btn_row = QHBoxLayout()
        ba = QPushButton("Add Row")
        ba.clicked.connect(lambda: self.cmd_table.insertRow(self.cmd_table.rowCount()))
        
        bd = QPushButton("Delete Selected Row")
        bd.clicked.connect(self.delete_command_row)
        
        btn_row.addWidget(ba)
        btn_row.addWidget(bd)
        l2.addLayout(btn_row)
        
        self.tabs.addTab(t2, "Commands")
        
        # ===== SETTINGS TAB =====
        sc = QScrollArea()
        cw = QWidget()
        main_layout = QVBoxLayout(cw)
        
        # --- AI Model Settings ---
        ai_group = QGroupBox("AI Model Settings")
        ai_layout = QFormLayout()
        
        self.cfg_model = QComboBox()
        self.cfg_model.addItems(WHISPER_MODELS)
        self.cfg_model.setCurrentText(self.config.settings["model"])
        ai_layout.addRow("Whisper Model:", self.cfg_model)
        
        self.cfg_lang = QComboBox()
        self.cfg_lang.addItems(list(LANG_MAP.keys()))
        self.cfg_lang.setCurrentText(self.config.settings["lang_name"])
        ai_layout.addRow("Language:", self.cfg_lang)
        
        self.cfg_ts = QCheckBox("Include timestamps")
        self.cfg_ts.setChecked(self.config.settings["timestamps"])
        ai_layout.addRow(self.cfg_ts)
        
        self.cfg_trans = QCheckBox("Translation mode (to English)")
        self.cfg_trans.setChecked(self.config.settings["translate"])
        ai_layout.addRow(self.cfg_trans)
        
        ai_group.setLayout(ai_layout)
        main_layout.addWidget(ai_group)
        
        # --- Audio Input Settings ---
        audio_group = QGroupBox("Audio Input Settings")
        audio_layout = QFormLayout()
        
        self.cfg_mic = QComboBox()
        self.pop_mics()
        audio_layout.addRow("Microphone:", self.cfg_mic)
        
        self.cfg_dict_m = QComboBox()
        self.cfg_dict_m.addItems(["Continuous", "Auto-Pause"])
        self.cfg_dict_m.setCurrentText(self.config.settings["dict_mode"])
        audio_layout.addRow("Detection Mode:", self.cfg_dict_m)
        
        self.cfg_p_sec = QDoubleSpinBox()
        self.cfg_p_sec.setRange(0.1, 5.0)
        self.cfg_p_sec.setValue(self.config.settings["auto_pause_sec"])
        self.cfg_p_sec.setSuffix(" sec")
        audio_layout.addRow("Silence Threshold:", self.cfg_p_sec)
        
        audio_group.setLayout(audio_layout)
        main_layout.addWidget(audio_group)
        
        # --- Microphone Calibration ---
        cal_group = QGroupBox("Microphone Calibration")
        cal_layout = QVBoxLayout()
        
        cal_layout.addWidget(QLabel("Live Input Level:"))
        self.live_meter = QProgressBar()
        self.live_meter.setRange(0, 5000)
        cal_layout.addWidget(self.live_meter)
        
        self.btn_cal = QPushButton("Run Auto-Calibration")
        self.btn_cal.clicked.connect(self.start_cal)
        cal_layout.addWidget(self.btn_cal)
        
        self.cal_prog = QProgressBar()
        cal_layout.addWidget(self.cal_prog)
        
        self.lbl_cal = QLabel("Idle")
        cal_layout.addWidget(self.lbl_cal)
        
        levels_layout = QHBoxLayout()
        levels_layout.addWidget(QLabel("Noise Floor:"))
        self.n_spin = QSpinBox()
        self.n_spin.setRange(0, 8000)
        self.n_spin.setValue(self.config.settings["noise_floor"])
        levels_layout.addWidget(self.n_spin)
        
        levels_layout.addWidget(QLabel("Speech Level:"))
        self.s_spin = QSpinBox()
        self.s_spin.setRange(0, 8000)
        self.s_spin.setValue(self.config.settings["speech_vol"])
        levels_layout.addWidget(self.s_spin)
        
        cal_layout.addLayout(levels_layout)
        cal_group.setLayout(cal_layout)
        main_layout.addWidget(cal_group)
        
        # --- Hotkeys ---
        hotkey_group = QGroupBox("Keyboard Shortcuts")
        hotkey_layout = QFormLayout()
        
        self.btn_hk1 = QPushButton(self.config.settings["hotkey"])
        self.btn_hk1.clicked.connect(lambda: self.cap_hk(self.btn_hk1, "hotkey"))
        hotkey_layout.addRow("Toggle Dictation:", self.btn_hk1)
        
        self.btn_hk2 = QPushButton(self.config.settings["ptt_key"])
        self.btn_hk2.clicked.connect(lambda: self.cap_hk(self.btn_hk2, "ptt_key"))
        hotkey_layout.addRow("Push-to-Talk:", self.btn_hk2)
        
        self.btn_hk_vis = QPushButton(self.config.settings["visibility_hotkey"])
        self.btn_hk_vis.clicked.connect(lambda: self.cap_hk(self.btn_hk_vis, "visibility_hotkey"))
        hotkey_layout.addRow("Show/Hide Window:", self.btn_hk_vis)
        
        hotkey_group.setLayout(hotkey_layout)
        main_layout.addWidget(hotkey_group)
        
        # --- Output & Behavior ---
        output_group = QGroupBox("Output & Behavior")
        output_layout = QFormLayout()
        
        self.cfg_p_win = QDoubleSpinBox()
        self.cfg_p_win.setRange(0.1, 5.0)
        self.cfg_p_win.setValue(self.config.settings["paste_delay"])
        self.cfg_p_win.setSuffix(" sec")
        output_layout.addRow("Paste Delay:", self.cfg_p_win)
        
        self.cfg_space = QCheckBox("Auto-append space after paste")
        self.cfg_space.setChecked(self.config.settings["auto_space"])
        output_layout.addRow(self.cfg_space)
        
        self.cfg_tray = QCheckBox("Minimize to system tray")
        self.cfg_tray.setChecked(self.config.settings["min_to_tray"])
        output_layout.addRow(self.cfg_tray)
        
        output_group.setLayout(output_layout)
        main_layout.addWidget(output_group)
        
        # --- File Storage ---
        storage_group = QGroupBox("File Storage")
        storage_layout = QFormLayout()
        
        rec_row = QHBoxLayout()
        self.cfg_folder = QLineEdit(self.config.settings["audio_folder"])
        b_f = QPushButton("Browse")
        b_f.clicked.connect(lambda: self.browse_f(self.cfg_folder))
        rec_row.addWidget(self.cfg_folder)
        rec_row.addWidget(b_f)
        storage_layout.addRow("Recordings Folder:", rec_row)
        
        mon_row = QHBoxLayout()
        self.cfg_mon = QLineEdit(self.config.settings["mon_folder"])
        b_m = QPushButton("Browse")
        b_m.clicked.connect(lambda: self.browse_f(self.cfg_mon))
        mon_row.addWidget(self.cfg_mon)
        mon_row.addWidget(b_m)
        storage_layout.addRow("Monitor Folder:", mon_row)
        
        self.cfg_ram = QCheckBox("RAM-only mode (no disk writes)")
        self.cfg_ram.setChecked(not self.config.settings["save_to_disk"])
        storage_layout.addRow(self.cfg_ram)
        
        self.cfg_clear = QCheckBox("Clear recordings on exit")
        self.cfg_clear.setChecked(self.config.settings["clear_exit"])
        storage_layout.addRow(self.cfg_clear)
        
        storage_group.setLayout(storage_layout)
        main_layout.addWidget(storage_group)
        
        # --- Visual Indicators ---
        visual_group = QGroupBox("Visual Indicators")
        visual_layout = QFormLayout()
        
        self.cfg_ind_show = QCheckBox("Enable status indicators")
        self.cfg_ind_show.setChecked(self.config.settings["ind_show"])
        visual_layout.addRow(self.cfg_ind_show)
        
        self.cfg_ind_type = QComboBox()
        self.cfg_ind_type.addItems(["Icons", "Bar", "Both"])
        self.cfg_ind_type.setCurrentText(self.config.settings["ind_type"])
        visual_layout.addRow("Indicator Type:", self.cfg_ind_type)
        
        self.cfg_ind_pos = QComboBox()
        self.cfg_ind_pos.addItems(["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"])
        self.cfg_ind_pos.setCurrentText(self.config.settings["ind_pos"])
        visual_layout.addRow("Icon Position:", self.cfg_ind_pos)
        
        self.cfg_bar_edge = QComboBox()
        self.cfg_bar_edge.addItems(["Top", "Bottom", "Left", "Right"])
        self.cfg_bar_edge.setCurrentText(self.config.settings["bar_edge"])
        visual_layout.addRow("Bar Edge:", self.cfg_bar_edge)
        
        self.cfg_ind_sz = QSpinBox()
        self.cfg_ind_sz.setRange(16, 256)
        self.cfg_ind_sz.setValue(self.config.settings["ind_size"])
        visual_layout.addRow("Icon Size:", self.cfg_ind_sz)
        
        self.cfg_ind_off = QSpinBox()
        self.cfg_ind_off.setRange(0, 256)
        self.cfg_ind_off.setValue(self.config.settings["ind_off"])
        visual_layout.addRow("Corner Offset:", self.cfg_ind_off)
        
        visual_group.setLayout(visual_layout)
        main_layout.addWidget(visual_group)
        
        # --- Advanced ---
        advanced_group = QGroupBox("Advanced")
        advanced_layout = QFormLayout()
        
        self.cfg_log_level = QComboBox()
        self.cfg_log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.cfg_log_level.setCurrentText(self.config.settings["log_level"])
        advanced_layout.addRow("Logging Level:", self.cfg_log_level)
        
        self.btn_setup = QPushButton("Download GPU Dependencies")
        self.btn_setup.setStyleSheet("background-color: #27ae60; color: white;")
        self.btn_setup.clicked.connect(self.setup_deps)
        advanced_layout.addRow(self.btn_setup)
        
        btn_open_log = QPushButton("Open Log File")
        btn_open_log.clicked.connect(self.open_log_file)
        advanced_layout.addRow(btn_open_log)
        
        advanced_group.setLayout(advanced_layout)
        main_layout.addWidget(advanced_group)
        
        # Save button
        btn_s = QPushButton("💾 SAVE ALL SETTINGS")
        btn_s.setFixedHeight(40)
        btn_s.setStyleSheet("background-color: #0078d7; color: white; font-weight: bold;")
        btn_s.clicked.connect(self.save_cfg)
        main_layout.addWidget(btn_s)
        
        # Version label at bottom
        version_label = QLabel(f"{APP_NAME} v{__version__}")
        version_label.setStyleSheet("color: #666; font-size: 8pt; padding: 5px;")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(version_label)
        
        main_layout.addStretch()
        
        sc.setWidget(cw)
        sc.setWidgetResizable(True)
        self.tabs.addTab(sc, "Settings")

    def delete_command_row(self):
        current_row = self.cmd_table.currentRow()
        if current_row >= 0:
            self.cmd_table.removeRow(current_row)
            app_logger.debug(f"Deleted command row {current_row}")
        else:
            QMessageBox.warning(self, "No Selection", "Please select a row to delete.")

    def pop_mics(self):
        p = pyaudio.PyAudio()
        self.cfg_mic.clear()
        sel = 0
        
        app_logger.debug("Scanning audio input devices:")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                h = p.get_host_api_info_by_index(info["hostApi"])["name"]
                name = f"{info['name']} ({h})"
                self.cfg_mic.addItem(name, i)
                app_logger.debug(f"  Device {i}: {name}")
                
                if info["name"] in self.config.settings["input_device_name"]:
                    sel = self.cfg_mic.count()-1
        
        self.cfg_mic.setCurrentIndex(sel)
        p.terminate()

    def update_meter(self):
        if self.recorder and self.recorder.active:
            return
        
        try:
            if not self.meter_stream:
                idx = self.cfg_mic.currentData()
                sr = int(self.pa_sys.get_device_info_by_index(idx)['defaultSampleRate'])
                self.meter_stream = self.pa_sys.open(
                    format=pyaudio.paInt16, 
                    channels=1, 
                    rate=sr, 
                    input=True, 
                    input_device_index=idx, 
                    frames_per_buffer=1024
                )
            
            d = self.meter_stream.read(1024, exception_on_overflow=False)
            rms = int(np.sqrt(np.mean(np.frombuffer(d, dtype=np.int16).astype(np.float64)**2)))
            self.live_meter.setValue(rms)
        except Exception as e:
            app_logger.debug(f"Meter update error: {e}")
            if self.meter_stream:
                self.meter_stream.close()
                self.meter_stream = None

    def save_cfg(self):
        # Collect commands from table
        cmds = {}
        for r in range(self.cmd_table.rowCount()):
            phrase_item = self.cmd_table.item(r, 0)
            cmd_item = self.cmd_table.item(r, 1)
            if phrase_item and cmd_item:
                phrase = phrase_item.text().strip()
                cmd = cmd_item.text().strip()
                if phrase and cmd:
                    cmds[phrase] = cmd
        
        # Update all settings
        self.config.settings.update({
            "model": self.cfg_model.currentText(),
            "lang_name": self.cfg_lang.currentText(),
            "lang_code": LANG_MAP[self.cfg_lang.currentText()],
            "audio_folder": self.cfg_folder.text(),
            "mon_folder": self.cfg_mon.text(),
            "clear_exit": self.cfg_clear.isChecked(),
            "save_to_disk": not self.cfg_ram.isChecked(),
            "input_device_name": self.cfg_mic.currentText(),
            "dict_mode": self.cfg_dict_m.currentText(),
            "auto_pause_sec": self.cfg_p_sec.value(),
            "paste_delay": self.cfg_p_win.value(),
            "hotkey": self.btn_hk1.text(),
            "ptt_key": self.btn_hk2.text(),
            "visibility_hotkey": self.btn_hk_vis.text(),
            "noise_floor": self.n_spin.value(),
            "speech_vol": self.s_spin.value(),
            "commands": cmds,
            "initial_prompt": self.prompt_edit.toPlainText(),
            "min_to_tray": self.cfg_tray.isChecked(),
            "auto_space": self.cfg_space.isChecked(),
            "ind_show": self.cfg_ind_show.isChecked(),
            "ind_type": self.cfg_ind_type.currentText(),
            "ind_pos": self.cfg_ind_pos.currentText(),
            "bar_edge": self.cfg_bar_edge.currentText(),
            "ind_size": self.cfg_ind_sz.value(),
            "ind_off": self.cfg_ind_off.value(),
            "timestamps": self.cfg_ts.isChecked(),
            "translate": self.cfg_trans.isChecked(),
            "log_level": self.cfg_log_level.currentText()
        })
        
        self.config.save()
        app_logger.set_level(self.config.settings["log_level"])
        self.scratchpad.append("✓ Settings saved successfully")
        
        # Restart hotkey listeners with new keys
        self.setup_logic()

    def start_cal(self):
        if self.recorder and self.recorder.active:
            QMessageBox.warning(self, "Recording Active", "Stop dictation before calibrating.")
            return
        
        if self.meter_stream:
            self.meter_stream.close()
            self.meter_stream = None
        
        self.btn_cal.setEnabled(False)
        self.cal_w = CalibrationWorker(self.cfg_mic.currentData())
        self.cal_w.progress.connect(self.cal_prog.setValue)
        self.cal_w.status_msg.connect(self.lbl_cal.setText)
        self.cal_w.finished.connect(self.on_calibration_finished)
        self.cal_w.start()
    
    def on_calibration_finished(self, noise, speech):
        self.n_spin.setValue(noise)
        self.s_spin.setValue(speech)
        self.btn_cal.setEnabled(True)
        self.lbl_cal.setText("✓ Calibration complete")
        app_logger.info(f"Calibration complete: noise={noise}, speech={speech}")

    def toggle_rec(self):
        if self.recorder and self.recorder.active:
            self.recorder.active = False
            self.btn_toggle.setText("Start Dictation")
            self.indicator.is_list = False
            self.indicator.is_rec = False
            app_logger.info("Dictation stopped")
        else:
            if self.meter_stream:
                self.meter_stream.close()
                self.meter_stream = None
            
            self.recorder = AudioRecorder(self.config)
            self.recorder.data_ready.connect(lambda d: self.transcriber.queue.put((d, "live")))
            self.recorder.speech_active.connect(lambda a: (setattr(self.indicator, 'is_rec', a), self.indicator.update()))
            self.recorder.volume_out.connect(self.live_meter.setValue)
            self.recorder.start()
            
            self.btn_toggle.setText("⏹ STOP DICTATION")
            self.indicator.is_list = True
            app_logger.info("Dictation started")
        
        self.indicator.update()

    def on_p(self, key):
        try:
            key_str = self.key_to_string(key)
            if key_str == self.config.settings["ptt_key"] and self.recorder:
                self.recorder.ptt_pressed = True
                app_logger.debug("PTT pressed")
        except Exception as e:
            app_logger.debug(f"PTT press error: {e}")
    
    def on_r(self, key):
        try:
            if self.recorder:
                self.recorder.ptt_pressed = False
                app_logger.debug("PTT released")
        except Exception as e:
            app_logger.debug(f"PTT release error: {e}")
    
    def key_to_string(self, key):
        """Convert pynput key to string format"""
        if hasattr(key, 'char') and key.char:
            return key.char.lower()
        elif hasattr(key, 'name'):
            return key.name.lower()
        return str(key).lower()
    
    def on_trans_status(self, active):
        self.indicator.is_trans = active
        self.indicator.update()
    
    def on_text(self, text, src):
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.scratchpad.append(f"[{timestamp}] {text}")
        
        if src == "live":
            p_text = text + " " if self.config.settings["auto_space"] else text
            
            try:
                pyperclip.copy(p_text)
                time.sleep(self.config.settings["paste_delay"])
                pyautogui.hotkey('ctrl', 'v')
                app_logger.info(f"Text pasted: '{text[:30]}...'")
            except Exception as e:
                app_logger.error(f"Paste error: {e}")
            
            # Check for voice commands
            for phrase, cmd in self.config.settings["commands"].items():
                if phrase.lower() in text.lower():
                    try:
                        subprocess.Popen(cmd, shell=True)
                        app_logger.info(f"Command executed: {cmd}")
                        self.scratchpad.append(f"[Command] Executed: {cmd}")
                    except Exception as e:
                        app_logger.error(f"Command execution failed: {e}")

    def setup_logic(self):
        # Stop existing listeners
        if hasattr(self, 'hk_l'):
            try:
                self.hk_l.stop()
            except:
                pass
        
        if hasattr(self, 'ptt_l'):
            try:
                self.ptt_l.stop()
            except:
                pass
        
        # Create hotkey mapping
        hotkey_map = {}
        
        try:
            toggle_hotkey = self.normalize_hotkey(self.config.settings["hotkey"])
            visibility_hotkey = self.normalize_hotkey(self.config.settings["visibility_hotkey"])
            
            hotkey_map[toggle_hotkey] = lambda: self.sig_toggle_rec.emit()
            hotkey_map[visibility_hotkey] = lambda: self.sig_toggle_vis.emit()
            
            self.hk_l = keyboard.GlobalHotKeys(hotkey_map)
            self.hk_l.start()
            app_logger.info(f"Hotkeys registered: {list(hotkey_map.keys())}")
        except Exception as e:
            app_logger.error(f"Hotkey registration failed: {e}")
            QMessageBox.warning(
                self, 
                "Hotkey Error", 
                f"Failed to register hotkeys:\n{e}\n\nPlease check your hotkey settings."
            )
        
        # Start PTT listener
        try:
            self.ptt_l = keyboard.Listener(on_press=self.on_p, on_release=self.on_r)
            self.ptt_l.start()
            app_logger.info("PTT listener started")
        except Exception as e:
            app_logger.error(f"PTT listener failed: {e}")
    
    def normalize_hotkey(self, hotkey_str):
        """Convert our hotkey format to pynput format"""
        # Our format: "ctrl+shift+w"
        # pynput format: "<ctrl>+<shift>+w"
        
        parts = hotkey_str.lower().split('+')
        normalized = []
        
        for part in parts:
            part = part.strip()
            if part in ['ctrl', 'shift', 'alt', 'cmd', 'win']:
                normalized.append(f'<{part}>')
            else:
                # Check if it's a function key
                if part.startswith('f') and len(part) > 1 and part[1:].isdigit():
                    normalized.append(f'<{part}>')
                else:
                    normalized.append(part)
        
        result = '+'.join(normalized)
        app_logger.debug(f"Normalized hotkey '{hotkey_str}' to '{result}'")
        return result

    def monitor_dirs(self):
        root = Path(self.config.settings["mon_folder"])
        
        if not root.exists():
            return
        
        proc_dir = root / "Processed"
        proc_dir.mkdir(exist_ok=True)
        
        for f in root.glob("*.*"):
            if f.suffix.lower() in ['.wav', '.mp3', '.m4a'] and f.parent != proc_dir:
                try:
                    target = proc_dir / f.name
                    shutil.move(str(f), str(target))
                    self.transcriber.queue.put((str(target.absolute()), "file"))
                    app_logger.info(f"File moved to processing: {f.name}")
                except Exception as e:
                    app_logger.error(f"Failed to process file {f.name}: {e}")
    
    def setup_deps(self):
        url = "https://github.com/purfview/whisper-standalone-win/releases/download/libs/cuBLAS_cuDNN_zlib.zip"
        
        reply = QMessageBox.question(
            self, 
            "Download Dependencies", 
            "Download NVIDIA GPU acceleration files?\n\n"
            "Size: ~500MB\n"
            "Required for: NVIDIA GPU users\n"
            "Optional for: CPU-only or AMD GPU users\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            def download_task():
                try:
                    file_path = os.path.join(BASE_DIR, "gpu_libs.zip")
                    app_logger.info(f"Downloading GPU dependencies from {url}")
                    
                    self.scratchpad.append("[Download] Starting download...")
                    urllib.request.urlretrieve(url, file_path)
                    
                    self.scratchpad.append("[Download] Extracting files...")
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(BASE_DIR)
                    
                    os.remove(file_path)
                    
                    self.scratchpad.append("[Download] ✓ Complete! Restart the app to use GPU acceleration.")
                    app_logger.info("GPU dependencies downloaded and extracted successfully")
                except Exception as e:
                    error_msg = f"[Download] ✗ Error: {e}"
                    self.scratchpad.append(error_msg)
                    app_logger.error(f"GPU dependencies download failed: {e}")
            
            threading.Thread(target=download_task, daemon=True).start()
    
    def open_log_file(self):
        try:
            if os.path.exists(app_logger.log_path):
                if os.name == 'nt':
                    os.startfile(app_logger.log_path)
                else:
                    subprocess.Popen(['xdg-open', app_logger.log_path])
            else:
                QMessageBox.information(self, "No Log File", "Log file not found.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open log file:\n{e}")

    def import_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, 
            "Select Audio Files", 
            "", 
            "Audio Files (*.wav *.mp3 *.m4a *.mp4)"
        )
        
        for p in paths:
            self.transcriber.queue.put((os.path.abspath(p), "file"))
            app_logger.info(f"File imported for transcription: {p}")

    def browse_f(self, line_edit):
        path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if path:
            line_edit.setText(path)

    def cap_hk(self, button, config_key):
        dialog = HotkeyCaptureDialog(self)
        dialog.key_captured.connect(button.setText)
        dialog.exec()

    def import_p(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Prompt", "", "Text Files (*.txt)")
        if path:
            try:
                self.prompt_edit.setText(Path(path).read_text(encoding='utf-8'))
                app_logger.info(f"Prompt imported from {path}")
            except Exception as e:
                app_logger.error(f"Failed to import prompt: {e}")

    def export_p(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Prompt", "", "Text Files (*.txt)")
        if path:
            try:
                Path(path).write_text(self.prompt_edit.toPlainText(), encoding='utf-8')
                app_logger.info(f"Prompt exported to {path}")
            except Exception as e:
                app_logger.error(f"Failed to export prompt: {e}")
    
    def closeEvent(self, event):
        app_logger.info("Application closing")
        
        # Clean up recordings if requested
        if self.config.settings["clear_exit"]:
            try:
                folder = Path(self.config.settings["audio_folder"])
                if folder.exists():
                    for f in folder.glob("*.wav"):
                        f.unlink()
                    app_logger.info("Recordings cleared on exit")
            except Exception as e:
                app_logger.error(f"Failed to clear recordings: {e}")
        
        # Stop workers
        if self.recorder:
            self.recorder.active = False
        
        self.transcriber.running = False
        
        event.accept()


class StatusOverlay(QWidget):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.WindowStaysOnTopHint | 
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        self.is_list = False
        self.is_rec = False
        self.is_trans = False
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_pos)
        self.timer.start(1000)
        
        self.update_pos()
        self.show()
    
    def update_pos(self):
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(0, 0, screen.width(), screen.height())
    
    def paintEvent(self, event):
        if not self.config.settings["ind_show"]:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Determine color based on state
        if (self.is_list or self.is_rec) and self.is_trans:
            color = QColor(128, 0, 128, 255)  # Purple: recording + transcribing
        elif self.is_rec:
            color = QColor(255, 0, 0, 255)     # Red: actively recording
        elif self.is_list:
            color = QColor(100, 0, 0, 255)     # Dark red: listening
        elif self.is_trans:
            color = QColor(0, 0, 255, 255)     # Blue: transcribing only
        else:
            color = QColor(128, 128, 128, 100) # Gray: idle
        
        screen_rect = self.rect()
        size = self.config.settings["ind_size"]
        offset = self.config.settings["ind_off"]
        position = self.config.settings["ind_pos"]
        
        # Draw icon indicator
        if "Icon" in self.config.settings["ind_type"] or "Both" in self.config.settings["ind_type"]:
            if "Left" in position:
                icon_x = offset
            else:
                icon_x = screen_rect.width() - size - offset
            
            if "Top" in position:
                icon_y = offset
            else:
                icon_y = screen_rect.height() - size - offset
            
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(icon_x, icon_y, size, size)
        
        # Draw bar indicator
        if "Bar" in self.config.settings["ind_type"] or "Both" in self.config.settings["ind_type"]:
            bar_size = self.config.settings["bar_size"]
            edge = self.config.settings["bar_edge"]
            
            painter.setBrush(color)
            
            if edge == "Top":
                painter.drawRect(0, 0, screen_rect.width(), bar_size)
            elif edge == "Bottom":
                painter.drawRect(0, screen_rect.height() - bar_size, screen_rect.width(), bar_size)
            elif edge == "Left":
                painter.drawRect(0, 0, bar_size, screen_rect.height())
            else:  # Right
                painter.drawRect(screen_rect.width() - bar_size, 0, bar_size, screen_rect.height())


if __name__ == "__main__":
    app_logger.info("="*60)
    app_logger.info(f"{APP_NAME} v{__version__} - Starting")
    app_logger.info(f"Python: {sys.version}")
    app_logger.info(f"Platform: {sys.platform}")
    app_logger.info("="*60)
    
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)
    
    window = WhisperRApp()
    window.show()
    
    exit_code = app.exec()
    app_logger.info(f"Application exited with code {exit_code}")
    sys.exit(exit_code)