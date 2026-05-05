# -*- coding: utf-8 -*-
"""
WhisperR Logging Utilities
Application logging and crash handling
"""

import os
import sys
import logging
import traceback
from datetime import datetime
from typing import Optional


class _FlushingFileHandler(logging.FileHandler):
    """FileHandler that flushes to disk after every record"""
    
    def emit(self, record):
        super().emit(record)
        self.flush()


class AppLogger:
    """
    Application logger with configurable levels
    """
    
    def __init__(self, log_path: str, name: str = "WhisperR"):
        self.log_path = log_path
        self.name = name
        self.level = logging.INFO
        self._file_handler: Optional[logging.FileHandler] = None
        self._disabled = False
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self._attach_file_handler()
        
        # Console handler (warnings/errors only)
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        self.logger.addHandler(ch)
    
    def _attach_file_handler(self):
        if self._file_handler:
            self.logger.removeHandler(self._file_handler)
            self._file_handler.close()
            self._file_handler = None
        
        fh = _FlushingFileHandler(self.log_path, mode='w', encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        self.logger.addHandler(fh)
        self._file_handler = fh
    
    def set_level(self, level_name: str):
        """Set logging level"""
        if level_name == "NONE":
            # Disable file logging entirely
            if self._file_handler:
                self.logger.removeHandler(self._file_handler)
                self._file_handler.close()
                self._file_handler = None
            
            # Delete existing log file
            try:
                if os.path.exists(self.log_path):
                    os.remove(self.log_path)
            except Exception:
                pass
            
            self._disabled = True
            self.logger.setLevel(logging.CRITICAL + 1)
        else:
            levels = {
                "DEBUG": logging.DEBUG,
                "INFO": logging.INFO,
                "WARNING": logging.WARNING,
                "ERROR": logging.ERROR
            }
            self._disabled = False
            if not self._file_handler:
                self._attach_file_handler()
            self.level = levels.get(level_name, logging.INFO)
            self.logger.setLevel(self.level)
    
    def debug(self, msg: str, exc_info: bool = False):
        if not self._disabled:
            self.logger.debug(msg, exc_info=exc_info)
    
    def info(self, msg: str, exc_info: bool = False):
        if not self._disabled:
            self.logger.info(msg, exc_info=exc_info)
    
    def warning(self, msg: str, exc_info: bool = False):
        if not self._disabled:
            self.logger.warning(msg, exc_info=exc_info)
    
    def error(self, msg: str, exc_info: bool = False):
        if not self._disabled:
            self.logger.error(msg, exc_info=exc_info)


def create_logger(base_dir: str) -> AppLogger:
    """Create application logger"""
    log_path = os.path.join(base_dir, "app_log.txt")
    return AppLogger(log_path)


def crash_logger(etype, value, tb):
    """
    Enhanced crash handler
    Collects comprehensive crash information
    """
    try:
        import platform
        import psutil
        
        # Gather system info
        info = {
            "timestamp": datetime.now().isoformat(),
            "version": "2.1.0",
            "os": f"{platform.system()} {platform.version()}",
            "python": sys.version.split()[0],
        }
        
        # Try to get PyQt version
        try:
            from PyQt6.QtCore import PYQT_VERSION_STR
            info["pyqt_version"] = PYQT_VERSION_STR
        except ImportError:
            info["pyqt_version"] = "unknown"
        
        # RAM info
        try:
            vm = psutil.virtual_memory()
            info["ram_total_gb"] = f"{vm.total / (1024**3):.1f} GB"
            info["ram_available_gb"] = f"{vm.available / (1024**3):.1f} GB"
            info["ram_percent"] = f"{vm.percent}%"
        except Exception:
            info["ram_info"] = "unavailable"
        
        # Read recent logs
        log_path = os.path.join(os.path.dirname(sys.executable), "app_log.txt")
        recent_logs = ""
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    recent_logs = "".join(lines[-50:])
            except Exception:
                recent_logs = "[Could not read log file]"
        
        # Write crash report
        crash_path = os.path.join(os.path.dirname(sys.executable), "CRASH_LOG.txt")
        
        with open(crash_path, "w", encoding="utf-8") as f:
            f.write("=" * 50 + "\n")
            f.write("WHISPERR CRASH REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Time: {info['timestamp']}\n")
            f.write(f"Version: {info['version']}\n")
            f.write(f"OS: {info['os']}\n")
            f.write(f"Python: {info['python']}\n")
            f.write(f"PyQt: {info['pyqt_version']}\n")
            
            if "ram_total_gb" in info:
                f.write(f"RAM: {info['ram_total_gb']} total, {info['ram_available_gb']} available ({info['ram_percent']} used)\n")
            
            f.write("\n--- Recent Logs ---\n")
            f.write(recent_logs if recent_logs else "(no logs)")
            f.write("\n\n--- Traceback ---\n")
            traceback.print_exception(etype, value, tb, file=f)
            
            f.write("\n\n--- Instructions ---\n")
            f.write("Please report this crash on GitHub:\n")
            f.write("https://github.com/WhisperR/WhisperR/issues\n")
        
        # Try to show crash dialog
        try:
            from PyQt6.QtWidgets import QMessageBox, QApplication
            from PyQt6.QtCore import Qt
            
            app = QApplication.instance()
            if app:
                msg = QMessageBox()
                msg.setWindowTitle("WhisperR Crashed")
                msg.setIcon(QMessageBox.Icon.Critical)
                msg.setText("WhisperR has encountered an unexpected error.")
                msg.setInformativeText(
                    f"A crash report has been saved to:\n{crash_path}\n\n"
                    "Please check the GitHub issues page for known issues."
                )
                msg.setDetailedText(
                    f"OS: {info['os']}\n"
                    f"Python: {info['python']}\n"
                    f"PyQt: {info['pyqt_version']}\n\n"
                    f"Error: {value}"
                )
                msg.exec()
        except Exception:
            pass
            
    except Exception as e:
        # Last resort: write to stderr
        try:
            traceback.print_exception(etype, value, tb)
        except Exception:
            pass


def setup_crash_handler():
    """Install the crash handler"""
    sys.excepthook = crash_logger