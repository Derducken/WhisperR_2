# -*- coding: utf-8 -*-
"""
WhisperR Configuration Module
Handles app settings, save/load, and path validation
"""

import os
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

# Default settings - used on first run or when config is corrupted
DEFAULT_SETTINGS: Dict[str, Any] = {
    "model": "tiny",
    "lang_name": "English",
    "lang_code": "en",
    "hf_cache_path": "",  # empty = use default HF cache location
    "translate": False,
    "timestamps": False,
    "initial_prompt": "General professional writing. The speaker may use technical terminology across a variety of fields. Software names: Microsoft Word, Excel, PowerPoint, Google Docs, Sheets, Slides, VS Code, GitHub, ChatGPT, WhisperR. Technology terms: API, JSON, XML, HTML, CSS, JavaScript, Python, SQL, GPU, CPU, RAM, SSD, HDD, USB, Wi-Fi, Bluetooth, HTTPS. Business terms: KPI, ROI, B2C, B2C, SaaS, MVP, NDA, CRM. Measurements and abbreviations: GB, TB, MHz, GHz, ms, fps, dpi, OK, AI, ML, AR, VR, UI, UX. When the speaker says 'okay' in a sentence, transcribe it as 'OK'.",
    "hotwords": [],
    "audio_folder": "",  # Will be set to default in __init__
    "mon_folder": "",    # Will be set to default in __init__
    "clear_exit": True,
    "save_to_disk": False,
    "auto_space": True,
    "min_to_tray": True,
    "input_device_name": "",
    "input_device_index": -1,
    "paste_delay": 0.5,
    "hotkey": "<ctrl>+<alt>+z",
    "ptt_key": "ctrl+shift+space",
    "visibility_hotkey": "ctrl+shift+alt+z",
    "editor_hotkey": "ctrl+shift+alt+a",
    "editor_edit_hotkey": "ctrl+shift+x",
    "rollback_hotkey": "ctrl+shift+z",
    "always_on_top": True,
    "aot_main": True,
    "aot_editor": True,
    "aot_notes": True,
    "aot_cheatsheet": True,
    "auto_backup_enabled": False,
    "auto_backup_interval": 10,
    "auto_backup_keep": 5,
    "cb_source_tag": True,
    "version_history_keep": 20,
    "version_history_infinite": False,
    "snapshots_enabled": False,
    "snapshots_mode": "count",
    "snapshots_keep_count": 60,
    "snapshots_keep_hours": 24,
    "snapshots_keep_unit": "hours",
    "live_mode": "Auto-Pause",
    "dict_mode": "Auto-Pause",
    "auto_pause_sec": 2.0,
    "noise_floor": 50,
    "speech_vol": 500,
    "commands": {"Launch Notepad": "notepad.exe"},
    "terms": {
        "whisper ar": "WhisperR",
        "youre": "you're",
        "dont": "don't",
        "cant": "can't",
        "wont": "won't",
        "its a": "it's a"
    },
    "hallucinations": [
        "thank you.",
        "thanks for watching.",
        "god bless.",
        "god bless you.",
        "subtitles by",
        "amara.org",
        "translated by",
        "transcribed by",
        "please subscribe",
        "don't forget to subscribe",
        "like and subscribe",
        "thanks for watching, and i'll see you",
        "thank you for watching",
        "this video was"
    ],
    "ind_show": True,
    "ind_type": "Both",
    "ind_pos": "Top-Left",
    "ind_size": 32,
    "ind_off": 5,
    "bar_edge": "Top",
    "bar_size": 5,
    "bar_thickness": 3,
    "ind_opacity": 220,
    "bar_opacity": 220,
    "ind_hide_idle": True,
    "log_level": "NONE",
    "use_vad": True,
    "vad_threshold": 0.5,
    "hotkey_cooldown_ms": 400,
    "manual_sentence_split": False,
    "mss_break_key": "shift",
    "harper": {
        "installed": False,
        "version": None,
        "user_dict_path": "",
        "vocabulary": [],
        "linters": {}
    },
    "vad_min_silence_ms": 2000,
    "vad_min_speech_ms": 250,
    "ft_output_folder": "",
    "ft_mon_folder": "",
    "ft_mon_enabled": False,
    "use_confidence": True,
    "min_confidence": 0.5,
    "editor_type_trigger": "whisper type, whisper write",
    "editor_edit_trigger": "whisper edit, whisper edit this",
    "editor_paste_trigger": "whisper paste, whisper done, whisper okay",
    "editor_hk_bold": "Ctrl+B",
    "editor_hk_italic": "Ctrl+I",
    "editor_hk_strike": "Ctrl+Shift+S",
    "editor_hk_highlight": "Ctrl+Shift+H",
    "editor_hk_code": "Ctrl+`",
    "editor_hk_h1": "Ctrl+1",
    "editor_hk_h2": "Ctrl+2",
    "editor_hk_h3": "Ctrl+3",
    "editor_hk_emdash": "Ctrl+Shift+Minus",
    "editor_hk_bullet": "Ctrl+Shift+B",
    "editor_hk_numlist": "Ctrl+Shift+N",
    "editor_hk_tasklist": "Ctrl+Shift+T",
    "editor_hk_kbd": "Ctrl+Shift+D",
    "editor_hk_tagwrap": "Ctrl+Shift+W",
    "editor_hk_link": "Ctrl+K",
    "editor_auto_save_interval": 30,
    "sendkeys_trigger": "whisper send keys",
    "select_trigger": "whisper select",
    "move_trigger": "whisper move",
    "movebefore_trigger": "whisper before",
    "moveafter_trigger": "whisper after",
    "replace_trigger": "whisper replace",
    "insertbefore_trigger": "whisper insert before",
    "insertafter_trigger": "whisper insert after",
    "fuzzy_threshold": 0.75,
    "theme": "dark",  # NEW: theme preference
}


class AppConfig:
    """
    WhisperR Configuration Manager
    Handles loading, saving, and validating app settings
    """
    
    def __init__(self, base_dir: str = None, logger = None):
        """
        Initialize configuration
        
        Args:
            base_dir: Application base directory (where config.json lives)
            logger: Logger instance for logging (optional, can be set later)
        """
        self._logger = logger
        
        # Set base directory
        if base_dir is None:
            import sys
            if getattr(sys, 'frozen', False):
                self._base_dir = os.path.dirname(sys.executable)
            else:
                self._base_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self._base_dir = base_dir
        
        self.path = os.path.join(self._base_dir, "config.json")
        
        # Initialize with defaults, then override with loaded settings
        self._init_defaults()
        
        # Load existing config if present
        self._first_run = not os.path.exists(self.path)
        self.load()
        
        # On first run, write defaults immediately
        if self._first_run:
            self.save()
            self._log("info", "First run — default config.json created.")
        
        # Apply log level after loading
        if self._logger:
            self._logger.set_level(self.settings.get("log_level", "NONE"))
    
    def _init_defaults(self):
        """Initialize default settings"""
        # Deep copy to avoid modifying the original
        self.settings = DEFAULT_SETTINGS.copy()
        
        # Set dynamic defaults (things that depend on user home directory)
        home = str(Path.home())
        self.settings["audio_folder"] = os.path.join(home, "WhisperR_Recordings")
        self.settings["mon_folder"] = os.path.join(home, "WhisperR_Watch")
        self.settings["ft_output_folder"] = os.path.join(home, "WhisperR_Output")
        self.settings["ft_mon_folder"] = os.path.join(home, "WhisperR_Watch")
    
    def _log(self, level: str, message: str):
        """Log a message using the configured logger"""
        if self._logger is None:
            return
        
        if level == "debug":
            self._logger.debug(message)
        elif level == "info":
            self._logger.info(message)
        elif level == "warning":
            self._logger.warning(message)
        elif level == "error":
            self._logger.error(message)
    
    def set_logger(self, logger):
        """Set the logger instance after initialization"""
        self._logger = logger
    
    def load(self) -> bool:
        """
        Load configuration from file
        
        Returns:
            True if loaded successfully, False if using defaults
        """
        if not os.path.exists(self.path):
            return False
        
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                # Merge loaded settings with defaults (preserves new defaults)
                self.settings.update(loaded)
            
            # Config migrations
            self._migrate_config()
            
            # Validate paths
            self._validate_paths()
            
            self._log("info", "Configuration loaded successfully")
            return True
            
        except Exception as e:
            self._log("error", f"Failed to load config: {e}")
            self._log("warning", "Using default configuration")
            return False
    
    def _migrate_config(self):
        """Handle config version migrations"""
        # v2.1.0: editor_hk_kbd Ctrl+Shift+K → Ctrl+Shift+D
        if self.settings.get("editor_hk_kbd", "") == "Ctrl+Shift+K":
            self.settings["editor_hk_kbd"] = "Ctrl+Shift+D"
        
        # Add new settings if missing (for future migrations)
        if "theme" not in self.settings:
            self.settings["theme"] = "dark"
        
        if "editor_auto_save_interval" not in self.settings:
            self.settings["editor_auto_save_interval"] = 30
    
    def _validate_paths(self):
        """Validate and fix paths in config"""
        path_keys = [
            "audio_folder",
            "mon_folder",
            "ft_output_folder",
            "ft_mon_folder",
            "hf_cache_path"
        ]
        
        for key in path_keys:
            path_str = self.settings.get(key, "")
            if not path_str:
                continue
            
            # Check for invalid characters
            if '\x00' in path_str or '\x01' in path_str:
                self._log("error", f"Invalid characters in path for {key}: {path_str}")
                path_str = ""
            
            # Check if path is absolute
            if path_str and not os.path.isabs(path_str):
                self._log("error", f"Path is not absolute for {key}: {path_str}")
                path_str = ""
            
            # Try to create directory if it doesn't exist
            if path_str:
                try:
                    Path(path_str).mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    self._log("error", f"Failed to create directory for {key}: {e}")
                    path_str = ""
            
            # Reset to default if invalid
            if not path_str:
                home = str(Path.home())
                if key == "audio_folder":
                    path_str = os.path.join(home, "WhisperR_Recordings")
                elif key in ("mon_folder", "ft_mon_folder"):
                    path_str = os.path.join(home, "WhisperR_Watch")
                elif key == "ft_output_folder":
                    path_str = os.path.join(home, "WhisperR_Output")
                else:
                    continue
                
                # Create the default directory
                try:
                    Path(path_str).mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    self._log("error", f"Failed to create default path: {e}")
                    # Last resort: use temp
                    import tempfile
                    path_str = os.path.join(tempfile.gettempdir(), f"WhisperR_{key}")
                    Path(path_str).mkdir(parents=True, exist_ok=True)
                
                self._log("warning", f"Resetting {key} to default: {path_str}")
                self.settings[key] = path_str
    
    def save(self) -> bool:
        """
        Save configuration to file
        
        Returns:
            True if saved successfully
            
        Raises:
            Exception: If save fails
        """
        try:
            # Create backup of existing config
            backup_path = self.path + ".backup"
            if os.path.exists(self.path):
                try:
                    shutil.copy2(self.path, backup_path)
                except:
                    pass  # Backup failed, but continue anyway
            
            # Write new config
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=4)
            
            self._log("info", "Configuration saved successfully")
            
            # Remove backup after successful save
            if os.path.exists(backup_path):
                try:
                    os.remove(backup_path)
                except:
                    pass
            
            return True
            
        except PermissionError as e:
            self._log("error", f"Permission denied saving config: {e}")
            raise Exception("Cannot save settings - permission denied. Try running as administrator.")
        except Exception as e:
            self._log("error", f"Failed to save config: {e}", exc_info=True)
            
            # Try to restore backup
            backup_path = self.path + ".backup"
            if os.path.exists(backup_path):
                try:
                    shutil.copy2(backup_path, self.path)
                    self._log("info", "Restored config from backup")
                except:
                    pass
            
            raise Exception(f"Failed to save settings: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        return self.settings.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set a configuration value"""
        self.settings[key] = value
    
    def update(self, updates: Dict[str, Any]):
        """Update multiple configuration values"""
        self.settings.update(updates)
    
    def get_path(self, key: str) -> str:
        """Get a path from config, returning empty string if not set"""
        return str(self.settings.get(key, ""))
    
    @property
    def audio_folder(self) -> str:
        return self.settings.get("audio_folder", "")
    
    @property
    def mon_folder(self) -> str:
        return self.settings.get("mon_folder", "")
    
    @property
    def model(self) -> str:
        return self.settings.get("model", "tiny")
    
    @property
    def language(self) -> str:
        return self.settings.get("lang_code", "en")
    
    @property
    def theme(self) -> str:
        return self.settings.get("theme", "dark")