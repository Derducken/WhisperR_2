# -*- coding: utf-8 -*-
"""
WhisperR Transcription Module
Handles Whisper model loading and transcription
"""

import os
import sys
import multiprocessing
from typing import Optional, List, Tuple, Dict, Any


# Whisper model configurations
WHISPER_MODELS = {
    "tiny": {"size_mb": 75, "vram_mb": 450, "speed": "very fast"},
    "base": {"size_mb": 150, "vram_mb": 700, "speed": "fast"},
    "small": {"size_mb": 490, "vram_mb": 1400, "speed": "medium"},
    "medium": {"size_mb": 1500, "vram_mb": 2800, "speed": "slow"},
    "large-v3": {"size_mb": 2900, "vram_mb": 5200, "speed": "slow"},
}


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get information about a Whisper model"""
    return WHISPER_MODELS.get(model_name, {})


def get_available_models() -> List[str]:
    """Get list of available model names"""
    return list(WHISPER_MODELS.keys())


def get_model_recommendation(has_gpu: bool = False, vram_gb: float = 0) -> str:
    """
    Get recommended model based on hardware
    
    Args:
        has_gpu: Whether user has NVIDIA GPU
        vram_gb: GPU VRAM in GB
        
    Returns:
        Recommended model name
    """
    if has_gpu and vram_gb >= 6:
        return "large-v3"
    elif has_gpu and vram_gb >= 3:
        return "medium"
    elif has_gpu and vram_gb >= 1.5:
        return "small"
    elif vram_gb >= 4:
        return "small"
    else:
        return "tiny"


# Language codes supported by Whisper
SUPPORTED_LANGUAGES = {
    "Auto": None,
    "English": "en",
    "Greek": "el",
    "German": "de",
    "French": "fr",
    "Spanish": "es",
    "Italian": "it",
    "Portuguese": "pt",
    "Dutch": "nl",
    "Russian": "ru",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Arabic": "ar",
    "Hindi": "hi",
    "Turkish": "tr",
    "Polish": "pl",
    "Swedish": "sv",
    "Norwegian": "no",
    "Danish": "da",
    "Finnish": "fi",
}


def get_language_name(code: str) -> Optional[str]:
    """Get language name from code"""
    for name, c in SUPPORTED_LANGUAGES.items():
        if c == code:
            return name
    return None


def get_language_code(name: str) -> Optional[str]:
    """Get language code from name"""
    return SUPPORTED_LANGUAGES.get(name)


class TranscriptionResult:
    """Container for transcription results"""
    
    def __init__(
        self,
        text: str,
        language: str = None,
        segments: List[Dict[str, Any]] = None,
        duration: float = 0.0
    ):
        self.text = text
        self.language = language
        self.segments = segments or []
        self.duration = duration
    
    def __repr__(self):
        return f"TranscriptionResult(text='{self.text[:50]}...', language='{self.language}')"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "segments": self.segments,
            "duration": self.duration
        }


class TranscriptionOptions:
    """Options for transcription"""
    
    def __init__(
        self,
        language: str = None,
        translate: bool = False,
        timestamps: bool = False,
        initial_prompt: str = "",
        hotwords: List[str] = None,
        min_confidence: float = 0.0,
        use_vad: bool = True,
        vad_params: Dict[str, Any] = None
    ):
        self.language = language
        self.translate = translate
        self.timestamps = timestamps
        self.initial_prompt = initial_prompt
        self.hotwords = hotwords or []
        self.min_confidence = min_confidence
        self.use_vad = use_vad
        self.vad_params = vad_params or {
            "threshold": 0.5,
            "min_silence_ms": 2000,
            "min_speech_ms": 250
        }


def create_transcription_task(
    model_name: str,
    language: str,
    compute_pref: str,
    audio_data,
    source: str,
    options: TranscriptionOptions
) -> Tuple:
    """
    Create a task tuple for the AI worker
    
    Returns:
        Tuple of (model_name, lang_code, compute_pref, audio_data, src, 
                 translate, use_vad, prompt, min_confidence, hotwords, vad_params)
    """
    return (
        model_name,
        language,
        compute_pref,
        audio_data,
        source,
        options.translate,
        options.use_vad,
        options.initial_prompt,
        options.min_confidence,
        options.hotwords,
        options.vad_params
    )


# Model download fallback files
HF_FALLBACK_FILES = {
    "tiny": (["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"],
             ["tokenizer_config.json", "special_tokens_map.json", "preprocessor_config.json"]),
    "base": (["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"],
             ["tokenizer_config.json", "special_tokens_map.json", "preprocessor_config.json"]),
    "small": (["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"],
              ["tokenizer_config.json", "special_tokens_map.json", "preprocessor_config.json"]),
    "medium": (["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"],
               ["tokenizer_config.json", "special_tokens_map.json", "preprocessor_config.json"]),
    "large-v2": (["config.json", "model.bin", "tokenizer.json"],
                 ["vocabulary.txt", "tokenizer_config.json", "special_tokens_map.json", "preprocessor_config.json"]),
    "large-v3": (["config.json", "model.bin", "tokenizer.json"],
                 ["vocabulary.txt", "tokenizer_config.json", "special_tokens_map.json", "preprocessor_config.json"]),
}


def get_hf_model_files(model_name: str) -> Tuple[List[str], List[str]]:
    """Get list of files to download for a model"""
    return HF_FALLBACK_FILES.get(
        model_name,
        (["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"], [])
    )


# Note: The actual _ai_worker_process remains in the main WhisperR.py
# because it contains complex multiprocessing setup that depends on 
# the full application context. This module provides utilities and
# data structures for working with transcription.


class TranscriptionStats:
    """Statistics about transcription operations"""
    
    def __init__(self):
        self.total_transcriptions = 0
        self.total_words = 0
        self.total_duration = 0.0
        self.avg_confidence = 0.0
    
    def record(self, result: TranscriptionResult, duration: float):
        """Record a completed transcription"""
        self.total_transcriptions += 1
        words = len(result.text.split())
        self.total_words += words
        self.total_duration += duration
        
        if result.segments:
            confidences = [s.get('confidence', 0) for s in result.segments]
            if confidences:
                avg = sum(confidences) / len(confidences)
                self.avg_confidence = (
                    (self.avg_confidence * (self.total_transcriptions - 1) + avg)
                    / self.total_transcriptions
                )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_transcriptions": self.total_transcriptions,
            "total_words": self.total_words,
            "total_duration": round(self.total_duration, 2),
            "avg_confidence": round(self.avg_confidence, 2)
        }