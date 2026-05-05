# -*- coding: utf-8 -*-
"""
WhisperR Audio Module
Audio recording utilities and device enumeration
"""

import os
import wave
import numpy as np
from typing import List, Optional, Tuple, Dict, Any


# Audio configuration constants
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_FORMAT = 'float32'


class AudioDevice:
    """Represents an audio input device"""
    
    def __init__(self, index: int, name: str, api: str = ""):
        self.index = index
        self.name = name
        self.api = api
    
    def __repr__(self):
        return f"AudioDevice({self.index}, '{self.name}', '{self.api}')"
    
    def __str__(self):
        return f"{self.name} [{self.api}]"


def enumerate_audio_devices() -> List[AudioDevice]:
    """
    Enumerate available audio input devices
    
    Returns:
        List of AudioDevice objects
    """
    devices = []
    
    try:
        import pyaudio
        
        p = pyaudio.PyAudio()
        
        # Count devices
        device_count = p.get_device_count()
        
        for i in range(device_count):
            try:
                info = p.get_device_info_by_index(i)
                
                # Only include input devices
                if info['maxInputChannels'] > 0:
                    name = info['name']
                    
                    # Determine API from name patterns
                    api = ""
                    if 'WASAPI' in name:
                        api = "Windows WASAPI"
                    elif 'MME' in name or 'DirectSound' in name:
                        api = "MME/DirectSound"
                    elif 'WDM-KS' in name:
                        api = "WDM-KS"
                    
                    devices.append(AudioDevice(i, name, api))
                    
            except Exception:
                continue
        
        p.terminate()
        
    except ImportError:
        pass
    
    return devices


def get_default_input_device() -> Optional[AudioDevice]:
    """Get the default system input device"""
    devices = enumerate_audio_devices()
    
    # Prefer WASAPI devices
    for device in devices:
        if 'WASAPI' in device.name:
            return device
    
    # Fall back to first available
    return devices[0] if devices else None


def save_audio_file(
    filepath: str,
    audio_data: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS
) -> bool:
    """
    Save audio data to WAV file
    
    Args:
        filepath: Output file path
        audio_data: Audio data as numpy array
        sample_rate: Sample rate in Hz
        channels: Number of channels
        
    Returns:
        True if saved successfully
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert float32 to int16 for WAV
        if audio_data.dtype == np.float32:
            audio_int16 = (audio_data * 32767).astype(np.int16)
        else:
            audio_int16 = audio_data.astype(np.int16)
        
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        
        return True
        
    except Exception as e:
        return False


def load_audio_file(filepath: str) -> Optional[Tuple[np.ndarray, int]]:
    """
    Load audio from WAV file
    
    Args:
        filepath: Path to WAV file
        
    Returns:
        Tuple of (audio_data, sample_rate) or None on error
    """
    try:
        with wave.open(filepath, 'rb') as wf:
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            
            # Convert to numpy array
            audio_data = np.frombuffer(frames, dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32767.0
            
            return (audio_data, sample_rate)
            
    except Exception:
        return None


def get_audio_duration(audio_data: np.ndarray, sample_rate: int) -> float:
    """Get audio duration in seconds"""
    return len(audio_data) / sample_rate


def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
    """Normalize audio to -1.0 to 1.0 range"""
    max_val = np.abs(audio_data).max()
    if max_val > 0:
        return audio_data / max_val
    return audio_data


def compute_audio_level(audio_data: np.ndarray) -> float:
    """
    Compute RMS level of audio
    
    Returns:
        RMS level (0.0 to 1.0)
    """
    # Compute RMS
    rms = np.sqrt(np.mean(audio_data ** 2))
    return rms


def is_silent(
    audio_data: np.ndarray,
    threshold: float = 0.01
) -> bool:
    """
    Check if audio is mostly silent
    
    Args:
        audio_data: Audio data
        threshold: Level below which is considered silent
        
    Returns:
        True if audio is silent
    """
    level = compute_audio_level(audio_data)
    return level < threshold


class AudioRecorder:
    """Simple audio recorder wrapper (requires pyaudio)"""
    
    def __init__(
        self,
        device_index: int = -1,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        chunk_size: int = DEFAULT_CHUNK_SIZE
    ):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self._stream = None
        self._pyaudio = None
        self._recording = False
    
    def start(self) -> bool:
        """Start recording"""
        try:
            import pyaudio
            
            self._pyaudio = pyaudio.PyAudio()
            
            self._stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size
            )
            
            self._recording = True
            return True
            
        except Exception:
            return False
    
    def read(self) -> Optional[np.ndarray]:
        """Read audio chunk"""
        if not self._recording or not self._stream:
            return None
        
        try:
            frames = self._stream.read(self.chunk_size)
            audio_data = np.frombuffer(frames, dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32767.0
            return audio_data
            
        except Exception:
            return None
    
    def stop(self) -> bool:
        """Stop recording"""
        try:
            if self._stream:
                self._stream.stop_stream()
                self._stream.close()
                self._stream = None
            
            if self._pyaudio:
                self._pyaudio.terminate()
                self._pyaudio = None
            
            self._recording = False
            return True
            
        except Exception:
            return False
    
    @property
    def is_recording(self) -> bool:
        return self._recording