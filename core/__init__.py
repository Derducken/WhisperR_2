# WhisperR Core Package
# Business logic modules (no PyQt imports)

from .config import AppConfig
from .audio import AudioRecorder
from .terms import TermsProcessor
from .diff_engine import compute_word_diff, compute_line_diff, get_diff_stats

__all__ = [
    'AppConfig',
    'AudioRecorder',
    'TermsProcessor',
    'compute_word_diff',
    'compute_line_diff',
    'get_diff_stats',
]