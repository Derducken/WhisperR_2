# WhisperR Themes Package

from .dark import THEME as DARK_THEME
from .dark_true_black import THEME as DARK_TRUE_BLACK_THEME
from .light import THEME as LIGHT_THEME

THEMES = {
    'dark': DARK_THEME,
    'dark_true_black': DARK_TRUE_BLACK_THEME,
    'light': LIGHT_THEME,
}

def get_theme(name: str):
    """Get theme by name"""
    return THEMES.get(name, DARK_THEME)

__all__ = ['THEMES', 'get_theme', 'DARK_THEME', 'DARK_TRUE_BLACK_THEME', 'LIGHT_THEME']