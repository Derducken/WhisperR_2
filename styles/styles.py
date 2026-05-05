# -*- coding: utf-8 -*-
"""
WhisperR Style Generator
Generates Qt Style Sheets from theme dictionaries
"""

from typing import Dict, Any


def generate_qss(theme: Dict[str, Any]) -> str:
    """
    Generate Qt Style Sheet from theme dictionary
    
    Args:
        theme: Theme dictionary with color values
        
    Returns:
        QSS string for PyQt6 styling
    """
    return f"""
/* Base styles */
QMainWindow, QDialog, QScrollArea, QTabWidget, QTabBar, QStackedWidget {{
    background-color: {theme['background']};
}}

QWidget {{
    background-color: {theme['background']};
    color: {theme['text']};
    font-family: 'Segoe UI';
    font-size: 9pt;
}}

QWidget > QMenu, QMenu {{
    background-color: {theme['surface']};
    color: {theme['text_secondary']};
}}

QFrame {{
    background-color: {theme['background']};
}}

QScrollArea > QWidget > QWidget {{
    background-color: {theme['background']};
}}

/* Tab bar */
QTabBar::tab {{
    background-color: {theme['surface']};
    color: {theme['text_secondary']};
    padding: 6px 14px;
    border: 1px solid {theme['border']};
    border-bottom: none;
    border-radius: 3px 3px 0 0;
}}

QTabBar::tab:selected {{
    background-color: {theme['primary']};
    color: {theme['text']};
}}

QTabBar::tab:hover {{
    background-color: {theme['surface_variant']};
}}

/* Text controls */
QTextEdit, QPlainTextEdit {{
    background-color: {theme['surface']};
    border: 1px solid {theme['border']};
    color: {theme['text']};
    border-radius: 4px;
    selection-background-color: {theme['selection_background']};
    selection-color: {theme['selection_text']};
}}

/* Buttons */
QPushButton {{
    background-color: {theme['button_background']};
    border: 1px solid {theme['border']};
    padding: 6px 16px;
    border-radius: 4px;
    color: {theme['text']};
}}

QPushButton:hover {{
    background-color: {theme['button_hover']};
    border: 1px solid {theme['primary']};
}}

QPushButton:pressed {{
    background-color: {theme['button_active']};
}}

QPushButton:disabled {{
    background-color: {theme['button_disabled']};
    color: {theme['text_disabled']};
}}

/* Group boxes */
QGroupBox {{
    border: 1px solid {theme['border']};
    margin-top: 12px;
    font-weight: bold;
    padding: 8px;
    border-radius: 4px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}}

/* Progress bar */
QProgressBar {{
    background: {theme['surface']};
    border: 1px solid {theme['border']};
    text-align: center;
    height: 12px;
    border-radius: 6px;
}}

QProgressBar::chunk {{
    background-color: {theme['primary']};
    border-radius: 6px;
}}

/* Headers */
QHeaderView::section {{
    background-color: {theme['surface_variant']};
    color: {theme['text']};
    padding: 4px;
    border: 1px solid {theme['border']};
}}

QTableWidget {{
    background-color: {theme['surface']};
    grid-color: {theme['border']};
}}

/* Combo boxes and line edits */
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {theme['input_background']};
    border: 1px solid {theme['input_border']};
    padding: 4px 28px 4px 6px;
    min-height: 22px;
    border-radius: 3px;
    color: {theme['text']};
}}

QComboBox:focus, QLineEdit:focus {{
    border: 1px solid {theme['input_focus_border']};
}}

QComboBox::drop-down {{
    border: none;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 3px solid transparent;
    border-right: 3px solid transparent;
    border-top: 5px solid {theme['text_secondary']};
    margin-right: 8px;
}}

QSpinBox::up-button, QDoubleSpinBox::up-button {{
    width: 0;
    border: none;
}}

QSpinBox::down-button, QDoubleSpinBox::down-button {{
    width: 0;
    border: none;
}}

/* Menus */
QMenu {{
    background-color: {theme['surface']};
    color: {theme['text_secondary']};
    border: 1px solid {theme['border']};
    border-radius: 4px;
}}

QMenu::item {{
    padding: 4px 20px;
}}

QMenu::item:selected {{
    background-color: {theme['primary']};
    color: {theme['text']};
}}

QMenu::item:hover {{
    background-color: {theme['primary']};
    color: {theme['text']};
}}

QMenu::item:pressed {{
    background-color: {theme['primary_active']};
}}

QMenu::separator {{
    height: 1px;
    background: {theme['border']};
    margin: 2px 0;
}}

QMenu::item:disabled {{
    color: {theme['text_disabled']};
}}

/* Tooltips */
QToolTip {{
    background-color: {theme['surface']};
    color: {theme['text']};
    border: 1px solid {theme['primary']};
    padding: 4px;
}}

/* Checkboxes and radios */
QCheckBox, QRadioButton {{
    color: {theme['text']};
}}

QCheckBox::indicator, QRadioButton::indicator {{
    background-color: {theme['surface']};
    border: 1px solid {theme['border']};
    border-radius: 3px;
}}

QCheckBox::indicator:checked {{
    background-color: {theme['primary']};
}}

QRadioButton::indicator:checked {{
    background-color: {theme['primary']};
}}

/* Scrollbars */
QScrollBar:vertical {{
    background: {theme['scrollbar_background']};
    width: 12px;
    border: none;
}}

QScrollBar::handle:vertical {{
    background: {theme['scrollbar_handle']};
    min-height: 20px;
    border-radius: 6px;
}}

QScrollBar::handle:vertical:hover {{
    background: {theme['scrollbar_handle_hover']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background: {theme['scrollbar_background']};
    height: 12px;
    border: none;
}}

QScrollBar::handle:horizontal {{
    background: {theme['scrollbar_handle']};
    min-width: 20px;
    border-radius: 6px;
}}

QScrollBar::handle:horizontal:hover {{
    background: {theme['scrollbar_handle_hover']};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* Sliders */
QSlider::groove:horizontal {{
    background: {theme['surface_variant']};
    height: 6px;
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background: {theme['primary']};
    width: 16px;
    margin: -5px 0;
    border-radius: 8px;
}}

QSlider::groove:vertical {{
    background: {theme['surface_variant']};
    width: 6px;
    border-radius: 3px;
}}

QSlider::handle:vertical {{
    background: {theme['primary']};
    height: 16px;
    margin: 0 -5px;
    border-radius: 8px;
}}

/* Lists */
QListWidget {{
    background-color: {theme['surface']};
    border: 1px solid {theme['border']};
    border-radius: 4px;
}}

QListWidget::item:selected {{
    background-color: {theme['primary']};
    color: {theme['text']};
}}

QListWidget::item:hover {{
    background-color: {theme['surface_variant']};
}}

/* Status bar */
QStatusBar {{
    background-color: {theme['surface']};
    color: {theme['text_secondary']};
}}

/* Splitters */
QSplitter::handle {{
    background-color: {theme['border']};
}}

QSplitter::handle:horizontal {{
    width: 2px;
}}

QSplitter::handle:vertical {{
    height: 2px;
}}

/* Tool bar */
QToolBar {{
    background-color: {theme['surface']};
    border: none;
    spacing: 3px;
    padding: 4px;
}}

QToolButton {{
    background-color: transparent;
    border: none;
    border-radius: 4px;
    padding: 4px;
}}

QToolButton:hover {{
    background-color: {theme['surface_variant']};
}}

QToolButton:pressed {{
    background-color: {theme['button_active']};
}}
"""


def generate_markdown_preview_css(theme: Dict[str, Any]) -> str:
    """
    Generate CSS for markdown preview in web view
    
    Args:
        theme: Theme dictionary with color values
        
    Returns:
        CSS string for HTML preview
    """
    return f"""
    <style>
        body {{
            background-color: {theme['surface']};
            color: {theme['text']};
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            line-height: 1.6;
            padding: 20px;
            margin: 0;
        }}
        
        h1, h2, h3, h4, h5, h6 {{
            color: {theme['md_heading']};
            margin-top: 24px;
            margin-bottom: 16px;
            font-weight: 600;
            line-height: 1.25;
        }}
        
        h1 {{ font-size: 2em; border-bottom: 1px solid {theme['border']}; padding-bottom: 0.3em; }}
        h2 {{ font-size: 1.5em; border-bottom: 1px solid {theme['border']}; padding-bottom: 0.3em; }}
        h3 {{ font-size: 1.25em; }}
        h4 {{ font-size: 1em; }}
        
        p {{
            margin-top: 0;
            margin-bottom: 16px;
        }}
        
        a {{
            color: {theme['md_link']};
            text-decoration: none;
        }}
        
        a:hover {{
            text-decoration: underline;
        }}
        
        code {{
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            background-color: {theme['code_background']};
            color: {theme['md_code']};
            padding: 0.2em 0.4em;
            border-radius: 3px;
            font-size: 85%;
        }}
        
        pre {{
            background-color: {theme['code_background']};
            padding: 16px;
            overflow: auto;
            border-radius: 6px;
            margin-bottom: 16px;
        }}
        
        pre code {{
            padding: 0;
            background-color: transparent;
        }}
        
        blockquote {{
            margin: 0 0 16px 0;
            padding: 0 16px;
            color: {theme['text_secondary']};
            border-left: 4px solid {theme['primary']};
        }}
        
        ul, ol {{
            margin-top: 0;
            margin-bottom: 16px;
            padding-left: 2em;
        }}
        
        li + li {{
            margin-top: 0.25em;
        }}
        
        hr {{
            height: 2px;
            padding: 0;
            margin: 24px 0;
            background-color: {theme['border']};
            border: 0;
        }}
        
        table {{
            border-collapse: collapse;
            margin-bottom: 16px;
            width: 100%;
        }}
        
        table th, table td {{
            padding: 8px 12px;
            border: 1px solid {theme['border']};
            text-align: left;
        }}
        
        table th {{
            background-color: {theme['surface_variant']};
            font-weight: 600;
        }}
        
        table tr:nth-child(even) {{
            background-color: {theme['surface_variant']};
        }}
        
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        
        /* Task list checkboxes */
        input[type="checkbox"] {{
            margin-right: 8px;
        }}
        
        /* Strong and emphasis */
        strong {{
            font-weight: 600;
            color: {theme['md_bold']};
        }}
        
        em {{
            font-style: italic;
            color: {theme['md_italic']};
        }}
        
        /* Kbd style */
        kbd {{
            background-color: {theme['code_background']};
            border: 1px solid {theme['border']};
            border-bottom: 2px solid {theme['border']};
            border-radius: 3px;
            padding: 2px 6px;
            font-family: 'Consolas', monospace;
            font-size: 85%;
        }}
    </style>
    """


def apply_theme(app, theme_name: str):
    """
    Apply a theme to the application
    
    Args:
        app: QApplication instance
        theme_name: Name of theme to apply
    """
    from themes import get_theme
    
    theme = get_theme(theme_name)
    qss = generate_qss(theme)
    app.setStyleSheet(qss)