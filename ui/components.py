# -*- coding: utf-8 -*-
"""
WhisperR UI Components
Reusable PyQt6 components with consistent styling and tooltips
"""

from PyQt6.QtWidgets import (
    QPushButton, QLineEdit, QSlider, QComboBox, 
    QTextEdit, QLabel, QWidget, QHBoxLayout, QVBoxLayout
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont


class TooltipMixin:
    """Mixin to add tooltip functionality to widgets"""
    
    def set_tip(self, text: str, detailed: str = ""):
        """
        Set tooltip with optional detailed text
        
        Args:
            text: Short tooltip text
            detailed: Optional detailed tooltip (shown on hover)
        """
        if detailed:
            self.setToolTip(f"{text}\n\n{detailed}")
        else:
            self.setToolTip(text)


class WhisperButton(QPushButton, TooltipMixin):
    """
    Styled button with consistent styling across the app
    """
    
    def __init__(self, text: str = "", icon: str = None, tooltip: str = ""):
        super().__init__(text)
        
        if icon:
            self.setText(f"{icon} {text}")
        
        if tooltip:
            self.set_tip(tooltip)
        
        self._apply_style()
    
    def _apply_style(self):
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(28)


class WhisperPrimaryButton(WhisperButton):
    """
    Primary action button (highlighted)
    """
    
    def _apply_style(self):
        super()._apply_style()
        self.setObjectName("primaryButton")


class WhisperDangerButton(WhisperButton):
    """
    Danger/Delete button (red accent)
    """
    
    def _apply_style(self):
        super()._apply_style()
        self.setObjectName("dangerButton")


class WhisperInput(QLineEdit, TooltipMixin):
    """
    Styled text input field
    """
    
    def __init__(self, placeholder: str = "", tooltip: str = ""):
        super().__init__()
        
        if placeholder:
            self.setPlaceholderText(placeholder)
        
        if tooltip:
            self.set_tip(tooltip)
        
        self.setMinimumHeight(26)


class WhisperSlider(QSlider, TooltipMixin):
    """
    Styled slider with value display
    """
    
    value_changed = pyqtSignal(int)
    
    def __init__(
        self,
        min_val: int = 0,
        max_val: int = 100,
        value: int = 50,
        tooltip: str = "",
        show_value: bool = True
    ):
        super().__init__(Qt.Orientation.Horizontal)
        
        self.setMinimum(min_val)
        self.setMaximum(max_val)
        self.setValue(value)
        
        if tooltip:
            self.set_tip(tooltip)
        
        self.show_value = show_value
        self.valueLabel = QLabel(str(value))
        self.valueLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.valueLabel.setMinimumWidth(40)
        
        self.valueChanged.connect(self._on_value_changed)
        self.valueChanged.connect(self.value_changed.emit)
    
    def _on_value_changed(self, value: int):
        if self.show_value:
            self.valueLabel.setText(str(value))
    
    def get_with_label(self) -> QWidget:
        """Return widget with slider and label"""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self)
        layout.addWidget(self.valueLabel)
        return container


class WhisperDropdown(QComboBox, TooltipMixin):
    """
    Styled dropdown/combobox
    """
    
    def __init__(self, items: list = None, tooltip: str = ""):
        super().__init__()
        
        if items:
            self.addItems(items)
        
        if tooltip:
            self.set_tip(tooltip)
        
        self.setMinimumHeight(26)


class WhisperTextEdit(QTextEdit, TooltipMixin):
    """
    Styled text edit area
    """
    
    def __init__(self, placeholder: str = "", tooltip: str = ""):
        super().__init__()
        
        if placeholder:
            self.setPlaceholderText(placeholder)
        
        if tooltip:
            self.set_tip(tooltip)


class WhisperLabel(QLabel, TooltipMixin):
    """
    Styled label
    """
    
    def __init__(self, text: str = "", tooltip: str = "", bold: bool = False):
        super().__init__(text)
        
        if tooltip:
            self.set_tip(tooltip)
        
        if bold:
            font = self.font()
            font.setBold(True)
            self.setFont(font)


class WhisperSection(QWidget):
    """
    Collapsible section with title
    """
    
    def __init__(self, title: str, collapsible: bool = True):
        super().__init__()
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.title_label = WhisperLabel(title, bold=True)
        self.title_label.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(self.title_label)
        
        self.content = QWidget()
        layout.addWidget(self.content)
        
        if collapsible:
            self.title_label.mousePressEvent = self._toggle
        
        self._collapsed = False
    
    def _toggle(self, event):
        self._collapsed = not self._collapsed
        self.content.setVisible(not self._collapsed)
    
    def get_content_layout(self) -> QVBoxLayout:
        """Get the content layout to add widgets to"""
        return QVBoxLayout(self.content)


class IconButton(WhisperButton):
    """
    Icon-only button (for toolbars)
    """
    
    def __init__(self, icon: str, tooltip: str = "", size: int = 32):
        super().__init__(icon, tooltip=tooltip)
        self.setFixedSize(size, size)
        self.setStyleSheet("""
            QPushButton {
                border: none;
                padding: 4px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #353535;
            }
        """)


class ToolButton(IconButton):
    """
    Toolbar button with icon and optional text
    """
    
    def __init__(self, icon: str, text: str = "", tooltip: str = ""):
        display = f"{icon} {text}" if text else icon
        super().__init__(icon, tooltip)
        self.setText(display)
        self.setFixedHeight(32)
        self.setStyleSheet("""
            QPushButton {
                border: none;
                padding: 4px 8px;
                border-radius: 4px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #353535;
            }
        """)


# Component factory for creating components from config


class ComponentFactory:
    """
    Factory for creating consistently styled components
    """
    
    @staticmethod
    def create_button(
        text: str,
        button_type: str = "default",
        tooltip: str = ""
    ) -> WhisperButton:
        """Create a styled button"""
        if button_type == "primary":
            return WhisperPrimaryButton(text, tooltip=tooltip)
        elif button_type == "danger":
            return WhisperDangerButton(text, tooltip=tooltip)
        else:
            return WhisperButton(text, tooltip=tooltip)
    
    @staticmethod
    def create_input(
        placeholder: str = "",
        tooltip: str = ""
    ) -> WhisperInput:
        """Create a styled input"""
        return WhisperInput(placeholder, tooltip)
    
    @staticmethod
    def create_slider(
        min_val: int = 0,
        max_val: int = 100,
        value: int = 50,
        tooltip: str = ""
    ) -> WhisperSlider:
        """Create a styled slider"""
        return WhisperSlider(min_val, max_val, value, tooltip)
    
    @staticmethod
    def create_dropdown(
        items: list = None,
        tooltip: str = ""
    ) -> WhisperDropdown:
        """Create a styled dropdown"""
        return WhisperDropdown(items, tooltip)
    
    @staticmethod
    def create_label(
        text: str = "",
        tooltip: str = "",
        bold: bool = False
    ) -> WhisperLabel:
        """Create a styled label"""
        return WhisperLabel(text, tooltip, bold)