# WhisperR Plugins Package
# Plugin system for extensibility (future)

class PluginBase:
    """Base class for WhisperR plugins"""
    def __init__(self):
        self.name = "Base Plugin"
        self.version = "1.0.0"
        self.description = ""
    
    def on_load(self):
        """Called when plugin is loaded"""
        pass
    
    def on_unload(self):
        """Called when plugin is unloaded"""
        pass
    
    def process_text(self, text: str) -> str:
        """Process transcribed text"""
        return text
    
    def get_menu_items(self):
        """Return list of (menu_path, callback) tuples"""
        return []

__all__ = ['PluginBase']