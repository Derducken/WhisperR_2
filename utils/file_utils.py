# -*- coding: utf-8 -*-
"""
WhisperR File Utilities
Project save/load, backup operations
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


class ProjectFile:
    """
    WhisperR project file handler (.wrp)
    """
    
    VERSION = "2.1"
    
    def __init__(self, filepath: str):
        self.filepath = filepath
    
    def save(
        self,
        content: str,
        notes: list = None,
        word_target: int = 0,
        notes_visible: bool = True,
        notes_color_filter: list = None
    ) -> bool:
        """
        Save project to file
        
        Args:
            content: Main text content
            notes: List of note dicts
            word_target: Word count target
            notes_visible: Notes panel visibility
            notes_color_filter: List of visible colors
            
        Returns:
            True if saved successfully
        """
        try:
            data = {
                "version": self.VERSION,
                "content": content,
                "notes": notes or [],
                "word_target": word_target,
                "notes_visible": notes_visible,
                "notes_color_filter": notes_color_filter or [],
                "saved_at": datetime.now().isoformat()
            }
            
            # Create backup of existing file
            if os.path.exists(self.filepath):
                backup = self.filepath + ".backup"
                shutil.copy2(self.filepath, backup)
            
            # Write new file
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            return False
    
    def load(self) -> Optional[Dict[str, Any]]:
        """
        Load project from file
        
        Returns:
            Dict with project data, or None on error
        """
        if not os.path.exists(self.filepath):
            return None
        
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Ensure all fields exist (for older projects)
            return {
                "version": data.get("version", "1.0"),
                "content": data.get("content", ""),
                "notes": data.get("notes", []),
                "word_target": data.get("word_target", 0),
                "notes_visible": data.get("notes_visible", True),
                "notes_color_filter": data.get("notes_color_filter", [])
            }
            
        except Exception:
            return None
    
    @staticmethod
    def is_project_file(filepath: str) -> bool:
        """Check if file is a valid project file"""
        if not filepath.endswith('.wrp'):
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return "version" in data and "content" in data
        except Exception:
            return False


def save_project(
    filepath: str,
    content: str,
    notes: list = None,
    word_target: int = 0,
    notes_visible: bool = True
) -> bool:
    """
    Convenience function to save project
    
    Args:
        filepath: Path to .wrp file
        content: Main text content
        notes: List of notes
        word_target: Word count target
        notes_visible: Notes panel visibility
        
    Returns:
        True if saved successfully
    """
    project = ProjectFile(filepath)
    return project.save(content, notes, word_target, notes_visible)


def load_project(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Convenience function to load project
    
    Args:
        filepath: Path to .wrp file
        
    Returns:
        Project data dict, or None on error
    """
    project = ProjectFile(filepath)
    return project.load()


def create_backup(
    source_path: str,
    backup_folder: str,
    max_backups: int = 5
) -> Optional[str]:
    """
    Create timestamped backup of a file
    
    Args:
        source_path: File to backup
        backup_folder: Destination folder
        max_backups: Maximum backups to keep
        
    Returns:
        Path to backup file, or None on error
    """
    if not os.path.exists(source_path):
        return None
    
    try:
        os.makedirs(backup_folder, exist_ok=True)
        
        # Generate timestamped filename
        filename = os.path.basename(source_path)
        name, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        backup_name = f"{name}_{timestamp}.bak"
        backup_path = os.path.join(backup_folder, backup_name)
        
        # Copy file
        shutil.copy2(source_path, backup_path)
        
        # Prune old backups
        _prune_backups(backup_folder, name, max_backups)
        
        return backup_path
        
    except Exception:
        return None


def _prune_backups(folder: str, name_prefix: str, max_keep: int):
    """Remove old backups, keeping only max_keep most recent"""
    try:
        # Find all backups matching the prefix
        backups = []
        for f in os.listdir(folder):
            if f.startswith(name_prefix) and f.endswith('.bak'):
                path = os.path.join(folder, f)
                mtime = os.path.getmtime(path)
                backups.append((mtime, path))
        
        # Sort by modification time (newest first)
        backups.sort(reverse=True)
        
        # Remove old backups
        for _, path in backups[max_keep:]:
            try:
                os.remove(path)
            except Exception:
                pass
                
    except Exception:
        pass


def get_backup_list(backup_folder: str, name_prefix: str) -> list:
    """
    Get list of backups for a file
    
    Returns:
        List of (filename, path, timestamp) tuples, sorted newest first
    """
    backups = []
    
    try:
        for f in os.listdir(backup_folder):
            if f.startswith(name_prefix) and f.endswith('.bak'):
                path = os.path.join(backup_folder, f)
                mtime = os.path.getmtime(path)
                timestamp = datetime.fromtimestamp(mtime)
                backups.append((f, path, timestamp))
    
        # Sort newest first
        backups.sort(key=lambda x: x[2], reverse=True)
        
    except Exception:
        pass
    
    return backups


def export_to_markdown(content: str, output_path: str) -> bool:
    """Export content as Markdown file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception:
        return False


def export_to_text(content: str, output_path: str) -> bool:
    """Export content as plain text"""
    try:
        # Strip markdown formatting for plain text
        import re
        text = content
        
        # Remove headers
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        
        # Remove bold/italic
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Remove links
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Remove list markers
        text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return True
    except Exception:
        return False