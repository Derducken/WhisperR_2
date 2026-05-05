# -*- coding: utf-8 -*-
"""
WhisperR Diff Engine
Provides word-level and line-level text comparison for version history
"""

import difflib
from typing import List, Tuple, Dict, Any


def compute_word_diff(text1: str, text2: str) -> List[Dict[str, Any]]:
    """
    Compare two texts at word level
    
    Args:
        text1: Original text
        text2: Modified text
        
    Returns:
        List of dicts with keys: 'type' (added/removed/unchanged), 'text'
    """
    # Split into words while preserving whitespace
    import re
    word_pattern = re.compile(r'(\s+|\S+)')
    words1 = word_pattern.findall(text1)
    words2 = word_pattern.findall(text2)
    
    # Use SequenceMatcher for word-level diff
    matcher = difflib.SequenceMatcher(None, words1, words2)
    
    result = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            result.append({
                'type': 'unchanged',
                'text': ''.join(words1[i1:i2])
            })
        elif tag == 'replace':
            result.append({
                'type': 'removed',
                'text': ''.join(words1[i1:i2])
            })
            result.append({
                'type': 'added',
                'text': ''.join(words2[j1:j2])
            })
        elif tag == 'delete':
            result.append({
                'type': 'removed',
                'text': ''.join(words1[i1:i2])
            })
        elif tag == 'insert':
            result.append({
                'type': 'added',
                'text': ''.join(words2[j1:j2])
            })
    
    return result


def compute_line_diff(text1: str, text2: str) -> List[Dict[str, Any]]:
    """
    Compare two texts at line level
    
    Args:
        text1: Original text
        text2: Modified text
        
    Returns:
        List of dicts with keys: 'type' (added/removed/unchanged), 'text', 'line_num'
    """
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()
    
    matcher = difflib.SequenceMatcher(None, lines1, lines2)
    
    result = []
    line_num = 0
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for line in lines1[i1:i2]:
                result.append({
                    'type': 'unchanged',
                    'text': line,
                    'line_num': line_num
                })
                line_num += 1
        elif tag == 'replace':
            for line in lines1[i1:i2]:
                result.append({
                    'type': 'removed',
                    'text': line,
                    'line_num': line_num
                })
            for line in lines2[j1:j2]:
                result.append({
                    'type': 'added',
                    'text': line,
                    'line_num': line_num
                })
                line_num += 1
        elif tag == 'delete':
            for line in lines1[i1:i2]:
                result.append({
                    'type': 'removed',
                    'text': line,
                    'line_num': line_num
                })
        elif tag == 'insert':
            for line in lines2[j1:j2]:
                result.append({
                    'type': 'added',
                    'text': line,
                    'line_num': line_num
                })
                line_num += 1
    
    return result


def generate_html_diff(text1: str, text2: str, mode: str = 'word') -> str:
    """
    Generate HTML representation of the diff
    
    Args:
        text1: Original text
        text2: Modified text
        mode: 'word' or 'line'
        
    Returns:
        HTML string with colored diff
    """
    if mode == 'word':
        diffs = compute_word_diff(text1, text2)
    else:
        diffs = compute_line_diff(text1, text2)
    
    html_parts = []
    
    for diff in diffs:
        text = diff['text']
        # Escape HTML special characters
        escaped = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        if diff['type'] == 'added':
            html_parts.append(
                f'<span class="diff-added" style="background-color: #1a3d1a; color: #4caf50;">{escaped}</span>'
            )
        elif diff['type'] == 'removed':
            html_parts.append(
                f'<span class="diff-removed" style="background-color: #3d1a1a; color: #f44336; text-decoration: line-through;">{escaped}</span>'
            )
        else:
            html_parts.append(f'<span class="diff-unchanged">{escaped}</span>')
    
    return ''.join(html_parts)


def get_diff_stats(text1: str, text2: str) -> Dict[str, int]:
    """
    Get statistics about the differences
    
    Returns:
        Dict with keys: additions, deletions, unchanged (word counts)
    """
    diffs = compute_word_diff(text1, text2)
    
    stats = {
        'additions': 0,
        'deletions': 0,
        'unchanged': 0
    }
    
    for diff in diffs:
        # Count words (non-whitespace sequences)
        words = diff['text'].split()
        count = len(words) if diff['text'].strip() else 0
        
        if diff['type'] == 'added':
            stats['additions'] += count
        elif diff['type'] == 'removed':
            stats['deletions'] += count
        else:
            stats['unchanged'] += count
    
    return stats


def summarize_changes(text1: str, text2: str, max_length: int = 100) -> str:
    """
    Create a brief summary of changes
    
    Args:
        text1: Original text
        text2: Modified text
        max_length: Maximum length of summary
        
    Returns:
        Short string describing the change
    """
    stats = get_diff_stats(text1, text2)
    
    parts = []
    if stats['additions'] > 0:
        parts.append(f"+{stats['additions']} words")
    if stats['deletions'] > 0:
        parts.append(f"-{stats['deletions']} words")
    
    if not parts:
        return "No changes"
    
    summary = ", ".join(parts)
    
    if len(summary) > max_length:
        summary = summary[:max_length-3] + "..."
    
    return summary