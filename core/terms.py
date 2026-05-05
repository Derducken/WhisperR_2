# -*- coding: utf-8 -*-
"""
WhisperR Terms Processor
Handles term replacement, commands, and hallucination filtering
"""

import re
from typing import Dict, List, Optional, Tuple


# Default hallucination patterns (case-insensitive substring matching)
DEFAULT_HALLUCINATIONS: List[str] = [
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
    "this video was",
]


class TermsProcessor:
    """
    Process terms, commands, and filter hallucinations
    """
    
    def __init__(
        self,
        terms: Dict[str, str] = None,
        commands: Dict[str, str] = None,
        hallucinations: List[str] = None
    ):
        """
        Initialize processor
        
        Args:
            terms: Dict of trigger -> replacement
            commands: Dict of phrase -> command
            hallucinations: List of hallucination patterns
        """
        self.terms = terms or {}
        self.commands = commands or {}
        self.hallucinations = hallucinations or DEFAULT_HALLUCINATIONS
        
        # Compile regex for terms (case-insensitive)
        self._build_term_regex()
    
    def _build_term_regex(self):
        """Build compiled regex patterns for terms"""
        # Sort by length (longest first) to match longest terms first
        sorted_terms = sorted(self.terms.keys(), key=len, reverse=True)
        
        if sorted_terms:
            # Create pattern that matches any term boundary
            # Word boundaries (\b) won't work for partial word matches,
            # so we use a more flexible approach
            patterns = [re.escape(term) for term in sorted_terms]
            pattern = '|'.join(patterns)
            self._term_regex = re.compile(pattern, re.IGNORECASE)
        else:
            self._term_regex = None
    
    def update_terms(self, terms: Dict[str, str]):
        """Update terms and rebuild regex"""
        self.terms = terms
        self._build_term_regex()
    
    def update_commands(self, commands: Dict[str, str]):
        """Update commands dict"""
        self.commands = commands
    
    def update_hallucinations(self, hallucinations: List[str]):
        """Update hallucination patterns"""
        self.hallucinations = hallucinations
    
    def process(self, text: str) -> Tuple[str, bool]:
        """
        Process text: apply terms and check for commands/hallucinations
        
        Args:
            text: Input text from transcription
            
        Returns:
            Tuple of (processed_text, is_command)
            - processed_text: text with terms replaced
            - is_command: True if text matched a command trigger
        """
        # Check for commands first (command phrases are NOT inserted as text)
        if self._is_command(text):
            return "", True
        
        # Check for hallucinations (don't insert)
        if self._is_hallucination(text):
            return "", False
        
        # Apply term replacements
        processed = self._apply_terms(text)
        
        return processed, False
    
    def _is_command(self, text: str) -> bool:
        """Check if text matches any command trigger"""
        text_lower = text.lower().strip()
        
        for trigger in self.commands.keys():
            if trigger.lower() in text_lower:
                return True
        
        return False
    
    def execute_command(self, text: str) -> Optional[str]:
        """
        Execute command if text matches
        
        Args:
            text: Input text
            
        Returns:
            Command to execute, or None if no command matched
        """
        text_lower = text.lower().strip()
        
        for trigger, command in self.commands.items():
            if trigger.lower() in text_lower:
                return command
        
        return None
    
    def _is_hallucination(self, text: str) -> bool:
        """Check if text is a hallucination"""
        text_stripped = text.strip()
        text_lower = text_stripped.lower()
        
        for pattern in self.hallucinations:
            pattern_lower = pattern.lower()
            # Check exact match or prefix match
            if text_lower == pattern_lower or text_lower.startswith(pattern_lower):
                return True
        
        return False
    
    def _apply_terms(self, text: str) -> str:
        """Apply term replacements to text"""
        if not self._term_regex:
            return text
        
        def replace_func(match):
            term = match.group(0)
            # Find the original case from terms dict
            for orig, replacement in self.terms.items():
                if orig.lower() == term.lower():
                    return replacement
            return term
        
        return self._term_regex.sub(replace_func, text)
    
    def apply_typing_expansion(self, text: str) -> str:
        """
        Apply text expansion while typing
        Called on space/punctuation to check for term triggers
        
        Args:
            text: Current text including the trigger word
            
        Returns:
            Expanded text if trigger matched, otherwise original
        """
        # Check each term to see if it appears at end of text
        text_lower = text.lower().strip()
        
        for trigger, replacement in self.terms.items():
            # Check if text ends with trigger (with word boundary)
            if text_lower.endswith(trigger.lower()):
                # Get the position of the trigger
                trigger_pos = len(text) - len(trigger)
                # Replace the trigger with the replacement
                result = text[:trigger_pos] + replacement
                return result
        
        return text
    
    def is_term_trigger(self, text: str) -> bool:
        """
        Check if the given text ends with a term trigger
        
        Args:
            text: Text to check
            
        Returns:
            True if text ends with a term trigger
        """
        text_lower = text.lower().strip()
        
        for trigger in self.terms.keys():
            if text_lower.endswith(trigger.lower()):
                return True
        
        return False


def create_default_processor() -> TermsProcessor:
    """Create a processor with default settings"""
    return TermsProcessor(
        terms={
            "whisper ar": "WhisperR",
            "youre": "you're",
            "dont": "don't",
            "cant": "can't",
            "wont": "won't",
            "its a": "it's a"
        },
        commands={
            "Launch Notepad": "notepad.exe"
        },
        hallucinations=DEFAULT_HALLUCINATIONS.copy()
    )