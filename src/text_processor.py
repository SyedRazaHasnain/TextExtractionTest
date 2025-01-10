"""Text processing utilities for regulatory text analysis."""

from typing import List, Tuple

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Ensure required NLTK data is available
nltk.download('punkt', quiet=True)

class TextProcessor:
    """Handles text processing operations."""

    @staticmethod
    def segment_text(text: str) -> List[Tuple[int, str]]:
        """
        Split text into sections based on section markers.

        Args:
            text (str): Input text to segment

        Returns:
            List[Tuple[int, str]]: List of (section number, section text) tuples
        """
        sections = []
        current_section = []
        section_number = 0

        for line in text.split('\n'):
            if line.strip().lower().startswith('section'):
                if current_section:
                    sections.append((section_number, '\n'.join(current_section)))
                current_section = [line]
                section_number += 1
            elif line.strip():
                current_section.append(line)

        if current_section:
            sections.append((section_number, '\n'.join(current_section)))

        return sections

    @staticmethod
    def get_text_metrics(text: str) -> dict:
        """
        Calculate metrics for the given text.

        Args:
            text (str): Input text

        Returns:
            dict: Dictionary containing text metrics
        """
        return {
            'word_count': len(word_tokenize(text)),
            'sentence_count': len(sent_tokenize(text)),
            'character_count': len(text),
            'paragraph_count': len([p for p in text.split('\n\n') if p.strip()])
        } 