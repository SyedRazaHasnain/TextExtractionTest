"""Basic tokenization utilities for fallback when NLTK is not available."""

import re
from typing import List, Set, Tuple
from functools import lru_cache

class TokenizationFallback:
    """Fallback tokenization methods when NLTK data is unavailable."""
    
    @staticmethod
    def sent_tokenize(text: str) -> List[str]:
        """Basic sentence tokenization."""
        # Split on common sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def word_tokenize(text: str) -> List[str]:
        """Basic word tokenization."""
        # Split on whitespace and punctuation
        return re.findall(r'\w+', text.lower())
    
    @staticmethod
    @lru_cache(maxsize=1)
    def get_basic_stopwords() -> Set[str]:
        """Get basic stopwords set."""
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
            'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on',
            'that', 'the', 'to', 'was', 'were', 'will', 'with'
        }

    @staticmethod
    def pos_tag(tokens: List[str]) -> List[Tuple[str, str]]:
        """Basic part-of-speech tagging."""
        # Simple rule-based tagging
        basic_tags = {
            'the': 'DT', 'a': 'DT', 'an': 'DT',
            'is': 'VBZ', 'are': 'VBP', 'was': 'VBD', 'were': 'VBD',
            'must': 'MD', 'shall': 'MD', 'should': 'MD', 'will': 'MD',
            'by': 'IN', 'for': 'IN', 'in': 'IN', 'of': 'IN'
        }
        return [(token, basic_tags.get(token.lower(), 'NN')) for token in tokens] 