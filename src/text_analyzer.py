"""Text analysis module implementing advanced NLP-based text analysis."""

import os
import re
import ssl
import time
import logging
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from collections import defaultdict
from pathlib import Path

from .tokenization import TokenizationFallback
from .patterns import (
    REQUIREMENT_INDICATORS,
    METRIC_PATTERNS,
    NLTK_PACKAGES,
)
from .constants import MAX_RETRIES, RETRY_DELAY, DOWNLOAD_TIMEOUT
from .exceptions import TextAnalyzerError, ValidationError, TextProcessingError

logger = logging.getLogger(__name__)

class TextAnalyzer:
    """Implements advanced NLP-based text analysis capabilities."""

    # Class-level constants
    MAX_RETRIES = 3
    RETRY_DELAY = 1
    DOWNLOAD_TIMEOUT = 30
    NLTK_PACKAGES = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'wordnet']

    def __init__(self, mode='simulated_nlp'):
        """Initialize the analyzer with specified mode."""
        if mode not in ['simulated_nlp', 'openai']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'simulated_nlp' or 'openai'")
            
        self.mode = mode
        self.using_fallback = False
        self.downloaded_packages = set()
        self.nltk_loaded = False
        
        if self.mode == 'simulated_nlp':
            self._setup_simulated_mode()
        else:
            self._setup_basic_components()
            
        logger.info(f"TextAnalyzer initialized in {mode} mode")

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text using advanced NLP techniques.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, Any]: Analysis results containing:
                - key_requirements: List of requirements with confidence scores
                - deadlines_and_metrics: List of metrics with context
                - hierarchical_summary: Structured summary of the text
            
        Raises:
            ValidationError: If input is invalid
            TextProcessingError: If analysis fails
        """
        if not isinstance(text, str):
            raise ValidationError("Input must be a string")
        
        logger.debug(f"Analyzing text of length {len(text)}")
        
        try:
            if not text.strip():
                return {
                    'key_requirements': [],
                    'deadlines_and_metrics': [],
                    'hierarchical_summary': []
                }

            if self.mode == 'simulated_nlp':
                preprocessed_text = self._preprocess_sentence(text)
                requirements = self._extract_requirements(preprocessed_text)
                metrics = self._extract_metrics(text)
                
                result = {
                    'key_requirements': [
                        {
                            'requirement': req,
                            'confidence': self._calculate_confidence(req)
                        }
                        for req in requirements
                    ],
                    'deadlines_and_metrics': [
                        {
                            'type': metric_type,
                            'value': value,
                            'context': context
                        }
                        for value, metric_type, context in metrics
                    ],
                    'hierarchical_summary': self._generate_hierarchical_summary(
                        self.sent_tokenize(text),
                        self._get_processed_sentences(text)
                    )
                }
                
                logger.info(f"Analysis complete: found {len(requirements)} requirements and {len(metrics)} metrics")
                return result
            else:
                return {
                    'key_requirements': [],
                    'deadlines_and_metrics': [],
                    'hierarchical_summary': []
                }
                
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            raise TextProcessingError(f"Failed to analyze text: {str(e)}") from e

    def _calculate_confidence(self, requirement: str) -> float:
        """
        Calculate confidence score for a requirement.

        Args:
            requirement (str): The requirement text

        Returns:
            float: Confidence score between 0 and 1
        """
        requirement_lower = requirement.lower()
        
        # Strong indicators get automatic 1.0
        strong_indicators = ['must', 'shall', 'required', 'mandatory']
        if any(indicator in requirement_lower for indicator in strong_indicators):
            return 1.0
            
        # Medium indicators
        medium_indicators = ['should', 'needs to']
        if any(indicator in requirement_lower for indicator in medium_indicators):
            return 0.7
            
        # Weak or no indicators
        return 0.5

    def _get_processed_sentences(self, text: str) -> List[List[Tuple[str, str]]]:
        """
        Process sentences for hierarchical summary.

        Args:
            text (str): Input text

        Returns:
            List[List[Tuple[str, str]]]: Processed sentences with POS tags
        """
        sentences = self.sent_tokenize(text)
        return [self.pos_tag(self.word_tokenize(sent)) for sent in sentences]

    def _setup_simulated_mode(self):
        """Set up all components for simulated NLP mode."""
        # Configure SSL context
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        # Load and initialize NLTK
        self._load_nltk()
        self._initialize_nltk()
        self._setup_tokenization()
        self._setup_indicators_and_patterns()

    def _setup_basic_components(self):
        """Set up basic components for OpenAI mode."""
        self.sent_tokenize = TokenizationFallback.sent_tokenize
        self.word_tokenize = TokenizationFallback.word_tokenize
        self.stop_words = TokenizationFallback.get_basic_stopwords()
        self.pos_tag = TokenizationFallback.pos_tag
        self.tokenizer = None
        self.lemmatizer = None
        self._setup_indicators_and_patterns()

    def _initialize_nltk(self) -> None:
        """Initialize NLTK components with proper error handling."""
        if not self.nltk_loaded:
            return

        # Create and set NLTK data directory
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        if nltk_data_dir not in nltk.data.path:
            nltk.data.path.append(nltk_data_dir)

        for package in NLTK_PACKAGES:
            if not self._download_nltk_package(package, nltk_data_dir):
                logger.warning(f"Using fallback for {package}")
                self.using_fallback = True

    def _setup_indicators_and_patterns(self):
        """Set up requirement indicators and metric patterns."""
        self.requirement_indicators = REQUIREMENT_INDICATORS
        self.metric_patterns = METRIC_PATTERNS

    def _load_nltk(self):
        """Lazy load NLTK only when needed."""
        if not self.nltk_loaded:
            global nltk, nltk_sent_tokenize, nltk_word_tokenize, stopwords, RegexpTokenizer
            global WordNetLemmatizer, wordnet, pos_tag
            
            import nltk
            from nltk.tokenize import sent_tokenize as nltk_sent_tokenize
            from nltk.tokenize import word_tokenize as nltk_word_tokenize
            from nltk.corpus import stopwords
            from nltk.tokenize import RegexpTokenizer
            from nltk.stem import WordNetLemmatizer
            from nltk.corpus import wordnet
            from nltk.tag import pos_tag
            
            self.nltk_loaded = True

    def _setup_tokenization(self) -> None:
        """Set up tokenization functions based on NLTK availability."""
        if self.using_fallback:
            self.sent_tokenize = TokenizationFallback.sent_tokenize
            self.word_tokenize = TokenizationFallback.word_tokenize
            self.stop_words = TokenizationFallback.get_basic_stopwords()
            self.pos_tag = TokenizationFallback.pos_tag
        else:
            try:
                self.sent_tokenize = nltk_sent_tokenize
                self.word_tokenize = nltk_word_tokenize
                self.stop_words = set(stopwords.words('english'))
                self.pos_tag = pos_tag
            except Exception as e:
                logger.warning(f"Error setting up NLTK functions: {str(e)}")
                logger.info("Falling back to basic tokenization")
                self.using_fallback = True
                self._setup_tokenization()

        self.tokenizer = RegexpTokenizer(r'\w+') if not self.using_fallback else None
        try:
            self.lemmatizer = WordNetLemmatizer() if not self.using_fallback else None
        except Exception as e:
            logger.warning(f"Could not initialize lemmatizer: {str(e)}")
            self.lemmatizer = None

    def _preprocess_sentence(self, text: str) -> str:
        """Preprocess text for analysis."""
        if self.mode != 'simulated_nlp':
            return text
            
        # Only import nltk if we're in simulated mode
        if not self.nltk_loaded:
            self._load_nltk()
            
        tokens = self.word_tokenize(text)
        tagged = self.pos_tag(tokens)
        return ' '.join(word for word, _ in tagged)

    @staticmethod
    def _get_wordnet_pos(treebank_tag: str) -> str:
        """Convert treebank POS tags to WordNet POS tags."""
        tag = treebank_tag[0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }
        return tag_dict.get(tag)

    def _extract_requirements(self, text: str) -> List[str]:
        """Extract requirements from text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            List[str]: List of requirement strings
        """
        requirements = []
        sentences = self.sent_tokenize(text)
        
        requirement_indicators = ['must', 'shall', 'required', 'needs to', 'should']
        for sentence in sentences:
            # Ensure we're working with the string content
            sentence_text = sentence if isinstance(sentence, str) else str(sentence)
            sentence_text = sentence_text.strip().lower()
            
            if any(indicator in sentence_text for indicator in requirement_indicators):
                requirements.append(sentence_text)
        
        return requirements

    def _generate_hierarchical_summary(self, sentences: List[str], preprocessed: List[List[Tuple[str, str]]]) -> str:
        """Generate a hierarchical summary focusing on key points."""
        # Score sentences based on multiple factors
        sentence_scores = defaultdict(float)
        
        for idx, (sentence, processed) in enumerate(zip(sentences, preprocessed)):
            # Base score from requirement indicators
            score = sum(self.requirement_indicators.get(word, 0) for word, _ in processed)
            
            # Position score (higher weight for first sentences in section)
            position_score = 1.0 / (idx + 1)
            
            # Length score (prefer medium-length sentences)
            length_score = min(1.0, len(processed) / 20.0)
            
            # Presence of numerical information
            has_numbers = any(tag == 'CD' for _, tag in processed)
            number_score = 1.5 if has_numbers else 1.0
            
            # Combined score with weights
            sentence_scores[sentence] = (score * 0.4 + position_score * 0.3 + 
                                      length_score * 0.2) * number_score

        # Select top sentences maintaining original order
        top_sentences = sorted(
            [(s, score) for s, score in sentence_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        summary_sentences = [s for s, _ in sorted(
            top_sentences,
            key=lambda x: sentences.index(x[0])
        )]
        
        return ' '.join(summary_sentences)

    def _extract_deadlines_with_context(self, sentences: List[str]) -> List[Dict]:
        """Extract deadlines and metrics with contextual information."""
        results = []
        
        for sentence in sentences:
            metrics_found = []
            
            # Check each category of patterns
            for category, patterns in self.metric_patterns.items():
                for pattern, weight in patterns:
                    matches = re.finditer(pattern, sentence, re.IGNORECASE)
                    for match in matches:
                        # Get context around the metric
                        start = max(0, sentence.find(match.group()) - 40)
                        end = min(len(sentence), sentence.find(match.group()) + len(match.group()) + 40)
                        context = sentence[start:end].strip()
                        
                        metrics_found.append({
                            'metric': match.group(),
                            'category': category,
                            'context': self._clean_sentence(context),
                            'importance': weight
                        })
            
            if metrics_found:
                # Sort by importance and remove duplicates
                unique_metrics = {m['metric']: m for m in metrics_found}
                results.extend(unique_metrics.values())
        
        # Sort by importance and return
        return sorted(results, key=lambda x: x['importance'], reverse=True)

    @staticmethod
    def _clean_sentence(sentence: str) -> str:
        """Clean and normalize a sentence."""
        # Remove multiple spaces
        sentence = re.sub(r'\s+', ' ', sentence)
        # Remove leading/trailing whitespace
        sentence = sentence.strip()
        # Ensure the sentence ends with proper punctuation
        if not sentence.endswith(('.', '!', '?')):
            sentence += '.'
        return sentence 

    def _extract_metrics(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract metrics and deadlines from text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            List[Tuple[str, str, str]]: List of (value, type, context) tuples
        """
        metrics = []
        sentences = text.split('.')
        
        # Define metric patterns
        metric_indicators = {
            '%': 'percentage',
            'percent': 'percentage',
            'milliseconds': 'time',
            'ms': 'time',
            'seconds': 'time',
            'minutes': 'time',
            'hours': 'time',
            'days': 'time',
            'weeks': 'time',
            'months': 'time',
            'years': 'time',
            'quarter': 'time',
            '$': 'currency',
            'dollars': 'currency',
            'budget': 'currency',
            'cost': 'currency',
            'price': 'currency',
            'units': 'units',
            'transactions': 'transactions'
        }
        
        for sentence in sentences:
            sentence = sentence.strip().lower()
            if not sentence:
                continue
                
            words = sentence.split()
            
            for i, word in enumerate(words):
                # Handle percentage patterns (e.g., "99.9%")
                if '%' in word:
                    value = word.strip('%').strip()
                    metrics.append((value, 'percentage', sentence))
                    continue
                
                # Handle currency patterns (e.g., "$500", "$1,500.50")
                if word.startswith('$'):
                    value = word[1:].replace(',', '')
                    if value.replace('.', '').isdigit():
                        # Keep decimal places for currency
                        metrics.append((value, 'currency', sentence))
                    continue
                
                # Handle numeric values followed by units
                if word.replace(',', '').replace('.', '').isdigit():
                    value = word.replace(',', '')
                    # Look ahead for unit indicators
                    found_unit = False
                    for j in range(i + 1, min(i + 4, len(words))):  # Look up to 3 words ahead
                        if words[j] in metric_indicators:
                            metrics.append((value, metric_indicators[words[j]], sentence))
                            found_unit = True
                            break
                        # Special case for currency when $ is missing but "budget/cost" is present
                        elif words[j] in ['budget', 'cost', 'price']:
                            metrics.append((value, 'currency', sentence))
                            found_unit = True
                            break
                    if found_unit:
                        continue
                
                # Handle "minimum/maximum of X" patterns
                if word in ['minimum', 'maximum'] and i + 2 < len(words) and words[i + 1] == 'of':
                    next_word = words[i + 2]
                    if next_word.replace(',', '').replace('.', '').isdigit():
                        value = next_word.replace(',', '')
                        # Look for unit after the number
                        if i + 3 < len(words) and words[i + 3] in metric_indicators:
                            metrics.append((value, metric_indicators[words[i + 3]], sentence))
        
        return metrics

    def _download_nltk_package(self, package: str, download_dir: str) -> bool:
        """
        Download a specific NLTK package with retries.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if package in self.downloaded_packages:
            return True

        for attempt in range(TextAnalyzer.MAX_RETRIES):
            try:
                # First try to find existing data
                try:
                    nltk.data.find(f'tokenizers/{package}' if package == 'punkt' 
                                 else f'corpora/{package}')
                    logger.info(f"Found NLTK data: {package}")
                    self.downloaded_packages.add(package)
                    return True
                except LookupError:
                    pass

                # If not found, try to download
                logger.info(f"Downloading NLTK data: {package}")
                nltk.download(package, download_dir=download_dir, quiet=True, 
                            raise_on_error=True)
                logger.info(f"Successfully downloaded {package}")
                self.downloaded_packages.add(package)
                return True

            except Exception as e:
                if attempt < TextAnalyzer.MAX_RETRIES - 1:
                    logger.info(f"Retry {attempt + 1} for {package}")
                    time.sleep(TextAnalyzer.RETRY_DELAY)
                else:
                    logger.warning(f"Failed to download {package} after {TextAnalyzer.MAX_RETRIES} "
                                 f"attempts: {str(e)}")
                    return False

    def _ensure_nltk_data(self) -> None:
        """Ensure NLTK data is available."""
        import nltk
        import ssl
        import os
        
        for attempt in range(TextAnalyzer.MAX_RETRIES):
            try:
                # Configure SSL context
                try:
                    _create_unverified_https_context = ssl._create_unverified_context
                except AttributeError:
                    pass
                else:
                    ssl._create_default_https_context = _create_unverified_https_context

                # Create and set NLTK data directory
                nltk_data_dir = os.path.expanduser('~/nltk_data')
                os.makedirs(nltk_data_dir, exist_ok=True)
                
                # Download required packages
                for package in NLTK_PACKAGES:
                    if not self._download_nltk_package(package, nltk_data_dir):
                        logger.warning(f"Failed to download {package}")
                        
                return  # Moved outside the package loop
                        
            except Exception as e:
                if attempt == TextAnalyzer.MAX_RETRIES - 1:
                    logger.warning(f"Failed to download after {TextAnalyzer.MAX_RETRIES} attempts: {str(e)}")
                else:
                    logger.info(f"Retry {attempt + 1}")
                    time.sleep(TextAnalyzer.RETRY_DELAY)