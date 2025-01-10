"""Constants used throughout the application."""

from typing import Dict, List, Set
from .patterns import REQUIREMENT_INDICATORS, METRIC_PATTERNS, NLTK_PACKAGES

# Analysis Configuration
DEFAULT_MAX_TOKENS = 1000
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MODEL = "gpt-4"

# Retry Configuration
MAX_RETRIES = 3
RETRY_DELAY = 1
DOWNLOAD_TIMEOUT = 30

# Logging Configuration
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_LEVEL = "INFO"

# File Configuration
DEFAULT_INPUT_FILE = "regulations.txt"
DEFAULT_OUTPUT_FILE = "extracted_requirements.json"
DEFAULT_ENCODING = "utf-8" 