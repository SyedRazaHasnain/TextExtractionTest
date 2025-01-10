"""Configuration management for the regulatory text processor."""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
from functools import lru_cache

from dotenv import load_dotenv

class AnalysisMode(str, Enum):
    """Supported analysis modes."""
    SIMULATED = "simulated"
    OPENAI = "openai"

    def __str__(self) -> str:
        """String representation."""
        return self.value

@dataclass(frozen=True)
class OpenAIConfig:
    """OpenAI-specific configuration."""
    api_key: str
    model: str = "gpt-4"
    temperature: float = 0.2
    max_tokens: int = 1000

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        if not isinstance(self.temperature, float) or not 0 <= self.temperature <= 1:
            raise ValueError("Temperature must be a float between 0 and 1")
        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise ValueError("Max tokens must be a positive integer")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

@dataclass(frozen=True)
class AppConfig:
    """Application configuration."""
    input_file: str
    output_file: str
    analysis_mode: AnalysisMode = AnalysisMode.SIMULATED
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(levelname)s - %(message)s"
    openai_config: Optional[OpenAIConfig] = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.input_file or not isinstance(self.input_file, str):
            raise ValueError("Input file path is required and must be a string")
        if not self.output_file or not isinstance(self.output_file, str):
            raise ValueError("Output file path is required and must be a string")
        if self.analysis_mode == AnalysisMode.OPENAI and not self.openai_config:
            raise ValueError("OpenAI configuration required for OpenAI mode")

class ConfigManager:
    """Manages application configuration with environment variable support."""
    
    def __init__(self) -> None:
        """Initialize configuration manager."""
        load_dotenv()
        self._validate_environment()
        
    @staticmethod
    def _validate_environment() -> None:
        """Validate required environment variables."""
        required_vars = {
            'LOG_LEVEL': ['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'],
        }
        
        for var, valid_values in required_vars.items():
            value = os.getenv(var)
            if value and value not in valid_values:
                raise ValueError(f"{var} must be one of {valid_values}")

    @lru_cache
    def get_openai_config(self) -> Optional[OpenAIConfig]:
        """Get OpenAI configuration if available."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return None
            
        return OpenAIConfig(
            api_key=api_key,
            model=os.getenv('OPENAI_MODEL', 'gpt-4'),
            temperature=float(os.getenv('OPENAI_TEMPERATURE', '0.2')),
            max_tokens=int(os.getenv('OPENAI_MAX_TOKENS', '1000'))
        )

    def get_app_config(
        self, 
        input_file: str, 
        output_file: str, 
        analysis_mode: Optional[str] = None
    ) -> AppConfig:
        """
        Get application configuration.
        
        Args:
            input_file: Path to input file
            output_file: Path to output file
            analysis_mode: Analysis mode ('simulated' or 'openai')
            
        Returns:
            AppConfig instance
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Determine analysis mode
        mode = AnalysisMode.SIMULATED
        openai_config = None
        
        if analysis_mode == 'openai':
            openai_config = self.get_openai_config()
            if not openai_config:
                raise ValueError("OpenAI configuration required but not available")
            mode = AnalysisMode.OPENAI
        
        return AppConfig(
            input_file=input_file,
            output_file=output_file,
            analysis_mode=mode,
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            log_format=os.getenv('LOG_FORMAT', '%(asctime)s - %(levelname)s - %(message)s'),
            openai_config=openai_config
        ) 