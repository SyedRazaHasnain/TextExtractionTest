"""Base class for LLM clients.

This module defines the abstract base interface for all LLM clients.
Each concrete implementation must provide methods for text analysis,
requirement extraction, and metric detection.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TypedDict, List, Protocol
from datetime import datetime

class Message(TypedDict):
    """Structure for LLM messages."""
    role: str
    content: str

class CacheEntry(TypedDict):
    """Structure for cache entries."""
    result: Any
    timestamp: datetime
    ttl: int

class HTTPClientProtocol(Protocol):
    """Protocol for HTTP clients."""
    async def close(self) -> None: ...
    async def __aenter__(self) -> 'HTTPClientProtocol': ...
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None: ...

class Requirement(TypedDict):
    """Structure for requirement data."""
    text: str
    priority: str
    confidence: float
    category: str

class Metric(TypedDict):
    """Structure for metric data."""
    type: str
    value: str
    context: str
    timeframe: Optional[str]

class AnalysisResult(TypedDict):
    """Structure for analysis results."""
    summary: str
    key_requirements: List[Requirement]
    deadlines_and_metrics: List[Metric]
    metadata: Dict[str, Any]

class BaseLLMClient(ABC):
    """Abstract base class for LLM clients.
    
    This class defines the interface that all LLM clients must implement.
    It provides type-safe methods for text analysis and information extraction.
    
    Implementation Notes:
        - All methods are asynchronous to support different LLM APIs
        - Error handling should be implemented by concrete classes
        - Results should be properly typed using the defined TypedDict classes
        - Implementations should handle rate limiting and caching
    """
    
    @abstractmethod
    async def setup(self) -> None:
        """Initialize client resources."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup client resources."""
        pass
    
    @abstractmethod
    async def analyze_text(self, text: str, *, use_cache: bool = True) -> AnalysisResult:
        """Analyze text using the LLM.
        
        Args:
            text: The text to analyze.
            use_cache: Whether to use cached results if available.
            
        Returns:
            AnalysisResult containing summary, requirements, and metrics.
            
        Raises:
            ValueError: If text is empty or invalid.
            ConnectionError: If LLM service is unavailable.
            RateLimitError: If service rate limit is exceeded.
        """
        pass
    
    @abstractmethod
    async def analyze_requirements(self, 
                                 text: str, 
                                 *, 
                                 use_cache: bool = True) -> List[Requirement]:
        """Extract requirements from text.
        
        Args:
            text: The text to analyze.
            use_cache: Whether to use cached results if available.
            
        Returns:
            List of requirements with metadata.
            
        Raises:
            ValueError: If text is empty or invalid.
            ConnectionError: If LLM service is unavailable.
            RateLimitError: If service rate limit is exceeded.
        """
        pass
    
    @abstractmethod
    async def analyze_metrics(self, 
                            text: str, 
                            *, 
                            use_cache: bool = True) -> List[Metric]:
        """Extract metrics from text.
        
        Args:
            text: The text to analyze.
            use_cache: Whether to use cached results if available.
            
        Returns:
            List of metrics with metadata.
            
        Raises:
            ValueError: If text is empty or invalid.
            ConnectionError: If LLM service is unavailable.
            RateLimitError: If service rate limit is exceeded.
        """
        pass

    @abstractmethod
    async def __aenter__(self) -> 'BaseLLMClient':
        """Async context manager entry."""
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        pass 