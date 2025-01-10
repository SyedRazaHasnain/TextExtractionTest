"""OpenAI LLM client implementation.

This module implements the OpenAI-specific LLM client with proper error handling,
retry logic, and comprehensive logging.
"""

import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, cast
from functools import wraps
from datetime import datetime, timedelta
from hashlib import sha256

from openai import AsyncOpenAI, APIError, RateLimitError
from aiohttp import ClientSession, TCPConnector
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_log,
    after_log
)

from .base_client import (
    BaseLLMClient, 
    AnalysisResult, 
    Requirement, 
    Metric, 
    Message,
    CacheEntry,
    HTTPClientProtocol
)
from ..config import OpenAIConfig
from ..exceptions import APIError as CustomAPIError, ValidationError

logger = logging.getLogger(__name__)

def validate_input(func):
    """Decorator for input validation."""
    @wraps(func)
    async def wrapper(self, text: str, *args, **kwargs):
        if not isinstance(text, str):
            raise ValidationError("Input must be a string")
        if not text.strip():
            raise ValidationError("Input cannot be empty")
        return await func(self, text, *args, **kwargs)
    return wrapper

class OpenAIClient(BaseLLMClient):
    """Client for interacting with OpenAI's LLM.
    
    This class implements the BaseLLMClient interface using OpenAI's API.
    It includes proper error handling, retry logic, and comprehensive logging.
    
    Attributes:
        config: OpenAI configuration settings
        client: AsyncOpenAI client instance
    """

    def __init__(self, 
                 config: OpenAIConfig, 
                 session: Optional[ClientSession] = None,
                 cache_ttl: int = 3600):
        """Initialize OpenAI client.
        
        Args:
            config: OpenAI configuration settings
            
        Raises:
            ValidationError: If config is invalid
        """
        if not isinstance(config, OpenAIConfig):
            raise ValidationError("config must be an instance of OpenAIConfig")
        if not config.api_key:
            raise ValidationError("OpenAI API key is required")
            
        self.config = config
        self._client: Optional[AsyncOpenAI] = None
        self._session = session
        self._setup_lock = asyncio.Lock()
        self._rate_limit = asyncio.Semaphore(50)  # 50 concurrent requests max
        self._cache: Dict[str, CacheEntry] = {}
        self._cache_ttl = cache_ttl
        
        logger.info("OpenAI client initialized with model: %s", config.model)

    async def __aenter__(self) -> 'OpenAIClient':
        """Async context manager entry."""
        await self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.cleanup()

    async def setup(self) -> None:
        """Setup client connection."""
        async with self._setup_lock:
            if self._client is None:
                if self._session is None:
                    self._session = ClientSession(
                        connector=TCPConnector(
                            limit=10,  # Connection pool size
                            ttl_dns_cache=300  # DNS cache TTL
                        )
                    )
                self._client = AsyncOpenAI(
                    api_key=self.config.api_key,
                    http_client=self._session
                )
                logger.debug("OpenAI client connection established")

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self._client:
            await self._client.close()
            self._client = None
        if self._session:
            await self._session.close()
            self._session = None
        logger.debug("OpenAI client connection closed")

    def _get_cache_key(self, text: str, method: str) -> str:
        """Generate cache key for text and method."""
        return sha256(f"{method}:{text}".encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result if valid."""
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if datetime.now() - entry['timestamp'] < timedelta(seconds=entry['ttl']):
                logger.debug("Cache hit for key: %s", cache_key)
                return entry['result']
            else:
                del self._cache[cache_key]
        return None

    def _cache_result(self, cache_key: str, result: Any) -> None:
        """Cache result with timestamp."""
        self._cache[cache_key] = CacheEntry(
            result=result,
            timestamp=datetime.now(),
            ttl=self._cache_ttl
        )

    @validate_input
    @retry(
        retry=retry_if_exception_type((APIError, RateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        before=before_log(logger, logging.DEBUG),
        after=after_log(logger, logging.DEBUG)
    )
    async def analyze_text(self, text: str, *, use_cache: bool = True) -> AnalysisResult:
        """Analyze text using OpenAI's GPT model.
        
        Args:
            text: The text to analyze
            
        Returns:
            AnalysisResult containing analysis details
            
        Raises:
            ValidationError: If input is invalid
            CustomAPIError: If API call fails
            ConnectionError: If service is unavailable
        """
        if use_cache:
            cache_key = self._get_cache_key(text, 'analyze_text')
            cached = self._get_cached_result(cache_key)
            if cached:
                return cast(AnalysisResult, cached)

        try:
            async with self._rate_limit:
                response = await self._get_completion([
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": self._get_analysis_prompt(text)}
                ])
                
                result = self._parse_response(response)
                if use_cache:
                    self._cache_result(cache_key, result)
                    
                logger.info("Successfully analyzed text of length %d", len(text))
                return cast(AnalysisResult, result)
                
        except APIError as e:
            logger.error("OpenAI API error: %s", str(e), exc_info=True)
            raise CustomAPIError(f"API call failed: {str(e)}") from e
        except RateLimitError as e:
            logger.warning("Rate limit hit: %s", str(e))
            raise CustomAPIError("Rate limit exceeded, please retry later") from e
        except json.JSONDecodeError as e:
            logger.error("Response parsing failed: %s", str(e))
            raise ValidationError("Invalid API response format") from e
        except Exception as e:
            logger.critical("Unexpected error: %s", str(e), exc_info=True)
            raise

    @validate_input
    async def analyze_requirements(self, text: str) -> List[Requirement]:
        """Extract requirements using OpenAI.
        
        Args:
            text: The text to analyze
            
        Returns:
            List of requirements with metadata
            
        Raises:
            ValidationError: If input is invalid
            CustomAPIError: If API call fails
        """
        try:
            response = await self._get_completion([
                {"role": "system", "content": "Extract requirements from the text."},
                {"role": "user", "content": text}
            ])
            
            result = self._parse_response(response)
            return cast(List[Requirement], result.get('requirements', []))
            
        except Exception as e:
            logger.error("Error extracting requirements: %s", str(e), exc_info=True)
            raise CustomAPIError(f"Requirement extraction failed: {str(e)}") from e

    @validate_input
    async def analyze_metrics(self, text: str) -> List[Metric]:
        """Extract metrics using OpenAI.
        
        Args:
            text: The text to analyze
            
        Returns:
            List of metrics with metadata
            
        Raises:
            ValidationError: If input is invalid
            CustomAPIError: If API call fails
        """
        try:
            response = await self._get_completion([
                {"role": "system", "content": "Extract metrics from the text."},
                {"role": "user", "content": text}
            ])
            
            result = self._parse_response(response)
            return cast(List[Metric], result.get('metrics', []))
            
        except Exception as e:
            logger.error("Error extracting metrics: %s", str(e), exc_info=True)
            raise CustomAPIError(f"Metric extraction failed: {str(e)}") from e

    async def _get_completion(self, messages: List[Message]) -> str:
        """Get completion from OpenAI with retry logic.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Completion response content
            
        Raises:
            CustomAPIError: If API call fails
        """
        if self._client is None:
            raise RuntimeError("Client not initialized. Use 'async with' or call setup()")
            
        try:
            response = await self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("API call failed: %s", str(e), exc_info=True)
            raise

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse OpenAI response into structured format.
        
        Args:
            response: Raw response string
            
        Returns:
            Parsed response dictionary
            
        Raises:
            CustomAPIError: If parsing fails
        """
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse response as JSON: %s", str(e))
            return {
                "text": response[:500],
                "error": "Failed to parse as JSON",
                "requirements": [],
                "metrics": []
            }

    @staticmethod
    def _get_system_prompt() -> str:
        """Get system prompt for analysis."""
        return """You are an expert regulatory compliance analyst with deep experience in:
1. Interpreting complex regulatory requirements
2. Identifying key compliance obligations
3. Extracting specific deadlines and metrics
4. Prioritizing requirements based on importance

Analyze the provided regulatory text and extract key information in a structured format."""

    @staticmethod
    def _get_analysis_prompt(text: str) -> str:
        """Get analysis prompt template."""
        return f"""Analyze the following regulatory text and provide:

1. A concise summary focusing on key compliance requirements
2. A prioritized list of specific, actionable requirements
3. All deadlines, metrics, and numerical thresholds
4. Risk levels and compliance priorities

Regulatory text:
{text}

Provide the analysis in JSON format with the following structure:
{{
    "summary": "Concise summary of key points",
    "requirements": [
        {{
            "text": "Requirement text",
            "priority": "high|medium|low",
            "risk_level": "critical|high|medium|low"
        }}
    ],
    "metrics": [
        {{
            "type": "deadline|metric|threshold",
            "value": "extracted value",
            "context": "surrounding context"
        }}
    ]
}}""" 