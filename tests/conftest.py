"""Test configuration and fixtures."""

import os
import sys
import pytest
import asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import Mock, AsyncMock

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import OpenAIConfig, AppConfig, AnalysisMode
from src.llm.openai_client import OpenAIClient
from src.text_analyzer import TextAnalyzer

@pytest.fixture
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for each test case."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def mock_openai_client() -> AsyncGenerator[OpenAIClient, None]:
    """Create a mock OpenAI client."""
    config = OpenAIConfig(
        api_key="test_key",
        model="test-model",
        temperature=0.5,
        max_tokens=100
    )
    
    client = OpenAIClient(config)
    client._client = AsyncMock()
    client._client.chat.completions.create = AsyncMock(return_value=Mock(
        choices=[Mock(message=Mock(content='{"test": "response"}'))]
    ))
    
    await client.setup()
    yield client
    await client.cleanup()

@pytest.fixture
def app_config() -> AppConfig:
    """Create a test application configuration."""
    return AppConfig(
        input_file="test_input.txt",
        output_file="test_output.json",
        analysis_mode=AnalysisMode.SIMULATED,
        log_level="DEBUG"
    )

@pytest.fixture
def sample_text() -> str:
    """Provide sample text for testing."""
    return """
    The system must maintain 99.9% uptime.
    Users shall change passwords every 90 days.
    All critical alerts must be responded to within 5 minutes.
    """

@pytest.fixture
def analyzer_sim() -> TextAnalyzer:
    """Create a simulated analyzer instance."""
    return TextAnalyzer(mode='simulated_nlp') 