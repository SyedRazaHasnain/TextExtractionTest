# Regulatory Text Analyzer

A production-grade Python application for analyzing regulatory documents using advanced NLP techniques and LLM integration. Built with modern Python practices, type safety, and enterprise-level error handling.

## Technical Overview

### Architecture
- **Clean Architecture** with clear separation of concerns
- **SOLID Principles** adherence
- **Dependency Injection** ready design
- **Abstract Base Classes** for extensibility
- **Async/Await** for optimal performance
- **Type Safety** using TypedDict and strict typing

### Key Technical Features
- **Advanced Error Handling**:
  - Custom exception hierarchy
  - Comprehensive error logging
  - Exception chaining
  - Input validation decorators

- **Performance Optimizations**:
  - Asynchronous API calls
  - Configurable retry logic
  - Rate limiting handling
  - Response streaming support

- **Type Safety**:
  - Full type hinting
  - Runtime type checking
  - TypedDict for structured data
  - Generic type support

- **Professional Logging**:
  - Structured log format
  - Multiple log levels
  - Contextual error information
  - Performance metrics

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd regulatory-text-analyzer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Create .env file with required configuration
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4  # or gpt-3.5-turbo
OPENAI_TEMPERATURE=0.2
OPENAI_MAX_TOKENS=1000
LOG_LEVEL=INFO
```

## Quick Start

### 1. Run in Simulated Mode (No API Key Required)
```bash
# Create a sample regulations file
echo "The system must maintain 99.9% uptime.
Users shall change passwords every 90 days.
All critical alerts must be responded to within 5 minutes." > regulations.txt

# Run the analyzer in simulated mode
python extract_requirements.py --mode simulated

# Check results
cat extracted_requirements.json
```

### 2. Run with OpenAI (API Key Required)
```bash
# Set your OpenAI API key
echo "OPENAI_API_KEY=your_key_here" > .env

# Run the analyzer with OpenAI
python extract_requirements.py --mode openai
```

### 3. Quick Test Suite Run
```bash
# Run core tests (fast)
pytest tests/test_text_analyzer.py -v

# Run full test suite
pytest tests/ -v

# Run with coverage report
pytest --cov=src --cov-report=term-missing tests/
```

### Default Configuration
The application comes with sensible defaults:
- Input file: `regulations.txt`
- Output file: `extracted_requirements.json`
- Mode: `simulated`
- Log level: `INFO`
- OpenAI model: `gpt-4`
- Temperature: `0.2`
- Max tokens: `1000`

### Minimal Python Example
```python
from src.text_analyzer import TextAnalyzer

# Simulated mode (no API key needed)
analyzer = TextAnalyzer(mode='simulated_nlp')
text = "The system must maintain 99.9% uptime."
result = analyzer.analyze_text(text)
print(result['key_requirements'])

# OpenAI mode (requires API key in .env)
from src.config import ConfigManager
from src.llm_client import LLMClient

config = ConfigManager().get_openai_config()
analyzer = LLMClient(config)
result = await analyzer.analyze_text(text)
print(result['key_requirements'])
```

## Usage Examples

### Basic Usage with Type Safety

```python
from src.text_analyzer import TextAnalyzer
from src.llm.base_client import AnalysisResult, Requirement, Metric

async def analyze_document(text: str) -> AnalysisResult:
    analyzer = TextAnalyzer(mode='simulated_nlp')
    return await analyzer.analyze_text(text)

# Example with type checking
result: AnalysisResult = await analyze_document(text)
requirements: List[Requirement] = result['key_requirements']
metrics: List[Metric] = result['deadlines_and_metrics']
```

### Advanced Usage with OpenAI Integration

```python
from src.config import OpenAIConfig
from src.llm.openai_client import OpenAIClient
from src.exceptions import ValidationError, CustomAPIError

async def analyze_with_openai(text: str) -> AnalysisResult:
    try:
        config = OpenAIConfig(
            api_key="your_api_key",
            model="gpt-4",
            temperature=0.2,
            max_tokens=1000
        )
        client = OpenAIClient(config)
        return await client.analyze_text(text)
    except ValidationError as e:
        logger.error("Validation error: %s", str(e))
        raise
    except CustomAPIError as e:
        logger.error("API error: %s", str(e), exc_info=True)
        raise
```

### Error Handling Example

```python
from src.exceptions import TextAnalyzerError

async def safe_analysis(text: str) -> Optional[AnalysisResult]:
    try:
        analyzer = TextAnalyzer()
        return await analyzer.analyze_text(text)
    except ValidationError as e:
        logger.warning("Invalid input: %s", str(e))
        return None
    except CustomAPIError as e:
        logger.error("API error occurred: %s", str(e), exc_info=True)
        raise
    except TextAnalyzerError as e:
        logger.error("Analysis failed: %s", str(e), exc_info=True)
        raise
```

## Development

### Code Quality Tools

```bash
# Type checking
mypy src/

# Style checking
flake8 src/

# Security analysis
bandit -r src/

# Format code
black src/
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test categories
pytest tests/test_text_analyzer.py -k "test_requirement"
```

### Adding New LLM Provider

1. Implement the `BaseLLMClient` interface:
```python
from src.llm.base_client import BaseLLMClient, AnalysisResult

class CustomLLMClient(BaseLLMClient):
    async def analyze_text(self, text: str) -> AnalysisResult:
        # Implementation
        pass

    async def analyze_requirements(self, text: str) -> List[Requirement]:
        # Implementation
        pass

    async def analyze_metrics(self, text: str) -> List[Metric]:
        # Implementation
        pass
```

2. Add proper error handling:
```python
try:
    # API calls
except Exception as e:
    logger.error("Error: %s", str(e), exc_info=True)
    raise CustomAPIError("Operation failed") from e
```

## Best Practices

### Error Handling
- Use custom exception hierarchy
- Always chain exceptions
- Include context in error messages
- Log at appropriate levels

### Type Safety
- Use TypedDict for structured data
- Add type hints to all functions
- Use runtime type checking
- Document type requirements

### Performance
- Use async/await for I/O operations
- Implement proper retry logic
- Handle rate limits gracefully
- Monitor and log performance metrics

### Testing
- Write unit tests for all components
- Include integration tests
- Use property-based testing
- Maintain high coverage

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - See LICENSE file for details 