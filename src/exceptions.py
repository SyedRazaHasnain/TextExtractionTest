"""Custom exceptions for the regulatory text analyzer."""

class TextAnalyzerError(Exception):
    """Base exception for all text analyzer errors."""
    pass

class ConfigurationError(TextAnalyzerError):
    """Raised when there is a configuration error."""
    pass

class APIError(TextAnalyzerError):
    """Raised when there is an error with external API calls."""
    pass

class TextProcessingError(TextAnalyzerError):
    """Raised when there is an error processing text."""
    pass

class ValidationError(TextAnalyzerError):
    """Raised when there is a validation error."""
    pass