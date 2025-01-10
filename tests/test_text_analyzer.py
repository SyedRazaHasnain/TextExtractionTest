import pytest
from src.text_analyzer import TextAnalyzer
from src.exceptions import ValidationError, TextProcessingError

@pytest.fixture
def analyzer_sim():
    """Fixture for simulated NLP analyzer."""
    return TextAnalyzer(mode='simulated_nlp')

class TestTextAnalyzer:
    """Test suite for TextAnalyzer class."""

    def test_hierarchical_summary_generation(self, analyzer_sim):
        """Test the hierarchical summary generation with prioritized content."""
        structured_text = """
        Section 1: Critical Requirements
        The system must maintain 99.9% uptime at all times.
        Regular maintenance is recommended during off-peak hours.
        All critical alerts must be responded to within 5 minutes.

        Section 2: Optional Features
        The UI should be user-friendly.
        Reports can be generated in PDF format.
        
        Section 3: Compliance
        Users must change passwords every 90 days.
        All financial transactions must be logged.
        """
        
        result = analyzer_sim.analyze_text(structured_text)
        summary = result['hierarchical_summary']
        
        # Summary should be a non-empty string
        assert isinstance(summary, str)
        assert len(summary) > 0
        
        # Summary should prioritize critical requirements and metrics
        assert '99.9% uptime' in summary.lower()
        assert '90 days' in summary.lower()
        
        # Summary should include high-priority requirements
        assert any(indicator in summary.lower() 
                  for indicator in ['must', 'required', 'shall'])
        
        # Summary should not be the entire text
        assert len(summary) < len(structured_text)

    def test_requirement_confidence_correlation(self, analyzer_sim):
        """Test if requirement confidence scores correlate with indicator strength."""
        test_requirements = [
            ("The system must maintain high availability.", 1.0),  # Strong - must
            ("Users shall authenticate properly.", 1.0),          # Strong - shall
            ("The system should log all errors.", 0.7),          # Medium - should
            ("It would be nice to have dark mode.", 0.5)         # Weak - no indicator
        ]
        
        for text, expected_min_confidence in test_requirements:
            result = analyzer_sim.analyze_text(text)
            if result['key_requirements']:
                confidence = result['key_requirements'][0]['confidence']
                assert confidence >= expected_min_confidence, \
                    f"Confidence {confidence} should be >= {expected_min_confidence} for '{text}'" 