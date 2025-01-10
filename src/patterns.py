"""Common patterns and constants for text analysis."""

# Requirement indicators with weights
REQUIREMENT_INDICATORS = {
    'must': 3.0,
    'shall': 3.0,
    'required': 2.5,
    'mandatory': 2.5,
    'will': 2.0,
    'should': 1.5,
    'maintain': 1.5,
    'implement': 1.5,
    'ensure': 1.5,
    'comply': 2.0,
    'responsible': 1.0,
    'necessary': 1.0,
    'obligation': 2.0,
    'requirement': 2.0
}

# Metric patterns with weights
METRIC_PATTERNS = {
    'time': [
        (r'\d+\s*(?:day|month|year|hour|week)s?', 2.0),
        (r'(?:daily|weekly|monthly|quarterly|annually)', 1.5),
        (r'(?:immediate|immediately)', 2.0)
    ],
    'percentage': [
        (r'\d+(?:\.\d+)?%', 1.5),
        (r'\d+(?:\.\d+)?\s*percent', 1.5)
    ],
    'monetary': [
        (r'€\d+(?:\.\d+)?(?:\s*million|\s*billion)?', 2.0),
        (r'\$\d+(?:\.\d+)?(?:\s*million|\s*billion)?', 2.0)
    ],
    'quantity': [
        (r'\d+(?:\.\d+)?\s*(?:million|billion|thousand)', 1.5),
        (r'\d+\s*records?', 1.0)
    ]
}

# NLTK configuration
NLTK_PACKAGES = [
    'punkt',
    'stopwords',
    'averaged_perceptron_tagger',
    'wordnet'
] 