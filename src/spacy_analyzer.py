"""Text analysis module implementing advanced NLP using spaCy."""

import re
from typing import Dict, List, Set, Tuple
from collections import defaultdict

import spacy
from spacy.tokens import Doc, Span, Token

class SpacyAnalyzer:
    """Implements advanced NLP-based text analysis using spaCy."""

    def __init__(self):
        """Initialize the spaCy analyzer."""
        # Load English language model with all components
        self.nlp = spacy.load('en_core_web_lg')
        
        # Enhanced requirement indicators with weights
        self.requirement_indicators = {
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
        
        # Metric patterns (same as before)
        self.metric_patterns = {
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

    def analyze_text(self, text: str) -> Dict:
        """
        Analyze text using spaCy's advanced NLP capabilities.

        Args:
            text (str): Text to analyze

        Returns:
            Dict: Analysis results including summary and requirements
        """
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract requirements with confidence scores
        requirements_with_scores = self._extract_requirements(doc)
        
        # Generate summary using dependency parsing and entity recognition
        summary = self._generate_summary(doc)
        
        # Extract deadlines and metrics with context
        deadlines = self._extract_deadlines_with_context(doc)

        return {
            "summary": summary,
            "key_requirements": [req for req, _ in requirements_with_scores[:5]],
            "deadlines_and_metrics": deadlines,
            "metadata": {
                "confidence_scores": {
                    "requirements": [score for _, score in requirements_with_scores[:5]],
                    "requirement_count": len(requirements_with_scores),
                    "metric_count": len(deadlines),
                    "entities": self._extract_entities(doc)
                }
            }
        }

    def _extract_requirements(self, doc: Doc) -> List[Tuple[str, float]]:
        """Extract requirements using dependency parsing."""
        requirements = []
        
        for sent in doc.sents:
            score = 0.0
            
            # Score based on requirement indicators
            for token in sent:
                if token.lemma_.lower() in self.requirement_indicators:
                    score += self.requirement_indicators[token.lemma_.lower()]
            
            # Boost score based on linguistic features
            if any(token.dep_ == 'ROOT' and token.pos_ == 'VERB' for token in sent):
                score *= 1.2  # Boost for action-centered sentences
            
            if any(token.dep_ == 'aux' and token.tag_ == 'MD' for token in sent):
                score *= 1.2  # Boost for modal verbs
                
            # Boost for proper grammatical structure
            if self._has_subject_verb_structure(sent):
                score *= 1.1
                
            if score > 1.0:
                requirements.append((sent.text, score))
        
        return sorted(requirements, key=lambda x: x[1], reverse=True)

    def _generate_summary(self, doc: Doc) -> str:
        """Generate summary using dependency parsing and entity recognition."""
        sentence_scores = defaultdict(float)
        
        for sent in doc.sents:
            # Base score from requirement indicators
            score = sum(self.requirement_indicators.get(token.lemma_.lower(), 0)
                       for token in sent)
            
            # Boost for named entities
            score += len(sent.ents) * 0.5
            
            # Boost for proper syntactic structure
            if self._has_subject_verb_structure(sent):
                score *= 1.2
            
            # Boost for sentences with numbers
            if any(token.like_num for token in sent):
                score *= 1.1
                
            sentence_scores[sent.text] = score

        # Select and order top sentences
        top_sentences = sorted(
            sentence_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        # Maintain original order
        summary_sentences = sorted(
            top_sentences,
            key=lambda x: doc.text.find(x[0])
        )
        
        return ' '.join(sent for sent, _ in summary_sentences)

    def _extract_deadlines_with_context(self, doc: Doc) -> List[Dict]:
        """Extract deadlines using both regex and entity recognition."""
        results = []
        
        for sent in doc.sents:
            metrics_found = []
            
            # Use spaCy's entity recognition for dates and numbers
            for ent in sent.ents:
                if ent.label_ in ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY']:
                    metrics_found.append({
                        'metric': ent.text,
                        'category': ent.label_.lower(),
                        'context': sent.text,
                        'importance': 2.0
                    })
            
            # Also use regex patterns for additional coverage
            for category, patterns in self.metric_patterns.items():
                for pattern, weight in patterns:
                    matches = re.finditer(pattern, sent.text, re.IGNORECASE)
                    for match in matches:
                        metrics_found.append({
                            'metric': match.group(),
                            'category': category,
                            'context': sent.text,
                            'importance': weight
                        })
        
            if metrics_found:
                # Remove duplicates keeping highest importance
                unique_metrics = {}
                for metric in metrics_found:
                    key = (metric['metric'], metric['category'])
                    if key not in unique_metrics or metric['importance'] > unique_metrics[key]['importance']:
                        unique_metrics[key] = metric
                results.extend(unique_metrics.values())
        
        return sorted(results, key=lambda x: x['importance'], reverse=True)

    @staticmethod
    def _has_subject_verb_structure(span: Span) -> bool:
        """Check if span has proper subject-verb structure."""
        has_subject = any(token.dep_ in {'nsubj', 'nsubjpass'} for token in span)
        has_verb = any(token.pos_ == 'VERB' for token in span)
        return has_subject and has_verb

    @staticmethod
    def _extract_entities(doc: Doc) -> Dict[str, List[str]]:
        """Extract named entities by category."""
        entities = defaultdict(list)
        for ent in doc.ents:
            entities[ent.label_.lower()].append(ent.text)
        return dict(entities) 