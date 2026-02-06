"""
Pattern Analyzer Module
Contains trained patterns for causal analysis
"""

import re
from typing import Dict, List, Tuple, Any


class PatternAnalyzer:
    """
    Pattern-based analyzer containing trained patterns for identifying
    causal relationships in customer service conversations.
    """
    
    def __init__(self):
        """Initialize with trained patterns"""
        self.outcome_patterns = self._load_outcome_patterns()
        self.causal_patterns = self._load_causal_patterns()
        self.entity_patterns = self._load_entity_patterns()
    
    def _load_outcome_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for outcome classification"""
        return {
            'escalation': [
                r'speak\s+(?:to|with)\s+(?:a\s+)?supervisor',
                r'escalat(?:e|ed|ing)',
                r'talk\s+to\s+(?:a\s+)?manager',
                r'file\s+(?:a\s+)?complaint',
                r'(?:been|calling)\s+for\s+(?:\d+\s+)?weeks?'
            ],
            'fraud': [
                r'fraud\s+alert',
                r'unauthorized\s+(?:charge|transaction|purchase)',
                r'didn\'?t\s+make\s+(?:this|that)\s+purchase',
                r'never\s+been\s+to',
                r'block(?:ed|ing)?\s+(?:my\s+)?card'
            ],
            'delivery_issue': [
                r'shows?\s+(?:as\s+)?delivered',
                r'never\s+received',
                r'package\s+(?:is\s+)?missing',
                r'not\s+(?:at\s+)?(?:my\s+)?door',
                r'wrong\s+address'
            ],
            'resolution': [
                r'send(?:ing)?\s+(?:a\s+)?replacement',
                r'(?:full\s+)?refund',
                r'expedited\s+(?:shipping|delivery)',
                r'no\s+(?:extra\s+)?charge',
                r'investigation\s+(?:started|initiated)'
            ]
        }
    
    def _load_causal_patterns(self) -> Dict[str, List[Tuple[str, str]]]:
        """Load patterns for causal factor identification"""
        return {
            'temporal': [
                (r'for\s+(\d+)\s+weeks?', 'Duration: {} week(s)'),
                (r'for\s+(\d+)\s+days?', 'Duration: {} day(s)'),
                (r'since\s+(\w+day)', 'Since {}'),
                (r'yesterday', 'Occurred yesterday'),
                (r'this\s+morning', 'Occurred this morning')
            ],
            'repetition': [
                (r'(\d+)\s+(?:times?|attempts?)', '{} previous attempts'),
                (r'multiple\s+(?:times?|calls?)', 'Multiple occurrences'),
                (r'(?:keep|keeps)\s+(?:happening|failing)', 'Recurring issue'),
                (r'again\s+and\s+again', 'Repeated issue')
            ],
            'emotional': [
                (r'(?:very\s+)?frustrated', 'Customer frustration'),
                (r'(?:very\s+)?upset', 'Customer upset'),
                (r'unacceptable', 'Expressed unacceptability'),
                (r'wasted?\s+(?:my\s+)?time', 'Perceived time waste')
            ],
            'technical': [
                (r'error\s+(?:code\s+)?(\d+)', 'Error code {}'),
                (r'(?:app|system)\s+(?:crash|fail)', 'System failure'),
                (r'login\s+fail', 'Authentication issue'),
                (r'can\'?t\s+(?:access|log\s*in)', 'Access problem')
            ]
        }
    
    def _load_entity_patterns(self) -> Dict[str, str]:
        """Load patterns for entity extraction"""
        return {
            'amount': r'\$[\d,]+\.?\d*',
            'order_number': r'(?:order\s*(?:#|number)?\s*)?(\d{7,})',
            'account_number': r'(?:account\s*(?:#|number)?\s*)?(\d{4}[-\s]?\d{4}[-\s]?\d{4})',
            'error_code': r'error\s*(?:code\s*)?(\d+)',
            'date': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            'time_period': r'(\d+)\s*(days?|weeks?|months?)'
        }
    
    def classify_outcome(self, text: str) -> Tuple[str, float]:
        """
        Classify the outcome type of a conversation.
        
        Args:
            text: Conversation text
            
        Returns:
            Tuple of (outcome_type, confidence)
        """
        text_lower = text.lower()
        scores = {}
        
        for outcome, patterns in self.outcome_patterns.items():
            matches = sum(1 for p in patterns if re.search(p, text_lower))
            scores[outcome] = matches / len(patterns)
        
        if not scores:
            return ('unknown', 0.0)
        
        best_outcome = max(scores, key=scores.get)
        return (best_outcome, scores[best_outcome])
    
    def extract_causal_factors(self, text: str) -> List[str]:
        """
        Extract causal factors from conversation text.
        
        Args:
            text: Conversation text
            
        Returns:
            List of identified causal factors
        """
        text_lower = text.lower()
        factors = []
        
        for category, patterns in self.causal_patterns.items():
            for pattern, template in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    if match.groups():
                        factor = template.format(*match.groups())
                    else:
                        factor = template
                    factors.append(f"{category.title()}: {factor}")
        
        return factors
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from conversation text.
        
        Args:
            text: Conversation text
            
        Returns:
            Dictionary of entity types to extracted values
        """
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[entity_type] = matches
        
        return entities
    
    def get_pattern_stats(self) -> Dict[str, int]:
        """Get statistics about loaded patterns"""
        return {
            'outcome_patterns': sum(len(p) for p in self.outcome_patterns.values()),
            'causal_patterns': sum(len(p) for p in self.causal_patterns.values()),
            'entity_patterns': len(self.entity_patterns)
        }