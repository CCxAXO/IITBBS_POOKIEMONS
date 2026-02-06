"""
Task 2: Causal Analysis and Explanation Generation
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Import ConversationTranscript from task1
try:
    from task1_retrieval import ConversationTranscript
except ImportError:
    # If running as module
    from .task1_retrieval import ConversationTranscript

logger = logging.getLogger(__name__)


@dataclass
class CausalExplanation:
    """Output structure for causal analysis"""
    query: str
    primary_cause: str
    supporting_factors: List[str]
    evidence_spans: List[Tuple[int, str]]
    confidence: float
    relevant_transcript_ids: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "query": self.query,
            "primary_cause": self.primary_cause,
            "supporting_factors": self.supporting_factors,
            "evidence_spans": [{"turn_id": t, "text": s} for t, s in self.evidence_spans],
            "confidence": self.confidence,
            "relevant_transcript_ids": self.relevant_transcript_ids,
            "timestamp": self.timestamp
        }


class CausalAnalyzer:
    """Pattern-based causal analyzer for customer service conversations"""
    
    def __init__(self):
        """Initialize the analyzer"""
        self.history: List[Dict] = []
        logger.info("CausalAnalyzer initialized")
    
    def analyze(
        self,
        query: str,
        transcripts: List[ConversationTranscript],
        include_history: bool = True
    ) -> CausalExplanation:
        """Analyze transcripts to generate causal explanation"""
        if not transcripts:
            return self._empty_explanation(query)
        
        # Determine outcome type
        outcome = transcripts[0].outcome
        
        # Generate analysis
        primary_cause = self._generate_primary_cause(outcome, transcripts)
        factors = self._extract_supporting_factors(outcome, transcripts)
        evidence = self._extract_evidence(query, transcripts)
        confidence = self._calculate_confidence(transcripts, factors)
        
        explanation = CausalExplanation(
            query=query,
            primary_cause=primary_cause,
            supporting_factors=factors,
            evidence_spans=evidence,
            confidence=confidence,
            relevant_transcript_ids=[t.transcript_id for t in transcripts]
        )
        
        # Update history
        if include_history:
            self.history.append({
                'query': query,
                'explanation': primary_cause,
                'timestamp': explanation.timestamp
            })
        
        return explanation
    
    def _empty_explanation(self, query: str) -> CausalExplanation:
        """Create empty explanation when no transcripts found"""
        return CausalExplanation(
            query=query,
            primary_cause="No relevant conversations found for analysis",
            supporting_factors=[],
            evidence_spans=[],
            confidence=0.3,
            relevant_transcript_ids=[]
        )
    
    def _generate_primary_cause(
        self, 
        outcome: str, 
        transcripts: List[ConversationTranscript]
    ) -> str:
        """Generate the primary causal explanation"""
        text = " ".join([t.get_full_text() for t in transcripts]).lower()
        reason = transcripts[0].metadata.get('reason_for_call', '')
        
        if 'escalation' in outcome:
            causes = []
            
            if 'three weeks' in text or 'weeks' in text:
                causes.append("prolonged issue duration (multiple weeks)")
            if 'multiple' in text or 'several' in text or 'repeated' in text:
                causes.append("multiple failed resolution attempts")
            if 'frustrated' in text or 'frustration' in text:
                causes.append("accumulated customer frustration")
            if 'nobody' in text or 'no one' in text:
                causes.append("previous agents unable to resolve")
            
            # Extract error codes
            error_match = re.search(r'error\s*(?:code\s*)?(\d+)', text)
            if error_match:
                causes.append(f"unresolved error code {error_match.group(1)}")
            
            if causes:
                return "Customer escalated due to: " + "; ".join(causes)
            return "Customer requested escalation to supervisor"
        
        elif 'fraud' in outcome:
            causes = []
            
            # Extract amount
            amount_match = re.search(r'\$[\d,]+\.?\d*', text)
            if amount_match:
                causes.append(f"unauthorized charge of {amount_match.group(0)}")
            
            # Location analysis
            if 'new york' in text.lower():
                causes.append("transaction in New York (customer never visited)")
            elif 'different location' in text:
                causes.append("transaction from different location")
            
            if 'fraud alert' in text:
                causes.append("automatic fraud detection triggered")
            
            if 'blocked' in text or 'block' in text:
                causes.append("card blocked for security")
            
            if causes:
                return "Fraud detected: " + "; ".join(causes)
            return "Fraudulent transaction identified and addressed"
        
        elif 'delivery' in outcome:
            causes = []
            
            if 'shows delivered' in text or 'marked delivered' in text:
                causes.append("package marked delivered in tracking")
            if 'never received' in text or 'not there' in text:
                causes.append("customer did not receive package")
            if 'camera' in text or 'neighbor' in text:
                causes.append("customer verified non-delivery")
            if 'wrong address' in text:
                causes.append("possible wrong address delivery")
            
            if causes:
                return "Delivery issue: " + "; ".join(causes)
            return "Package delivery discrepancy reported"
        
        # Default case
        if reason:
            return f"Issue identified: {reason}"
        return f"Issue type: {outcome}"
    
    def _extract_supporting_factors(
        self, 
        outcome: str, 
        transcripts: List[ConversationTranscript]
    ) -> List[str]:
        """Extract supporting factors from transcripts"""
        text = " ".join([t.get_full_text() for t in transcripts]).lower()
        factors = []
        
        # Time-based factors
        if 'three weeks' in text or 'weeks' in text:
            factors.append("Extended duration: issue persisted for weeks")
        if 'yesterday' in text or 'today' in text:
            factors.append("Recent occurrence: within last 24 hours")
        
        # Repetition factors
        if 'multiple' in text or 'several' in text:
            factors.append("Multiple occurrences or attempts documented")
        if 'repeated' in text or 'again' in text:
            factors.append("Repeated failures noted")
        
        # Emotional factors
        if 'frustrated' in text or 'frustration' in text:
            factors.append("Customer expressed frustration")
        if 'upset' in text or 'angry' in text:
            factors.append("Customer emotional distress")
        
        # Action factors
        if 'checked' in text or 'verified' in text:
            factors.append("Customer performed verification steps")
        if 'supervisor' in text or 'manager' in text:
            factors.append("Escalation to supervisor requested")
        
        # Response factors
        if 'expedited' in text or 'immediately' in text:
            factors.append("Agent provided swift response")
        if 'blocked' in text or 'reversed' in text:
            factors.append("Immediate security action taken")
        
        return factors[:6]  # Return top 6 factors
    
    def _extract_evidence(
        self, 
        query: str, 
        transcripts: List[ConversationTranscript]
    ) -> List[Tuple[int, str]]:
        """Extract relevant evidence spans from transcripts"""
        evidence = []
        query_terms = set(w.lower() for w in query.split() if len(w) > 3)
        
        # Key indicators to look for
        key_indicators = [
            'escalate', 'supervisor', 'fraud', 'unauthorized', 
            'delivered', 'error', 'frustrated', 'weeks', 'multiple'
        ]
        
        for transcript in transcripts:
            for turn in transcript.turns:
                text_lower = turn.text.lower()
                
                # Check for query term matches
                query_matches = sum(1 for t in query_terms if t in text_lower)
                
                # Check for key indicators
                indicator_matches = sum(1 for k in key_indicators if k in text_lower)
                
                if (query_matches > 0 or indicator_matches > 0) and len(evidence) < 4:
                    display = turn.text[:120] + "..." if len(turn.text) > 120 else turn.text
                    evidence.append((turn.turn_id, f"[{turn.speaker}] {display}"))
        
        return evidence
    
    def _calculate_confidence(
        self, 
        transcripts: List[ConversationTranscript], 
        factors: List[str]
    ) -> float:
        """Calculate confidence score for the analysis"""
        confidence = 0.6  # Base confidence
        
        # Transcript availability
        confidence += min(len(transcripts) * 0.05, 0.15)
        
        # Factor count
        confidence += min(len(factors) * 0.03, 0.15)
        
        # Metadata availability
        if transcripts and transcripts[0].metadata.get('reason_for_call'):
            confidence += 0.1
        
        return min(max(confidence, 0.6), 0.95)
    
    def get_history(self) -> List[Dict]:
        """Get analysis history"""
        return self.history.copy()
    
    def clear_history(self):
        """Clear analysis history"""
        self.history = []