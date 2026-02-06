"""
Task 1: Conversation Retrieval System
"""

import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try importing optional packages
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_EMBEDDINGS = True
    logger.info("Sentence transformers available")
except ImportError:
    HAS_EMBEDDINGS = False
    logger.info("Using keyword-based retrieval (no sentence-transformers)")


@dataclass
class ConversationTurn:
    """Single turn in conversation"""
    turn_id: int
    speaker: str
    text: str
    timestamp: Optional[str] = None


@dataclass
class ConversationTranscript:
    """Complete conversation transcript"""
    transcript_id: str
    domain: str
    outcome: str
    turns: List[ConversationTurn]
    metadata: Dict[str, Any]
    
    def get_full_text(self) -> str:
        """Get concatenated text from all turns"""
        return " ".join([turn.text for turn in self.turns])


class ConversationRetriever:
    """Retrieves relevant conversations based on queries"""
    
    def __init__(self, use_embeddings: bool = True):
        """Initialize the retriever"""
        self.conversations_by_id: Dict[str, ConversationTranscript] = {}
        self.embeddings: Dict[str, Any] = {}
        self.has_embeddings = HAS_EMBEDDINGS and use_embeddings
        self.model = None
        
        if self.has_embeddings:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded embedding model")
            except Exception as e:
                logger.warning(f"Could not load embedding model: {e}")
                self.has_embeddings = False
        
        logger.info(f"ConversationRetriever initialized (embeddings: {self.has_embeddings})")
    
    def load_conversations(self, data: Any) -> int:
        """Load conversations from JSON data"""
        conversations = self._extract_conversations(data)
        
        for idx, conv_data in enumerate(conversations):
            try:
                transcript = self._parse_conversation(conv_data, idx)
                self.conversations_by_id[transcript.transcript_id] = transcript
                
                # Create embeddings if available
                if self.has_embeddings and self.model:
                    try:
                        text = transcript.get_full_text()
                        embedding = self.model.encode(text, convert_to_tensor=True)
                        self.embeddings[transcript.transcript_id] = embedding
                    except:
                        pass
                    
            except Exception as e:
                logger.warning(f"Could not parse conversation {idx}: {e}")
        
        logger.info(f"Loaded {len(self.conversations_by_id)} conversations")
        return len(self.conversations_by_id)
    
    def _extract_conversations(self, data: Any) -> List[Dict]:
        """Extract conversation list from various JSON formats"""
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            if 'transcripts' in data:
                return data['transcripts']
            elif 'conversations' in data:
                return data['conversations']
            elif 'conversation' in data or 'transcript_id' in data:
                return [data]
        return []
    
    def _parse_conversation(self, conv_data: Dict[str, Any], idx: int) -> ConversationTranscript:
        """Parse a single conversation into structured format"""
        # Extract turns
        turns = []
        conversation_data = conv_data.get("conversation", conv_data.get("turns", []))
        
        for i, turn in enumerate(conversation_data):
            if isinstance(turn, dict):
                turns.append(ConversationTurn(
                    turn_id=i,
                    speaker=turn.get("speaker", f"Speaker{i%2+1}"),
                    text=turn.get("text", turn.get("utterance", "")),
                    timestamp=turn.get("timestamp")
                ))
            elif isinstance(turn, str):
                turns.append(ConversationTurn(
                    turn_id=i,
                    speaker=f"Speaker{i%2+1}",
                    text=turn,
                    timestamp=None
                ))
        
        # Determine outcome from intent
        intent = conv_data.get("intent", "")
        outcome = self._parse_intent_to_outcome(intent)
        
        # Build metadata
        metadata = {
            "time_of_interaction": conv_data.get("time_of_interaction"),
            "intent": intent,
            "reason_for_call": conv_data.get("reason_for_call", "")
        }
        
        return ConversationTranscript(
            transcript_id=conv_data.get("transcript_id", f"conv_{idx}"),
            domain=conv_data.get("domain", "unknown"),
            outcome=outcome,
            turns=turns,
            metadata=metadata
        )
    
    def _parse_intent_to_outcome(self, intent: str) -> str:
        """Map intent string to outcome category"""
        intent_lower = intent.lower()
        
        if 'escalation' in intent_lower:
            return 'escalation'
        elif 'fraud' in intent_lower:
            return 'fraud_resolved'
        elif 'delivery' in intent_lower:
            return 'delivery_investigation'
        elif 'resolved' in intent_lower or 'compensation' in intent_lower:
            return 'resolved_with_compensation'
        else:
            return intent if intent else 'general_inquiry'
    
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve relevant conversation IDs for a query"""
        if self.has_embeddings and self.embeddings:
            return self._retrieve_semantic(query, top_k)
        else:
            return self._retrieve_keyword(query, top_k)
    
    def _retrieve_semantic(self, query: str, top_k: int) -> List[str]:
        """Semantic search using embeddings"""
        try:
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            scores = {}
            
            for tid, embedding in self.embeddings.items():
                similarity = util.pytorch_cos_sim(query_embedding, embedding)[0][0].item()
                scores[tid] = similarity
            
            sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return [tid for tid, _ in sorted_ids[:top_k]]
            
        except Exception as e:
            logger.warning(f"Semantic retrieval failed: {e}")
            return self._retrieve_keyword(query, top_k)
    
    def _retrieve_keyword(self, query: str, top_k: int) -> List[str]:
        """Keyword-based retrieval with domain scoring"""
        query_lower = query.lower()
        scores = {}
        
        for tid, transcript in self.conversations_by_id.items():
            score = 0
            
            # Get all searchable text
            all_text = transcript.get_full_text().lower()
            reason = transcript.metadata.get('reason_for_call', '').lower()
            all_content = all_text + " " + reason
            
            # Word matching
            query_words = [w for w in query_lower.split() if len(w) > 2]
            if query_words:
                matches = sum(1 for w in query_words if w in all_content)
                score = (matches / len(query_words)) * 100
            
            # Domain-specific boosting
            if 'escalat' in query_lower and ('escalat' in all_content or 'supervisor' in all_content):
                score += 50
            if 'fraud' in query_lower and 'fraud' in all_content:
                score += 50
            if 'delivery' in query_lower and 'delivery' in all_content:
                score += 50
            if 'error' in query_lower and 'error' in all_content:
                score += 30
            
            scores[tid] = score
        
        # Sort and return
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        result = [tid for tid, score in sorted_ids[:top_k] if score > 0]
        
        return result if result else list(self.conversations_by_id.keys())[:top_k]
    
    def get_transcript(self, transcript_id: str) -> Optional[ConversationTranscript]:
        """Get transcript by ID"""
        return self.conversations_by_id.get(transcript_id)
    
    def get_all_transcripts(self) -> List[ConversationTranscript]:
        """Get all loaded transcripts"""
        return list(self.conversations_by_id.values())