"""
Main Entry Point for Causal Analysis System
"""

import os
import sys
import json
import logging
from datetime import datetime

# Import from current directory since we're in src
from task1_retrieval import ConversationRetriever
from task2_causal_analysis import CausalAnalyzer, CausalExplanation
from utils.helpers import format_explanation, load_json_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_data():
    """Create sample conversation data"""
    return {
        "transcripts": [
            {
                "transcript_id": "HC-001",
                "domain": "Healthcare Services",
                "intent": "Escalation - Repeated Service Failures",
                "reason_for_call": "Customer experiencing login issues for three weeks",
                "conversation": [
                    {"speaker": "Agent", "text": "Thank you for calling. How can I help you?"},
                    {"speaker": "Customer", "text": "I've been trying to resolve a login issue for three weeks now and I'm not getting any real help."},
                    {"speaker": "Agent", "text": "I'm sorry to hear about your ongoing issue. Let me check your account."},
                    {"speaker": "Customer", "text": "I've explained this multiple times. Each time I'm told it's fixed, but it's not. I need to speak with a supervisor."},
                    {"speaker": "Agent", "text": "I understand your frustration. I see error code 3309 in your account."},
                    {"speaker": "Customer", "text": "Nobody can tell me what that means or how to fix it!"},
                    {"speaker": "Agent", "text": "I'll transfer you to my supervisor right away."}
                ]
            },
            {
                "transcript_id": "FR-002",
                "domain": "Banking",
                "intent": "Fraud Alert Investigation",
                "reason_for_call": "Unauthorized charge detected",
                "conversation": [
                    {"speaker": "Agent", "text": "Fraud Department, how can I help?"},
                    {"speaker": "Customer", "text": "I got a fraud alert about a charge I didn't make."},
                    {"speaker": "Agent", "text": "I see a charge for $356.82 in New York. Did you make this?"},
                    {"speaker": "Customer", "text": "No, I've never been to New York."},
                    {"speaker": "Agent", "text": "I'm blocking your card and reversing the charge. You'll get a new card in 2-3 days."}
                ]
            }
        ]
    }


class CausalAnalysisSystem:
    """Complete causal analysis system"""
    
    def __init__(self):
        """Initialize the system"""
        self.retriever = ConversationRetriever()
        self.analyzer = CausalAnalyzer()
        self.loaded = False
    
    def load_data(self) -> bool:
        """Load conversation data"""
        # Try to find data file
        data = None
        paths_to_try = [
            "../data/sample_conversations.json",
            "data/sample_conversations.json",
            "sample_conversations.json"
        ]
        
        for path in paths_to_try:
            if os.path.exists(path):
                logger.info(f"Loading data from: {path}")
                data = load_json_file(path)
                if data:
                    break
        
        if not data:
            logger.info("No data file found. Creating sample data...")
            data = create_sample_data()
            
            # Save for next time
            try:
                os.makedirs("../data", exist_ok=True)
                with open("../data/sample_conversations.json", "w") as f:
                    json.dump(data, f, indent=2)
                logger.info("Saved sample data to ../data/sample_conversations.json")
            except:
                pass
        
        count = self.retriever.load_conversations(data)
        self.loaded = count > 0
        return self.loaded
    
    def process_query(self, query: str, top_k: int = 3) -> CausalExplanation:
        """Process a user query"""
        if not self.loaded:
            if not self.load_data():
                return self.analyzer._empty_explanation(query)
        
        # Task 1: Retrieve
        relevant_ids = self.retriever.retrieve(query, top_k=top_k)
        transcripts = [
            self.retriever.get_transcript(tid) 
            for tid in relevant_ids 
            if self.retriever.get_transcript(tid)
        ]
        
        # Task 2: Analyze
        return self.analyzer.analyze(query, transcripts)
    
    def list_transcripts(self):
        """Display all transcripts"""
        print("\n📑 Available Transcripts:")
        print("-" * 60)
        for t in self.retriever.get_all_transcripts():
            print(f"  ID: {t.transcript_id}")
            print(f"  Domain: {t.domain}")
            print(f"  Outcome: {t.outcome}")
            if t.metadata.get('reason_for_call'):
                print(f"  Reason: {t.metadata['reason_for_call']}")
            print("-" * 60)


def main():
    """Main entry point"""
    print("\n" + "=" * 80)
    print("🔍 CAUSAL ANALYSIS SYSTEM")
    print("=" * 80)
    
    system = CausalAnalysisSystem()
    
    if not system.load_data():
        print("⚠️  Using sample data")
    
    print("\n💡 Commands: 'quit', 'list', 'help'\n")
    
    while True:
        try:
            query = input("🔎 Query: ").strip()
            
            if not query:
                continue
            
            if query.lower() == 'quit':
                print("\n👋 Goodbye!")
                break
            
            if query.lower() == 'list':
                system.list_transcripts()
                continue
            
            if query.lower() == 'help':
                print("\n📖 Example Queries:")
                print("  • Why did the healthcare conversation escalate?")
                print("  • What was the fraud amount?")
                print("  • What error code was mentioned?")
                print("  • How long did the issue persist?\n")
                continue
            
            print("\n⏳ Processing...")
            explanation = system.process_query(query)
            print(format_explanation(explanation))
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()