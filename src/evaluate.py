"""
Evaluation Script for Causal Analysis System
Evaluates both Task 1 (Retrieval) and Task 2 (Causal Analysis)
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.task1_retrieval import ConversationRetriever
from src.task2_causal_analysis import CausalAnalyzer
from src.utils.helpers import load_json_file, save_results


class SystemEvaluator:
    """Evaluates the causal analysis system"""
    
    def __init__(self):
        self.retriever = ConversationRetriever()
        self.analyzer = CausalAnalyzer()
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'task1_retrieval': {},
            'task2_causal_analysis': {},
            'overall': {}
        }
    
    def load_data(self, conv_path: str, query_path: str) -> bool:
        """Load conversation and query data"""
        conv_data = load_json_file(conv_path)
        if conv_data:
            self.retriever.load_conversations(conv_data)
        
        self.queries = load_json_file(query_path)
        return conv_data is not None and self.queries is not None
    
    def evaluate_task1(self) -> Dict[str, Any]:
        """Evaluate Task 1: Conversation Retrieval"""
        print("\n📊 Evaluating Task 1: Conversation Retrieval")
        print("-" * 50)
        
        results = {
            'total_queries': 0,
            'successful_retrievals': 0,
            'domain_accuracy': 0,
            'avg_retrieval_time_ms': 0,
            'details': []
        }
        
        queries = self.queries.get('queries', [])
        total_time = 0
        domain_correct = 0
        
        for query_data in queries:
            query = query_data['query']
            expected_domain = query_data.get('expected_domain', '')
            
            start_time = time.time()
            retrieved_ids = self.retriever.retrieve(query, top_k=1)
            elapsed = (time.time() - start_time) * 1000
            
            total_time += elapsed
            results['total_queries'] += 1
            
            if retrieved_ids:
                results['successful_retrievals'] += 1
                transcript = self.retriever.get_transcript(retrieved_ids[0])
                if transcript and expected_domain.lower() in transcript.domain.lower():
                    domain_correct += 1
                
                results['details'].append({
                    'query_id': query_data.get('query_id'),
                    'query': query,
                    'retrieved': retrieved_ids[0],
                    'expected_domain': expected_domain,
                    'actual_domain': transcript.domain if transcript else 'N/A',
                    'time_ms': round(elapsed, 2)
                })
        
        if results['total_queries'] > 0:
            results['retrieval_rate'] = results['successful_retrievals'] / results['total_queries']
            results['domain_accuracy'] = domain_correct / results['total_queries']
            results['avg_retrieval_time_ms'] = round(total_time / results['total_queries'], 2)
        
        print(f"   Total Queries: {results['total_queries']}")
        print(f"   Successful Retrievals: {results['successful_retrievals']}")
        print(f"   Retrieval Rate: {results.get('retrieval_rate', 0):.1%}")
        print(f"   Domain Accuracy: {results['domain_accuracy']:.1%}")
        print(f"   Avg Retrieval Time: {results['avg_retrieval_time_ms']:.2f}ms")
        
        return results
    
    def evaluate_task2(self) -> Dict[str, Any]:
        """Evaluate Task 2: Causal Analysis"""
        print("\n📊 Evaluating Task 2: Causal Analysis")
        print("-" * 50)
        
        results = {
            'total_analyses': 0,
            'avg_confidence': 0,
            'avg_factors_found': 0,
            'avg_evidence_spans': 0,
            'cause_coverage': 0,
            'details': []
        }
        
        queries = self.queries.get('queries', [])
        total_confidence = 0
        total_factors = 0
        total_evidence = 0
        cause_matches = 0
        
        for query_data in queries:
            query = query_data['query']
            expected_causes = query_data.get('expected_causes', [])
            
            # Retrieve and analyze
            retrieved_ids = self.retriever.retrieve(query, top_k=1)
            transcripts = [
                self.retriever.get_transcript(tid)
                for tid in retrieved_ids
                if self.retriever.get_transcript(tid)
            ]
            
            explanation = self.analyzer.analyze(query, transcripts)
            
            results['total_analyses'] += 1
            total_confidence += explanation.confidence
            total_factors += len(explanation.supporting_factors)
            total_evidence += len(explanation.evidence_spans)
            
            # Check cause coverage
            if expected_causes:
                primary_lower = explanation.primary_cause.lower()
                matched = sum(1 for c in expected_causes if c.lower() in primary_lower)
                coverage = matched / len(expected_causes)
                cause_matches += coverage
            
            results['details'].append({
                'query_id': query_data.get('query_id'),
                'query': query,
                'primary_cause': explanation.primary_cause,
                'confidence': explanation.confidence,
                'factors_count': len(explanation.supporting_factors),
                'evidence_count': len(explanation.evidence_spans)
            })
        
        if results['total_analyses'] > 0:
            n = results['total_analyses']
            results['avg_confidence'] = round(total_confidence / n, 3)
            results['avg_factors_found'] = round(total_factors / n, 2)
            results['avg_evidence_spans'] = round(total_evidence / n, 2)
            results['cause_coverage'] = round(cause_matches / n, 3)
        
        print(f"   Total Analyses: {results['total_analyses']}")
        print(f"   Avg Confidence: {results['avg_confidence']:.1%}")
        print(f"   Avg Factors Found: {results['avg_factors_found']:.1f}")
        print(f"   Avg Evidence Spans: {results['avg_evidence_spans']:.1f}")
        print(f"   Cause Coverage: {results['cause_coverage']:.1%}")
        
        return results
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation"""
        print("\n" + "=" * 60)
        print("🔬 CAUSAL ANALYSIS SYSTEM EVALUATION")
        print("=" * 60)
        
        # Evaluate both tasks
        self.results['task1_retrieval'] = self.evaluate_task1()
        self.results['task2_causal_analysis'] = self.evaluate_task2()
        
        # Calculate overall metrics
        t1 = self.results['task1_retrieval']
        t2 = self.results['task2_causal_analysis']
        
        self.results['overall'] = {
            'combined_score': round(
                (t1.get('retrieval_rate', 0) * 0.3 + 
                 t1.get('domain_accuracy', 0) * 0.2 +
                 t2.get('avg_confidence', 0) * 0.3 +
                 t2.get('cause_coverage', 0) * 0.2), 3
            ),
            'total_transcripts': len(self.retriever.conversations_by_id),
            'total_queries_evaluated': t1.get('total_queries', 0)
        }
        
        print("\n" + "=" * 60)
        print("📈 OVERALL RESULTS")
        print("=" * 60)
        print(f"   Combined Score: {self.results['overall']['combined_score']:.1%}")
        print(f"   Transcripts Loaded: {self.results['overall']['total_transcripts']}")
        print(f"   Queries Evaluated: {self.results['overall']['total_queries_evaluated']}")
        print("=" * 60)
        
        return self.results
    
    def save_results(self, output_path: str = "evaluation_results.json"):
        """Save evaluation results to file"""
        save_results(self.results, output_path)
        print(f"\n✅ Results saved to: {output_path}")


def main():
    """Main evaluation entry point"""
    evaluator = SystemEvaluator()
    
    # Load data
    if not evaluator.load_data(
        "data/sample_conversations.json",
        "data/query_dataset.json"
    ):
        print("❌ Failed to load data files")
        return
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    # Save results
    evaluator.save_results("evaluation_results.json")


if __name__ == "__main__":
    main()