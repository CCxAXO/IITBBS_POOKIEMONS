"""
Helper utilities for the causal analysis system
"""

import json
import os
from typing import Any, Dict, Optional


def format_explanation(explanation: Any, use_emoji: bool = True) -> str:
    """Format a CausalExplanation for display."""
    lines = [
        "=" * 80,
        "CAUSAL ANALYSIS RESULT",
        "=" * 80,
        f"\n📋 Query: {explanation.query}\n",
        f"🎯 PRIMARY CAUSE:",
        f"   {explanation.primary_cause}\n"
    ]
    
    if explanation.supporting_factors:
        lines.append("📊 SUPPORTING FACTORS:")
        for i, factor in enumerate(explanation.supporting_factors, 1):
            lines.append(f"   {i}. {factor}")
        lines.append("")
    
    if explanation.evidence_spans:
        lines.append("💬 EVIDENCE FROM CONVERSATION:")
        for turn_id, text in explanation.evidence_spans:
            lines.append(f"   Turn {turn_id}: {text}")
        lines.append("")
    
    lines.extend([
        f"📈 CONFIDENCE: {explanation.confidence:.0%}",
        f"   Relevant Transcripts: {', '.join(explanation.relevant_transcript_ids)}",
        "=" * 80
    ])
    
    return "\n".join(lines)


def load_json_file(filepath: str) -> Optional[Dict]:
    """Load JSON data from file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {filepath}: {e}")
        return None