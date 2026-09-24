# src/escalator.py

import os
import json
import logging
import hashlib
import re
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timezone
from functools import lru_cache
from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.prompts import ESCALATION_PROMPT, HANDOFF_SUMMARY_PROMPT

# ============================================
# LOGGING SETUP
# ============================================

logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================

ESCALATION_CONFIG = {
    "min_persona_confidence": 0.65,
    "min_retrieval_confidence": 0.50,
    "max_turns_before_escalation": 3,
    "model": "gemini-2.5-flash-lite",
    "cache_size": 128,
    "enable_llm_escalation": True,
    "frustration_threshold": 3,
    "sensitive_keywords": [
        "refund", "billing", "payment", "legal", "lawsuit",
        "cancel", "terminate", "compliance", "gdpr", "privacy",
        "security breach", "account lock", "fraud", "unauthorized",
        "chargeback", "dispute", "violation"
    ],
    "frustration_keywords": [
        "angry", "frustrated", "annoyed", "terrible", "worst",
        "useless", "tired", "stupid", "awful", "horrible",
        "unacceptable", "disappointed", "upset", "nothing works",
        "waste", "never works", "still broken", "ridiculous"
    ],
    "force_escalation_keywords": [
        "emergency", "urgent", "security breach", "legal action",
        "lawyer", "attorney", "lawsuit", "immediate action"
    ]
}

# ============================================
# CLIENT INITIALIZATION
# ============================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.warning("GEMINI_API_KEY not found in environment variables")

client = genai.Client(api_key=api_key)

# ============================================
# HELPER FUNCTIONS
# ============================================

def sanitize_query(query: str) -> str:
    """
    Sanitize user query for safe processing.
    """
    if not query:
        return ""
    
    # Remove excessive whitespace
    cleaned = ' '.join(query.strip().split())
    
    # Limit length to prevent abuse
    max_length = 2000
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]
    
    return cleaned


def extract_json_from_response(text: str) -> Optional[Dict]:
    """
    Robust JSON extraction from Gemini responses.
    Handles markdown, explanations, and malformed JSON.
    """
    if not text:
        return None
    
    # Clean the response
    cleaned = text.strip()
    
    # Remove markdown code fences
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    
    # Try to find JSON in the response
    json_start = cleaned.find('{')
    json_end = cleaned.rfind('}') + 1
    
    if json_start != -1 and json_end > json_start:
        json_str = cleaned[json_start:json_end]
    else:
        json_str = cleaned
    
    # Try parsing
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try to fix common issues
        try:
            # Fix trailing commas
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            return json.loads(json_str)
        except:
            return None


def generate_cache_key(query: str, persona: str, confidence: float) -> str:
    """
    Generate a cache key for escalation decisions.
    """
    key = f"{query}_{persona}_{confidence:.2f}"
    return hashlib.md5(key.encode()).hexdigest()


def get_timestamp() -> str:
    """
    Get current UTC timestamp in ISO format.
    """
    return datetime.now(timezone.utc).isoformat()


# ============================================
# ESCALATION MANAGER
# ============================================

class EscalationManager:
    """
    Manages escalation decisions and handoff summaries.
    Optimized with early exit and minimal LLM calls.
    """
    
    def __init__(self):
        self.conversation_history: List[Dict[str, str]] = []
        self.escalation_count: int = 0
        self.max_turns_before_escalation: int = ESCALATION_CONFIG["max_turns_before_escalation"]
        self._cache: Dict[str, Tuple[bool, str]] = {}
    
    def add_to_history(self, user_query: str, response: str, persona: str) -> None:
        """
        Add a turn to conversation history.
        """
        self.conversation_history.append({
            "user": user_query[:500],  # Truncate to prevent memory issues
            "assistant": response[:500],
            "persona": persona
        })
        
        # Keep only last 10 turns to prevent memory growth
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def check_escalation(self, query: str, persona_data: Dict, 
                         retrieved_docs: List[Dict]) -> Tuple[bool, str]:
        """
        Check if the query should be escalated.
        Uses rule-based check first, then LLM only if needed.
        """
        # Sanitize input
        query = sanitize_query(query)
        if not query:
            return False, "No escalation needed (empty query)"
        
        # Check cache first
        cache_key = generate_cache_key(
            query,
            persona_data.get("persona", "Unknown"),
            persona_data.get("confidence", 0)
        )
        if cache_key in self._cache:
            logger.debug(f"Cache hit for escalation: {cache_key}")
            return self._cache[cache_key]
        
        # 1. Rule-based checks (fast, deterministic)
        rule_result = self._rule_based_check(query, persona_data, retrieved_docs)
        if rule_result[0]:
            # Escalation determined by rules
            self._cache[cache_key] = rule_result
            return rule_result
        
        # 2. LLM-based check (only for ambiguous cases)
        if ESCALATION_CONFIG["enable_llm_escalation"]:
            # Check if LLM is needed (borderline cases)
            confidence = persona_data.get("confidence", 0)
            if 0.60 <= confidence <= 0.70:  # Borderline confidence
                logger.info("LLM escalation check triggered (borderline confidence)")
                llm_result = self._llm_based_check(query, persona_data, retrieved_docs)
                self._cache[cache_key] = llm_result
                return llm_result
            
            # Check if frustration level is high but not extreme
            frustration_score = self._calculate_frustration_score(query)
            if 2 <= frustration_score <= 3:  # Moderate frustration
                logger.info("LLM escalation check triggered (moderate frustration)")
                llm_result = self._llm_based_check(query, persona_data, retrieved_docs)
                self._cache[cache_key] = llm_result
                return llm_result
        
        # No escalation needed
        result = (False, "No escalation needed")
        self._cache[cache_key] = result
        return result
    
    def _rule_based_check(self, query: str, persona_data: Dict, 
                          retrieved_docs: List[Dict]) -> Tuple[bool, str]:
        """
        Rule-based escalation checks.
        Returns (escalate, reason) if escalation is determined.
        """
        query_lower = query.lower()
        confidence = persona_data.get("confidence", 0)
        
        # Check 1: Force escalation keywords (highest priority)
        for keyword in ESCALATION_CONFIG["force_escalation_keywords"]:
            if keyword in query_lower:
                return True, f"Force escalation: {keyword}"
        
        # Check 2: No documents retrieved
        if not retrieved_docs:
            return True, "No relevant documents found in knowledge base"
        
        # Check 3: Low persona confidence
        if confidence < ESCALATION_CONFIG["min_persona_confidence"]:
            return True, f"Low confidence in persona detection ({confidence:.2f})"
        
        # Check 4: Sensitive topics
        for keyword in ESCALATION_CONFIG["sensitive_keywords"]:
            if keyword in query_lower:
                return True, f"Sensitive topic detected: {keyword}"
        
        # Check 5: High frustration
        if persona_data.get("persona") == "Frustrated User":
            frustration_score = self._calculate_frustration_score(query)
            if frustration_score >= ESCALATION_CONFIG["frustration_threshold"]:
                return True, f"High frustration level detected (score: {frustration_score})"
        
        # Check 6: Multiple unresolved turns
        if len(self.conversation_history) >= self.max_turns_before_escalation:
            return True, f"Issue unresolved after {self.max_turns_before_escalation} interactions"
        
        # Check 7: Retrieval confidence low
        if self._calculate_retrieval_confidence(retrieved_docs) < ESCALATION_CONFIG["min_retrieval_confidence"]:
            return True, "Low retrieval confidence"
        
        # No rule-based escalation
        return False, "No escalation needed"
    
    def _llm_based_check(self, query: str, persona_data: Dict, 
                         retrieved_docs: List[Dict]) -> Tuple[bool, str]:
        """
        LLM-based escalation check for nuanced cases.
        Only called when rules are ambiguous.
        """
        try:
            # Build minimal context (avoid token waste)
            context = f"""
Query: {query[:200]}
Persona: {persona_data.get('persona', 'Unknown')}
Confidence: {persona_data.get('confidence', 0):.2f}
Documents: {len(retrieved_docs)}

{ESCALATION_PROMPT}
"""
            
            response = client.models.generate_content(
                model=ESCALATION_CONFIG["model"],
                contents=context
            )
            
            # Extract JSON robustly
            result = extract_json_from_response(response.text)
            
            if result is None:
                logger.warning("Failed to parse LLM escalation response")
                return False, "LLM response parsing failed"
            
            escalate = result.get("escalate", False)
            reason = result.get("reason", "LLM escalation decision")
            
            logger.info(f"LLM escalation decision: escalate={escalate}, reason={reason}")
            return escalate, reason
            
        except Exception as e:
            logger.error(f"LLM escalation check error: {e}")
            # Fallback: no escalation to avoid false positives
            return False, "LLM check failed, no escalation"
    
    def _calculate_frustration_score(self, query: str) -> int:
        """
        Calculate frustration score based on keyword matches.
        """
        query_lower = query.lower()
        score = 0
        
        for keyword in ESCALATION_CONFIG["frustration_keywords"]:
            if keyword in query_lower:
                score += 1
        
        return min(score, 5)  # Cap at 5
    
    def _calculate_retrieval_confidence(self, retrieved_docs: List[Dict]) -> float:
        """
        Calculate overall retrieval confidence from document scores.
        """
        if not retrieved_docs:
            return 0.0
        
        # Average similarity scores
        scores = [doc.get("score", 0) for doc in retrieved_docs]
        avg_score = sum(scores) / len(scores)
        
        # Boost if we have multiple documents
        if len(scores) >= 2:
            avg_score = min(1.0, avg_score + 0.1)
        
        return avg_score
    
    def generate_handoff_summary(self, query: str, persona_data: Dict, 
                                 retrieved_docs: List[Dict]) -> Dict:
        """
        Generate a structured handoff summary for human support.
        """
        # Build documents used
        documents_used = list(set([
            doc.get("source", "Unknown") for doc in retrieved_docs[:5]  # Limit to 5
        ]))
        
        # Build attempted steps
        attempted_steps = ["Knowledge base retrieval"]
        if len(self.conversation_history) > 0:
            attempted_steps.extend([
                "AI response generation",
                f"{len(self.conversation_history)} conversation turns"
            ])
        
        # Try LLM-generated summary
        try:
            # Build conversation summary (limited to last 3 turns)
            conversation_summary = "\n".join([
                f"User: {turn['user'][:100]}..."
                for turn in self.conversation_history[-3:]
            ])
            
            prompt = f"""
{HANDOFF_SUMMARY_PROMPT}

Input Data:
- Persona: {persona_data.get('persona', 'Unknown')}
- Confidence: {persona_data.get('confidence', 0):.2f}
- User Query: {query[:300]}
- Documents Used: {documents_used}
- Conversation Summary: {conversation_summary}
- Attempted Steps: {attempted_steps}
- Escalation Count: {self.escalation_count}

Generate handoff summary in JSON format.
"""
            
            response = client.models.generate_content(
                model=ESCALATION_CONFIG["model"],
                contents=prompt
            )
            
            # Extract JSON robustly
            summary = extract_json_from_response(response.text)
            
            if summary:
                # Add metadata
                summary["escalation_count"] = self.escalation_count
                summary["timestamp"] = get_timestamp()
                return summary
            
        except Exception as e:
            logger.error(f"Handoff summary generation error: {e}")
        
        # Fallback: structured summary
        return {
            "persona": persona_data.get("persona", "Unknown"),
            "confidence": persona_data.get("confidence", 0),
            "issue_summary": query[:200],
            "conversation_summary": f"{len(self.conversation_history)} conversation turns",
            "documents_used": documents_used,
            "attempted_steps": attempted_steps,
            "escalation_reason": "Manual escalation triggered",
            "recommended_next_step": "Human support intervention required",
            "escalation_count": self.escalation_count,
            "timestamp": get_timestamp()
        }
    
    def clear_cache(self) -> None:
        """
        Clear the escalation decision cache.
        """
        self._cache.clear()
        logger.info("Escalation cache cleared")


# ============================================
# BACKWARD COMPATIBILITY FUNCTIONS
# ============================================

def check_escalation(query: str, persona_data: Dict, 
                     retrieved_docs: List[Dict]) -> Tuple[bool, str]:
    """
    Legacy function for backward compatibility.
    """
    manager = EscalationManager()
    return manager.check_escalation(query, persona_data, retrieved_docs)


def generate_handoff_summary(query: str, persona_data: Dict, 
                             retrieved_docs: List[Dict]) -> Dict:
    """
    Legacy function for backward compatibility.
    """
    manager = EscalationManager()
    return manager.generate_handoff_summary(query, persona_data, retrieved_docs)


# ============================================
# TESTING
# ============================================

if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("ESCALATOR TEST (UPDATED - google.genai)")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    manager = EscalationManager()
    
    # Test cases
    test_cases = [
        {
            "query": "I need a refund for my duplicate payment",
            "persona": "Frustrated User",
            "confidence": 0.85,
            "docs": [{"source": "billing_policy.txt", "score": 0.9}]
        },
        {
            "query": "How do I reset my password?",
            "persona": "Technical Expert",
            "confidence": 0.90,
            "docs": [{"source": "password_reset_guide.pdf", "score": 0.95}]
        },
        {
            "query": "This is a security breach!",
            "persona": "Business Executive",
            "confidence": 0.75,
            "docs": [{"source": "security_policy.md", "score": 0.8}]
        },
        {
            "query": "I've been trying for hours and nothing works!",
            "persona": "Frustrated User",
            "confidence": 0.55,
            "docs": []
        }
    ]
    
    for test in test_cases:
        print(f"\n📝 Query: {test['query']}")
        persona_data = {
            "persona": test['persona'],
            "confidence": test['confidence']
        }
        
        escalated, reason = manager.check_escalation(
            test['query'],
            persona_data,
            test['docs']
        )
        
        print(f"   Escalated: {escalated}")
        print(f"   Reason: {reason}")
        
        if escalated:
            summary = manager.generate_handoff_summary(
                test['query'],
                persona_data,
                test['docs']
            )
            print("   Summary:", json.dumps(summary, indent=2))
    
    print("\n" + "=" * 60)
    print("CACHE PERFORMANCE")
    print("=" * 60)
    
    # Test cache
    test_query = "I need a refund"
    test_persona = {"persona": "Frustrated User", "confidence": 0.85}
    test_docs = [{"source": "billing.txt", "score": 0.9}]
    
    # Warm up cache
    manager.check_escalation(test_query, test_persona, test_docs)
    
    start = time.time()
    for _ in range(3):
        manager.check_escalation(test_query, test_persona, test_docs)
    cached_time = time.time() - start
    
    # Clear cache for comparison
    manager.clear_cache()
    start = time.time()
    for _ in range(3):
        manager.check_escalation(test_query, test_persona, test_docs)
    uncached_time = time.time() - start
    
    print(f"Cached (3 calls): {cached_time:.3f}s")
    print(f"Uncached (3 calls): {uncached_time:.3f}s")
    if cached_time > 0 and uncached_time > 0:
        print(f"Speedup: {uncached_time/cached_time:.1f}x")
    else:
        print("Speedup: N/A (cache hit time too small)")