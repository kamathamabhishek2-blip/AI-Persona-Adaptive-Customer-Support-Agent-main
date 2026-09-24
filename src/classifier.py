# src/classifier.py

import os
import json
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ============================================
# CONFIGURATION
# ============================================

CLASSIFIER_CONFIG = {
    "model": "gemini-2.5-flash-lite",
    "temperature": 0.1,
    "max_retries": 3,
    "retry_delay": 1,
    "cache_size": 128,
    "max_query_length": 2000,
    "confidence_thresholds": {
        "Technical Expert": 0.70,
        "Frustrated User": 0.60,
        "Business Executive": 0.65
    },
    "fallback_confidence": 0.5,
    "min_fallback_confidence": 0.3
}

# ============================================
# CLIENT INITIALIZATION
# ============================================

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)

# ============================================
# SYSTEM PROMPT - Persona Classification
# ============================================

PERSONA_CLASSIFIER_PROMPT = """
You are an expert customer support persona classification engine.

Your task is to identify the user's dominant support persona.

Available Personas:

1. Technical Expert
Characteristics:
- Uses technical terminology
- Mentions APIs, logs, debugging, integrations, configurations
- Requests root cause analysis
- Wants detailed explanations

2. Frustrated User
Characteristics:
- Emotional language
- Complaints or dissatisfaction
- Urgent requests
- Repeated failures
- Uses words like frustrated, angry, upset, tired, disappointed

3. Business Executive
Characteristics:
- Focuses on business impact
- Interested in timelines and risk
- Wants concise communication
- Avoids deep technical details
- Mentions customers, operations, revenue, downtime

Instructions:
- Analyze intent, tone, vocabulary and goals.
- Select exactly ONE persona.
- Provide confidence score between 0 and 1.
- Explain reasoning briefly.

IMPORTANT:
Return ONLY valid JSON.
Do not include markdown.
Do not include explanations.
Do not wrap in ```json blocks.

Schema:

{
  "persona": "",
  "confidence": 0.0,
  "reasoning": ""
}
"""

# ============================================
# HELPER FUNCTIONS
# ============================================

def preprocess_query(query: str) -> str:
    """
    Clean and normalize input query for optimal LLM processing.
    """
    if not query or not query.strip():
        return ""
    
    # Remove extra whitespace
    cleaned = ' '.join(query.strip().split())
    
    # Truncate to safe length
    max_len = CLASSIFIER_CONFIG["max_query_length"]
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len] + "..."
    
    return cleaned


def clean_json_response(raw: str) -> str:
    """
    Clean markdown code fences from Gemini response.
    """
    if not raw:
        return ""
    
    cleaned = raw.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    return cleaned


def validate_persona_response(response: Dict) -> bool:
    """
    Validate that the response contains all required fields.
    """
    required_keys = ["persona", "confidence", "reasoning"]
    valid_personas = ["Technical Expert", "Frustrated User", "Business Executive"]
    
    if not all(key in response for key in required_keys):
        return False
    
    if response["persona"] not in valid_personas:
        return False
    
    if not 0 <= response["confidence"] <= 1:
        return False
    
    return True


def with_retry(func, max_retries: int = None, delay: int = None) -> Optional[Dict]:
    """
    Execute function with retry logic for transient failures.
    """
    max_retries = max_retries or CLASSIFIER_CONFIG["max_retries"]
    delay = delay or CLASSIFIER_CONFIG["retry_delay"]
    
    last_error = None
    for attempt in range(max_retries):
        try:
            result = func()
            if result:
                return result
        except (json.JSONDecodeError, ValueError, Exception) as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))
                continue
            break
    
    return None


# ============================================
# CLASSIFICATION FUNCTIONS
# ============================================

def build_persona_prompt(user_message: str) -> str:
    """
    Efficiently build the classification prompt.
    """
    query = preprocess_query(user_message)
    return f"{PERSONA_CLASSIFIER_PROMPT}\n\nUser Message: {query}"


@lru_cache(maxsize=128)
def classify_persona_cached(user_message: str) -> Optional[Dict]:
    """
    Cached version of persona classification.
    """
    return classify_persona(user_message)


def classify_persona(user_message: str) -> Optional[Dict]:
    """
    Classify customer persona using Gemini 2.5 Flash.
    Returns JSON with persona, confidence, and reasoning.
    """
    if not user_message or not user_message.strip():
        return _fallback_classification("", "Empty query")

    def _call_api():
        prompt = build_persona_prompt(user_message)
        
        response = client.models.generate_content(
            model=CLASSIFIER_CONFIG["model"],
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=CLASSIFIER_CONFIG["temperature"],
            )
        )
        
        raw = clean_json_response(response.text)
        result = json.loads(raw)
        
        # Validate confidence range
        result["confidence"] = min(1.0, max(0.0, result.get("confidence", CLASSIFIER_CONFIG["fallback_confidence"])))
        
        if validate_persona_response(result):
            return result
        
        return None

    try:
        result = with_retry(_call_api)
        if result:
            return result
        
        print("⚠️ Classification failed, using fallback")
        return _fallback_classification(user_message, "API returned invalid response")
        
    except Exception as e:
        print(f"❌ Classification error: {e}")
        return _fallback_classification(user_message, f"API error: {str(e)}")


def _fallback_classification(user_message: str, reason: str = "API fallback") -> Dict:
    """
    Rule-based classification as fallback when Gemini API fails.
    """
    if not user_message:
        return {
            "persona": "Business Executive",
            "confidence": CLASSIFIER_CONFIG["min_fallback_confidence"],
            "reasoning": f"Empty query. {reason}"
        }
    
    message_lower = user_message.lower()
    
    # Technical keywords
    tech_keywords = {
        "api": 2, "sdk": 2, "endpoint": 2, "authentication": 2,
        "log": 2, "logs": 2, "debug": 2, "debugging": 2,
        "config": 2, "configuration": 2, "error": 1, "exception": 1,
        "integration": 2, "pipeline": 1, "deployment": 1,
        "root cause": 3, "stack trace": 3, "database": 1,
        "code": 1, "syntax": 1, "compile": 1, "server": 1
    }
    
    # Frustration keywords
    frustration_keywords = {
        "angry": 3, "frustrated": 3, "annoyed": 2,
        "terrible": 2, "worst": 2, "useless": 2,
        "tired": 2, "stupid": 2, "awful": 2,
        "never works": 3, "waste": 2, "unacceptable": 3,
        "horrible": 2, "disappointed": 2, "upset": 2,
        "nothing works": 3, "still broken": 3, "again": 1
    }
    
    # Business keywords
    business_keywords = {
        "impact": 3, "business": 3, "revenue": 3,
        "operations": 3, "timeline": 3, "roi": 3,
        "stakeholder": 3, "budget": 2, "deadline": 2,
        "customers": 2, "sales": 2, "production": 2,
        "downtime": 3, "sla": 3, "priority": 2,
        "cost": 2, "efficiency": 2, "strategy": 2
    }
    
    # Calculate scores
    tech_score = sum(weight for word, weight in tech_keywords.items() 
                    if word in message_lower)
    frustration_score = sum(weight for word, weight in frustration_keywords.items() 
                          if word in message_lower)
    business_score = sum(weight for word, weight in business_keywords.items() 
                        if word in message_lower)
    
    scores = {
        "Technical Expert": tech_score,
        "Frustrated User": frustration_score,
        "Business Executive": business_score
    }
    
    # Find best persona
    best_persona = max(scores, key=scores.get)
    max_score = scores[best_persona]
    
    # Calculate confidence based on score
    if max_score == 0:
        return {
            "persona": "Business Executive",
            "confidence": CLASSIFIER_CONFIG["min_fallback_confidence"],
            "reasoning": f"Default classification (no keywords matched). {reason}"
        }
    elif max_score <= 2:
        confidence = 0.4 + (max_score / 10)
    elif max_score <= 5:
        confidence = 0.6 + (max_score / 20)
    else:
        confidence = min(0.95, 0.8 + (max_score / 50))
    
    # Adjust if multiple personas have close scores
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) > 1 and sorted_scores[0] - sorted_scores[1] < 2:
        confidence = max(0.5, confidence - 0.2)
    
    return {
        "persona": best_persona,
        "confidence": min(0.95, max(CLASSIFIER_CONFIG["min_fallback_confidence"], confidence)),
        "reasoning": f"Rule-based classification. Keywords matched: {best_persona}. {reason}"
    }


def classify_persona_with_fallback(user_message: str) -> Dict:
    """
    Enhanced classification with automatic fallback if confidence is low.
    """
    # Try primary classification with caching
    result = classify_persona_cached(user_message)
    
    if not result:
        return _fallback_classification(user_message, "Classification returned None")
    
    # Check if confidence meets threshold
    persona = result.get("persona", "Business Executive")
    confidence = result.get("confidence", CLASSIFIER_CONFIG["fallback_confidence"])
    threshold = CLASSIFIER_CONFIG["confidence_thresholds"].get(persona, 0.65)
    
    if confidence < threshold:
        fallback_result = _fallback_classification(
            user_message, 
            f"Low confidence ({confidence:.2f} < {threshold:.2f})"
        )
        
        if fallback_result["confidence"] > confidence:
            return fallback_result
    
    return result


def classify_batch(messages: List[str]) -> List[Dict]:
    """
    Classify multiple messages in batch.
    Useful for testing and evaluation.
    """
    results = []
    for msg in messages:
        results.append(classify_persona_with_fallback(msg))
    return results


def evaluate_classification(test_cases: List[str], expected: List[str]) -> Dict:
    """
    Evaluate classification accuracy with test cases.
    """
    if not test_cases or len(test_cases) != len(expected):
        return {"accuracy": 0.0, "results": []}
    
    correct = 0
    results = []
    
    for msg, expected_persona in zip(test_cases, expected):
        result = classify_persona_with_fallback(msg)
        is_correct = result["persona"] == expected_persona
        results.append({
            "query": msg[:100] + ("..." if len(msg) > 100 else ""),
            "expected": expected_persona,
            "predicted": result["persona"],
            "confidence": result["confidence"],
            "correct": is_correct
        })
        if is_correct:
            correct += 1
    
    return {
        "accuracy": correct / len(test_cases) if test_cases else 0.0,
        "results": results
    }


# ============================================
# CLEAR CACHE FUNCTION
# ============================================

def clear_classifier_cache():
    """
    Clear the LRU cache for testing or memory management.
    """
    classify_persona_cached.cache_clear()
    print("✅ Classifier cache cleared")


# ============================================
# TEST CASES
# ============================================

if __name__ == "__main__":
    import time
    
    test_queries = [
        "Can you explain why the API authentication is failing? I'm getting a 401 error with the following logs: [ERROR] Authentication failed for user_id=12345.",
        "What's the best way to debug database connection issues? We're using PostgreSQL with connection pooling.",
        "I've tried everything and nothing works! I'm so frustrated right now.",
        "This is the third time this week I'm facing the same issue. Your support is terrible.",
        "How will this service outage impact our operations? We're losing customers.",
        "What's the ROI of implementing this new feature? I need a concise breakdown.",
        "The API error is causing 500 errors on production. Affecting 1000+ customers.",
    ]
    
    print("=" * 60)
    print("PERSONA CLASSIFICATION TEST (OPTIMIZED)")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n📝 Query: {query[:80]}...")
        result = classify_persona_with_fallback(query)
        print(f"   Persona: {result['persona']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Reasoning: {result['reasoning']}")
        print("-" * 40)
    
    # Test evaluation
    print("\n" + "=" * 60)
    print("CLASSIFICATION EVALUATION")
    print("=" * 60)
    
    test_cases = [
        "API authentication failed with error code 401",
        "I'm really frustrated with this issue",
        "What's the business impact of this delay?"
    ]
    expected = [
        "Technical Expert",
        "Frustrated User",
        "Business Executive"
    ]
    
    eval_result = evaluate_classification(test_cases, expected)
    print(f"\nAccuracy: {eval_result['accuracy']:.2%}")
    for result in eval_result['results']:
        status = "✅" if result['correct'] else "❌"
        print(f"{status} Expected: {result['expected']}, "
              f"Predicted: {result['predicted']}, "
              f"Confidence: {result['confidence']:.2%}")
    
    # Test cache performance
    print("\n" + "=" * 60)
    print("CACHE PERFORMANCE TEST")
    print("=" * 60)
    
    test_query = "API authentication failed"
    
    # Warm up cache
    classify_persona_cached(test_query)
    
    start = time.time()
    for _ in range(3):
        classify_persona_cached(test_query)
    cached_time = time.time() - start
    
    # Clear cache for comparison
    clear_classifier_cache()
    
    start = time.time()
    for _ in range(3):
        classify_persona(test_query)
    uncached_time = time.time() - start
    
    print(f"Cached calls (3): {cached_time:.3f}s")
    print(f"Uncached calls (3): {uncached_time:.3f}s")
    if cached_time > 0 and uncached_time > 0:
        print(f"Speedup: {uncached_time/cached_time:.1f}x")
    else:
        print("Speedup: N/A (cache hit time too small)")