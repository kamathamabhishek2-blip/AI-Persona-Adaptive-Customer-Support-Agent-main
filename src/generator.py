# src/generator.py

import os
import json
import logging
import hashlib
import re
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.prompts import (
    get_persona_prompt,
    GROUNDING_PROMPT,
    QUERY_EXPANSION_PROMPT
)
from src.rag_pipeline import retrieve_chunks

# ============================================
# LOGGING SETUP
# ============================================

logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================

GENERATOR_CONFIG = {
    "model": "gemini-2.5-flash-lite",
    "temperature": 0.7,
    "top_k": 3,
    "min_retrieval_confidence": 0.3,
    "enable_query_expansion": True,
    "expansion_threshold": 0.4,  # Only expand if confidence below this
    "cache_size": 128,
    "max_context_length": 2000,
    "max_query_length": 500,
    "complexity_threshold": 3,  # Number of clauses to trigger expansion
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
    Sanitize and normalize user query.
    """
    if not query:
        return ""
    
    # Clean whitespace
    cleaned = ' '.join(query.strip().split())
    
    # Truncate if too long
    max_len = GENERATOR_CONFIG["max_query_length"]
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len] + "..."
    
    return cleaned


def calculate_query_complexity(query: str) -> int:
    """
    Calculate query complexity based on clauses and keywords.
    """
    if not query:
        return 0
    
    # Count logical operators
    operators = ["and", "or", "with", "without", "but", "however"]
    operator_count = sum(1 for op in operators if op in query.lower())
    
    # Count question words
    question_words = ["how", "why", "what", "when", "where", "who", "which"]
    question_count = sum(1 for word in question_words if word in query.lower())
    
    # Count technical terms (basic heuristic)
    technical_terms = ["api", "error", "config", "log", "debug", "code", "server"]
    tech_count = sum(1 for term in technical_terms if term in query.lower())
    
    return operator_count + question_count + (tech_count // 2)


def calculate_retrieval_confidence(chunks: List[Dict]) -> float:
    """
    Calculate overall retrieval confidence from chunk scores.
    """
    if not chunks:
        return 0.0
    
    scores = [chunk.get("score", 0) for chunk in chunks]
    avg_score = sum(scores) / len(scores)
    
    # Boost if we have multiple high-quality chunks
    high_quality = sum(1 for s in scores if s > 0.5)
    if high_quality >= 2:
        avg_score = min(1.0, avg_score + 0.1)
    
    return avg_score


def extract_json_from_response(text: str) -> Optional[Dict]:
    """
    Robust JSON extraction from Gemini responses.
    """
    if not text:
        return None
    
    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    
    # Find JSON
    json_start = cleaned.find('{')
    json_end = cleaned.rfind('}') + 1
    
    if json_start != -1 and json_end > json_start:
        json_str = cleaned[json_start:json_end]
    else:
        json_str = cleaned
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try to fix common issues
        try:
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            return json.loads(json_str)
        except:
            return None


def generate_cache_key(query: str, persona: str) -> str:
    """
    Generate cache key for query expansion.
    """
    key = f"{query}_{persona}"
    return hashlib.md5(key.encode()).hexdigest()


# ============================================
# QUERY EXPANSION (OPTIONAL)
# ============================================

@lru_cache(maxsize=128)
def expand_query_cached(query: str) -> List[str]:
    """
    Cached version of query expansion.
    """
    return expand_query(query)


def expand_query(query: str) -> List[str]:
    """
    Expand user query into multiple search variations.
    Only called when necessary (low confidence or complex query).
    """
    try:
        response = client.models.generate_content(
            model=GENERATOR_CONFIG["model"],
            contents=f"""
            {QUERY_EXPANSION_PROMPT}
            
            Original Query: {query}
            """
        )

        raw = response.text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        result = json.loads(raw)
        expanded = result.get("queries", [query])
        
        # Ensure we have at least the original query
        if query not in expanded:
            expanded.insert(0, query)
        
        logger.debug(f"Query expansion: {query} → {len(expanded)} variations")
        return expanded

    except Exception as e:
        logger.warning(f"Query expansion error: {e}")
        return [query]


# ============================================
# RETRIEVAL FUNCTIONS
# ============================================

def retrieve_chunks_optimized(query: str, top_k: int = 3) -> Tuple[List[Dict], float]:
    """
    Retrieve chunks with confidence scoring.
    Returns (chunks, confidence_score).
    """
    if not query:
        return [], 0.0
    
    # Single retrieval first (fast)
    chunks = retrieve_chunks(query, top_k=top_k)
    confidence = calculate_retrieval_confidence(chunks)
    
    # Check if we need expansion
    complexity = calculate_query_complexity(query)
    need_expansion = (
        GENERATOR_CONFIG["enable_query_expansion"] and
        (confidence < GENERATOR_CONFIG["expansion_threshold"] or
         complexity >= GENERATOR_CONFIG["complexity_threshold"])
    )
    
    if need_expansion:
        logger.info(f"Query expansion triggered (confidence={confidence:.2f}, complexity={complexity})")
        expanded_queries = expand_query_cached(query)
        
        # Retrieve for each expanded query
        all_chunks = []
        seen_texts = set()
        
        for expanded_query in expanded_queries:
            chunks = retrieve_chunks(expanded_query, top_k=2)
            for chunk in chunks:
                text = chunk.get("text", "")
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    all_chunks.append(chunk)
        
        # Sort and return top-k
        all_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
        chunks = all_chunks[:top_k]
        confidence = calculate_retrieval_confidence(chunks)
        
        logger.debug(f"Expanded retrieval: {len(chunks)} chunks, confidence={confidence:.2f}")
    
    return chunks, confidence


# ============================================
# RESPONSE GENERATION
# ============================================

def build_context(chunks: List[Dict], max_length: int = 2000) -> str:
    """
    Build context string from retrieved chunks with length limit.
    """
    if not chunks:
        return "No relevant documents found."
    
    context_parts = []
    total_length = 0
    
    for chunk in chunks:
        source = chunk.get("source", "Unknown")
        text = chunk.get("text", "")
        
        part = f"Source: {source}\n{text}\n"
        
        if total_length + len(part) <= max_length:
            context_parts.append(part)
            total_length += len(part)
        else:
            # Truncate to fit
            remaining = max_length - total_length
            if remaining > 50:
                truncated = text[:remaining-50] + "..."
                context_parts.append(f"Source: {source}\n{truncated}\n")
            break
    
    return "\n".join(context_parts) if context_parts else "No relevant documents found."


def generate_response(user_query: str, persona_data: Dict) -> Dict:
    """
    Generate a persona-adaptive response using RAG.
    Optimized with single retrieval + optional expansion.
    """
    # Sanitize input
    query = sanitize_query(user_query)
    persona = persona_data.get("persona", "Business Executive")
    confidence = persona_data.get("confidence", 0.5)
    
    if not query:
        return {
            "persona": persona,
            "confidence": confidence,
            "retrieved_chunks": [],
            "response": "Please provide a valid query.",
            "insufficient_info": True,
            "retrieval_confidence": 0.0
        }
    
    # Retrieve chunks with adaptive expansion
    retrieved_chunks, retrieval_conf = retrieve_chunks_optimized(query, top_k=GENERATOR_CONFIG["top_k"])
    
    # Build context
    context = build_context(retrieved_chunks, max_length=GENERATOR_CONFIG["max_context_length"])
    
    # Check if we have good retrieval
    if not retrieved_chunks or retrieval_conf < 0.1:
        return {
            "persona": persona,
            "confidence": confidence,
            "retrieved_chunks": [],
            "response": "I could not find sufficient information in the support knowledge base. This issue should be escalated to a human support specialist.",
            "insufficient_info": True,
            "retrieval_confidence": retrieval_conf
        }
    
    # Get persona-specific prompt
    persona_prompt = get_persona_prompt(persona)
    
    # Build full prompt with grounding
    full_prompt = f"""
{persona_prompt}

{GROUNDING_PROMPT}

Knowledge Base Context:
{context}

User Query:
{query}

Remember:
- Answer ONLY from the context provided above
- If the answer is not in the context, say so
- Do not use external knowledge
- Match the tone to the {persona} persona
"""

    try:
        response = client.models.generate_content(
            model=GENERATOR_CONFIG["model"],
            contents=full_prompt
        )
        response_text = response.text
        
        # Check for insufficient info
        insufficient_keywords = [
            "could not find sufficient information",
            "not found in the knowledge base",
            "information is not available",
            "unable to find"
        ]
        
        insufficient_info = any(
            keyword in response_text.lower() 
            for keyword in insufficient_keywords
        )
        
        return {
            "persona": persona,
            "confidence": confidence,
            "retrieved_chunks": retrieved_chunks,
            "response": response_text,
            "insufficient_info": insufficient_info,
            "retrieval_confidence": retrieval_conf
        }
        
    except Exception as e:
        logger.error(f"Response generation error: {e}")
        return {
            "persona": persona,
            "confidence": confidence,
            "retrieved_chunks": retrieved_chunks,
            "response": f"I'm having trouble generating a response. Error: {str(e)}",
            "insufficient_info": True,
            "retrieval_confidence": retrieval_conf
        }


def generate_grounded_response(user_query: str, persona_data: Dict, 
                              context: str, retrieved_chunks: List[Dict]) -> str:
    """
    Generate a response with explicit grounding instructions.
    """
    persona = persona_data.get("persona", "Business Executive")
    query = sanitize_query(user_query)
    
    if not context or not retrieved_chunks:
        return "I could not find sufficient information to answer your question."
    
    prompt = f"""
{GROUNDING_PROMPT}

Persona: {persona}

Knowledge Base Context:
{context[:2000]}

User Query:
{query}

Instructions:
1. ONLY use information from the context above
2. If the answer isn't in the context, say so
3. Match your tone to the {persona} persona
4. Cite sources when possible
"""
    
    try:
        response = client.models.generate_content(
            model=GENERATOR_CONFIG["model"],
            contents=prompt
        )
        return response.text
    except Exception as e:
        logger.error(f"Grounded response generation error: {e}")
        return f"Error generating response: {str(e)}"


def clear_generator_cache() -> None:
    """
    Clear the query expansion cache.
    """
    expand_query_cached.cache_clear()
    logger.info("Generator cache cleared")


# ============================================
# TESTING
# ============================================

if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("GENERATOR TEST (UPDATED - google.genai)")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    test_queries = [
        "How do I reset my password?",
        "API authentication failure with error logs",
        "What is the refund policy?",
        "I'm frustrated and nothing works!",
        "Complex query: How does the billing work and what happens if payment fails?"
    ]
    
    test_personas = [
        {"persona": "Frustrated User", "confidence": 0.85},
        {"persona": "Technical Expert", "confidence": 0.90},
        {"persona": "Business Executive", "confidence": 0.75},
        {"persona": "Frustrated User", "confidence": 0.60},
        {"persona": "Technical Expert", "confidence": 0.80}
    ]
    
    for i, (query, persona_data) in enumerate(zip(test_queries, test_personas)):
        print(f"\n📝 Query {i+1}: {query}")
        print(f"   Persona: {persona_data['persona']}")
        
        start = time.time()
        result = generate_response(query, persona_data)
        elapsed = time.time() - start
        
        print(f"   Response: {result['response'][:150]}...")
        print(f"   Retrieved: {len(result['retrieved_chunks'])} chunks")
        print(f"   Retrieval Confidence: {result.get('retrieval_confidence', 0):.2%}")
        print(f"   Insufficient Info: {result.get('insufficient_info', False)}")
        print(f"   ⏱️  {elapsed:.2f}s")
        print("-" * 40)
    
    # Test cache performance
    print("\n" + "=" * 60)
    print("CACHE PERFORMANCE TEST")
    print("=" * 60)
    
    test_query = "How do I reset my password?"
    test_persona = {"persona": "Technical Expert", "confidence": 0.85}
    
    # Warm up cache
    generate_response(test_query, test_persona)
    
    start = time.time()
    for _ in range(3):
        generate_response(test_query, test_persona)
    cached_time = time.time() - start
    
    # Clear cache
    clear_generator_cache()
    
    start = time.time()
    for _ in range(3):
        generate_response(test_query, test_persona)
    uncached_time = time.time() - start
    
    print(f"Cached (3 calls): {cached_time:.3f}s")
    print(f"Uncached (3 calls): {uncached_time:.3f}s")
    if cached_time > 0 and uncached_time > 0:
        print(f"Speedup: {uncached_time/cached_time:.1f}x")
    else:
        print("Speedup: N/A (cache hit time too small)")
    
    # Test query complexity
    print("\n" + "=" * 60)
    print("QUERY COMPLEXITY ANALYSIS")
    print("=" * 60)
    
    sample_queries = [
        "Reset password",
        "How do I reset my password?",
        "I need to reset my password because I forgot it and can't login",
        "How does billing work when payment fails and what happens to my account?",
        "Can you explain the API authentication process, including the error handling and logging?"
    ]
    
    for q in sample_queries:
        complexity = calculate_query_complexity(q)
        print(f"'{q[:40]}...' → Complexity: {complexity}")