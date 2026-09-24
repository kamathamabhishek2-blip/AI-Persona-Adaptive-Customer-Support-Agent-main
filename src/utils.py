"""
Utility functions for the Persona-Adaptive Customer Support Agent.

This module provides common utilities for:
- Logging configuration
- Text processing
- File operations
- JSON handling
- Cryptography
- Time management
- Validation
- Error handling
- Metrics collection
- Streamlit UI helpers
- Conversation management
"""

import os
import json
import logging
import hashlib
import re
import threading
from functools import lru_cache
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from collections import Counter

import streamlit as st

from src.config import config

# ============================================
# CONSTANTS
# ============================================

# Precompiled regex patterns for performance
_WHITESPACE_PATTERN = re.compile(r'\s+')
_SPECIAL_CHARS_PATTERN = re.compile(r'[^\w\s.,!?\'"()-]')
_EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
_PHONE_PATTERN = re.compile(r'[\s\-\(\)]')
_WORD_PATTERN = re.compile(r'\b[a-zA-Z]+\b')

# Stop words for keyword extraction
_STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
    'for', 'of', 'with', 'without', 'by', 'from', 'up', 'down',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
}

# ============================================
# LOGGING UTILITIES (Thread-Safe)
# ============================================

_logger_lock = threading.Lock()
_logger_initialized = False


def setup_logging(name: str = "persona_support") -> logging.Logger:
    """
    Set up a logger with the specified name.
    Thread-safe with duplicate handler prevention.
    """
    global _logger_initialized
    
    with _logger_lock:
        logger = logging.getLogger(name)
        
        # Only add handlers if not already configured
        if not _logger_initialized:
            logger.setLevel(config.logging.level)
            
            # Remove existing handlers to prevent duplication
            if logger.handlers:
                logger.handlers.clear()
            
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(config.logging.console_level)
            
            # Create file handler with rotation support
            log_dir = os.path.dirname(config.logging.log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            
            try:
                from logging.handlers import RotatingFileHandler
                file_handler = RotatingFileHandler(
                    config.logging.log_file,
                    maxBytes=config.logging.max_file_size,
                    backupCount=config.logging.backup_count
                )
            except ImportError:
                file_handler = logging.FileHandler(config.logging.log_file)
            
            file_handler.setLevel(config.logging.file_level)
            
            # Create formatter - using config.logging.format and config.logging.date_format
            formatter = logging.Formatter(
                config.logging.format,
                config.logging.date_format
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            # Add handlers
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
            
            # Prevent propagation to root logger
            logger.propagate = False
            
            _logger_initialized = True
    
    return logger


# Global logger instance
logger = setup_logging()


# ============================================
# TEXT PROCESSING UTILITIES (Cached)
# ============================================

@lru_cache(maxsize=1024)
def clean_text_cached(text: str) -> str:
    """Cached version of clean_text."""
    return clean_text(text)


def clean_text(text: str) -> str:
    """
    Clean and normalize text with optimized regex.
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = _WHITESPACE_PATTERN.sub(' ', text)
    
    # Remove special characters but keep basic punctuation
    text = _SPECIAL_CHARS_PATTERN.sub('', text)
    
    return text.strip()


@lru_cache(maxsize=1024)
def truncate_text_cached(text: str, max_length: int = 500) -> str:
    """Cached version of truncate_text."""
    return truncate_text(text, max_length)


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length, preserving sentence boundaries.
    """
    if not text or len(text) <= max_length:
        return text
    
    truncated = text[:max_length]
    last_period = truncated.rfind('.')
    last_space = truncated.rfind(' ')
    
    cut_point = max(last_period, last_space)
    if cut_point > max_length // 2:
        return truncated[:cut_point + 1] + suffix
    
    return truncated + suffix


@lru_cache(maxsize=512)
def extract_keywords_cached(text: str, max_keywords: int = 10) -> Tuple[str, ...]:
    """Cached version of extract_keywords."""
    return tuple(extract_keywords(text, max_keywords))


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text using frequency-based approach.
    """
    if not text:
        return []
    
    words = _WORD_PATTERN.findall(text.lower())
    words = [w for w in words if w not in _STOP_WORDS and len(w) > 2]
    
    if not words:
        return []
    
    word_counts = Counter(words)
    return [word for word, _ in word_counts.most_common(max_keywords)]


# ============================================
# FILE UTILITIES (Safe)
# ============================================

def sanitize_path(path: str) -> str:
    """
    Sanitize file path to prevent directory traversal.
    """
    if not path:
        return ""
    
    normalized = os.path.normpath(path)
    
    if '..' in normalized.split(os.sep):
        raise ValueError("Directory traversal detected in path")
    
    return normalized


def ensure_directory(path: str) -> bool:
    """
    Ensure a directory exists, create if it doesn't.
    """
    try:
        safe_path = sanitize_path(path)
        os.makedirs(safe_path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        return False


def get_file_size(file_path: str) -> str:
    """
    Get human-readable file size.
    """
    try:
        safe_path = sanitize_path(file_path)
        size = os.path.getsize(safe_path)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    except Exception:
        return "Unknown"


def read_file(file_path: str) -> Optional[str]:
    """
    Read a file and return its content with safe path handling.
    """
    try:
        safe_path = sanitize_path(file_path)
        with open(safe_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        return None


def write_file(file_path: str, content: str) -> bool:
    """
    Write content to a file with safe path handling.
    """
    try:
        safe_path = sanitize_path(file_path)
        ensure_directory(os.path.dirname(safe_path))
        with open(safe_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        logger.error(f"Failed to write file {file_path}: {e}")
        return False


# ============================================
# JSON UTILITIES
# ============================================

def safe_json_loads(json_str: str) -> Optional[Dict]:
    """
    Safely load JSON from a string.
    """
    if not json_str:
        return None
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.debug(f"JSON decode error: {e}")
        return None


def safe_json_dumps(data: Any, indent: int = 2) -> str:
    """
    Safely dump JSON with error handling and sensitive data masking.
    """
    if data is None:
        return "{}"
    
    sensitive_keys = {'api_key', 'password', 'secret', 'token', 'key'}
    
    def mask_sensitive(obj):
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                if k.lower() in sensitive_keys and v:
                    result[k] = "***MASKED***"
                else:
                    result[k] = mask_sensitive(v)
            return result
        elif isinstance(obj, list):
            return [mask_sensitive(item) for item in obj]
        else:
            return obj
    
    try:
        masked = mask_sensitive(data)
        return json.dumps(masked, indent=indent, default=str)
    except Exception as e:
        logger.error(f"JSON dump error: {e}")
        return json.dumps({"error": "Failed to serialize data"})


def merge_json(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries.
    """
    if not base:
        return override.copy() if override else {}
    if not override:
        return base.copy()
    
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_json(result[key], value)
        else:
            result[key] = value
    return result


# ============================================
# CRYPTOGRAPHY UTILITIES
# ============================================

@lru_cache(maxsize=1024)
def generate_hash_cached(text: str, algorithm: str = "sha256") -> str:
    """Cached version of generate_hash."""
    return generate_hash(text, algorithm)


def generate_hash(text: str, algorithm: str = "sha256") -> str:
    """
    Generate a hash of the text.
    """
    if not text:
        return ""
    
    try:
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(text.encode('utf-8'))
        return hash_obj.hexdigest()
    except ValueError:
        hash_obj = hashlib.sha256()
        hash_obj.update(text.encode('utf-8'))
        return hash_obj.hexdigest()


def generate_id(text: str) -> str:
    """
    Generate a short ID from text.
    """
    return generate_hash(text, "md5")[:12]


# ============================================
# TIME UTILITIES
# ============================================

def get_timestamp() -> str:
    """
    Get current timestamp in ISO format with UTC timezone.
    """
    return datetime.now(timezone.utc).isoformat()


@lru_cache(maxsize=1024)
def format_timestamp_cached(timestamp: str) -> str:
    """Cached version of format_timestamp."""
    return format_timestamp(timestamp)


def format_timestamp(timestamp: str) -> str:
    """
    Format a timestamp for display.
    """
    if not timestamp:
        return ""
    
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return timestamp


@lru_cache(maxsize=1024)
def time_ago_cached(timestamp: str) -> str:
    """Cached version of time_ago."""
    return time_ago(timestamp)


def time_ago(timestamp: str) -> str:
    """
    Get human-readable time difference.
    """
    if not timestamp:
        return "unknown time"
    
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        diff = now - dt
        
        seconds = diff.total_seconds()
        
        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif seconds < 86400:
            hours = int(seconds // 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif seconds < 604800:
            days = int(seconds // 86400)
            return f"{days} day{'s' if days != 1 else ''} ago"
        else:
            weeks = int(seconds // 604800)
            return f"{weeks} week{'s' if weeks != 1 else ''} ago"
    except (ValueError, TypeError):
        return "unknown time"


# ============================================
# VALIDATION UTILITIES
# ============================================

def validate_email(email: str) -> bool:
    """
    Validate email address format.
    """
    if not email:
        return False
    return bool(_EMAIL_PATTERN.match(email.strip()))


def validate_phone(phone: str) -> bool:
    """
    Validate phone number format.
    """
    if not phone:
        return False
    
    cleaned = _PHONE_PATTERN.sub('', phone)
    return len(cleaned) >= 10 and len(cleaned) <= 15 and cleaned.isdigit()


def validate_query(query: str) -> bool:
    """
    Validate user query.
    """
    if not query:
        return False
    
    cleaned = query.strip()
    if len(cleaned) < 2:
        return False
    
    if len(cleaned) > 5000:
        return False
    
    return True


# ============================================
# ERROR HANDLING UTILITIES
# ============================================

class AppError(Exception):
    """Custom application error."""
    def __init__(self, message: str, code: int = 500, details: Optional[Dict] = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict:
        return {
            "error": self.message,
            "code": self.code,
            "details": self.details
        }


def handle_error(error: Exception, context: str = "") -> Dict:
    """
    Handle and format errors for display.
    """
    logger.error(f"Error in {context}: {error}")
    
    if isinstance(error, AppError):
        return {
            "success": False,
            "error": error.message,
            "code": error.code,
            "details": error.details
        }
    
    error_msg = str(error)
    if config.environment.value != "development":
        error_msg = "An internal error occurred. Please try again."
    
    return {
        "success": False,
        "error": error_msg,
        "code": 500,
        "details": {"context": context}
    }


# ============================================
# METRICS COLLECTOR (Thread-Safe)
# ============================================

class MetricsCollector:
    """
    Thread-safe metrics collector for monitoring.
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._metrics = {
            "total_queries": 0,
            "persona_counts": {},
            "escalation_counts": 0,
            "avg_confidence": 0.0,
            "total_confidence": 0.0,
            "total_tokens_used": 0,
            "avg_latency": 0.0,
            "total_latency": 0.0,
        }
    
    def record_query(self, persona: str, confidence: float, escalated: bool,
                     tokens: Optional[int] = None, latency: Optional[float] = None):
        """Record a query and its metrics (thread-safe)."""
        with self._lock:
            self._metrics["total_queries"] += 1
            
            self._metrics["persona_counts"][persona] = \
                self._metrics["persona_counts"].get(persona, 0) + 1
            
            if escalated:
                self._metrics["escalation_counts"] += 1
            
            self._metrics["total_confidence"] += confidence
            self._metrics["avg_confidence"] = \
                self._metrics["total_confidence"] / self._metrics["total_queries"]
            
            if tokens:
                self._metrics["total_tokens_used"] += tokens
            
            if latency:
                self._metrics["total_latency"] += latency
                self._metrics["avg_latency"] = \
                    self._metrics["total_latency"] / self._metrics["total_queries"]
    
    def get_metrics(self) -> Dict:
        """Get current metrics (thread-safe)."""
        with self._lock:
            metrics = self._metrics.copy()
            if metrics["total_queries"] > 0:
                metrics["escalation_rate"] = \
                    metrics["escalation_counts"] / metrics["total_queries"]
            return metrics
    
    def reset(self) -> None:
        """Reset all metrics (thread-safe)."""
        with self._lock:
            self._metrics = {
                "total_queries": 0,
                "persona_counts": {},
                "escalation_counts": 0,
                "avg_confidence": 0.0,
                "total_confidence": 0.0,
                "total_tokens_used": 0,
                "avg_latency": 0.0,
                "total_latency": 0.0,
            }


# Global metrics instance
metrics = MetricsCollector()


# ============================================
# STREAMLIT UI UTILITIES (Cached)
# ============================================

@st.cache_data(ttl=60)
def get_cached_sources(sources: tuple) -> List[Dict]:
    """Cache sources for display."""
    return list(sources)


def display_metric(label: str, value: Any, delta: Optional[Any] = None):
    """Display a metric with consistent styling."""
    st.metric(label=label, value=value, delta=delta)


def display_sources(sources: List[Dict], max_preview_length: int = 300):
    """Display retrieved sources in a consistent format."""
    if not sources:
        st.info("No sources retrieved.")
        return
    
    with st.expander(f"📚 Retrieved Sources ({len(sources)})"):
        for idx, source in enumerate(sources, 1):
            st.write(f"**{idx}. Source: {source.get('source', 'Unknown')}**")
            
            score = source.get('score', 0)
            st.caption(f"Relevance Score: {score:.2%}")
            
            if source.get('page'):
                st.caption(f"Page: {source['page']}")
            
            text = source.get('text', '')
            preview = truncate_text_cached(text, max_preview_length)
            st.write(preview)
            st.write("---")


def display_response(response: str, persona: str, confidence: float):
    """Display the AI response with metadata."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(response)
    
    with col2:
        st.caption(f"🎭 Persona: {persona}")
        st.caption(f"Confidence: {confidence:.1%}")


# ============================================
# CONVERSATION UTILITIES
# ============================================

def format_conversation(conversation: List[Dict]) -> str:
    """Format conversation history as a string."""
    if not conversation:
        return "No conversation history"
    
    formatted = []
    for turn in conversation:
        user = turn.get('user', '')
        assistant = turn.get('assistant', '')
        persona = turn.get('persona', 'Unknown')
        
        formatted.append(f"User: {user}")
        formatted.append(f"Assistant: {assistant[:200]}..." if len(assistant) > 200 else f"Assistant: {assistant}")
        formatted.append(f"Persona: {persona}")
        formatted.append("---")
    
    return "\n".join(formatted)


def get_conversation_summary(conversation: List[Dict]) -> Dict:
    """Generate a summary of the conversation."""
    if not conversation:
        return {"turns": 0, "personas": [], "topics": []}
    
    turns = len(conversation)
    personas = list(set(turn.get('persona', 'Unknown') for turn in conversation))
    
    topics = []
    for turn in conversation:
        user_msg = turn.get('user', '')
        if user_msg:
            keywords = extract_keywords_cached(user_msg, 3)
            topics.extend(keywords)
    
    return {
        "turns": turns,
        "personas": personas,
        "topics": list(set(topics))[:10]
    }


# ============================================
# TESTING
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("UTILITIES TEST (FIXED)")
    print("=" * 60)
    
    # Test text utilities
    print("\n📝 Text Utilities:")
    text = "This is a test sentence with some extra   whitespace."
    print(f"Cleaned: {clean_text(text)}")
    print(f"Truncated: {truncate_text(text, 30)}")
    print(f"Keywords: {extract_keywords(text)}")
    
    # Test file utilities
    print("\n📁 File Utilities:")
    print(f"Current Directory: {os.getcwd()}")
    ensure_directory("test_dir")
    print("Directory created: test_dir")
    
    # Test hash utilities
    print("\n🔐 Hash Utilities:")
    test_str = "Hello World"
    print(f"MD5: {generate_hash(test_str, 'md5')}")
    print(f"SHA256: {generate_hash(test_str)}")
    print(f"Short ID: {generate_id(test_str)}")
    
    # Test time utilities
    print("\n⏰ Time Utilities:")
    timestamp = get_timestamp()
    print(f"Timestamp: {timestamp}")
    print(f"Formatted: {format_timestamp(timestamp)}")
    print(f"Time ago: {time_ago(timestamp)}")
    
    # Test validation
    print("\n✅ Validation Utilities:")
    print(f"Email valid: {validate_email('test@example.com')}")
    print(f"Phone valid: {validate_phone('+1-234-567-8900')}")
    print(f"Query valid: {validate_query('How do I reset my password?')}")
    
    # Test metrics
    print("\n📊 Metrics (Thread-Safe):")
    metrics.record_query("Technical Expert", 0.85, False, tokens=150, latency=2.3)
    metrics.record_query("Frustrated User", 0.75, True, tokens=200, latency=3.1)
    metrics.record_query("Business Executive", 0.90, False, tokens=120, latency=1.8)
    print(json.dumps(metrics.get_metrics(), indent=2))
    
    # Clean up
    import shutil
    if os.path.exists("test_dir"):
        shutil.rmtree("test_dir")
    
    print("\n✅ All utilities tested successfully")