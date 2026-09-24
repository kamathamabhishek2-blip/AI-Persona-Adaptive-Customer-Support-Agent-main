# src/config.py

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from enum import Enum
from dotenv import load_dotenv

# ============================================
# ENUMS FOR TYPE SAFETY
# ============================================

class PersonaType(str, Enum):
    """Supported persona types."""
    TECHNICAL_EXPERT = "Technical Expert"
    FRUSTRATED_USER = "Frustrated User"
    BUSINESS_EXECUTIVE = "Business Executive"


class Environment(str, Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class ModelProvider(str, Enum):
    """Supported LLM providers."""
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


# ============================================
# BASE CONFIGURATION
# ============================================

@dataclass
class BaseConfig:
    """Base configuration with common validation."""
    
    def validate(self) -> None:
        """Validate configuration values. Override in subclasses."""
        pass
    
    def to_dict(self, mask_secrets: bool = True) -> Dict[str, Any]:
        """Convert to dictionary with optional secret masking."""
        result = {}
        for key, value in self.__dict__.items():
            if mask_secrets and "key" in key.lower() and value:
                result[key] = "***MASKED***"
            elif "password" in key.lower() and value:
                result[key] = "***MASKED***"
            else:
                result[key] = value
        return result


# ============================================
# CONFIGURATION DATACLASSES
# ============================================

@dataclass
class PersonaConfig(BaseConfig):
    """Configuration for persona detection."""
    
    # Confidence thresholds for each persona
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "Technical Expert": 0.70,
        "Frustrated User": 0.60,
        "Business Executive": 0.65
    })
    
    # Default persona when classification fails
    fallback_persona: str = "Business Executive"
    
    # Minimum confidence for classification to be valid
    min_confidence: float = 0.50
    
    # Maximum query length for classification
    max_query_length: int = 2000
    
    # Persona-specific keywords for rule-based fallback
    keyword_weights: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "Technical Expert": {
            "api": 2, "sdk": 2, "endpoint": 2, "authentication": 2,
            "log": 2, "logs": 2, "debug": 2, "debugging": 2,
            "config": 2, "configuration": 2, "error": 1, "exception": 1,
            "integration": 2, "pipeline": 1, "deployment": 1,
            "root cause": 3, "stack trace": 3, "database": 1,
            "code": 1, "syntax": 1, "compile": 1, "server": 1
        },
        "Frustrated User": {
            "angry": 3, "frustrated": 3, "annoyed": 2,
            "terrible": 2, "worst": 2, "useless": 2,
            "tired": 2, "stupid": 2, "awful": 2,
            "never works": 3, "waste": 2, "unacceptable": 3,
            "horrible": 2, "disappointed": 2, "upset": 2,
            "nothing works": 3, "still broken": 3, "again": 1
        },
        "Business Executive": {
            "impact": 3, "business": 3, "revenue": 3,
            "operations": 3, "timeline": 3, "roi": 3,
            "stakeholder": 3, "budget": 2, "deadline": 2,
            "customers": 2, "sales": 2, "production": 2,
            "downtime": 3, "sla": 3, "priority": 2,
            "cost": 2, "efficiency": 2, "strategy": 2
        }
    })
    
    # Minimum keyword score for classification
    min_keyword_score: int = 2
    
    def validate(self) -> None:
        """Validate persona configuration."""
        for persona, threshold in self.thresholds.items():
            if not isinstance(persona, str):
                raise ValueError(f"Persona key must be string, got {type(persona)}")
            if not 0 <= threshold <= 1:
                raise ValueError(f"Invalid threshold for {persona}: {threshold}")
        
        if self.fallback_persona not in self.thresholds:
            raise ValueError(f"Fallback persona '{self.fallback_persona}' not in thresholds")
        
        if not 0 <= self.min_confidence <= 1:
            raise ValueError(f"Invalid min_confidence: {self.min_confidence}")


@dataclass
class RAGConfig(BaseConfig):
    """Configuration for RAG pipeline."""
    
    chunk_size: int = 500
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])
    
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    embedding_batch_size: int = 32
    
    vector_db_path: str = "./chroma_db"
    collection_name: str = "support_kb"
    
    top_k: int = 3
    min_similarity_score: float = 0.3
    max_retrieval_distance: float = 1.0
    
    enable_query_expansion: bool = True
    expansion_variations: int = 4
    
    supported_extensions: List[str] = field(default_factory=lambda: [".txt", ".md", ".pdf"])
    data_folder: str = "data"
    max_file_size_mb: int = 50
    
    def validate(self) -> None:
        if self.chunk_size < self.min_chunk_size:
            raise ValueError(f"Chunk size {self.chunk_size} < minimum {self.min_chunk_size}")
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"Chunk overlap {self.chunk_overlap} >= chunk size {self.chunk_size}")
        
        if self.top_k < 1:
            raise ValueError(f"Invalid top_k: {self.top_k}")
        
        if self.vector_db_path and not os.path.exists(self.vector_db_path):
            try:
                os.makedirs(self.vector_db_path, exist_ok=True)
            except Exception as e:
                raise ValueError(f"Cannot create vector DB path: {e}")
        
        if self.data_folder and not os.path.exists(self.data_folder):
            try:
                os.makedirs(self.data_folder, exist_ok=True)
            except Exception as e:
                raise ValueError(f"Cannot create data folder: {e}")


@dataclass
class EscalationConfig(BaseConfig):
    """Configuration for escalation logic."""
    
    min_persona_confidence: float = 0.65
    min_retrieval_confidence: float = 0.50
    min_response_confidence: float = 0.60
    
    max_turns_before_escalation: int = 3
    max_escalation_attempts: int = 2
    
    frustration_threshold: int = 3
    frustration_keywords: List[str] = field(default_factory=lambda: [
        "angry", "frustrated", "annoyed", "terrible", "worst",
        "useless", "tired", "stupid", "awful", "horrible",
        "unacceptable", "disappointed", "upset", "nothing works",
        "waste", "never works", "still broken"
    ])
    
    sensitive_keywords: List[str] = field(default_factory=lambda: [
        "refund", "billing", "payment", "legal", "lawsuit",
        "cancel", "terminate", "compliance", "gdpr", "privacy",
        "security breach", "account lock", "fraud", "unauthorized",
        "chargeback", "dispute", "violation", "policy"
    ])
    
    enable_llm_escalation: bool = True
    
    force_escalation_keywords: List[str] = field(default_factory=lambda: [
        "emergency", "urgent", "security breach", "legal action",
        "lawyer", "attorney", "lawsuit"
    ])
    
    def validate(self) -> None:
        if not 0 <= self.min_persona_confidence <= 1:
            raise ValueError(f"Invalid min_persona_confidence: {self.min_persona_confidence}")
        
        if self.max_turns_before_escalation < 1:
            raise ValueError(f"Invalid max_turns_before_escalation: {self.max_turns_before_escalation}")


@dataclass
class LLMConfig(BaseConfig):
    """Configuration for LLM integration."""
    
    provider: ModelProvider = ModelProvider.GEMINI
    model_name: str = "gemini-2.5-flash-lite"
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95
    
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    api_base: Optional[str] = None
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: int = 1
    max_retry_delay: int = 10
    
    max_response_length: int = 2000
    min_response_length: int = 50
    response_timeout: int = 60
    
    enable_caching: bool = True
    cache_max_size: int = 128
    cache_ttl_seconds: int = 3600
    
    def validate(self) -> None:
        if self.provider in [ModelProvider.GEMINI, ModelProvider.OPENAI]:
            if not self.api_key or len(self.api_key) < 10:
                raise ValueError(f"Invalid API key for {self.provider.value}")
        
        if not 0 <= self.temperature <= 1:
            raise ValueError(f"Invalid temperature: {self.temperature}")
        
        if self.max_tokens < 1:
            raise ValueError(f"Invalid max_tokens: {self.max_tokens}")


@dataclass
class UIConfig(BaseConfig):
    """Configuration for the user interface."""
    
    page_title: str = "Persona Support Agent"
    page_icon: str = "🤖"
    layout: str = "wide"
    initial_sidebar_state: str = "auto"
    
    show_sources: bool = True
    show_confidence: bool = True
    show_escalation_status: bool = True
    show_retrieved_chunks: bool = True
    max_conversation_turns: int = 20
    max_display_text_length: int = 500
    
    primary_color: str = "#FF4B4B"
    background_color: str = "#FFFFFF"
    secondary_background_color: str = "#F0F2F6"
    text_color: str = "#262730"
    font: str = "sans serif"
    
    def validate(self) -> None:
        if self.max_conversation_turns < 1:
            raise ValueError(f"Invalid max_conversation_turns: {self.max_conversation_turns}")


@dataclass
class LoggingConfig(BaseConfig):
    """Configuration for logging."""
    
    level: str = "INFO"
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    
    log_file: str = "logs/app.log"
    max_file_size: int = 10_485_760  # 10MB
    backup_count: int = 5
    
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    def validate(self) -> None:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        for level_attr in ["level", "console_level", "file_level"]:
            level = getattr(self, level_attr, "")
            if level and level.upper() not in valid_levels:
                raise ValueError(f"Invalid log level for {level_attr}: {level}")
        
        if self.log_file:
            log_dir = os.path.dirname(self.log_file)
            if log_dir and not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir, exist_ok=True)
                except Exception as e:
                    raise ValueError(f"Cannot create log directory: {e}")


# ============================================
# MAIN APPLICATION CONFIGURATION
# ============================================

@dataclass
class AppConfig(BaseConfig):
    """Main application configuration."""
    
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    
    persona: PersonaConfig = field(default_factory=PersonaConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    escalation: EscalationConfig = field(default_factory=EscalationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    app_name: str = "Persona Support Agent"
    app_version: str = "1.0.0"
    author: str = "Adsparkx AI"
    description: str = "AI-powered persona-adaptive customer support agent"
    
    def __post_init__(self):
        self._load_environment_config()
        self.validate()
    
    def _load_environment_config(self) -> None:
        env = os.getenv("ENVIRONMENT", "development").lower()
        
        if env == "production":
            self.environment = Environment.PRODUCTION
            self.debug = False
            self.logging.level = "WARNING"
            self.logging.console_level = "WARNING"
            self.llm.temperature = 0.3
            self.llm.enable_caching = True
            self.escalation.enable_llm_escalation = True
            self.rag.enable_query_expansion = True
            
        elif env == "testing":
            self.environment = Environment.TESTING
            self.debug = True
            self.logging.level = "DEBUG"
            self.rag.top_k = 5
            self.rag.chunk_size = 300
            self.llm.enable_caching = False
            self.escalation.enable_llm_escalation = True
            
        else:
            self.environment = Environment.DEVELOPMENT
            self.debug = True
            self.logging.level = "INFO"
            self.llm.temperature = 0.7
            self.llm.enable_caching = True
            self.escalation.enable_llm_escalation = False
            self.rag.enable_query_expansion = True
    
    def validate(self) -> None:
        if not isinstance(self.environment, Environment):
            raise ValueError(f"Invalid environment: {self.environment}")
        
        try:
            self.persona.validate()
            self.rag.validate()
            self.escalation.validate()
            self.llm.validate()
            self.ui.validate()
            self.logging.validate()
        except Exception as e:
            raise ValueError(f"Config validation failed: {e}")
    
    def to_dict(self, mask_secrets: bool = True) -> Dict[str, Any]:
        return {
            "environment": self.environment.value,
            "debug": self.debug,
            "app_name": self.app_name,
            "app_version": self.app_version,
            "author": self.author,
            "persona": self.persona.to_dict(mask_secrets),
            "rag": self.rag.to_dict(mask_secrets),
            "escalation": self.escalation.to_dict(mask_secrets),
            "llm": self.llm.to_dict(mask_secrets),
            "ui": self.ui.to_dict(mask_secrets),
            "logging": self.logging.to_dict(mask_secrets),
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        return {
            "model": self.llm.model_name,
            "temperature": self.llm.temperature,
            "max_tokens": self.llm.max_tokens,
            "top_p": self.llm.top_p,
            "timeout": self.llm.timeout,
            "retry_attempts": self.llm.retry_attempts,
        }
    
    def get_rag_config(self) -> Dict[str, Any]:
        return {
            "chunk_size": self.rag.chunk_size,
            "chunk_overlap": self.rag.chunk_overlap,
            "top_k": self.rag.top_k,
            "min_similarity": self.rag.min_similarity_score,
            "embedding_model": self.rag.embedding_model,
            "enable_expansion": self.rag.enable_query_expansion,
        }
    
    def get_escalation_config(self) -> Dict[str, Any]:
        return {
            "min_persona_conf": self.escalation.min_persona_confidence,
            "min_retrieval_conf": self.escalation.min_retrieval_confidence,
            "max_turns": self.escalation.max_turns_before_escalation,
            "sensitive_keywords": self.escalation.sensitive_keywords,
            "enable_llm": self.escalation.enable_llm_escalation,
        }


# ============================================
# CONFIG MANAGER (SINGLETON)
# ============================================

class ConfigManager:
    """Singleton configuration manager."""
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[AppConfig] = None
    _loaded: bool = False
    
    def __new__(cls) -> 'ConfigManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._loaded:
            self._load_config()
    
    def _load_config(self) -> None:
        try:
            load_dotenv()
            self._config = AppConfig()
            self._loaded = True
            logging.info(f"✅ Config loaded: {self._config.environment.value}")
        except Exception as e:
            logging.error(f"❌ Failed to load config: {e}")
            raise
    
    @property
    def config(self) -> AppConfig:
        if not self._loaded:
            self._load_config()
        return self._config
    
    def reload(self) -> None:
        self._loaded = False
        self._load_config()
    
    def get(self) -> AppConfig:
        return self.config


# ============================================
# GLOBAL CONFIG INSTANCE
# ============================================

_config_manager = ConfigManager()
config = _config_manager.config

# ============================================
# CONFIG HELPERS
# ============================================

def get_config_section(section: str) -> Dict[str, Any]:
    sections = {
        "persona": config.persona,
        "rag": config.rag,
        "escalation": config.escalation,
        "llm": config.llm,
        "ui": config.ui,
        "logging": config.logging,
    }
    
    section_obj = sections.get(section)
    if section_obj:
        return section_obj.to_dict(mask_secrets=True)
    
    return {}


def update_config(section: str, **kwargs: Any) -> None:
    sections = {
        "persona": config.persona,
        "rag": config.rag,
        "escalation": config.escalation,
        "llm": config.llm,
        "ui": config.ui,
        "logging": config.logging,
    }
    
    section_obj = sections.get(section)
    if not section_obj:
        raise ValueError(f"Unknown section: {section}")
    
    for key, value in kwargs.items():
        if hasattr(section_obj, key):
            setattr(section_obj, key, value)
        else:
            raise ValueError(f"Unknown config key: {key}")


def print_config(mask_secrets: bool = True) -> None:
    print(json.dumps(config.to_dict(mask_secrets=mask_secrets), indent=2))


def reload_config() -> None:
    global config, _config_manager
    _config_manager.reload()
    config = _config_manager.config
    print("✅ Configuration reloaded")


def validate_environment() -> bool:
    try:
        config.validate()
        return True
    except Exception as e:
        logging.error(f"Environment validation failed: {e}")
        return False


def get_environment_info() -> Dict[str, Any]:
    return {
        "environment": config.environment.value,
        "debug": config.debug,
        "app_version": config.app_version,
        "model": config.llm.model_name,
        "chunk_size": config.rag.chunk_size,
        "top_k": config.rag.top_k,
        "escalation_enabled": config.escalation.enable_llm_escalation,
        "api_key_set": bool(config.llm.api_key),
    }


# ============================================
# TESTING
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("CONFIGURATION TEST (FIXED)")
    print("=" * 60)
    
    print(f"\nEnvironment: {config.environment.value}")
    print(f"Debug Mode: {config.debug}")
    print(f"App Version: {config.app_version}")
    
    print("\nPersona Thresholds:")
    for persona, threshold in config.persona.thresholds.items():
        print(f"  {persona}: {threshold}")
    
    print(f"\nRAG Configuration:")
    print(f"  Chunk Size: {config.rag.chunk_size}")
    print(f"  Top-K: {config.rag.top_k}")
    print(f"  Embedding Model: {config.rag.embedding_model}")
    print(f"  Data Folder: {config.rag.data_folder}")
    
    print(f"\nEscalation Configuration:")
    print(f"  Min Confidence: {config.escalation.min_persona_confidence}")
    print(f"  Max Turns: {config.escalation.max_turns_before_escalation}")
    
    print(f"\nLLM Configuration:")
    print(f"  Model: {config.llm.model_name}")
    print(f"  Temperature: {config.llm.temperature}")
    print(f"  Cache Enabled: {config.llm.enable_caching}")
    print(f"  API Key Set: {'✅' if config.llm.api_key else '❌'}")
    
    print(f"\nLogging Configuration:")
    print(f"  Level: {config.logging.level}")
    print(f"  Format: {config.logging.format}")
    print(f"  Date Format: {config.logging.date_format}")
    
    print("\n" + "=" * 60)
    print("✅ Configuration loaded successfully")