"""Configuration settings for PDF processing backend"""

import os
from typing import Dict, Any

class Config:
    """Application configuration"""
    
    # API Settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # Processing Settings
    DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 100 * 1024 * 1024))  # 100MB
    MAX_FILES_PER_BATCH = int(os.getenv("MAX_FILES_PER_BATCH", 10))
    
    # Memory Management
    MEMORY_OPTIMIZATION = os.getenv("MEMORY_OPTIMIZATION", "true").lower() == "true"
    MAX_CONCURRENT_PROCESSING = int(os.getenv("MAX_CONCURRENT", 3))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Temporary Files
    TEMP_DIR = os.getenv("TEMP_DIR", "/tmp")
    CLEANUP_TEMP_FILES = os.getenv("CLEANUP_TEMP", "true").lower() == "true"
    
    # Text Processing
    LANGUAGE_DETECTION = os.getenv("LANGUAGE_DETECTION", "true").lower() == "true"
    NLP_PREPROCESSING = os.getenv("NLP_PREPROCESSING", "true").lower() == "true"
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith('_') and not callable(getattr(cls, key))
        }