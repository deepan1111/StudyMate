
import os
import logging
from typing import List
from dataclasses import dataclass

@dataclass
class SecurityConfig:
    cors_origins: List[str]
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    max_files_per_batch: int = 10

@dataclass  
class ProcessingConfig:
    default_chunk_size: int = 1000
    default_chunk_overlap: int = 200
    max_context_length: int = 3000
    max_concurrent_processing: int = 3
    memory_optimization: bool = True
    language_detection: bool = True
    nlp_preprocessing: bool = True

@dataclass
class SearchConfig:
    semantic_model: str = 'all-MiniLM-L6-v2'
    index_path: str = 'data/faiss_index.bin'
    metadata_path: str = 'data/chunk_metadata.json'
    score_threshold: float = 0.1

@dataclass
class APIConfig:
    openrouter_api_key: str = 'sk-test-key'  # Dummy key for testing
    openrouter_url: str = 'https://openrouter.ai/api/v1/chat/completions'
    hf_api_key: str = None
    working_models: List[str] = None

class SimpleConfig:
    def __init__(self):
        self.api = APIConfig(
            working_models=['microsoft/wizardlm-2-8x22b']
        )
        
        self.processing = ProcessingConfig()
        
        self.search = SearchConfig()
        
        self.security = SecurityConfig(
            cors_origins=[
                "http://localhost:5173",
                "http://localhost:3000", 
                "http://127.0.0.1:5173",
                "http://127.0.0.1:3000",
                "*"  # Allow all for testing
            ]
        )
        
        self.app = {
            'host': 'localhost',
            'port': 8000,
            'debug': True,
            'temp_dir': '/tmp',
            'data_dir': 'data',
            'cleanup_temp_files': True,
            'log_level': 'INFO',
            'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    
    def setup_logging(self):
        logging.basicConfig(
            level=getattr(logging, self.app['log_level']),
            format=self.app['log_format']
        )
    
    def to_dict(self):
        return {
            'processing': self.processing.__dict__,
            'search': self.search.__dict__,
            'security': {'cors_origins': self.security.cors_origins},
            'app': self.app
        }

# Use this instead of the complex config for testing
config = SimpleConfig()