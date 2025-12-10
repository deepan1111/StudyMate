# """
# Centralized configuration management for StudyMate backend
# """
# import os
# import logging
# from typing import List, Dict, Any, Optional
# from dataclasses import dataclass
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# @dataclass
# class SecurityConfig:
#     """Security-related configuration"""
#     cors_origins: List[str]
#     max_file_size: int
#     max_files_per_batch: int
#     api_key_validation: bool = True

# @dataclass  
# class ProcessingConfig:
#     """Text processing configuration"""
#     default_chunk_size: int
#     default_chunk_overlap: int
#     max_context_length: int
#     max_concurrent_processing: int
#     memory_optimization: bool
#     language_detection: bool
#     nlp_preprocessing: bool

# @dataclass
# class SearchConfig:
#     """Search engine configuration"""
#     semantic_model: str
#     index_path: str
#     metadata_path: str
#     score_threshold: float = 0.1

# @dataclass
# class APIConfig:
#     """External API configuration"""
#     openrouter_api_key: str
#     openrouter_url: str
#     hf_api_key: Optional[str]
#     working_models: List[str]

# class Config:
#     """Main configuration manager with validation"""
    
#     def __init__(self):
#         self._validate_environment()
#         self._load_configurations()
    
#     def _validate_environment(self) -> None:
#         """Validate required environment variables"""
#         required_vars = [
#             'OPENROUTER_API_KEY'
#         ]
        
#         missing_vars = []
#         for var in required_vars:
#             if not os.getenv(var):
#                 missing_vars.append(var)
        
#         if missing_vars:
#             raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
#     def _load_configurations(self) -> None:
#         """Load all configuration sections"""
#         self.api = self._load_api_config()
#         self.processing = self._load_processing_config()
#         self.search = self._load_search_config()
#         self.security = self._load_security_config()
#         self.app = self._load_app_config()
    
#     def _load_api_config(self) -> APIConfig:
#         """Load API configuration"""
#         api_key = os.getenv('OPENROUTER_API_KEY')
#         if not api_key or not api_key.startswith('sk-'):
#             raise ValueError("Invalid OPENROUTER_API_KEY format")
        
#         models_str = os.getenv('WORKING_MODELS', '')
#         models = [m.strip() for m in models_str.split(',') if m.strip()] if models_str else [
#             "microsoft/wizardlm-2-8x22b",
#             "microsoft/wizardlm-2-7b",
#             "huggingfaceh4/zephyr-7b-beta",
#             "openchat/openchat-7b"
#         ]
        
#         return APIConfig(
#             openrouter_api_key=api_key,
#             openrouter_url=os.getenv('OPENROUTER_URL', 'https://openrouter.ai/api/v1/chat/completions'),
#             hf_api_key=os.getenv('HF_API_KEY'),
#             working_models=models
#         )
    
#     def _load_processing_config(self) -> ProcessingConfig:
#         """Load processing configuration"""
#         chunk_size = int(os.getenv('DEFAULT_CHUNK_SIZE', '1000'))
#         chunk_overlap = int(os.getenv('DEFAULT_CHUNK_OVERLAP', '200'))
        
#         # Validate processing parameters
#         if not (100 <= chunk_size <= 5000):
#             raise ValueError("DEFAULT_CHUNK_SIZE must be between 100 and 5000")
        
#         if chunk_overlap < 0 or chunk_overlap >= chunk_size:
#             raise ValueError("Invalid DEFAULT_CHUNK_OVERLAP value")
        
#         return ProcessingConfig(
#             default_chunk_size=chunk_size,
#             default_chunk_overlap=chunk_overlap,
#             max_context_length=int(os.getenv('MAX_CONTEXT_LENGTH', '3000')),
#             max_concurrent_processing=int(os.getenv('MAX_CONCURRENT_PROCESSING', '3')),
#             memory_optimization=os.getenv('MEMORY_OPTIMIZATION', 'true').lower() == 'true',
#             language_detection=os.getenv('LANGUAGE_DETECTION', 'true').lower() == 'true',
#             nlp_preprocessing=os.getenv('NLP_PREPROCESSING', 'true').lower() == 'true'
#         )
    
#     def _load_search_config(self) -> SearchConfig:
#         """Load search configuration"""
#         return SearchConfig(
#             semantic_model=os.getenv('SEMANTIC_MODEL', 'all-MiniLM-L6-v2'),
#             index_path=os.getenv('INDEX_PATH', 'data/faiss_index.bin'),
#             metadata_path=os.getenv('METADATA_PATH', 'data/chunk_metadata.json')
#         )
    
#     def _load_security_config(self) -> SecurityConfig:
#         """Load security configuration"""
#         cors_origins_str = os.getenv('CORS_ORIGINS', '*')
#         cors_origins = [origin.strip() for origin in cors_origins_str.split(',') if origin.strip()]
        
#         return SecurityConfig(
#             cors_origins=cors_origins,
#             max_file_size=int(os.getenv('MAX_FILE_SIZE', str(100 * 1024 * 1024))),  # 100MB
#             max_files_per_batch=int(os.getenv('MAX_FILES_PER_BATCH', '10'))
#         )
    
#     def _load_app_config(self) -> Dict[str, Any]:
#         """Load application configuration"""
#         return {
#             'host': os.getenv('API_HOST', '0.0.0.0'),
#             'port': int(os.getenv('API_PORT', '8000')),
#             'debug': os.getenv('DEBUG', 'false').lower() == 'true',
#             'temp_dir': os.getenv('TEMP_DIR', '/tmp'),
#             'data_dir': os.getenv('DATA_DIR', 'data'),
#             'cleanup_temp_files': os.getenv('CLEANUP_TEMP_FILES', 'true').lower() == 'true',
#             'log_level': os.getenv('LOG_LEVEL', 'INFO'),
#             'log_format': os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#         }
    
#     def setup_logging(self) -> None:
#         """Setup logging configuration"""
#         logging.basicConfig(
#             level=getattr(logging, self.app['log_level']),
#             format=self.app['log_format']
#         )
    
#     def validate_file_upload(self, file_size: int, filename: str) -> bool:
#         """Validate file upload parameters"""
#         if file_size > self.security.max_file_size:
#             return False
        
#         if not filename.lower().endswith('.pdf'):
#             return False
        
#         return True
    
#     def to_dict(self) -> Dict[str, Any]:
#         """Convert configuration to dictionary (without sensitive data)"""
#         return {
#             'processing': {
#                 'chunk_size': self.processing.default_chunk_size,
#                 'chunk_overlap': self.processing.default_chunk_overlap,
#                 'max_context_length': self.processing.max_context_length,
#                 'memory_optimization': self.processing.memory_optimization
#             },
#             'search': {
#                 'model': self.search.semantic_model,
#                 'score_threshold': self.search.score_threshold
#             },
#             'security': {
#                 'max_file_size': self.security.max_file_size,
#                 'max_files_per_batch': self.security.max_files_per_batch
#             },
#             'app': {
#                 'debug': self.app['debug'],
#                 'log_level': self.app['log_level']
#             }
#         }

# # Global configuration instance
# config = Config()

"""
Simplified configuration for testing CORS issues
"""
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