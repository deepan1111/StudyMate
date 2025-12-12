# """
# StudyMate Backend - Complete Clean Implementation
# All secrets moved to environment variables
# """
# from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Body
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from typing import List, Optional, Dict, Any
# import aiofiles
# import os
# import tempfile
# import logging
# import uuid
# from datetime import datetime
# from pathlib import Path  # Add this import

# from config import config
# from pdf_processor import PDFProcessor, ProcessedChunk, DocumentMetadata
# from llm_service import LLMService, LLMResponse
# from semantic_search import SemanticSearchEngine, convert_search_results_to_dict
# import ssl

# # ==================== NLTK Setup ====================
# import nltk

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# def setup_nltk():
#     try:
#         nltk.data.find("tokenizers/punkt")
#         logging.info("NLTK punkt tokenizer found")
#     except LookupError:
#         try:
#             logging.info("Downloading NLTK punkt tokenizer...")
#             nltk.download("punkt", quiet=True)
#             logging.info("NLTK punkt tokenizer downloaded successfully")
#         except Exception as e:
#             logging.warning(f"Failed to download NLTK data: {e}")

# setup_nltk()

# # ==================== Configuration & Logging ====================
# config.setup_logging()
# logger = logging.getLogger(__name__)

# # ==================== FastAPI App ====================
# app = FastAPI(
#     title="StudyMate Backend",
#     description="AI-Powered Academic Assistant for PDF Processing and Q&A",
#     version="2.1.0",
#     debug=config.app['debug']
# )

# # ==================== CORS Middleware ====================
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=config.security.cors_origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ==================== Components ====================
# pdf_processor = PDFProcessor(
#     chunk_size=config.processing.default_chunk_size,
#     chunk_overlap=config.processing.default_chunk_overlap
# )
# llm_service = LLMService()

# search_engine = SemanticSearchEngine(
#     model_name=config.search.semantic_model,
#     index_path=config.search.index_path,
#     metadata_path=config.search.metadata_path
# )

# # ==================== Storage ====================
# document_store: Dict[str, Dict] = {}
# session_store: Dict[str, Dict] = {}

# # ==================== Helpers ====================

# async def _is_valid_pdf(file: UploadFile) -> bool:
#     """Validate PDF file"""
#     try:
#         await file.seek(0)
#         content = await file.read(1024)
#         await file.seek(0)
#         return content.startswith(b'%PDF')
#     except Exception as e:
#         logger.error(f"PDF validation error: {e}")
#         return False

# async def _save_temp_file(file: UploadFile) -> str:
#     """Save uploaded file temporarily with proper path handling"""
#     temp_dir = Path(config.app['temp_dir'])
    
#     # Ensure temp directory exists
#     temp_dir.mkdir(parents=True, exist_ok=True)
    
#     # Create temp file path using pathlib for cross-platform compatibility
#     safe_filename = f"pdf_{uuid.uuid4()}_{file.filename}"
#     temp_path = temp_dir / safe_filename
    
#     try:
#         # Convert to string for aiofiles
#         temp_path_str = str(temp_path)
        
#         async with aiofiles.open(temp_path_str, 'wb') as temp_file:
#             content = await file.read()
#             await temp_file.write(content)
            
#         logger.info(f"Saved temp file: {temp_path_str}")
#         return temp_path_str
        
#     except Exception as e:
#         logger.error(f"Failed to save temp file: {e}")
#         raise

# def search_chunks_semantic(
#     query: str, 
#     document_ids: List[str] = None, 
#     limit: int = 5,
#     use_hybrid: bool = True
# ) -> List[Dict]:
#     """Enhanced semantic search"""
#     try:
#         if use_hybrid:
#             results = search_engine.hybrid_search(
#                 query=query,
#                 k=limit,
#                 document_ids=document_ids,
#                 semantic_weight=0.7,
#                 keyword_weight=0.3
#             )
#         else:
#             results = search_engine.search(
#                 query=query,
#                 k=limit,
#                 document_ids=document_ids,
#                 score_threshold=config.search.score_threshold
#             )
        
#         return convert_search_results_to_dict(results)
        
#     except Exception as e:
#         logger.error(f"Semantic search error: {e}")
#         return search_chunks_fallback(query, document_ids, limit)

# def search_chunks_fallback(query: str, document_ids: List[str] = None, limit: int = 5) -> List[Dict]:
#     """Fallback keyword search"""
#     query_terms = query.lower().split()
#     results = []
    
#     logger.info(f"Using fallback search for: {query}")
    
#     for doc_id, doc_data in document_store.items():
#         if document_ids and doc_id not in document_ids:
#             continue
            
#         chunks = doc_data.get("chunks", [])
        
#         for chunk in chunks:
#             try:
#                 if hasattr(chunk, 'text'):
#                     chunk_text = chunk.text
#                     chunk_id = chunk.chunk_id
#                     page_numbers = getattr(chunk, 'page_numbers', [])
#                     key_terms = getattr(chunk, 'key_terms', [])
#                     metadata = chunk.metadata if hasattr(chunk, 'metadata') else {}
#                 else:
#                     chunk_text = chunk.get('text', '')
#                     chunk_id = chunk.get('chunk_id', '')
#                     page_numbers = chunk.get('page_numbers', [])
#                     key_terms = chunk.get('key_terms', [])
#                     metadata = chunk.get('metadata', {})
                
#                 if not chunk_text:
#                     continue
                
#                 chunk_text_lower = chunk_text.lower()
#                 keyword_score = sum(1 for term in query_terms if term in chunk_text_lower)
#                 key_term_score = sum(2 for term in query_terms 
#                                    if any(term in kt.lower() for kt in key_terms))
                
#                 total_score = keyword_score + key_term_score
                
#                 if total_score > 0:
#                     results.append({
#                         "chunk_id": chunk_id,
#                         "text": chunk_text,
#                         "document": doc_data["filename"],
#                         "score": total_score,
#                         "metadata": metadata,
#                         "page_numbers": page_numbers,
#                         "key_terms": key_terms,
#                         "embedding_score": 0.0
#                     })
                    
#             except Exception as e:
#                 logger.error(f"Error processing chunk in fallback search: {e}")
#                 continue
    
#     results.sort(key=lambda x: x["score"], reverse=True)
#     return results[:limit]

# # ==================== Document Management ====================

# @app.post("/upload/single")
# async def upload_single_pdf(
#     file: UploadFile = File(...),
#     chunk_size: Optional[int] = Form(None),
#     chunk_overlap: Optional[int] = Form(None)
# ):
#     """Upload and process a single PDF file"""
#     logger.info(f"Processing upload: {file.filename}")
    
#     # Validate file
#     if not await _is_valid_pdf(file):
#         raise HTTPException(status_code=400, detail="Invalid PDF file")
    
#     # Use config defaults if not provided
#     chunk_size = chunk_size or config.processing.default_chunk_size
#     chunk_overlap = chunk_overlap or config.processing.default_chunk_overlap
    
#     # Validate parameters
#     if not (100 <= chunk_size <= 5000):
#         raise HTTPException(status_code=400, detail="Chunk size must be between 100 and 5000")
    
#     if chunk_overlap < 0 or chunk_overlap >= chunk_size:
#         raise HTTPException(status_code=400, detail="Invalid chunk overlap")
    
#     pdf_processor.chunk_size = chunk_size
#     pdf_processor.chunk_overlap = chunk_overlap
    
#     temp_path = None
#     try:
#         temp_path = await _save_temp_file(file)
#         logger.info(f"Saved temp file: {temp_path}")
        
#         # Extract text and metadata
#         text, metadata = pdf_processor.extract_text_and_metadata(temp_path)
#         logger.info(f"Extracted {len(text)} characters of text")
        
#         # Create chunks
#         chunks = pdf_processor.chunk_text(text, metadata)
#         logger.info(f"Created {len(chunks)} chunks")
        
#         # Get stats
#         stats = pdf_processor.get_processing_stats(chunks)
        
#         # Generate unique document ID
#         doc_id = str(uuid.uuid4())
        
#         # Store document
#         document_info = {
#             "id": doc_id,
#             "filename": file.filename,
#             "metadata": metadata.__dict__,
#             "chunks": chunks,
#             "stats": stats,
#             "uploaded_at": datetime.utcnow().isoformat()
#         }
        
#         document_store[doc_id] = document_info
        
#         # Add to search index
#         try:
#             search_engine.add_document_chunks(chunks, document_info)
#             logger.info(f"Added {len(chunks)} chunks to search index")
#         except Exception as e:
#             logger.error(f"Failed to add chunks to search index: {e}")
        
#         # Clean up temp file
#         if temp_path and os.path.exists(temp_path) and config.app['cleanup_temp_files']:
#             os.unlink(temp_path)
        
#         # Prepare response
#         response_chunks = []
#         for chunk in chunks:
#             chunk_data = {
#                 "chunk_id": chunk.chunk_id,
#                 "text": chunk.text,
#                 "word_count": chunk.word_count,
#                 "char_count": chunk.char_count,
#                 "metadata": chunk.metadata,
#                 "page_numbers": getattr(chunk, 'page_numbers', []),
#                 "key_terms": getattr(chunk, 'key_terms', [])
#             }
#             response_chunks.append(chunk_data)
        
#         return JSONResponse({
#             "success": True,
#             "document_id": doc_id,
#             "filename": file.filename,
#             "metadata": metadata.__dict__,
#             "processing_stats": stats,
#             "chunks_count": len(chunks),
#             "chunks": response_chunks,
#             "search_index_updated": True
#         })
        
#     except Exception as e:
#         logger.error(f"Error processing PDF: {str(e)}")
#         if temp_path and os.path.exists(temp_path):
#             try:
#                 os.unlink(temp_path)
#             except OSError:
#                 pass
#         raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

# @app.get("/documents")
# async def get_documents():
#     """Get all uploaded documents"""
#     try:
#         documents = []
#         for doc_id, doc_data in document_store.items():
#             documents.append({
#                 "id": doc_id,
#                 "filename": doc_data["filename"],
#                 "metadata": doc_data["metadata"],
#                 "stats": doc_data["stats"],
#                 "uploaded_at": doc_data["uploaded_at"],
#                 "chunks_count": len(doc_data.get("chunks", []))
#             })
#         return {
#             "documents": documents,
#             "search_stats": search_engine.get_stats()
#         }
#     except Exception as e:
#         logger.error(f"Error retrieving documents: {e}")
#         raise HTTPException(status_code=500, detail="Failed to retrieve documents")

# @app.delete("/documents/{document_id}")
# async def delete_document(document_id: str):
#     """Delete a document"""
#     if document_id not in document_store:
#         raise HTTPException(status_code=404, detail="Document not found")
    
#     try:
#         # Remove from search index
#         try:
#             search_engine.remove_document(document_id)
#             logger.info(f"Removed document {document_id} from search index")
#         except Exception as e:
#             logger.error(f"Failed to remove document from search index: {e}")
        
#         # Remove from store
#         del document_store[document_id]
#         logger.info(f"Deleted document {document_id}")
        
#         return {
#             "success": True,
#             "message": "Document deleted",
#             "search_stats": search_engine.get_stats()
#         }
#     except Exception as e:
#         logger.error(f"Error deleting document: {e}")
#         raise HTTPException(status_code=500, detail="Failed to delete document")

# # ==================== Question Answering ====================

# @app.post("/ask")
# async def ask_question(request: Dict[str, Any] = Body(...)):
#     """Ask a question about uploaded documents"""
#     question = request.get("question", "").strip()
#     session_id = request.get("session_id")
#     document_ids = request.get("document_ids", None)
#     max_context_chunks = request.get("max_context_chunks", 5)
#     temperature = request.get("temperature", 0.3)
#     use_semantic_search = request.get("use_semantic_search", True)
    
#     # Validation
#     if not question or len(question) < 3:
#         raise HTTPException(status_code=400, detail="Question must be at least 3 characters")
    
#     if not document_store:
#         raise HTTPException(status_code=400, detail="No documents uploaded")
    
#     if not (0.0 <= temperature <= 1.0):
#         raise HTTPException(status_code=400, detail="Temperature must be between 0.0 and 1.0")
    
#     try:
#         start_time = datetime.utcnow()
        
#         # Search for relevant chunks
#         if use_semantic_search:
#             relevant_chunks = search_chunks_semantic(
#                 question, 
#                 document_ids=document_ids, 
#                 limit=max_context_chunks,
#                 use_hybrid=True
#             )
#         else:
#             relevant_chunks = search_chunks_fallback(
#                 question, 
#                 document_ids=document_ids, 
#                 limit=max_context_chunks
#             )
        
#         logger.info(f"Found {len(relevant_chunks)} relevant chunks for question: {question}")
        
#         if not relevant_chunks:
#             return {
#                 "question": question,
#                 "answer": "I couldn't find relevant information in your documents to answer this question.",
#                 "sources": [],
#                 "confidence": 0.1,
#                 "processing_time": 0.1,
#                 "chunks_used": 0,
#                 "search_method": "semantic" if use_semantic_search else "keyword"
#             }
        
#         # Generate answer
#         llm_response = await llm_service.generate_answer(
#             question=question,
#             context_chunks=relevant_chunks,
#             temperature=temperature
#         )
        
#         end_time = datetime.utcnow()
#         processing_time = (end_time - start_time).total_seconds()
        
#         # Prepare sources
#         sources = []
#         for chunk in relevant_chunks:
#             excerpt = chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"]
#             sources.append({
#                 "document": chunk.get("document", "Unknown"),
#                 "chunk_id": chunk.get("chunk_id", ""),
#                 "excerpt": excerpt,
#                 "page_numbers": chunk.get("page_numbers", []),
#                 "score": chunk.get("score", 0),
#                 "embedding_score": chunk.get("embedding_score", 0),
#                 "key_terms": chunk.get("key_terms", [])
#             })
        
#         response_data = {
#             "question": question,
#             "answer": llm_response.answer,
#             "sources": sources,
#             "confidence": llm_response.confidence,
#             "processing_time": processing_time,
#             "chunks_used": len(relevant_chunks),
#             "search_method": "semantic" if use_semantic_search else "keyword",
#             "llm_stats": {
#                 "model_used": llm_response.model_used,
#                 "token_count": llm_response.token_count,
#                 "llm_processing_time": llm_response.processing_time,
#                 "error": llm_response.error
#             },
#             "timestamp": end_time.isoformat()
#         }
        
#         # Store in session if provided
#         if session_id and session_id in session_store:
#             try:
#                 session_store[session_id]["messages"].append(response_data)
#             except Exception as session_error:
#                 logger.warning(f"Failed to save to session {session_id}: {session_error}")
        
#         return response_data
        
#     except Exception as e:
#         logger.error(f"Error answering question: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to answer question: {str(e)}")

# # ==================== Search Endpoints ====================

# @app.post("/search")
# async def search_documents(request: Dict[str, Any] = Body(...)):
#     """Search documents"""
#     query = request.get("query", "").strip()
#     document_ids = request.get("document_ids", None)
#     limit = request.get("limit", 10)
#     use_semantic_search = request.get("use_semantic_search", True)
    
#     if not query or len(query) < 2:
#         raise HTTPException(status_code=400, detail="Query must be at least 2 characters")
    
#     try:
#         start_time = datetime.utcnow()
        
#         if use_semantic_search:
#             results = search_chunks_semantic(query, document_ids=document_ids, limit=limit)
#         else:
#             results = search_chunks_fallback(query, document_ids=document_ids, limit=limit)
        
#         end_time = datetime.utcnow()
#         search_time = (end_time - start_time).total_seconds()
        
#         return {
#             "query": query,
#             "search_method": "semantic" if use_semantic_search else "keyword",
#             "search_time": search_time,
#             "results": results,
#             "total_results": len(results)
#         }
#     except Exception as e:
#         logger.error(f"Error searching documents: {e}")
#         raise HTTPException(status_code=500, detail="Search failed")

# @app.get("/search/stats")
# async def get_search_stats():
#     """Get search engine statistics"""
#     try:
#         search_stats = search_engine.get_stats()
        
#         # Add additional stats
#         total_documents = len(document_store)
#         total_chunks = sum(len(doc.get("chunks", [])) for doc in document_store.values())
        
#         return {
#             "search_engine": search_stats,
#             "documents": {
#                 "total_documents": total_documents,
#                 "total_chunks": total_chunks,
#                 "document_list": [
#                     {
#                         "id": doc_id,
#                         "filename": doc_data["filename"],
#                         "chunks_count": len(doc_data.get("chunks", []))
#                     }
#                     for doc_id, doc_data in document_store.items()
#                 ]
#             },
#             "status": "active",
#             "timestamp": datetime.utcnow().isoformat()
#         }
#     except Exception as e:
#         logger.error(f"Error getting search stats: {e}")
#         return {
#             "search_engine": {"status": "error", "message": str(e)},
#             "documents": {"total_documents": 0, "total_chunks": 0},
#             "status": "error",
#             "timestamp": datetime.utcnow().isoformat()
#         }

# # ==================== Health & Configuration ====================

# @app.get("/stats")
# async def get_system_stats():
#     """Get comprehensive system statistics"""
#     try:
#         # Document statistics
#         total_documents = len(document_store)
#         total_chunks = sum(len(doc.get("chunks", [])) for doc in document_store.values())
#         total_words = sum(
#             sum(chunk.word_count if hasattr(chunk, 'word_count') else 0 
#                 for chunk in doc.get("chunks", []))
#             for doc in document_store.values()
#         )
        
#         # Search engine stats
#         search_stats = search_engine.get_stats()
        
#         # Session statistics
#         active_sessions = len(session_store)
        
#         # Memory usage (optional - requires psutil)
#         try:
#             import psutil
#             process = psutil.Process()
#             memory_info = {
#                 "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
#                 "cpu_percent": process.cpu_percent()
#             }
#         except ImportError:
#             memory_info = {"status": "psutil not available"}
        
#         return {
#             "documents": total_documents,
#             "active_sessions": active_sessions,
#             "total_chunks": total_chunks,
#             "total_words": total_words,
#             "search_engine": search_stats,
#             "system": memory_info,
#             "timestamp": datetime.utcnow().isoformat(),
#             "version": "2.1.0"
#         }
        
#     except Exception as e:
#         logger.error(f"Error getting system stats: {e}")
#         return {
#             "documents": 0,
#             "active_sessions": 0,
#             "total_chunks": 0,
#             "total_words": 0,
#             "search_engine": {"status": "error"},
#             "system": {"status": "error"},
#             "error": str(e),
#             "timestamp": datetime.utcnow().isoformat()
#         }

# @app.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     try:
#         llm_health = await llm_service.health_check()
#         search_stats = search_engine.get_stats()
        
#         return {
#             "status": "healthy",
#             "service": "StudyMate Backend",
#             "version": "2.1.0",
#             "documents_loaded": len(document_store),
#             "active_sessions": len(session_store),
#             "total_chunks": sum(len(doc.get("chunks", [])) for doc in document_store.values()),
#             "search_engine": search_stats,
#             "llm_service": llm_health,
#             "timestamp": datetime.utcnow().isoformat()
#         }
#     except Exception as e:
#         logger.error(f"Health check error: {e}")
#         return {
#             "status": "degraded",
#             "error": str(e),
#             "documents_loaded": len(document_store),
#             "active_sessions": len(session_store)
#         }

# @app.get("/config")
# async def get_configuration():
#     """Get current configuration (without sensitive data)"""
#     return {
#         "configuration": config.to_dict(),
#         "features": {
#             "semantic_search": True,
#             "llm_integration": True,
#             "pdf_processing": True,
#             "memory_optimization": config.processing.memory_optimization,
#             "language_detection": config.processing.language_detection
#         }
#     }

# # ==================== Session Management ====================

# @app.post("/sessions/create")
# async def create_session():
#     """Create a new session"""
#     try:
#         session_id = str(uuid.uuid4())
#         session_store[session_id] = {
#             "id": session_id,
#             "created_at": datetime.utcnow().isoformat(),
#             "messages": [],
#             "context": []
#         }
#         return {"session_id": session_id}
#     except Exception as e:
#         logger.error(f"Error creating session: {e}")
#         raise HTTPException(status_code=500, detail="Failed to create session")

# @app.get("/sessions")
# async def get_sessions():
#     """Get all sessions"""
#     return {"sessions": list(session_store.values())}

# @app.delete("/sessions/{session_id}")
# async def delete_session(session_id: str):
#     """Delete a session"""
#     if session_id not in session_store:
#         raise HTTPException(status_code=404, detail="Session not found")
#     del session_store[session_id]
#     return {"success": True}

# # ==================== Application Startup ====================

# if __name__ == "__main__":
#     import uvicorn
    
#     # Ensure data directory exists
#     os.makedirs(config.app['data_dir'], exist_ok=True)
    
#     # Start server
#     uvicorn.run(
#         "main:app", 
#         host=config.app['host'], 
#         port=config.app['port'], 
#         reload=config.app['debug']
#     )

"""
StudyMate Backend - Complete Clean Implementation
All secrets moved to environment variables
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import aiofiles
import os
import tempfile
import logging
import uuid
from datetime import datetime
from pathlib import Path

from config import config
from pdf_processor import PDFProcessor, ProcessedChunk, DocumentMetadata
from llm_service import LLMService, LLMResponse

# Try to import semantic search - make it optional
try:
    from semantic_search import SemanticSearchEngine, convert_search_results_to_dict
    SEMANTIC_SEARCH_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Semantic search not available: {e}")
    print("⚠️  Running in fallback mode with keyword search only")
    SEMANTIC_SEARCH_AVAILABLE = False
    SemanticSearchEngine = None
    convert_search_results_to_dict = None

import ssl

# ==================== NLTK Setup ====================
import nltk

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def setup_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
        logging.info("NLTK punkt tokenizer found")
    except LookupError:
        try:
            logging.info("Downloading NLTK punkt tokenizer...")
            nltk.download("punkt", quiet=True)
            logging.info("NLTK punkt tokenizer downloaded successfully")
        except Exception as e:
            logging.warning(f"Failed to download NLTK data: {e}")

setup_nltk()

# ==================== Configuration & Logging ====================
config.setup_logging()
logger = logging.getLogger(__name__)

# ==================== FastAPI App ====================
app = FastAPI(
    title="StudyMate Backend",
    description="AI-Powered Academic Assistant for PDF Processing and Q&A",
    version="2.1.0",
    debug=config.app['debug']
)

# ==================== CORS Middleware ====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.security.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Components ====================
pdf_processor = PDFProcessor(
    chunk_size=config.processing.default_chunk_size,
    chunk_overlap=config.processing.default_chunk_overlap
)
llm_service = LLMService()

# Only initialize search engine if available
if SEMANTIC_SEARCH_AVAILABLE:
    search_engine = SemanticSearchEngine(
        model_name=config.search.semantic_model,
        index_path=config.search.index_path,
        metadata_path=config.search.metadata_path
    )
    logger.info("✅ Semantic search engine initialized")
else:
    search_engine = None
    logger.warning("⚠️  Running without semantic search - only keyword search available")

# ==================== Storage ====================
document_store: Dict[str, Dict] = {}
session_store: Dict[str, Dict] = {}

# ==================== Helpers ====================

async def _is_valid_pdf(file: UploadFile) -> bool:
    """Validate PDF file"""
    try:
        await file.seek(0)
        content = await file.read(1024)
        await file.seek(0)
        return content.startswith(b'%PDF')
    except Exception as e:
        logger.error(f"PDF validation error: {e}")
        return False

async def _save_temp_file(file: UploadFile) -> str:
    """Save uploaded file temporarily with proper path handling"""
    temp_dir = Path(config.app['temp_dir'])
    
    # Ensure temp directory exists
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temp file path using pathlib for cross-platform compatibility
    safe_filename = f"pdf_{uuid.uuid4()}_{file.filename}"
    temp_path = temp_dir / safe_filename
    
    try:
        # Convert to string for aiofiles
        temp_path_str = str(temp_path)
        
        async with aiofiles.open(temp_path_str, 'wb') as temp_file:
            content = await file.read()
            await temp_file.write(content)
            
        logger.info(f"Saved temp file: {temp_path_str}")
        return temp_path_str
        
    except Exception as e:
        logger.error(f"Failed to save temp file: {e}")
        raise

def search_chunks_semantic(
    query: str, 
    document_ids: List[str] = None, 
    limit: int = 5,
    use_hybrid: bool = True
) -> List[Dict]:
    """Enhanced semantic search - falls back to keyword if unavailable"""
    
    # If semantic search not available, use fallback immediately
    if not SEMANTIC_SEARCH_AVAILABLE or search_engine is None:
        logger.warning("Semantic search not available, using keyword fallback")
        return search_chunks_fallback(query, document_ids, limit)
    
    try:
        if use_hybrid:
            results = search_engine.hybrid_search(
                query=query,
                k=limit,
                document_ids=document_ids,
                semantic_weight=0.7,
                keyword_weight=0.3
            )
        else:
            results = search_engine.search(
                query=query,
                k=limit,
                document_ids=document_ids,
                score_threshold=config.search.score_threshold
            )
        
        return convert_search_results_to_dict(results)
        
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        return search_chunks_fallback(query, document_ids, limit)

def search_chunks_fallback(query: str, document_ids: List[str] = None, limit: int = 5) -> List[Dict]:
    """Fallback keyword search"""
    query_terms = query.lower().split()
    results = []
    
    logger.info(f"Using fallback search for: {query}")
    
    for doc_id, doc_data in document_store.items():
        if document_ids and doc_id not in document_ids:
            continue
            
        chunks = doc_data.get("chunks", [])
        
        for chunk in chunks:
            try:
                if hasattr(chunk, 'text'):
                    chunk_text = chunk.text
                    chunk_id = chunk.chunk_id
                    page_numbers = getattr(chunk, 'page_numbers', [])
                    key_terms = getattr(chunk, 'key_terms', [])
                    metadata = chunk.metadata if hasattr(chunk, 'metadata') else {}
                else:
                    chunk_text = chunk.get('text', '')
                    chunk_id = chunk.get('chunk_id', '')
                    page_numbers = chunk.get('page_numbers', [])
                    key_terms = chunk.get('key_terms', [])
                    metadata = chunk.get('metadata', {})
                
                if not chunk_text:
                    continue
                
                chunk_text_lower = chunk_text.lower()
                keyword_score = sum(1 for term in query_terms if term in chunk_text_lower)
                key_term_score = sum(2 for term in query_terms 
                                   if any(term in kt.lower() for kt in key_terms))
                
                total_score = keyword_score + key_term_score
                
                if total_score > 0:
                    results.append({
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "document": doc_data["filename"],
                        "score": total_score,
                        "metadata": metadata,
                        "page_numbers": page_numbers,
                        "key_terms": key_terms,
                        "embedding_score": 0.0
                    })
                    
            except Exception as e:
                logger.error(f"Error processing chunk in fallback search: {e}")
                continue
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]

# ==================== Document Management ====================

@app.post("/upload/single")
async def upload_single_pdf(
    file: UploadFile = File(...),
    chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None)
):
    """Upload and process a single PDF file"""
    logger.info(f"Processing upload: {file.filename}")
    
    # Validate file
    if not await _is_valid_pdf(file):
        raise HTTPException(status_code=400, detail="Invalid PDF file")
    
    # Use config defaults if not provided
    chunk_size = chunk_size or config.processing.default_chunk_size
    chunk_overlap = chunk_overlap or config.processing.default_chunk_overlap
    
    # Validate parameters
    if not (100 <= chunk_size <= 5000):
        raise HTTPException(status_code=400, detail="Chunk size must be between 100 and 5000")
    
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise HTTPException(status_code=400, detail="Invalid chunk overlap")
    
    pdf_processor.chunk_size = chunk_size
    pdf_processor.chunk_overlap = chunk_overlap
    
    temp_path = None
    try:
        temp_path = await _save_temp_file(file)
        logger.info(f"Saved temp file: {temp_path}")
        
        # Extract text and metadata
        text, metadata = pdf_processor.extract_text_and_metadata(temp_path)
        logger.info(f"Extracted {len(text)} characters of text")
        
        # Create chunks
        chunks = pdf_processor.chunk_text(text, metadata)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Get stats
        stats = pdf_processor.get_processing_stats(chunks)
        
        # Generate unique document ID
        doc_id = str(uuid.uuid4())
        
        # Store document
        document_info = {
            "id": doc_id,
            "filename": file.filename,
            "metadata": metadata.__dict__,
            "chunks": chunks,
            "stats": stats,
            "uploaded_at": datetime.utcnow().isoformat()
        }
        
        document_store[doc_id] = document_info
        
        # Add to search index (only if semantic search available)
        search_index_updated = False
        if SEMANTIC_SEARCH_AVAILABLE and search_engine is not None:
            try:
                search_engine.add_document_chunks(chunks, document_info)
                logger.info(f"Added {len(chunks)} chunks to search index")
                search_index_updated = True
            except Exception as e:
                logger.error(f"Failed to add chunks to search index: {e}")
        
        # Clean up temp file
        if temp_path and os.path.exists(temp_path) and config.app['cleanup_temp_files']:
            os.unlink(temp_path)
        
        # Prepare response
        response_chunks = []
        for chunk in chunks:
            chunk_data = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "word_count": chunk.word_count,
                "char_count": chunk.char_count,
                "metadata": chunk.metadata,
                "page_numbers": getattr(chunk, 'page_numbers', []),
                "key_terms": getattr(chunk, 'key_terms', [])
            }
            response_chunks.append(chunk_data)
        
        return JSONResponse({
            "success": True,
            "document_id": doc_id,
            "filename": file.filename,
            "metadata": metadata.__dict__,
            "processing_stats": stats,
            "chunks_count": len(chunks),
            "chunks": response_chunks,
            "search_index_updated": search_index_updated,
            "semantic_search_available": SEMANTIC_SEARCH_AVAILABLE
        })
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

@app.get("/documents")
async def get_documents():
    """Get all uploaded documents"""
    try:
        documents = []
        for doc_id, doc_data in document_store.items():
            documents.append({
                "id": doc_id,
                "filename": doc_data["filename"],
                "metadata": doc_data["metadata"],
                "stats": doc_data["stats"],
                "uploaded_at": doc_data["uploaded_at"],
                "chunks_count": len(doc_data.get("chunks", []))
            })
        
        # Get search stats only if available
        search_stats = {}
        if SEMANTIC_SEARCH_AVAILABLE and search_engine is not None:
            try:
                search_stats = search_engine.get_stats()
            except Exception as e:
                logger.error(f"Error getting search stats: {e}")
                search_stats = {"status": "error", "message": str(e)}
        else:
            search_stats = {"status": "unavailable", "message": "Semantic search disabled"}
        
        return {
            "documents": documents,
            "search_stats": search_stats,
            "semantic_search_available": SEMANTIC_SEARCH_AVAILABLE
        }
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document"""
    if document_id not in document_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Remove from search index (only if available)
        if SEMANTIC_SEARCH_AVAILABLE and search_engine is not None:
            try:
                search_engine.remove_document(document_id)
                logger.info(f"Removed document {document_id} from search index")
            except Exception as e:
                logger.error(f"Failed to remove document from search index: {e}")
        
        # Remove from store
        del document_store[document_id]
        logger.info(f"Deleted document {document_id}")
        
        # Get search stats only if available
        search_stats = {}
        if SEMANTIC_SEARCH_AVAILABLE and search_engine is not None:
            try:
                search_stats = search_engine.get_stats()
            except Exception as e:
                search_stats = {"status": "error"}
        
        return {
            "success": True,
            "message": "Document deleted",
            "search_stats": search_stats
        }
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")

# ==================== Question Answering ====================

@app.post("/ask")
async def ask_question(request: Dict[str, Any] = Body(...)):
    """Ask a question about uploaded documents"""
    question = request.get("question", "").strip()
    session_id = request.get("session_id")
    document_ids = request.get("document_ids", None)
    max_context_chunks = request.get("max_context_chunks", 5)
    temperature = request.get("temperature", 0.3)
    use_semantic_search = request.get("use_semantic_search", True)
    
    # Validation
    if not question or len(question) < 3:
        raise HTTPException(status_code=400, detail="Question must be at least 3 characters")
    
    if not document_store:
        raise HTTPException(status_code=400, detail="No documents uploaded")
    
    if not (0.0 <= temperature <= 1.0):
        raise HTTPException(status_code=400, detail="Temperature must be between 0.0 and 1.0")
    
    try:
        start_time = datetime.utcnow()
        
        # Search for relevant chunks
        # Force fallback if semantic search requested but unavailable
        if use_semantic_search and SEMANTIC_SEARCH_AVAILABLE:
            relevant_chunks = search_chunks_semantic(
                question, 
                document_ids=document_ids, 
                limit=max_context_chunks,
                use_hybrid=True
            )
            search_method_used = "semantic"
        else:
            relevant_chunks = search_chunks_fallback(
                question, 
                document_ids=document_ids, 
                limit=max_context_chunks
            )
            search_method_used = "keyword"
        
        logger.info(f"Found {len(relevant_chunks)} relevant chunks for question: {question}")
        
        if not relevant_chunks:
            return {
                "question": question,
                "answer": "I couldn't find relevant information in your documents to answer this question.",
                "sources": [],
                "confidence": 0.1,
                "processing_time": 0.1,
                "chunks_used": 0,
                "search_method": search_method_used,
                "semantic_search_available": SEMANTIC_SEARCH_AVAILABLE
            }
        
        # Generate answer
        llm_response = await llm_service.generate_answer(
            question=question,
            context_chunks=relevant_chunks,
            temperature=temperature
        )
        
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        
        # Prepare sources
        sources = []
        for chunk in relevant_chunks:
            excerpt = chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"]
            sources.append({
                "document": chunk.get("document", "Unknown"),
                "chunk_id": chunk.get("chunk_id", ""),
                "excerpt": excerpt,
                "page_numbers": chunk.get("page_numbers", []),
                "score": chunk.get("score", 0),
                "embedding_score": chunk.get("embedding_score", 0),
                "key_terms": chunk.get("key_terms", [])
            })
        
        response_data = {
            "question": question,
            "answer": llm_response.answer,
            "sources": sources,
            "confidence": llm_response.confidence,
            "processing_time": processing_time,
            "chunks_used": len(relevant_chunks),
            "search_method": search_method_used,
            "semantic_search_available": SEMANTIC_SEARCH_AVAILABLE,
            "llm_stats": {
                "model_used": llm_response.model_used,
                "token_count": llm_response.token_count,
                "llm_processing_time": llm_response.processing_time,
                "error": llm_response.error
            },
            "timestamp": end_time.isoformat()
        }
        
        # Store in session if provided
        if session_id and session_id in session_store:
            try:
                session_store[session_id]["messages"].append(response_data)
            except Exception as session_error:
                logger.warning(f"Failed to save to session {session_id}: {session_error}")
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {str(e)}")

# ==================== Search Endpoints ====================

@app.post("/search")
async def search_documents(request: Dict[str, Any] = Body(...)):
    """Search documents"""
    query = request.get("query", "").strip()
    document_ids = request.get("document_ids", None)
    limit = request.get("limit", 10)
    use_semantic_search = request.get("use_semantic_search", True)
    
    if not query or len(query) < 2:
        raise HTTPException(status_code=400, detail="Query must be at least 2 characters")
    
    try:
        start_time = datetime.utcnow()
        
        # Use semantic if available and requested, otherwise use fallback
        if use_semantic_search and SEMANTIC_SEARCH_AVAILABLE:
            results = search_chunks_semantic(query, document_ids=document_ids, limit=limit)
            search_method_used = "semantic"
        else:
            results = search_chunks_fallback(query, document_ids=document_ids, limit=limit)
            search_method_used = "keyword"
        
        end_time = datetime.utcnow()
        search_time = (end_time - start_time).total_seconds()
        
        return {
            "query": query,
            "search_method": search_method_used,
            "semantic_search_available": SEMANTIC_SEARCH_AVAILABLE,
            "search_time": search_time,
            "results": results,
            "total_results": len(results)
        }
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail="Search failed")

@app.get("/search/stats")
async def get_search_stats():
    """Get search engine statistics"""
    try:
        # Get search stats only if available
        if SEMANTIC_SEARCH_AVAILABLE and search_engine is not None:
            try:
                search_stats = search_engine.get_stats()
            except Exception as e:
                search_stats = {"status": "error", "message": str(e)}
        else:
            search_stats = {"status": "unavailable", "message": "Semantic search disabled"}
        
        # Add additional stats
        total_documents = len(document_store)
        total_chunks = sum(len(doc.get("chunks", [])) for doc in document_store.values())
        
        return {
            "search_engine": search_stats,
            "semantic_search_available": SEMANTIC_SEARCH_AVAILABLE,
            "documents": {
                "total_documents": total_documents,
                "total_chunks": total_chunks,
                "document_list": [
                    {
                        "id": doc_id,
                        "filename": doc_data["filename"],
                        "chunks_count": len(doc_data.get("chunks", []))
                    }
                    for doc_id, doc_data in document_store.items()
                ]
            },
            "status": "active",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting search stats: {e}")
        return {
            "search_engine": {"status": "error", "message": str(e)},
            "semantic_search_available": SEMANTIC_SEARCH_AVAILABLE,
            "documents": {"total_documents": 0, "total_chunks": 0},
            "status": "error",
            "timestamp": datetime.utcnow().isoformat()
        }

# ==================== Health & Configuration ====================

@app.get("/stats")
async def get_system_stats():
    """Get comprehensive system statistics"""
    try:
        # Document statistics
        total_documents = len(document_store)
        total_chunks = sum(len(doc.get("chunks", [])) for doc in document_store.values())
        total_words = sum(
            sum(chunk.word_count if hasattr(chunk, 'word_count') else 0 
                for chunk in doc.get("chunks", []))
            for doc in document_store.values()
        )
        
        # Search engine stats (only if available)
        if SEMANTIC_SEARCH_AVAILABLE and search_engine is not None:
            try:
                search_stats = search_engine.get_stats()
            except Exception as e:
                search_stats = {"status": "error", "message": str(e)}
        else:
            search_stats = {"status": "unavailable", "message": "Semantic search disabled"}
        
        # Session statistics
        active_sessions = len(session_store)
        
        # Memory usage (optional - requires psutil)
        try:
            import psutil
            process = psutil.Process()
            memory_info = {
                "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_percent": process.cpu_percent()
            }
        except ImportError:
            memory_info = {"status": "psutil not available"}
        
        return {
            "documents": total_documents,
            "active_sessions": active_sessions,
            "total_chunks": total_chunks,
            "total_words": total_words,
            "search_engine": search_stats,
            "semantic_search_available": SEMANTIC_SEARCH_AVAILABLE,
            "system": memory_info,
            "timestamp": datetime.utcnow().isoformat(),
            "version": "2.1.0"
        }
        
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        return {
            "documents": 0,
            "active_sessions": 0,
            "total_chunks": 0,
            "total_words": 0,
            "search_engine": {"status": "error"},
            "semantic_search_available": SEMANTIC_SEARCH_AVAILABLE,
            "system": {"status": "error"},
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        llm_health = await llm_service.health_check()
        
        # Get search stats only if available
        if SEMANTIC_SEARCH_AVAILABLE and search_engine is not None:
            try:
                search_stats = search_engine.get_stats()
            except Exception as e:
                search_stats = {"status": "error", "message": str(e)}
        else:
            search_stats = {"status": "unavailable", "message": "Semantic search disabled"}
        
        return {
            "status": "healthy",
            "service": "StudyMate Backend",
            "version": "2.1.0",
            "documents_loaded": len(document_store),
            "active_sessions": len(session_store),
            "total_chunks": sum(len(doc.get("chunks", [])) for doc in document_store.values()),
            "search_engine": search_stats,
            "semantic_search_available": SEMANTIC_SEARCH_AVAILABLE,
            "llm_service": llm_health,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "degraded",
            "error": str(e),
            "documents_loaded": len(document_store),
            "active_sessions": len(session_store),
            "semantic_search_available": SEMANTIC_SEARCH_AVAILABLE
        }

@app.get("/config")
async def get_configuration():
    """Get current configuration (without sensitive data)"""
    return {
        "configuration": config.to_dict(),
        "features": {
            "semantic_search": SEMANTIC_SEARCH_AVAILABLE,
            "llm_integration": True,
            "pdf_processing": True,
            "memory_optimization": config.processing.memory_optimization,
            "language_detection": config.processing.language_detection
        }
    }

# ==================== Session Management ====================

@app.post("/sessions/create")
async def create_session():
    """Create a new session"""
    try:
        session_id = str(uuid.uuid4())
        session_store[session_id] = {
            "id": session_id,
            "created_at": datetime.utcnow().isoformat(),
            "messages": [],
            "context": []
        }
        return {"session_id": session_id}
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create session")

@app.get("/sessions")
async def get_sessions():
    """Get all sessions"""
    return {"sessions": list(session_store.values())}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")
    del session_store[session_id]
    return {"success": True}

# ==================== Application Startup ====================

if __name__ == "__main__":
    import uvicorn
    
    # Ensure data directory exists
    os.makedirs(config.app['data_dir'], exist_ok=True)
    
    # Log startup info
    print("=" * 60)
    print(f"🚀 StudyMate Backend v2.1.0")
    print(f"📊 Semantic Search: {'✅ Enabled' if SEMANTIC_SEARCH_AVAILABLE else '⚠️  Disabled (keyword search only)'}")
    print("=" * 60)
    
    # Start server
    uvicorn.run(
        "main:app", 
        host=config.app['host'], 
        port=config.app['port'], 
        reload=config.app['debug']
    )