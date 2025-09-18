

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
from pdf_processor import PDFProcessor, ProcessedChunk, DocumentMetadata
from llm_service import LLMService, LLMResponse
from semantic_search import SemanticSearchEngine, convert_search_results_to_dict
import asyncio
import ssl

# ==================== NLTK Setup (FIXED) ====================
import nltk

# Handle SSL issues on Windows
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

# ==================== Logging ====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== FastAPI App ====================
app = FastAPI(
    title="StudyMate Backend",
    description="AI-Powered Academic Assistant for PDF Processing and Q&A with Semantic Search",
    version="2.1.0"
)

# ==================== CORS Middleware ====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Components ====================
pdf_processor = PDFProcessor(chunk_size=1000, chunk_overlap=200)
llm_service = LLMService()

# Initialize semantic search engine
search_engine = SemanticSearchEngine(
    model_name="all-MiniLM-L6-v2",  # Fast and accurate model
    index_path="data/faiss_index.bin",
    metadata_path="data/chunk_metadata.json"
)

# ==================== In-Memory Storage ====================
document_store: Dict[str, Dict] = {}
session_store: Dict[str, Dict] = {}

# ==================== HELPERS ====================

async def _is_valid_pdf(file: UploadFile) -> bool:
    try:
        await file.seek(0)
        content = await file.read(1024)
        await file.seek(0)
        return content.startswith(b'%PDF')
    except Exception as e:
        logger.error(f"PDF validation error: {e}")
        return False

async def _save_temp_file(file: UploadFile) -> str:
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"pdf_{uuid.uuid4()}_{file.filename}")
    try:
        async with aiofiles.open(temp_path, 'wb') as temp_file:
            content = await file.read()
            await temp_file.write(content)
        return temp_path
    except Exception as e:
        logger.error(f"Failed to save temp file: {e}")
        raise

def search_chunks_semantic(
    query: str, 
    document_ids: List[str] = None, 
    limit: int = 5,
    use_hybrid: bool = True
) -> List[Dict]:
    """
    Enhanced semantic search using FAISS and SentenceTransformers
    """
    try:
        if use_hybrid:
            # Use hybrid search for best results
            results = search_engine.hybrid_search(
                query=query,
                k=limit,
                document_ids=document_ids,
                semantic_weight=0.7,
                keyword_weight=0.3
            )
        else:
            # Pure semantic search
            results = search_engine.search(
                query=query,
                k=limit,
                document_ids=document_ids,
                score_threshold=0.1
            )
        
        # Convert to dictionary format for API compatibility
        return convert_search_results_to_dict(results)
        
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        # Fallback to basic keyword search if semantic search fails
        return search_chunks_fallback(query, document_ids, limit)

def search_chunks_fallback(query: str, document_ids: List[str] = None, limit: int = 5) -> List[Dict]:
    """Fallback keyword search if semantic search fails"""
    query_terms = query.lower().split()
    results = []
    
    logger.info(f"Using fallback search for: {query}")
    
    for doc_id, doc_data in document_store.items():
        if document_ids and doc_id not in document_ids:
            continue
            
        chunks = doc_data.get("chunks", [])
        
        for chunk in chunks:
            try:
                # Handle both ProcessedChunk objects and dict representations
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
                
                # Calculate relevance score
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
                        "embedding_score": 0.0  # No embedding score for fallback
                    })
                    
            except Exception as e:
                logger.error(f"Error processing chunk in fallback search: {e}")
                continue
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]

# ==================== DOCUMENT MANAGEMENT ====================


@app.post("/upload/single")
async def upload_single_pdf(
    file: UploadFile = File(...),
    chunk_size: Optional[int] = Form(1000),
    chunk_overlap: Optional[int] = Form(200)
):
    logger.info(f"Processing upload: {file.filename}")
    
    if not await _is_valid_pdf(file):
        raise HTTPException(status_code=400, detail="Invalid PDF file")
    
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
        
        # Store document with full chunk data
        document_info = {
            "id": doc_id,
            "filename": file.filename,
            "metadata": metadata.__dict__,
            "chunks": chunks,
            "stats": stats,
            "uploaded_at": datetime.utcnow().isoformat()
        }
        
        document_store[doc_id] = document_info
        
        # Add chunks to semantic search index
        try:
            search_engine.add_document_chunks(chunks, document_info)
            logger.info(f"Added {len(chunks)} chunks to semantic search index")
        except Exception as e:
            logger.error(f"Failed to add chunks to search index: {e}")
            # Continue without failing the upload
        
        logger.info(f"Stored document {doc_id} with {len(chunks)} chunks")
        
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        
        # Prepare response with all chunk data
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
            "search_index_updated": True
        })
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        # Clean up temp file on error
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

@app.get("/documents")
async def get_documents():
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
        return {
            "documents": documents,
            "search_stats": search_engine.get_stats()
        }
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    if document_id not in document_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Remove from search index first
        try:
            search_engine.remove_document(document_id)
            logger.info(f"Removed document {document_id} from search index")
        except Exception as e:
            logger.error(f"Failed to remove document from search index: {e}")
        
        # Remove from document store
        del document_store[document_id]
        logger.info(f"Deleted document {document_id}")
        
        return {
            "success": True,
            "message": "Document deleted",
            "search_stats": search_engine.get_stats()
        }
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")

# ==================== QUESTION ANSWERING WITH SEMANTIC SEARCH ====================

@app.post("/ask")
async def ask_question(request: Dict[str, Any] = Body(...)):
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
        
        # Search for relevant chunks using semantic search
        if use_semantic_search:
            relevant_chunks = search_chunks_semantic(
                question, 
                document_ids=document_ids, 
                limit=max_context_chunks,
                use_hybrid=True
            )
        else:
            # Fallback to keyword search
            relevant_chunks = search_chunks_fallback(
                question, 
                document_ids=document_ids, 
                limit=max_context_chunks
            )
        
        logger.info(f"Found {len(relevant_chunks)} relevant chunks for question: {question}")
        
        if not relevant_chunks:
            return {
                "question": question,
                "answer": "I couldn't find relevant information in your documents to answer this question. Please try rephrasing or check if your documents contain information about this topic.",
                "sources": [],
                "confidence": 0.1,
                "processing_time": 0.1,
                "chunks_used": 0,
                "search_method": "semantic" if use_semantic_search else "keyword",
                "llm_stats": {"model_used": "none", "error": "No relevant chunks found"}
            }
        
        # Generate answer using LLM
        llm_response = await llm_service.generate_answer(
            question=question,
            context_chunks=relevant_chunks,
            temperature=temperature
        )
        
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        
        # Prepare sources with enhanced metadata from semantic search
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
                "key_terms": chunk.get("key_terms", []),
                "relevance_type": "semantic" if use_semantic_search else "keyword"
            })
        
        response_data = {
            "question": question,
            "answer": llm_response.answer,
            "sources": sources,
            "confidence": llm_response.confidence,
            "processing_time": processing_time,
            "chunks_used": len(relevant_chunks),
            "search_method": "semantic" if use_semantic_search else "keyword",
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

# ==================== ENHANCED SEARCH ENDPOINTS ====================

@app.post("/search")
async def search_documents(request: Dict[str, Any] = Body(...)):
    query = request.get("query", "").strip()
    document_ids = request.get("document_ids", None)
    limit = request.get("limit", 10)
    include_key_terms = request.get("include_key_terms", True)
    use_semantic_search = request.get("use_semantic_search", True)
    search_method = request.get("search_method", "hybrid")  # "semantic", "keyword", "hybrid"
    
    if not query or len(query) < 2:
        raise HTTPException(status_code=400, detail="Query must be at least 2 characters")
    
    if not (1 <= limit <= 50):
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 50")
    
    try:
        start_time = datetime.utcnow()
        
        # Choose search method
        if use_semantic_search and search_method == "semantic":
            results = search_engine.search(
                query=query,
                k=limit,
                document_ids=document_ids,
                score_threshold=0.1
            )
            results = convert_search_results_to_dict(results)
        elif use_semantic_search and search_method == "hybrid":
            results = search_chunks_semantic(
                query, 
                document_ids=document_ids, 
                limit=limit,
                use_hybrid=True
            )
        else:
            # Keyword search
            results = search_chunks_fallback(query, document_ids=document_ids, limit=limit)
        
        end_time = datetime.utcnow()
        search_time = (end_time - start_time).total_seconds()
        
        return {
            "query": query,
            "search_method": search_method if use_semantic_search else "keyword",
            "search_time": search_time,
            "results": [
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "text": chunk["text"],
                    "document": chunk.get("document"),
                    "score": chunk.get("score", 0),
                    "embedding_score": chunk.get("embedding_score", 0),
                    "page_numbers": chunk.get("page_numbers", []),
                    "key_terms": chunk.get("key_terms", []) if include_key_terms else [],
                    "metadata": chunk.get("metadata", {})
                }
                for chunk in results
            ],
            "total_results": len(results),
            "search_stats": search_engine.get_stats()
        }
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail="Search failed")

@app.post("/search/similar")
async def find_similar_chunks(request: Dict[str, Any] = Body(...)):
    """Find chunks similar to a given chunk ID using semantic search"""
    chunk_id = request.get("chunk_id", "").strip()
    limit = request.get("limit", 5)
    
    if not chunk_id:
        raise HTTPException(status_code=400, detail="Chunk ID is required")
    
    try:
        # Find the original chunk
        original_chunk = None
        for doc_data in document_store.values():
            for chunk in doc_data.get("chunks", []):
                if hasattr(chunk, 'chunk_id') and chunk.chunk_id == chunk_id:
                    original_chunk = chunk
                    break
                elif isinstance(chunk, dict) and chunk.get('chunk_id') == chunk_id:
                    original_chunk = chunk
                    break
            if original_chunk:
                break
        
        if not original_chunk:
            raise HTTPException(status_code=404, detail="Chunk not found")
        
        # Get text for similarity search
        if hasattr(original_chunk, 'text'):
            chunk_text = original_chunk.text
        else:
            chunk_text = original_chunk.get('text', '')
        
        # Use the chunk text as query to find similar chunks
        results = search_engine.search(
            query=chunk_text,
            k=limit + 1,  # +1 to exclude the original chunk
            score_threshold=0.2
        )
        
        # Filter out the original chunk
        similar_chunks = [
            result for result in results 
            if result.chunk_id != chunk_id
        ][:limit]
        
        return {
            "original_chunk_id": chunk_id,
            "similar_chunks": convert_search_results_to_dict(similar_chunks),
            "total_found": len(similar_chunks)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding similar chunks: {e}")
        raise HTTPException(status_code=500, detail="Failed to find similar chunks")

# ==================== SEARCH INDEX MANAGEMENT ====================

@app.post("/search/reindex")
async def reindex_documents():
    """Rebuild the entire search index from current documents"""
    try:
        logger.info("Starting search index rebuild...")
        
        # Clear existing index
        search_engine.clear_index()
        
        # Re-add all documents
        reindexed_docs = 0
        total_chunks = 0
        
        for doc_id, doc_data in document_store.items():
            try:
                chunks = doc_data.get("chunks", [])
                if chunks:
                    search_engine.add_document_chunks(chunks, doc_data)
                    reindexed_docs += 1
                    total_chunks += len(chunks)
                    logger.info(f"Reindexed document {doc_data['filename']} with {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Failed to reindex document {doc_id}: {e}")
        
        logger.info(f"Search index rebuilt: {reindexed_docs} documents, {total_chunks} chunks")
        
        return {
            "success": True,
            "message": "Search index rebuilt successfully",
            "reindexed_documents": reindexed_docs,
            "total_chunks": total_chunks,
            "search_stats": search_engine.get_stats()
        }
        
    except Exception as e:
        logger.error(f"Error rebuilding search index: {e}")
        raise HTTPException(status_code=500, detail="Failed to rebuild search index")

@app.get("/search/stats")
async def get_search_stats():
    """Get detailed search engine statistics"""
    try:
        stats = search_engine.get_stats()
        
        # Add additional statistics
        doc_distribution = {}
        for doc_id, doc_data in document_store.items():
            doc_distribution[doc_data['filename']] = len(doc_data.get('chunks', []))
        
        return {
            "search_engine_stats": stats,
            "document_distribution": doc_distribution,
            "total_documents_in_store": len(document_store),
            "total_sessions": len(session_store)
        }
    except Exception as e:
        logger.error(f"Error getting search stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get search statistics")

# ==================== SESSION MANAGEMENT ====================
# Add this endpoint to your main.py file

@app.post("/process/text")
async def process_text(request: Dict[str, Any] = Body(...)):
    """Process raw text input for analysis"""
    text = request.get("text", "").strip()
    
    if not text or len(text) < 3:
        raise HTTPException(status_code=400, detail="Text must be at least 3 characters")
    
    try:
        # Basic text processing
        word_count = len(text.split())
        char_count = len(text)
        
        # You can add more processing here, such as:
        # - Sentiment analysis
        # - Key phrase extraction
        # - Text summarization
        # - Language detection
        
        # For now, let's do basic analysis
        sentences = text.split('.')
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Average words per sentence
        avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
        
        # Basic readability (simple word count based)
        readability = "Easy" if avg_words_per_sentence <= 15 else "Medium" if avg_words_per_sentence <= 25 else "Complex"
        
        response = {
            "success": True,
            "original_text": text,
            "analysis": {
                "word_count": word_count,
                "character_count": char_count,
                "sentence_count": sentence_count,
                "avg_words_per_sentence": round(avg_words_per_sentence, 2),
                "readability": readability
            },
            "processed_text": text,  # You can modify this for actual processing
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text processing failed: {str(e)}")
    
@app.post("/sessions/create")
async def create_session():
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
    return {"sessions": list(session_store.values())}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")
    del session_store[session_id]
    return {"success": True}

# ==================== HEALTH & STATS ====================

@app.get("/health")
async def health_check():
    try:
        llm_health = await llm_service.health_check()
        search_stats = search_engine.get_stats()
        
        return {
            "status": "healthy",
            "service": "StudyMate Backend",
            "version": "2.1.0",
            "features": ["semantic_search", "llm_integration", "pdf_processing"],
            "documents_loaded": len(document_store),
            "active_sessions": len(session_store),
            "total_chunks": sum(len(doc.get("chunks", [])) for doc in document_store.values()),
            "search_engine": search_stats,
            "llm_service": llm_health,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "degraded",
            "error": str(e),
            "documents_loaded": len(document_store),
            "active_sessions": len(session_store)
        }

@app.get("/stats")
async def get_system_stats():
    try:
        total_chunks = sum(len(doc.get("chunks", [])) for doc in document_store.values())
        total_words = sum(doc.get("stats", {}).get("total_words", 0) for doc in document_store.values())
        
        return {
            "documents": len(document_store),
            "active_sessions": len(session_store),
            "total_chunks": total_chunks,
            "total_words": total_words,
            "search_engine": search_engine.get_stats(),
            "llm_service": await llm_service.health_check()
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {"error": str(e)}

# ==================== LLM ENDPOINTS ====================

@app.get("/llm/health")
async def llm_health_check():
    try:
        return {"status": "healthy", "llm_service": await llm_service.health_check()}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/llm/cost-estimate")
async def estimate_llm_cost(request: Dict[str, Any] = Body(...)):
    """Estimate cost for a potential query using semantic search preview"""
    question = request.get("question", "").strip()
    context_chunks = request.get("context_chunks", 5)
    
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    try:
        # Get sample chunks using semantic search
        sample_chunks = search_chunks_semantic(question, limit=context_chunks)
        
        # Build sample context
        sample_context = ""
        for chunk in sample_chunks[:context_chunks]:
            sample_context += f"Document: {chunk['document']}\n"
            sample_context += f"Content: {chunk['text'][:200]}...\n\n"
        
        # Create sample prompt
        prompt = f"""You are StudyMate. Answer this question: {question}

Context from uploaded documents:
{sample_context}

Answer:"""
        
        cost_estimate = llm_service.get_cost_estimate(prompt)
        
        return {
            "question": question,
            "estimated_costs": cost_estimate,
            "context_chunks_found": len(sample_chunks),
            "sample_context_length": len(sample_context)
        }
        
    except Exception as e:
        logger.error(f"Error estimating cost: {e}")
        return {
            "error": str(e),
            "estimated_costs": {"huggingface": 0}
        }

# ==================== DEBUG ENDPOINT ====================

@app.get("/debug/document/{document_id}")
async def debug_document(document_id: str):
    """Debug endpoint to check document structure"""
    if document_id not in document_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_data = document_store[document_id]
    chunks = doc_data.get("chunks", [])
    
    # Check if document is in search index
    in_search_index = any(
        meta['document_id'] == document_id 
        for meta in search_engine.chunk_metadata
    )
    
    return {
        "document_id": document_id,
        "filename": doc_data.get("filename"),
        "chunks_count": len(chunks),
        "in_search_index": in_search_index,
        "chunks_in_search_index": sum(
            1 for meta in search_engine.chunk_metadata 
            if meta['document_id'] == document_id
        ),
        "sample_chunk": {
            "text": chunks[0].text[:200] if chunks else "No chunks",
            "chunk_id": chunks[0].chunk_id if chunks else "None",
            "type": str(type(chunks[0])) if chunks else "None"
        } if chunks else None,
        "metadata": doc_data.get("metadata"),
        "stats": doc_data.get("stats"),
        "search_stats": search_engine.get_stats()
    }

if __name__ == "__main__":
    import uvicorn
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)