import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import logging
import pickle
import os
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Container for search results with metadata"""
    chunk_id: str
    text: str
    document: str
    score: float
    page_numbers: List[int]
    key_terms: List[str]
    metadata: Dict
    embedding_score: float

class SemanticSearchEngine:
    """
    FAISS-based semantic search engine using SentenceTransformers
    for accurate document chunk retrieval in StudyMate
    """
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        index_path: str = "faiss_index.bin",
        metadata_path: str = "chunk_metadata.json"
    ):
        """
        Initialize the semantic search engine
        
        Args:
            model_name: SentenceTransformers model for embeddings
            index_path: Path to save/load FAISS index
            metadata_path: Path to save/load chunk metadata
        """
        self.model_name = model_name
        self.index_path = index_path
        self.metadata_path = metadata_path
        
        # Initialize SentenceTransformer model
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for cosine similarity
        
        # Store chunk metadata (text, document info, etc.)
        self.chunk_metadata: List[Dict] = []
        self.chunk_id_to_index: Dict[str, int] = {}
        
        # Load existing index if available
        self.load_index()
    
    def add_document_chunks(self, chunks: List[Dict], document_info: Dict) -> None:
        """
        Add document chunks to the search index
        
        Args:
            chunks: List of chunk dictionaries with text, chunk_id, etc.
            document_info: Document metadata (filename, id, etc.)
        """
        logger.info(f"Adding {len(chunks)} chunks from document: {document_info.get('filename', 'Unknown')}")
        
        # Extract text for embedding
        texts = []
        chunk_metadata = []
        
        for chunk in chunks:
            # Handle both ProcessedChunk objects and dictionaries
            if hasattr(chunk, 'text'):
                text = chunk.text
                chunk_id = chunk.chunk_id
                page_numbers = getattr(chunk, 'page_numbers', [])
                key_terms = getattr(chunk, 'key_terms', [])
                metadata = chunk.metadata if hasattr(chunk, 'metadata') else {}
            else:
                text = chunk.get('text', '')
                chunk_id = chunk.get('chunk_id', '')
                page_numbers = chunk.get('page_numbers', [])
                key_terms = chunk.get('key_terms', [])
                metadata = chunk.get('metadata', {})
            
            if not text or not chunk_id:
                logger.warning(f"Skipping chunk with missing text or ID")
                continue
            
            texts.append(text)
            
            # Store comprehensive metadata
            chunk_meta = {
                'chunk_id': chunk_id,
                'text': text,
                'document_id': document_info.get('id', ''),
                'document': document_info.get('filename', ''),
                'page_numbers': page_numbers,
                'key_terms': key_terms,
                'metadata': metadata,
                'word_count': len(text.split()),
                'char_count': len(text)
            }
            chunk_metadata.append(chunk_meta)
        
        if not texts:
            logger.warning("No valid chunks to add")
            return
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        start_index = len(self.chunk_metadata)
        self.index.add(embeddings)
        
        # Update metadata and mapping
        for i, meta in enumerate(chunk_metadata):
            current_index = start_index + i
            self.chunk_metadata.append(meta)
            self.chunk_id_to_index[meta['chunk_id']] = current_index
        
        logger.info(f"Added {len(texts)} chunks to search index. Total chunks: {len(self.chunk_metadata)}")
        
        # Save updated index
        self.save_index()
    
    def search(
        self, 
        query: str, 
        k: int = 5, 
        document_ids: Optional[List[str]] = None,
        score_threshold: float = 0.1
    ) -> List[SearchResult]:
        """
        Perform semantic search using FAISS
        
        Args:
            query: Search query
            k: Number of results to return
            document_ids: Optional list of document IDs to filter by
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of SearchResult objects
        """
        if not query.strip():
            return []
        
        if len(self.chunk_metadata) == 0:
            logger.warning("No chunks in search index")
            return []
        
        try:
            # Generate query embedding
            logger.info(f"Searching for: '{query}' (k={k})")
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Perform FAISS search
            # Search more than needed to allow for filtering
            search_k = min(k * 3, len(self.chunk_metadata))
            scores, indices = self.index.search(query_embedding, search_k)
            
            # Process results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                if score < score_threshold:
                    continue
                
                if idx >= len(self.chunk_metadata):
                    logger.warning(f"Invalid index {idx}, skipping")
                    continue
                
                chunk_meta = self.chunk_metadata[idx]
                
                # Filter by document IDs if specified
                if document_ids and chunk_meta['document_id'] not in document_ids:
                    continue
                
                # Create SearchResult
                result = SearchResult(
                    chunk_id=chunk_meta['chunk_id'],
                    text=chunk_meta['text'],
                    document=chunk_meta['document'],
                    score=float(score),
                    page_numbers=chunk_meta['page_numbers'],
                    key_terms=chunk_meta['key_terms'],
                    metadata=chunk_meta['metadata'],
                    embedding_score=float(score)
                )
                results.append(result)
                
                # Stop when we have enough results
                if len(results) >= k:
                    break
            
            logger.info(f"Found {len(results)} relevant chunks")
            return results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def hybrid_search(
        self, 
        query: str, 
        k: int = 5, 
        document_ids: Optional[List[str]] = None,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[SearchResult]:
        """
        Combine semantic search with keyword matching for better results
        
        Args:
            query: Search query
            k: Number of results to return
            document_ids: Optional document filter
            semantic_weight: Weight for semantic similarity scores
            keyword_weight: Weight for keyword matching scores
            
        Returns:
            List of SearchResult objects with combined scores
        """
        # Get semantic search results
        semantic_results = self.search(query, k=k*2, document_ids=document_ids)
        
        if not semantic_results:
            return []
        
        # Calculate keyword scores
        query_terms = query.lower().split()
        
        for result in semantic_results:
            # Keyword matching score
            text_lower = result.text.lower()
            keyword_score = sum(1 for term in query_terms if term in text_lower)
            
            # Key terms matching (higher weight)
            key_term_score = sum(2 for term in query_terms 
                               if any(term in kt.lower() for kt in result.key_terms))
            
            # Normalize keyword score
            max_keyword_score = len(query_terms) + len(result.key_terms) * 2
            normalized_keyword_score = (keyword_score + key_term_score) / max(max_keyword_score, 1)
            
            # Combine scores
            combined_score = (
                semantic_weight * result.embedding_score + 
                keyword_weight * normalized_keyword_score
            )
            
            result.score = combined_score
        
        # Sort by combined score and return top k
        semantic_results.sort(key=lambda x: x.score, reverse=True)
        return semantic_results[:k]
    
    def remove_document(self, document_id: str) -> None:
        """
        Remove all chunks from a specific document
        Note: FAISS doesn't support efficient deletion, so we rebuild the index
        """
        logger.info(f"Removing document: {document_id}")
        
        # Filter out chunks from the specified document
        filtered_metadata = [
            meta for meta in self.chunk_metadata 
            if meta['document_id'] != document_id
        ]
        
        if len(filtered_metadata) == len(self.chunk_metadata):
            logger.warning(f"Document {document_id} not found in index")
            return
        
        # Rebuild index without the removed document
        self._rebuild_index(filtered_metadata)
        
        logger.info(f"Removed document {document_id}. Index now has {len(self.chunk_metadata)} chunks")
    
    def _rebuild_index(self, metadata_list: List[Dict]) -> None:
        """Rebuild the FAISS index from scratch with given metadata"""
        if not metadata_list:
            # Empty index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.chunk_metadata = []
            self.chunk_id_to_index = {}
            self.save_index()
            return
        
        logger.info(f"Rebuilding index with {len(metadata_list)} chunks...")
        
        # Extract texts and generate new embeddings
        texts = [meta['text'] for meta in metadata_list]
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)
        
        # Create new index
        new_index = faiss.IndexFlatIP(self.embedding_dim)
        new_index.add(embeddings)
        
        # Update instance variables
        self.index = new_index
        self.chunk_metadata = metadata_list
        self.chunk_id_to_index = {
            meta['chunk_id']: i for i, meta in enumerate(metadata_list)
        }
        
        # Save the rebuilt index
        self.save_index()
    
    def save_index(self) -> None:
        """Save FAISS index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)
            
            # Save metadata
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'chunk_metadata': self.chunk_metadata,
                    'chunk_id_to_index': self.chunk_id_to_index,
                    'embedding_dim': self.embedding_dim,
                    'model_name': self.model_name
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved search index with {len(self.chunk_metadata)} chunks")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def load_index(self) -> None:
        """Load FAISS index and metadata from disk"""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                # Load FAISS index
                self.index = faiss.read_index(self.index_path)
                
                # Load metadata
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.chunk_metadata = data['chunk_metadata']
                self.chunk_id_to_index = data['chunk_id_to_index']
                
                # Verify consistency
                if len(self.chunk_metadata) != self.index.ntotal:
                    logger.warning("Index and metadata size mismatch, rebuilding...")
                    self._rebuild_index(self.chunk_metadata)
                else:
                    logger.info(f"Loaded search index with {len(self.chunk_metadata)} chunks")
            else:
                logger.info("No existing index found, starting fresh")
                
        except Exception as e:
            logger.error(f"Failed to load index: {e}, starting fresh")
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.chunk_metadata = []
            self.chunk_id_to_index = {}
    
    def get_stats(self) -> Dict:
        """Get search engine statistics"""
        return {
            'total_chunks': len(self.chunk_metadata),
            'embedding_dimension': self.embedding_dim,
            'model_name': self.model_name,
            'index_size_mb': os.path.getsize(self.index_path) / (1024*1024) if os.path.exists(self.index_path) else 0,
            'documents': len(set(meta['document_id'] for meta in self.chunk_metadata))
        }
    
    def clear_index(self) -> None:
        """Clear all data from the search index"""
        logger.info("Clearing search index")
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.chunk_metadata = []
        self.chunk_id_to_index = {}
        
        # Remove saved files
        for path in [self.index_path, self.metadata_path]:
            if os.path.exists(path):
                os.remove(path)

# Utility functions for integration with existing codebase
def convert_search_results_to_dict(results: List[SearchResult]) -> List[Dict]:
    """Convert SearchResult objects to dictionaries for API responses"""
    return [
        {
            'chunk_id': result.chunk_id,
            'text': result.text,
            'document': result.document,
            'score': result.score,
            'page_numbers': result.page_numbers,
            'key_terms': result.key_terms,
            'metadata': result.metadata,
            'embedding_score': result.embedding_score
        }
        for result in results
    ]

def create_search_engine(config: Optional[Dict] = None) -> SemanticSearchEngine:
    """Factory function to create a configured search engine"""
    default_config = {
        'model_name': 'all-MiniLM-L6-v2',  # Fast and accurate
        'index_path': 'data/faiss_index.bin',
        'metadata_path': 'data/chunk_metadata.json'
    }
    
    if config:
        default_config.update(config)
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(default_config['index_path']), exist_ok=True)
    
    return SemanticSearchEngine(**default_config)