# import fitz  # PyMuPDF
# import re
# import nltk
# from typing import List, Dict, Any, Optional
# from dataclasses import dataclass
# from datetime import datetime
# import hashlib
# import logging
# from pathlib import Path

# # Download required NLTK data
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')

# @dataclass
# class DocumentMetadata:
#     """Document metadata structure"""
#     title: Optional[str] = None
#     author: Optional[str] = None
#     subject: Optional[str] = None
#     creator: Optional[str] = None
#     producer: Optional[str] = None
#     creation_date: Optional[str] = None
#     modification_date: Optional[str] = None
#     page_count: int = 0
#     file_size: int = 0
#     file_hash: Optional[str] = None
#     language: Optional[str] = None
#     word_count: int = 0

# @dataclass
# class ProcessedChunk:
#     """Processed text chunk with metadata"""
#     text: str
#     chunk_id: str
#     page_number: int
#     chunk_index: int
#     word_count: int
#     char_count: int
#     metadata: Dict[str, Any]

# class PDFProcessor:
#     """Advanced PDF processing with PyMuPDF"""
    
#     def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         self.logger = logging.getLogger(__name__)
        
#     def extract_text_and_metadata(self, pdf_path: str) -> tuple[str, DocumentMetadata]:
#         """Extract text and metadata from PDF"""
#         try:
#             doc = fitz.open(pdf_path)
            
#             # Extract metadata
#             metadata = self._extract_metadata(doc, pdf_path)
            
#             # Extract text with optimization for large PDFs
#             full_text = ""
#             for page_num in range(len(doc)):
#                 page = doc.load_page(page_num)
#                 text = page.get_text()
#                 full_text += f"\n--- Page {page_num + 1} ---\n{text}"
            
#             doc.close()
#             return full_text, metadata
            
#         except Exception as e:
#             self.logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
#             raise
    
#     def _extract_metadata(self, doc: fitz.Document, pdf_path: str) -> DocumentMetadata:
#         """Extract comprehensive metadata from PDF"""
#         pdf_metadata = doc.metadata
#         file_path = Path(pdf_path)
        
#         # Calculate file hash for deduplication
#         file_hash = self._calculate_file_hash(pdf_path)
        
#         # Get file size
#         file_size = file_path.stat().st_size if file_path.exists() else 0
        
#         return DocumentMetadata(
#             title=pdf_metadata.get('title', ''),
#             author=pdf_metadata.get('author', ''),
#             subject=pdf_metadata.get('subject', ''),
#             creator=pdf_metadata.get('creator', ''),
#             producer=pdf_metadata.get('producer', ''),
#             creation_date=pdf_metadata.get('creationDate', ''),
#             modification_date=pdf_metadata.get('modDate', ''),
#             page_count=len(doc),
#             file_size=file_size,
#             file_hash=file_hash
#         )
    
#     def _calculate_file_hash(self, file_path: str) -> str:
#         """Calculate SHA-256 hash of file"""
#         hash_sha256 = hashlib.sha256()
#         with open(file_path, "rb") as f:
#             for chunk in iter(lambda: f.read(4096), b""):
#                 hash_sha256.update(chunk)
#         return hash_sha256.hexdigest()
    
#     def clean_text(self, text: str) -> str:
#         """Clean and preprocess extracted text"""
#         # Remove multiple whitespaces
#         text = re.sub(r'\s+', ' ', text)
        
#         # Remove page headers/footers patterns
#         text = re.sub(r'--- Page \d+ ---', '', text)
        
#         # Fix common PDF extraction issues
#         text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between joined words
#         text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Add space after punctuation
        
#         # Remove excessive line breaks
#         text = re.sub(r'\n+', '\n', text)
        
#         # Trim whitespace
#         text = text.strip()
        
#         return text
    
#     def chunk_text(self, text: str, metadata: DocumentMetadata) -> List[ProcessedChunk]:
#         """Chunk text with overlap and metadata tagging"""
#         # Clean the text first
#         cleaned_text = self.clean_text(text)
        
#         # Split into sentences for better chunking
#         sentences = nltk.sent_tokenize(cleaned_text)
        
#         chunks = []
#         current_chunk = ""
#         chunk_index = 0
        
#         for sentence in sentences:
#             # Check if adding this sentence would exceed chunk size
#             if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
#                 # Create chunk
#                 chunk = self._create_chunk(
#                     current_chunk, 
#                     chunk_index, 
#                     metadata
#                 )
#                 chunks.append(chunk)
                
#                 # Start new chunk with overlap
#                 overlap_text = self._get_overlap_text(current_chunk)
#                 current_chunk = overlap_text + " " + sentence
#                 chunk_index += 1
#             else:
#                 current_chunk += " " + sentence if current_chunk else sentence
        
#         # Add final chunk
#         if current_chunk:
#             chunk = self._create_chunk(current_chunk, chunk_index, metadata)
#             chunks.append(chunk)
        
#         return chunks
    
#     def _create_chunk(self, text: str, chunk_index: int, metadata: DocumentMetadata) -> ProcessedChunk:
#         """Create a processed chunk with metadata"""
#         chunk_id = f"{metadata.file_hash}_{chunk_index}" if metadata.file_hash else f"chunk_{chunk_index}"
        
#         return ProcessedChunk(
#             text=text.strip(),
#             chunk_id=chunk_id,
#             page_number=0,  # Could be enhanced to track actual page numbers
#             chunk_index=chunk_index,
#             word_count=len(text.split()),
#             char_count=len(text),
#             metadata={
#                 "source_file": metadata.title or "unknown",
#                 "author": metadata.author,
#                 "creation_date": metadata.creation_date,
#                 "total_pages": metadata.page_count
#             }
#         )
    
#     def _get_overlap_text(self, text: str) -> str:
#         """Get overlap text from the end of current chunk"""
#         if len(text) <= self.chunk_overlap:
#             return text
        
#         # Find the last complete sentence within overlap size
#         overlap_candidate = text[-self.chunk_overlap:]
#         last_sentence_end = overlap_candidate.rfind('.')
        
#         if last_sentence_end != -1:
#             return text[-(self.chunk_overlap - last_sentence_end - 1):]
#         else:
#             return overlap_candidate
    
#     def process_multiple_pdfs(self, pdf_paths: List[str]) -> Dict[str, List[ProcessedChunk]]:
#         """Process multiple PDFs with deduplication"""
#         results = {}
#         processed_hashes = set()
        
#         for pdf_path in pdf_paths:
#             try:
#                 text, metadata = self.extract_text_and_metadata(pdf_path)
                
#                 # Skip duplicates based on file hash
#                 if metadata.file_hash in processed_hashes:
#                     self.logger.info(f"Skipping duplicate file: {pdf_path}")
#                     continue
                
#                 processed_hashes.add(metadata.file_hash)
#                 chunks = self.chunk_text(text, metadata)
#                 results[pdf_path] = chunks
                
#                 self.logger.info(f"Processed {pdf_path}: {len(chunks)} chunks")
                
#             except Exception as e:
#                 self.logger.error(f"Failed to process {pdf_path}: {str(e)}")
#                 results[pdf_path] = []
        
#         return results
    
#     def get_processing_stats(self, chunks: List[ProcessedChunk]) -> Dict[str, Any]:
#         """Get statistics about processed chunks"""
#         if not chunks:
#             return {}
        
#         total_words = sum(chunk.word_count for chunk in chunks)
#         total_chars = sum(chunk.char_count for chunk in chunks)
        
#         return {
#             "total_chunks": len(chunks),
#             "total_words": total_words,
#             "total_characters": total_chars,
#             "avg_chunk_size": total_words / len(chunks),
#             "max_chunk_size": max(chunk.word_count for chunk in chunks),
#             "min_chunk_size": min(chunk.word_count for chunk in chunks)
#         }

import fitz  # PyMuPDF
import re
import nltk
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import logging
from pathlib import Path
import json

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

@dataclass
class DocumentMetadata:
    """Enhanced document metadata structure"""
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    page_count: int = 0
    file_size: int = 0
    file_hash: Optional[str] = None
    language: Optional[str] = None
    word_count: int = 0
    document_type: Optional[str] = None  # textbook, paper, notes, etc.
    
@dataclass
class ProcessedChunk:
    """Enhanced processed chunk with better metadata"""
    text: str
    chunk_id: str
    page_numbers: List[int]  # Can span multiple pages
    chunk_index: int
    word_count: int
    char_count: int
    metadata: Dict[str, Any]
    entities: List[str] = field(default_factory=list)  # Named entities
    key_terms: List[str] = field(default_factory=list)  # Important terms

class PDFProcessor:
    """Enhanced PDF processor with better text extraction"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)
        
    def extract_text_and_metadata(self, pdf_path: str) -> Tuple[str, DocumentMetadata]:
        """Enhanced text extraction with structure preservation"""
        try:
            doc = fitz.open(pdf_path)
            metadata = self._extract_metadata(doc, pdf_path)
            
            # Extract structured text
            full_text = ""
            page_texts = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract text blocks with position info
                blocks = page.get_text("dict")
                page_text = self._extract_structured_text(blocks)
                
                page_texts.append(page_text)
                full_text += f"\n[Page {page_num + 1}]\n{page_text}"
            
            # Update word count
            metadata.word_count = len(full_text.split())
            
            # Detect document type
            metadata.document_type = self._detect_document_type(full_text, metadata)
            
            doc.close()
            return full_text, metadata
            
        except Exception as e:
            self.logger.error(f"Error processing PDF: {str(e)}")
            raise
    
    def _extract_structured_text(self, blocks: Dict) -> str:
        """Extract text while preserving structure"""
        text_parts = []
        
        for block in blocks.get("blocks", []):
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                    
                    if line_text.strip():
                        text_parts.append(line_text.strip())
        
        return "\n".join(text_parts)
    
    def _detect_document_type(self, text: str, metadata: DocumentMetadata) -> str:
        """Detect the type of academic document"""
        text_lower = text.lower()
        
        # Check for common patterns
        if "abstract" in text_lower and "references" in text_lower:
            return "research_paper"
        elif "chapter" in text_lower and "section" in text_lower:
            return "textbook"
        elif "lecture" in text_lower or "slide" in text_lower:
            return "lecture_notes"
        elif "exam" in text_lower or "quiz" in text_lower:
            return "exam_material"
        else:
            return "general_document"
    
    def _extract_metadata(self, doc: fitz.Document, pdf_path: str) -> DocumentMetadata:
        """Extract comprehensive metadata"""
        pdf_metadata = doc.metadata
        file_path = Path(pdf_path)
        file_hash = self._calculate_file_hash(pdf_path)
        file_size = file_path.stat().st_size if file_path.exists() else 0
        
        # Extract keywords if available
        keywords = []
        if pdf_metadata.get('keywords'):
            keywords = [k.strip() for k in pdf_metadata['keywords'].split(',')]
        
        return DocumentMetadata(
            title=pdf_metadata.get('title', file_path.stem),
            author=pdf_metadata.get('author', ''),
            subject=pdf_metadata.get('subject', ''),
            keywords=keywords,
            creation_date=pdf_metadata.get('creationDate', ''),
            modification_date=pdf_metadata.get('modDate', ''),
            page_count=len(doc),
            file_size=file_size,
            file_hash=file_hash
        )
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common PDF issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        
        # Preserve page markers
        text = re.sub(r'\[Page (\d+)\]', r'\n[PAGE_\1]\n', text)
        
        # Remove hyphenation at line ends
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        return text.strip()
    
    def chunk_text(self, text: str, metadata: DocumentMetadata) -> List[ProcessedChunk]:
        """Smart chunking with context preservation"""
        cleaned_text = self.clean_text(text)
        
        # Track page boundaries
        page_pattern = re.compile(r'\[PAGE_(\d+)\]')
        
        # Split into semantic units (paragraphs/sections)
        sections = self._split_into_sections(cleaned_text)
        
        chunks = []
        current_chunk = ""
        current_pages = set()
        chunk_index = 0
        
        for section in sections:
            # Extract page numbers from section
            page_matches = page_pattern.findall(section)
            if page_matches:
                current_pages.update(int(p) for p in page_matches)
            
            # Clean section text
            section_text = page_pattern.sub('', section).strip()
            
            if not section_text:
                continue
            
            # Check if adding section exceeds chunk size
            if len(current_chunk) + len(section_text) > self.chunk_size and current_chunk:
                # Create chunk
                chunk = self._create_enhanced_chunk(
                    current_chunk,
                    chunk_index,
                    list(current_pages),
                    metadata
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = self._get_smart_overlap(current_chunk)
                current_chunk = overlap_text + " " + section_text
                current_pages = set(page_matches) if page_matches else current_pages
                chunk_index += 1
            else:
                current_chunk += " " + section_text if current_chunk else section_text
        
        # Add final chunk
        if current_chunk:
            chunk = self._create_enhanced_chunk(
                current_chunk,
                chunk_index,
                list(current_pages),
                metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_into_sections(self, text: str) -> List[str]:
        """Split text into semantic sections"""
        # Split by double newlines or page markers
        sections = re.split(r'\n\n+|\[PAGE_\d+\]', text)
        
        # Further split very long sections by sentences
        final_sections = []
        for section in sections:
            if len(section) > self.chunk_size * 1.5:
                sentences = nltk.sent_tokenize(section)
                temp_section = ""
                for sent in sentences:
                    if len(temp_section) + len(sent) > self.chunk_size:
                        if temp_section:
                            final_sections.append(temp_section)
                        temp_section = sent
                    else:
                        temp_section += " " + sent if temp_section else sent
                if temp_section:
                    final_sections.append(temp_section)
            else:
                final_sections.append(section)
        
        return final_sections
    
    def _create_enhanced_chunk(
        self,
        text: str,
        chunk_index: int,
        page_numbers: List[int],
        metadata: DocumentMetadata
    ) -> ProcessedChunk:
        """Create enhanced chunk with additional metadata"""
        chunk_id = f"{metadata.file_hash}_{chunk_index}" if metadata.file_hash else f"chunk_{chunk_index}"
        
        # Extract key terms (simple frequency-based for now)
        key_terms = self._extract_key_terms(text)
        
        return ProcessedChunk(
            text=text.strip(),
            chunk_id=chunk_id,
            page_numbers=page_numbers or [0],
            chunk_index=chunk_index,
            word_count=len(text.split()),
            char_count=len(text),
            metadata={
                "source_file": metadata.title or "unknown",
                "author": metadata.author,
                "document_type": metadata.document_type,
                "creation_date": metadata.creation_date,
                "total_pages": metadata.page_count,
                "pages_covered": page_numbers
            },
            key_terms=key_terms,
            entities=[]  # Placeholder for NER
        )
    
    def _extract_key_terms(self, text: str, top_n: int = 5) -> List[str]:
        """Extract key terms from text"""
        # Simple frequency-based extraction
        words = re.findall(r'\b[a-z]+\b', text.lower())
        
        # Filter common words (simple stopword list)
        common_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'could', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just'}
        
        words = [w for w in words if w not in common_words and len(w) > 3]
        
        # Count frequencies
        from collections import Counter
        word_freq = Counter(words)
        
        # Return top terms
        return [word for word, _ in word_freq.most_common(top_n)]
    
    def _get_smart_overlap(self, text: str) -> str:
        """Get intelligent overlap preserving sentence boundaries"""
        if len(text) <= self.chunk_overlap:
            return text
        
        # Try to find a sentence boundary
        overlap_candidate = text[-self.chunk_overlap:]
        sentences = nltk.sent_tokenize(overlap_candidate)
        
        if len(sentences) > 1:
            # Keep last complete sentence(s)
            return ' '.join(sentences[-2:]) if len(' '.join(sentences[-2:])) <= self.chunk_overlap else sentences[-1]
        else:
            return overlap_candidate
    
    def get_processing_stats(self, chunks: List[ProcessedChunk]) -> Dict[str, Any]:
        """Get detailed processing statistics"""
        if not chunks:
            return {}
        
        total_words = sum(chunk.word_count for chunk in chunks)
        total_chars = sum(chunk.char_count for chunk in chunks)
        all_pages = set()
        for chunk in chunks:
            all_pages.update(chunk.page_numbers)
        
        return {
            "total_chunks": len(chunks),
            "total_words": total_words,
            "total_characters": total_chars,
            "avg_chunk_size": total_words / len(chunks) if chunks else 0,
            "max_chunk_size": max(chunk.word_count for chunk in chunks),
            "min_chunk_size": min(chunk.word_count for chunk in chunks),
            "pages_processed": len(all_pages),
            "key_terms_extracted": len(set(term for chunk in chunks for term in chunk.key_terms))
        }