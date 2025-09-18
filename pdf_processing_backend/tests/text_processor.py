"""Advanced text processing and NLP utilities"""

import re
import nltk
from typing import List, Dict, Any, Optional, Tuple
from langdetect import detect, LangDetectError
import logging
from dataclasses import dataclass

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.stem import PorterStemmer, WordNetLemmatizer

@dataclass
class TextAnalysis:
    """Text analysis results"""
    language: Optional[str] = None
    word_count: int = 0
    sentence_count: int = 0
    avg_sentence_length: float = 0.0
    readability_score: Optional[float] = None
    top_keywords: List[Tuple[str, int]] = None
    entities: List[str] = None
    sentiment: Optional[str] = None

class TextProcessor:
    """Advanced text processing with NLP capabilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize stop words for common languages
        self.stop_words = {
            'english': set(stopwords.words('english')),
        }
        
        # Add more languages if needed
        try:
            self.stop_words['spanish'] = set(stopwords.words('spanish'))
            self.stop_words['french'] = set(stopwords.words('french'))
            self.stop_words['german'] = set(stopwords.words('german'))
        except:
            pass
    
    def detect_language(self, text: str) -> Optional[str]:
        """Detect text language"""
        try:
            # Use a sample of text for better performance on large documents
            sample_text = text[:1000] if len(text) > 1000 else text
            language = detect(sample_text)
            return language
        except LangDetectError:
            self.logger.warning("Could not detect language, defaulting to English")
            return 'en'
    
    def clean_text(self, text: str, aggressive: bool = False) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Basic cleaning
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        if aggressive:
            text = re.sub(r'[^\w\s.,!?;:\-\'\"]', '', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Space between joined words
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Space after punctuation
        
        # Remove multiple consecutive punctuation
        text = re.sub(r'[.]{2,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.!?,:;])', r'\1', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        return text.strip()
    
    def extract_keywords(self, text: str, top_k: int = 10, min_word_length: int = 3) -> List[Tuple[str, int]]:
        """Extract top keywords from text"""
        if not text:
            return []
        
        # Detect language for appropriate stop words
        language = self.detect_language(text)
        lang_key = 'english'  # Default
        
        if language:
            lang_map = {'en': 'english', 'es': 'spanish', 'fr': 'french', 'de': 'german'}
            lang_key = lang_map.get(language, 'english')
        
        stop_words = self.stop_words.get(lang_key, self.stop_words['english'])
        
        # Tokenize and clean
        words = word_tokenize(text.lower())
        
        # Filter words
        filtered_words = [
            word for word in words 
            if (word.isalpha() and 
                len(word) >= min_word_length and 
                word not in stop_words)
        ]
        
        # Count frequency
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return top_keywords[:top_k]
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities (basic pattern-based approach)"""
        if not text:
            return []
        
        entities = []
        
        # Extract potential person names (Title Case patterns)
        person_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b'
        persons = re.findall(person_pattern, text)
        entities.extend([('PERSON', p) for p in persons])
        
        # Extract dates
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY or MM-DD-YYYY
            r'\b[A-Z][a-z]+ \d{1,2},? \d{4}\b',    # January 1, 2023
            r'\b\d{1,2} [A-Z][a-z]+ \d{4}\b'       # 1 January 2023
        ]
        
        for pattern in date_patterns:
            dates = re.findall(pattern, text)
            entities.extend([('DATE', d) for d in dates])
        
        # Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        entities.extend([('EMAIL', e) for e in emails])
        
        # Extract phone numbers
        phone_pattern = r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'
        phones = re.findall(phone_pattern, text)
        entities.extend([('PHONE', '-'.join(p)) for p in phones])
        
        return entities
    
    def analyze_text(self, text: str) -> TextAnalysis:
        """Perform comprehensive text analysis"""
        if not text:
            return TextAnalysis()
        
        # Basic metrics
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        
        word_count = len([w for w in words if w.isalpha()])
        sentence_count = len(sentences)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Language detection
        language = self.detect_language(text)
        
        # Extract keywords
        keywords = self.extract_keywords(text