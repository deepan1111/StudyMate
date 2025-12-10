from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    document_ids: Optional[List[str]] = None
    max_context_chunks: Optional[int] = 5
    temperature: Optional[float] = 0.3

class SearchRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None
    limit: Optional[int] = 10
    include_key_terms: Optional[bool] = True