import logging
from typing import Any, Dict, List, Optional, TypedDict

class AstroState(TypedDict):
    session_id: str
    user_profile: Dict[str, Any]  # name, age, gender
    selected_row_refs: List[Dict[str, Any]]
    session_documents: List[str]
    messages: List[Dict[str, str]]
    retrieved_docs: List[str]
    retrieved_sources: List[Dict[str, Any]]
    final_answer: str
    thinking: str
    suggestions: List[str]
    # Runtime settings (set by UI)
    k_value: int
    llm_model: str
    embedding_model: str
    session_embedding_model: str
    answer_length: str
    # Added for record passing
    records: Optional[List[Dict[str, Any]]]
    records_data: Optional[str]
    # Phase 43: Advanced RAG
    expanded_queries: List[str]
    detected_topic: Optional[str]
    predictions: Optional[List[Any]]
    m_predictions: Optional[List[Any]]
    # RAG Pipeline fields
    search_requests: Optional[List[Dict[str, Any]]]
    remote_text: Optional[str]
    remote_type: Optional[str]
    timing_records: Optional[List[Dict[str, Any]]]
    raw_texts: Optional[List[Dict[str, Any]]]
