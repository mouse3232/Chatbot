import logging
import os
import sqlite3
import uuid
import json
from typing import Dict, Any, List

from backend.config import settings, PROJECT_ROOT
from backend.common.state import AstroState
from backend.common.parser import resolve_rows_from_text, resolve_rows_from_records

logger = logging.getLogger(__name__)

_embedding_cache = {}
_cached_device = None

def get_embedding_device() -> str:
    """Detect the best available device for embeddings (CUDA if available, else CPU)."""
    from backend.config import GPU_AVAILABLE
    return "cuda" if GPU_AVAILABLE else "cpu"

def get_embedding(model_name: str = "BAAI/bge-large-en-v1.5"):
    """Get or create a cached embedding model instance (HuggingFace only)."""
    # 1. Try to use pre-loaded RAG model from app.py to avoid redundant loading
    try:
        from backend.app import RAG_EMBED_MODEL
        if RAG_EMBED_MODEL and (model_name == "BAAI/bge-large-en-v1.5" or model_name == "BAAI/bge-small-en-v1.5"):
            return RAG_EMBED_MODEL
    except ImportError:
        pass

    if model_name in _embedding_cache:
        return _embedding_cache[model_name]

    from langchain_huggingface import HuggingFaceEmbeddings
    device = get_embedding_device()

    try:
        emb = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
        )
        logger.info("Loaded HuggingFace embedding model: %s on %s", model_name, device)
    except Exception:
        fallback = "BAAI/bge-large-en-v1.5"
        if model_name == fallback:
            raise
        logger.warning("Primary embedding failed, falling back to %s", fallback)
        return get_embedding(fallback)

    _embedding_cache[model_name] = emb
    return emb

def calculate_age(dob: str) -> int:
    import datetime
    try:
        # Assuming DD-MM-YYYY format based on UI default
        parts = dob.split('-')
        if len(parts) == 3:
            day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
            birth_date = datetime.date(year, month, day)
            today = datetime.date.today()
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            return max(0, age)
    except Exception:
        pass
    return 25 # Default fallback age if parsing fails

def collect_user_profile(state: AstroState) -> dict:
    """Validate user profile and generate a session ID."""
    # Create a copy to prevent destructive side effects but keep all fields (dob, tob, name2, etc.)
    up = (state.get("user_profile") or {}).copy()
    name = up.get("name", "User").strip()
    age = up.get("age", 25)
    gender = up.get("gender", "Male").strip()
    
    session_id = state.get("session_id") or str(uuid.uuid4())[:8]
    logger.info(
        "Node[collect]: %s (%s, age %d) -> session=%s",
        name, gender, age, session_id,
    )

    # Update only the primary validated fields
    up["name"] = name
    up["age"] = age
    up["gender"] = gender

    return {
        "session_id": session_id,
        "user_profile": up,
    }

# ── Interaction Logging (One-Row per Interaction) ────────────────────────────────
_pending_logs: Dict[str, Dict[str, Any]] = {}

def get_ist_time():
    """Returns current time in IST (UTC+5:30) as ISO string."""
    import datetime
    ist_delta = datetime.timedelta(hours=5, minutes=30)
    ist_tz = datetime.timezone(ist_delta)
    return datetime.datetime.now(ist_tz).isoformat()

def log_interaction(session_id: str, model_name: str, question: str, answer: str):
    """
    Starts a new interaction record. Flushes the PREVIOUS record if it exists.
    """
    # 1. Flush any existing interaction for this session before starting new one
    flush_interaction(session_id)
    
    # 2. Buffer the new interaction
    _pending_logs[session_id] = {
        "timestamp": get_ist_time(),
        "session_id": session_id,
        "model": model_name,
        "question": question,
        "answer": answer,
        "feedback": "none" # Default value
    }
    logger.info(f"Buffered interaction for {session_id}")

def log_feedback(session_id: str, feedback: str):
    """Updates the CURRENTLY BUFFERED interaction with user feedback."""
    if session_id in _pending_logs:
        _pending_logs[session_id]["feedback"] = feedback
        # Optional: Flush immediately upon feedback if you want
        flush_interaction(session_id)

def flush_interaction(session_id: str):
    """Permanently writes the buffered interaction for a session to the global file."""
    import json
    
    if session_id not in _pending_logs:
        return
        
    log_data = _pending_logs.pop(session_id)
    log_path = settings.GLOBAL_LOG_PATH
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data) + "\n")
    except Exception as e:
        logger.error(f"Failed to flush global interaction: {e}")

# ── New Consolidated Logging Architecture (v3.1) ────────────────────────────────

def _get_session_dir() -> str:
    """Returns the unified directory for all session logs."""
    path = settings.SESSION_LOGS_ROOT
    os.makedirs(path, exist_ok=True)
    return path

def log_general_timeline(message: str, category: str = "INFO", extra: Any = None):
    """
    Records a terminal-style log entry into the audit_logs table in sessions.db.
    """
    db_path = os.path.join(PROJECT_ROOT, "logs", "sessions.db")
    
    # Try to extract session_id from extra if it's a dict
    session_id = None
    if isinstance(extra, dict):
        session_id = extra.get("session_id")
    
    # Check if session_id is in the message (common pattern: "[Session: xxxxxxxx]")
    if not session_id and "[Session: " in message:
        import re
        match = re.search(r"\[Session: ([a-zA-Z0-9_-]+)\]", message)
        if match:
            session_id = match.group(1)

    entry = {
        "timestamp": get_ist_time(),
        "category": category,
        "message": message,
        "session_id": session_id,
        "metadata": json.dumps(extra) if extra else None
    }
    
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO audit_logs (timestamp, category, message, session_id, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (entry["timestamp"], entry["category"], entry["message"], entry["session_id"], entry["metadata"]))
        conn.commit()
        conn.close()
    except Exception:
        # Fallback to console if DB fails
        pass

def log_session_activity(session_id: str, event_type: str, payload: Any = None, response: Any = None, extra: Any = None):
    """
    (DISABLED) Legacy consolidated logging: Used to record every session event into a single JSON file.
    File: logs/session_log/{session_id}_activity.json
    """
    pass

# Shims for backward compatibility
def log_session_init(session_id: str, payload: Dict[str, Any]):
    log_session_activity(session_id, "INIT", payload=payload)

def log_session_timeline(session_id: str, method: str, path: str, request_data: Any, response_data: Any, extra_info: Any = None):
    log_session_activity(session_id, "API_CALL", payload={"method": method, "path": path, "req": request_data}, response=response_data, extra=extra_info)

def log_backend_trace(session_id: str, message: str):
    log_session_activity(session_id, "TRACE", payload=message)

def log_api_call(session_id, method, path, request_body, status_code, response_body, source_info=None):
    log_session_timeline(session_id, method, path, request_body, response_body, source_info)

def log_session_conv(session_id, question, answer):
    log_session_activity(session_id, "CONV", payload={"Q": question, "A": answer})

def save_llm_context(interaction_id: str, payload: Dict[str, Any]):
    """
    Save the raw LLM request context to a temporary directory for debugging.
    Files are stored in /logs/temp_context/{interaction_id}.json
    """
    try:
        from backend.config import PROJECT_ROOT
        temp_dir = os.path.join(PROJECT_ROOT, "logs", "temp_context")
        os.makedirs(temp_dir, exist_ok=True)
        
        file_path = os.path.join(temp_dir, f"{interaction_id}.json")
        
        # Add timestamp if not present
        if "timestamp" not in payload:
            from datetime import datetime
            payload["timestamp"] = datetime.now().isoformat()
            
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            
        logger.debug(f"[Utils] LLM context saved: {interaction_id}.json")
    except Exception as e:
        logger.error(f"[Utils] Failed to save LLM context: {e}")

def resolve_rows_from_remote(state: AstroState) -> dict:
    """Resolve dataset references from records or fallback to remote server."""
    
    records = state.get("records")
    records_data = state.get("records_data")

    if records:
        logger.info("Node[resolve]: Using DatasetReferences from client API")
        from backend.common.parser import resolve_rows_from_records
        refs = resolve_rows_from_records(records_list=records)
    elif records_data:
        logger.info("Node[resolve]: Parsing raw records_data for DatasetReferences")
        from backend.common.parser import resolve_rows_from_text
        refs = resolve_rows_from_text(content=records_data)
    else:
        if state.get("selected_row_refs"):
            return {}
        refs = []

    return {"selected_row_refs": refs}


