import logging
import uuid
import re
import datetime
from typing import Dict, List, Optional, Any, Union
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import json
import httpx
import urllib.parse
import asyncio

# Adjust import paths for the backend modules
import sys
import os
import re

def clean_non_ascii(text: Any) -> Any:
    """Recursively remove non-ASCII arrow characters from strings/lists/dicts."""
    if isinstance(text, str):
        return text.replace("\u2192", "")
    elif isinstance(text, list):
        return [clean_non_ascii(i) for i in text]
    elif isinstance(text, dict):
        return {k: clean_non_ascii(v) for k, v in text.items()}
    return text

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ── Startup Mode Flag ─────────────────────────────────────────────────────────
# Usage: python -m backend.app --t   → Timing-only mode
TIMING_ONLY_MODE = "--t" in sys.argv
if TIMING_ONLY_MODE:
    print("\n[MODE] TIMING-ONLY MODE ACTIVE (--t flag detected)\n")

from config import LLM_MODELS, EMBEDDING_CONFIGS, settings, PROJECT_ROOT
from standard.workflow import build_setup_graph_standard as standard_setup
from backend.standard.suggestion_pool import get_random_suggestions, generate_initial_suggestions

from backend.common.session_manager import SessionManager
from backend.common.utils import log_api_call, log_session_conv, log_interaction
from backend.common.llm import create_llm, strip_think_tags

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Silence noisy third-party libraries
for lib in ["httpx", "sentence_transformers", "huggingface_hub", "urllib3"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

# ── Middleware (ASGI Implementation for Streaming Compatibility) ──
class SessionLoggingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # 1. Capture Request Body without breaking streaming
        body = b""
        async def receive_with_logging():
            nonlocal body
            message = await receive()
            if message["type"] == "http.request":
                body += message.get("body", b"")
            return message

        # 2. Log Request (Simple)
        path = scope.get("path", "")
        if "/api/" in path:
             logger.info(f"--- API CALL: {scope.get('method')} {path} ---")

        # 3. Pass the logging-aware receive function
        await self.app(scope, receive_with_logging, send)

app = FastAPI(title="Astrology AI Backend API")

app.add_middleware(SessionLoggingMiddleware)
# Add CORS so frontend can communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Session Manager with SQLite database
session_manager = SessionManager(os.path.join(PROJECT_ROOT, "logs", "sessions.db"))

# ── RAG Model Globals (loaded at startup, shared across all sessions) ─────────
RAG_EMBED_MODEL = None      # SentenceTransformer: BAAI/bge-large-en-v1.5
RAG_RERANKER = None         # CrossEncoder: BAAI/bge-reranker-large
RAG_LABEL_VECS = None       # Pre-computed label description vectors
rag_session_manager = None  # RAGSessionManager instance

@app.on_event("startup")
async def startup_event():
    """Execute startup tasks."""
    global RAG_EMBED_MODEL, RAG_RERANKER, RAG_LABEL_VECS, rag_session_manager

    # 1. Start General Timeline
    from backend.common.utils import log_general_timeline
    log_general_timeline("SYSTEM_START: Astrology AI Backend Services Initialized.")

    # 1a. Run data retention cleanup
    try:
        session_manager.cleanup_retention()
    except Exception as e:
        logger.error(f"Retention cleanup failed: {e}")


    # 3. Load RAG Embedding Model + Reranker (replaces BERT + T5)
    try:
        import torch
        from sentence_transformers import SentenceTransformer, CrossEncoder
        from backend.astro_rag.pipeline.labeler import build_label_description_vectors
        from backend.astro_rag.session.rag_session_manager import RAGSessionManager

        device = "cuda" if torch.cuda.is_available() else "cpu"
        mode_label = "TIMING-ONLY" if TIMING_ONLY_MODE else "ALL DOMAINS"
        print("\n" + "*"*60)
        print(f"[RAG] PRE-LOADING RAG MODELS (device: {device} | mode: {mode_label})...")

        # 3a. Embedding model
        RAG_EMBED_MODEL = SentenceTransformer("BAAI/bge-large-en-v1.5")
        RAG_EMBED_MODEL.to(device)
        print("  [OK] bge-large-en-v1.5 loaded")

        # 3b. Reranker model
        RAG_RERANKER = CrossEncoder("BAAI/bge-reranker-large")
        RAG_RERANKER.model.to(device)
        print("  [OK] bge-reranker-large loaded")

        # 3c. Pre-compute label description vectors
        RAG_LABEL_VECS = build_label_description_vectors(RAG_EMBED_MODEL)
        print(f"  [OK] Label vectors computed ({RAG_LABEL_VECS.shape})")

        # 3d. Initialize RAG session manager
        rag_session_manager = RAGSessionManager()
        print("  [OK] RAGSessionManager initialized")

        print("*"*60 + "\n")
    except Exception as e:
        logger.error(f"RAG model loading failed: {e}")

    # Silence specific library warnings
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
    warnings.filterwarnings("ignore", message=".*allowed_objects.*")




# ── Pydantic Models ──────────────────────────────────────────────

class UserProfile(BaseModel):
    # Primary User
    name: str = "User"
    age: int = 25
    gender: str = "Male"
    dob: Optional[str] = None
    tob: Optional[str] = None
    city: Optional[str] = None
    countrycode: Optional[str] = None
    latitude: Optional[str] = None
    longitude: Optional[str] = None
    zone: Optional[str] = None
    tcor: Optional[str] = None
    
    # Partner / Secondary User
    name2: Optional[str] = None
    age2: Optional[int] = None
    gender2: Optional[str] = None
    dob2: Optional[str] = None
    tob2: Optional[str] = None
    city2: Optional[str] = None
    countrycode2: Optional[str] = None
    latitude2: Optional[str] = None
    longitude2: Optional[str] = None
    zone2: Optional[str] = None
    tcor2: Optional[str] = None
    
    # Misc
    languagecode: Optional[Union[int, str]] = None
    
    # Matching Specifics
    m_predictions: Optional[List[str]] = None
    ashtkoot_guna: Optional[Dict[str, Any]] = None
    
    # Remedies Specifics
    predictions: Optional[List[str]] = None

    class Config:
        extra = "allow" # Capture all provided fields dynamically

class RecordTable(BaseModel):
    table: str
    keys: List[str]
    values: List[List[Any]]

class PredictionItem(BaseModel):
    table: str
    keys: List[str]
    predictions: List[str]

class RawTextItem(BaseModel):
    """Raw text blob to be semantically chunked alongside DB records."""
    label: str = "Custom Text"
    text: str
    domain: Optional[str] = None  # "astrology" | "remedies" | "timing" | None=auto-detect

class InitRequest(BaseModel):
    user_profile: Optional[UserProfile] = None
    records: Optional[List[RecordTable]] = None
    records2: Optional[List[RecordTable]] = None
    matchrecords: Optional[List[RecordTable]] = None
    remedies: Optional[List[RecordTable]] = None
    timing_records: Optional[List[RecordTable]] = None  # Dedicated timing records
    predictions: Optional[List[PredictionItem]] = None
    m_predictions: Optional[List[PredictionItem]] = None
    ashtkoot_guna: Optional[Dict[str, Any]] = None
    raw_texts: Optional[List[RawTextItem]] = None  # Free-form text for semantic chunking
    session_id: Optional[str] = None
    mobile_number: Optional[str] = None

class ChatRequest(BaseModel):
    session_id: str
    message: str
    parent_id: Optional[str] = None # Link to previous interaction_id
    intent_override: Optional[str] = None # Optional: "KUNDALI" or "MATCHING"
    secondary_language_code: Optional[Union[int, str]] = None # Optional: Overrides profile language
    horary_number: Optional[int] = None
    probability_feedback: Optional[str] = None # e.g. "Chancers of happening this event is 80%"
    intent_preamble: Optional[str] = None # The preamble text generated in step 1
    stream: bool = False
    chart_data: Optional[Dict[str, Any]] = None
    debug: bool = False  # When True, return decomposer output + RAG chunks in SSE

class SettingsUpdateRequest(BaseModel):
    session_id: str
    k_value: Optional[int] = None
    llm_model: Optional[str] = None

class MatchingRequest(BaseModel):
    session_id: Optional[str] = None
    user_profile: Optional[UserProfile] = None
    records: Optional[List[RecordTable]] = None
    records2: Optional[List[RecordTable]] = None
    matchrecords: Optional[List[RecordTable]] = None
    remedies: Optional[List[RecordTable]] = None
    predictions: Optional[List[PredictionItem]] = None
    m_predictions: Optional[str] = None
    ashtkoot_guna: Optional[Dict[str, Any]] = None
    query: Optional[str] = None # Quick query support for one-shot init

class MatchingQueryRequest(BaseModel):
    session_id: str
    query: str
    language: Optional[str] = "English"
    k_value: Optional[int] = None
    llm_model: Optional[str] = None
    chart_data: Optional[Dict[str, Any]] = None

class FeedbackRequest(BaseModel):
    session_id: str
    question: Optional[str] = None
    answer: Optional[str] = None
    feedback: str # "good" or "bad"

class LanguageRequest(BaseModel):
    language: str # e.g., "Hindi", "English", "Marathi", etc.

# Global dict to store LIVE session state (state objects, graph runners)
ACTIVE_SESSIONS: Dict[str, Dict] = {}

# Persistent defaults for new sessions (can be updated via UI)
GLOBAL_DEFAULTS = {
    "k_value": 5,
    "llm_model": "sarvam/sarvam-30b"
}

SUPPORTED_LANGUAGES = [
    "English", "Hindi", "Hinglish", "Gujarati", "Marathi", 
    "Bengali", "Nepali", "Telugu", "Malayalam", "Kannada", "Tamil", "Oriya", "Spanish"
]

# ── Utilities ──────────────────────────────────────────────────
def is_valid_session_id(sid: Optional[str]) -> bool:
    """Check if sid is a valid 8-character hex string with at least one letter."""
    if not sid or len(sid) != 8:
        return False
    # Must be 8-char hex AND contain at least one alpha (a-f) to be distinct from numbers
    return bool(re.fullmatch(r"[0-9a-fA-F]{8}", sid)) and any(c.isalpha() for c in sid)

def generate_session_id() -> str:
    """Generate a valid 8-character hex session ID with at least one letter."""
    import uuid
    for _ in range(10): # Safety loop
        sid = str(uuid.uuid4())[:8]
        if is_valid_session_id(sid):
            return sid
    return "a1b2c3d4" # Hard fallback

# ── Endpoints ────────────────────────────────────────────────────

@app.get("/api/config/models")
async def get_models():
    """Return configured models and languages."""
    return {
        "llms": list(LLM_MODELS.keys()),
        "embeddings": list(EMBEDDING_CONFIGS.keys()),
        "languages": SUPPORTED_LANGUAGES,
        "timing_only_mode": TIMING_ONLY_MODE
    }

@app.get("/api/sessions")
def list_sessions():
    """List sessions from the persistent DB."""
    return {"sessions": session_manager.list_sessions()}

@app.get("/api/sessions/{session_id}")
def get_session_details(session_id: str):
    """Get full history from the persistent DB and clean internal references."""
    data = session_manager.get_session(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Session not found")

    # 1. Clean row_refs and other internal fields from the response
    data.pop("row_refs", None)
    
    # 2. STRIP HEAVY DATA: Remove records from the profile for the frontend response
    if "user_profile" in data:
        profile = data["user_profile"]
        profile.pop("records", None)
        profile.pop("records2", None)
        profile.pop("remedies", None)
        profile.pop("predictions", None)
    
    # 3. Clean (Reference: ...) blocks from historical messages for current UI display
    if "messages" in data:
        # Deduplicate and Clean
        seen = set()
        unique_messages = []
        for msg in data["messages"]:
            content = msg.get("content", "")
            role = msg.get("role")
            
            # Simple deduplication (Role + Content)
            msg_key = f"{role}:{content}"
            if msg_key in seen:
                continue
            seen.add(msg_key)

            if role == "assistant" and content:
                # Remove (Reference: ...) and any trailing whitespace
                content = re.sub(r"\s*\(Reference:.*?\)\s*", "", content, flags=re.DOTALL).strip()
            
            unique_messages.append({
                "role": role,
                "content": content
            })
        data["messages"] = unique_messages
                
    return data

@app.post("/api/sessions/init")
async def init_session(payload: Union[InitRequest, List[InitRequest]], request: Request):
    """
    Manually initialize or resume a session with profile and records.
    Returns: {"session_id": "...", "status": "initialized" | "resumed", ...}
    """
    # Robustness: Handle list-wrapped payloads [ { ... } ]
    req = payload[0] if isinstance(payload, list) else payload
    # 1. Validate or Generate Session ID
    session_id = req.session_id if is_valid_session_id(req.session_id) else generate_session_id()
    
    # 2. Check for history and logs
    db_data = session_manager.get_session(session_id)
    history = db_data["messages"] if db_data else []
    
    # 2. Determine profile (Use request if provided, else DB)
    if req.user_profile:
        user_p = req.user_profile.model_dump()
    elif db_data:
        user_p = db_data["user_profile"]
    else:
        raise HTTPException(status_code=400, detail="Session not found in DB and no profile provided.")

    # 3. Save/Update DB
    # We store the records inside the user_profile so agents can self-hydrate
    if req.records:
        user_p["records"] = [r.model_dump() for r in req.records]
    if req.records2:
        user_p["records2"] = [r.model_dump() for r in req.records2]
    if req.matchrecords:
        user_p["matchrecords"] = [r.model_dump() for r in req.matchrecords]
    if req.remedies:
        user_p["remedies"] = [r.model_dump() for r in req.remedies]
        
    # Flatten and store predictions
    if req.predictions:
        user_p["predictions"] = [p.model_dump() for p in req.predictions]
    if req.m_predictions:
        user_p["m_predictions"] = [p.model_dump() for p in req.m_predictions]
        
    if req.ashtkoot_guna:
        user_p["ashtkoot_guna"] = req.ashtkoot_guna.model_dump() if hasattr(req.ashtkoot_guna, "model_dump") else req.ashtkoot_guna
    if req.timing_records:
        user_p["timing_records"] = [r.model_dump() for r in req.timing_records]
    if req.raw_texts:
        user_p["raw_texts"] = [r.model_dump() for r in req.raw_texts]
    
    session_manager.save_session(session_id, user_p, mobile_number=req.mobile_number)
    
    initial_state = {
        "user_profile": user_p,
        "session_id": session_id,
        "records": [r.model_dump() for r in req.records] if req.records else None,
        "records2": [r.model_dump() for r in req.records2] if req.records2 else None,
        "matchrecords": [r.model_dump() for r in req.matchrecords] if req.matchrecords else None,
        "remedies": [r.model_dump() for r in req.remedies] if req.remedies else None,
        
        # Keep structured prediction dicts for VDB tagging
        "predictions": user_p.get("predictions"),
        "m_predictions": user_p.get("m_predictions"),
        
        "ashtkoot_guna": req.ashtkoot_guna,
        "timing_records": [r.model_dump() for r in req.timing_records] if req.timing_records else None,
        "raw_texts": [r.model_dump() for r in req.raw_texts] if req.raw_texts else None,
        "selected_row_refs": [], # Will be populated by setup graph
        "messages": history,
        "k_value": GLOBAL_DEFAULTS["k_value"],
        "llm_model": GLOBAL_DEFAULTS["llm_model"],
        "language": "English"
    }

    # 4. Strict Payload-Driven PKL Hydration
    # We will build exactly the .pkl files needed based on the session payload.
    from backend.common.parser import parse_compressed_records as r_to_refs
    
    setup_graph = standard_setup()
    
    is_remedies = bool(req.remedies)
    is_matching = bool(req.matchrecords)
    
    state = initial_state.copy()
    
    if is_remedies:
        # Remedies Session -> Exactly 2 files
        if state["records"]:
            state["session_id"] = session_id + "_records"
            setup_graph.invoke(state)
        
        state["session_id"] = session_id + "_remedies"
        state["records"] = state.get("remedies")
        state["predictions"] = state.get("predictions")
        state = setup_graph.invoke(state)
        
    elif is_matching:
        # Matching Session -> Exactly 3 files
        if state["records"]:
            state["session_id"] = session_id + "_r1"
            setup_graph.invoke(state)
            
        if state["records2"]:
            state["session_id"] = session_id + "_r2"
            state["records"] = state.get("records2")
            setup_graph.invoke(state)
            
        state["session_id"] = session_id + "_mr"
        state["records"] = state.get("matchrecords")
        state["predictions"] = state.get("m_predictions")
        state = setup_graph.invoke(state)
        
    else:
        # Kundali Session -> Exactly 1 file
        state["session_id"] = session_id + "_records"
        state = setup_graph.invoke(state)
    
    ACTIVE_SESSIONS[session_id] = state

    # 3. Save/Update DB with generated row_refs
    session_manager.save_session(session_id, user_p, row_refs=state.get("selected_row_refs"))

    # 4. New Consolidated Logging
    from common.utils import log_session_activity, log_general_timeline
    log_session_activity(session_id, "INIT", payload=req.model_dump())
    msg = f"Session Initialized: {session_id} (docs_loaded={len(state.get('session_documents', []))})"
    log_general_timeline(msg, category="SESSION_INIT", extra={"session_id": session_id})
    logger.info(msg)

    # Capture source info for logging timeline (processed by middleware)
    request.state.source_info = {
        "initial_tables": list(set(r.get("table") for r in state.get("selected_row_refs", []) if r.get("table")))
    }

    # 6. Context-Aware Suggestions (Static Pool)
    category = "KUNDALI"
    if req.records2 or req.matchrecords or req.ashtkoot_guna: category = "MATCHING"
    elif req.remedies: category = "REMEDIES"
    
    initial_suggestions = get_random_suggestions(category=category, count=3)

    return clean_non_ascii({
        "session_id": session_id,
        "status": "resumed" if db_data else "initialized",
        "initial_suggestions": initial_suggestions
    })

@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    # 1. Validate Session ID (Force new if invalid)
    session_id = req.session_id if is_valid_session_id(req.session_id) else generate_session_id()
    
    if session_id not in ACTIVE_SESSIONS:
        # ATTEMPT RESUMPTION FROM DB
        db_data = session_manager.get_session(session_id)
        if db_data:
            from backend.common.utils import log_backend_trace
            from backend.common.parser import parse_compressed_records as r_to_refs
            log_backend_trace(req.session_id, f"Resuming session from JSON logs")
            logger.info(f"Resuming session {req.session_id} from DB")
            
            # Extract partitioned records from user_profile if they exist
            profile = db_data.get("user_profile", {})
            male_refs = r_to_refs(profile.get("records", []))
            female_refs = r_to_refs(profile.get("records2", []))
            remedies_refs = r_to_refs(profile.get("remedies", []))
            predictions = profile.get("predictions", [])
            
            if not male_refs:
                male_refs = r_to_refs(db_data.get("records", []))
            if not female_refs:
                female_refs = r_to_refs(db_data.get("records2", []))

            initial_state = {
                "user_profile": db_data["user_profile"],
                "session_id": req.session_id,
                "selected_row_refs": db_data.get("row_refs", []),
                "messages": db_data["messages"],
                "k_value": GLOBAL_DEFAULTS["k_value"],
                "llm_model": GLOBAL_DEFAULTS["llm_model"]
            }
            # 4. Execute Setup Graph
            # Note: standard_setup is the function build_setup_graph_standard
            setup_graph = standard_setup()
            state = setup_graph.invoke(initial_state)
            ACTIVE_SESSIONS[req.session_id] = state
        else:
            raise HTTPException(status_code=404, detail="Session not found")

    state = ACTIVE_SESSIONS[req.session_id]
    
    # 1. Multi-lingual Processing: Pre-process non-English/Hindi queries
    processed_message = req.message
    
    # Priority: Secondary Language Code (Request) > Language Code (Profile)
    # Ensure we don't end up with string "None" by providing a fallback for the value itself
    active_lang_code = str(req.secondary_language_code) if req.secondary_language_code is not None else str(state.get("user_profile", {}).get("languagecode") or "0")
    
    # Mapping for Orchestrator synthesis
    LANGUAGE_MAP = {
        "0": "English", "1": "Hindi", "2": "Gujarati", "3": "Marathi",
        "4": "Tamil", "5": "Telugu", "6": "Assamese", "7": "Bengali",
        "8": "Punjabi", "9": "Nepali", "10": "Odia", "11": "Spanish",
        "12": "Malayalam", "13": "Kannada", "14": "Hinglish"
    }
    target_lang = LANGUAGE_MAP.get(active_lang_code, "English")

    try:
        from backend.agents.orchestrator import orchestrate_astro_query
        
        # FORCED GLOBAL STREAMING: Everything is a stream by default
        force_stream = True 

        # SINGLE ENTRY POINT: Everything flows through the Orchestrator
        agent_res = await orchestrate_astro_query(
            session_id=session_id,
            query=processed_message,
            user_profile=state["user_profile"],
            language=target_lang,
            language_code=int(active_lang_code),
            intent_override=req.intent_override,
            stream=force_stream,
            chart_data=req.chart_data,
            debug=req.debug,
            parent_id=req.parent_id
        )

        if force_stream:
            from fastapi.responses import StreamingResponse
            
            async def event_generator():
                # agent_res is the stream_generator from orchestrator
                logger.info(f"Streaming response started for session {session_id}...")
                async for chunk in agent_res:
                    if isinstance(chunk, dict) and "suggestions" in chunk:
                        sugs = chunk["suggestions"]
                        logger.info(f"Sent Suggestions: {sugs}")
                        yield f"data: {json.dumps({'suggestions': sugs})}\n\n"
                    elif isinstance(chunk, dict) and "debug" in chunk:
                        yield f"data: {json.dumps({'debug': chunk['debug']})}\n\n"
                    else:
                        token = str(chunk)
                        print(token, end="", flush=True)
                        yield f"data: {json.dumps({'answer': token})}\n\n"
                print("\n") # New line after stream ends
                logger.info(f"Streaming response completed.")
            
            response = StreamingResponse(event_generator(), media_type="text/event-stream")
            response.headers["X-Accel-Buffering"] = "no"
            response.headers["Cache-Control"] = "no-cache"
            return response
        
        answer = agent_res["answer"]
        suggestions = agent_res.get("suggestions", [])
        intent = agent_res.get("intent", "KUNDALI")

        # Combined Persistence: Add interaction to unified interactions table
        meta_ids = session_manager.save_interaction(
            session_id=session_id,
            category=intent.lower(),
            query=req.message,
            response=answer,
            parent_id=req.parent_id,
            interaction_id=agent_res.get("interaction_id")
        )

        # Final Cleaned Response for API
        return {
            "session_id": session_id,
            "answer": answer,
            "suggestions": suggestions,
            "interaction_metadata": meta_ids
        }
    except Exception as e:
        logger.error(f"Orchestrator failed: {e}", exc_info=True)
        from backend.common.utils import log_general_timeline
        log_general_timeline(f"ERROR: {str(e)}", category="CHAT_ERROR", extra={"session_id": session_id})
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        from backend.common.utils import log_general_timeline
        log_general_timeline(f"ERROR: Chat failed for {req.session_id}: {str(e)}")
        logger.error(f"Chat failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sessions/{session_id}/settings")
async def update_settings(session_id: str, req: SettingsUpdateRequest):
    if session_id not in ACTIVE_SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    state = ACTIVE_SESSIONS[session_id]
    # Removed embedding override logic. 
    # System strictly uses BGE-small with GPU detection from config.

    logger.info(f"Updated settings for session {session_id}: k={req.k_value}, model={req.llm_model}")
    return {
        "status": "success", 
        "session_id": session_id,
        "rebuilt": False
    }

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    # Flush log buffer
    from backend.common.utils import flush_interaction
    flush_interaction(session_id)
    
    # Delete from Memory
    if session_id in ACTIVE_SESSIONS:
        del ACTIVE_SESSIONS[session_id]
        
    # Delete from DB
    session_manager.delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}

@app.post("/api/sessions/{session_id}/language")
async def update_language(session_id: str, req: LanguageRequest):
    if session_id not in ACTIVE_SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    state = ACTIVE_SESSIONS[session_id]
    state["language"] = req.language
    logger.info(f"Language for session {session_id} updated to: {req.language}")

    return {"status": "success", "session_id": session_id, "language": req.language}

@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    """Submit user feedback for an interaction."""
    from backend.common.utils import log_feedback, log_backend_trace
    log_backend_trace(req.session_id, f"User Feedback: {req.feedback}")
    log_feedback(
        session_id=req.session_id,
        feedback=req.feedback
    )
    return {"status": "success"}

@app.post("/api/horary/sessions/init")
async def init_horary_session(req: InitRequest, request: Request):
    """
    Dedicated Horary Session Start or Resume.
    Accepts: { session_id, user_profile, records }
    Returns: { session_id, status, initial_suggestions }
    """
    # 1. Validate or Generate Session ID
    session_id = req.session_id if is_valid_session_id(req.session_id) else generate_session_id()
    
    # 2. Check for history and logs
    db_data = session_manager.get_session(session_id)
    
    # 3. Determine profile (Use request if provided, else DB)
    if req.user_profile:
        user_p = req.user_profile.model_dump()
    elif db_data:
        user_p = db_data["user_profile"]
    else:
        raise HTTPException(status_code=400, detail="Session not found and no profile provided.")

    # 4. Save/Update DB
    session_manager.save_session(session_id, user_p)
    
    # 5. Active State Sync
    if session_id not in ACTIVE_SESSIONS:
        ACTIVE_SESSIONS[session_id] = {
            "session_id": session_id,
            "user_profile": user_p,
            "messages": db_data.get("messages", []) if db_data else [],
            "language": "English"
        }

    # 6. Predefined Horary Questions (Static Pool)
    initial_suggestions = get_random_suggestions(category="HORARY", count=5)

    # 7. Logging
    from common.utils import log_session_activity, log_general_timeline
    log_session_activity(session_id, "HORARY_INIT", payload=req.model_dump())
    msg = f"Horary Session Started/Resumed: {session_id}"
    log_general_timeline(msg, category="HORARY_INIT", extra={"session_id": session_id})
    logger.info(msg)

    return {
        "session_id": session_id,
        "status": "resumed" if db_data else "initialized",
        "initial_suggestions": initial_suggestions
    }

# ── Horary Classifier ─────────────────────────────────────────────────────
class HoraryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    user_profile: Optional[UserProfile] = None # Optional for standalone init
    chart_data: Optional[Dict[str, Any]] = None

class TimingRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    user_profile: Optional[UserProfile] = None # Optional for standalone init
    chart_data: Optional[Dict[str, Any]] = None

@app.post("/api/matching/init")
async def init_matching_session(req: MatchingRequest, request: Request):
    """
    Initialize a Matching session with three independent datasets.
    Creates 3 isolated vector stores for records, records2, and matchrecords.
    """
    from backend.common.matching_store import create_matching_triple_store
    
    session_id = req.session_id if is_valid_session_id(req.session_id) else generate_session_id()
    db_data = session_manager.get_session(session_id)
    user_p = req.user_profile.model_dump() if req.user_profile else (db_data["user_profile"] if db_data else {})
    
    # Update DB
    session_manager.save_session(session_id, user_p)
    
    # Partitioned records for sub-sessions (Male, Female, Match)
    records_m = [r.model_dump() for r in req.records] if req.records else None
    records_f = [r.model_dump() for r in req.records2] if req.records2 else None
    records_mr = [r.model_dump() for r in req.matchrecords] if req.matchrecords else None
    predictions = [p.model_dump() for p in req.predictions] if req.predictions else None

    # Update profile with partitioned data for agent self-hydration
    user_p["records"] = records_m
    user_p["records2"] = records_f
    user_p["match_records"] = records_mr

    # Persist the profile so agents can see the partitioned data
    session_manager.save_session(session_id, user_p)
    
    # Create the Triple Store (Pre-embedded ingestion)
    stats = create_matching_triple_store(
        session_id=session_id,
        records=records_m,
        records2=records_f,
        matchrecords=records_mr,
        predictions=predictions
    )
    
    ACTIVE_SESSIONS[session_id] = {
        "session_id": session_id,
        "user_profile": user_p,
        "messages": db_data.get("messages", []) if db_data else [],
        "language": "English"
    }
    
    return {
        "session_id": session_id,
        "status": "initialized",
        "loaded_stats": stats
    }

async def analyze_matching_intent(query: str, male_name: str, female_name: str) -> str:
    """Classify the query intent into 'male', 'female', 'matching', or 'third_party'."""
    from backend.common.model_registry import ModelTask
    llm = create_llm(task=ModelTask.REASONING)
    
    prompt = f"""
    Analyze the intent of this astrological query. 
    Profile 1 (Male): {male_name}
    Profile 2 (Female): {female_name}
    
    Question: "{query}"
    
    Categorize into ONLY ONE of these:
    - 'male': Specifically asking about {male_name} (him/his).
    - 'female': Specifically asking about {female_name} (her/hers).
    - 'matching': Asking about BOTH, compatibility, relationship, or a general match.
    - 'third_party': Asking about someone NOT named {male_name} or {female_name} (e.g. spouse, brother, boss).
    
    Return ONLY THE CATEGORY WORD.
    """
    try:
        res = llm.invoke(prompt).content.strip().lower()
        # Extract first word to avoid any fluff
        import re
        match = re.search(r"(male|female|matching|third_party)", res)
        return match.group(1) if match else "matching"
    except Exception as e:
        logger.error(f"Intent Error: {e}")
        return "matching"

@app.post("/api/matching/query")
async def matching_query(req: MatchingQueryRequest, request: Request):
    """
    Execute a Matching Query via the Multi-Agent Orchestrator.
    Routing: Intent Classifier (Agent 0) -> Dataset Agents (1, 2, 3) -> Synthesis (Agent 0).
    """
    
    session_id = req.session_id
    if session_id not in ACTIVE_SESSIONS:
        db_data = session_manager.get_session(session_id)
        if not db_data:
            raise HTTPException(status_code=404, detail="Matching session not found.")
        ACTIVE_SESSIONS[session_id] = {
            "session_id": session_id,
            "user_profile": db_data["user_profile"],
            "messages": db_data["messages"],
            "language": req.language or "English"
        }

    state = ACTIVE_SESSIONS[session_id]
    
    # Ensure starting row_refs are hydrated for RAG analysis
    if "selected_row_refs" not in state:
        db_data = session_manager.get_session(session_id)
        state["selected_row_refs"] = db_data.get("row_refs", [])
        logger.info(f"Hydrated matching session {session_id} with {len(state['selected_row_refs'])} row_refs")

    # 2. Delegate to Orchestrator
    try:
        from backend.agents.orchestrator import orchestrate_astro_query
        
        force_stream = True

        agent_res = await orchestrate_astro_query(
            session_id=session_id,
            query=req.query,
            user_profile=state["user_profile"],
            language=state.get("language", "English"),
            is_matching=True,
            stream=force_stream
        )
        
        if force_stream:
            from fastapi.responses import StreamingResponse
            import json
            async def event_generator():
                logger.info(f"Streaming Matching response started for session {session_id}...")
                async for chunk in agent_res:
                    if isinstance(chunk, dict) and "suggestions" in chunk:
                        sugs = chunk["suggestions"]
                        logger.info(f"Sent Suggestions: {sugs}")
                        yield f"data: {json.dumps({'suggestions': sugs})}\n\n"
                    else:
                        token = str(chunk)
                        print(token, end="", flush=True)
                        yield f"data: {json.dumps({'answer': token})}\n\n"
                        await asyncio.sleep(0.03)
                print("\n")
                logger.info("Streaming Matching response completed.")
            
            response = StreamingResponse(event_generator(), media_type="text/event-stream")
            response.headers["X-Accel-Buffering"] = "no"
            response.headers["Cache-Control"] = "no-cache"
            return response

        answer = agent_res["answer"]
        thinking = agent_res.get("thinking", "")
        
        # Persistence
        session_manager.add_message(session_id, "user", req.query)
        session_manager.add_message(session_id, "assistant", answer)
        
        return {
            "session_id": session_id,
            "answer": answer,
            "thinking": thinking,
            "intent": agent_res.get("intent", "MATCHING"),
            "suggestions": agent_res.get("suggestions", [])
        }
    except Exception as e:
        logger.error(f"Multi-Agent Matching Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # Make sure we read host/port if needed, defaulting to 8001
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
