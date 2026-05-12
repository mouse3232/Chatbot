#!/usr/bin/env python3
"""
session_store_v2.py – On-demand per-session vector store using Pickle/FAISS.

Architecture:
    FptEnglish.db  (source of truth)
        ↓  [session init: SQL filter → 300-500 rows]
    Embed on-the-fly with bge-large-en-v1.5
        ↓  [build FAISS index]
    sessions/{session_id}.pkl  (persisted per-session)
        ↓  [query time: hybrid search]
    Top-K documents

Memory Management:
    - Sessions auto-flush from RAM after 120s of inactivity
    - Disk .pkl persists for reload on demand
    - Conversation buffer (last 5 Q/A) sent to LLM
    - Summarization runs async after every 5 queries
    - Rolling window: max 20 queries of summary history
"""

import json
import logging
import os
import pickle
import sqlite3
import threading
import time
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
SESSION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "sessions")
os.makedirs(SESSION_DIR, exist_ok=True)

# BGE document prefix for context-injected embedding
DOC_PREFIX = "Represent this astrological reading for retrieval: "

# TTL: flush from RAM after 120 seconds of inactivity
SESSION_TTL_SECONDS = 120
# Summarization triggers after every N interactions
SUMMARIZE_EVERY = 5
# Maximum queries covered by rolling summary
MAX_SUMMARY_QUERIES = 20

# Schema cache (loaded once from schema_FptEnglish.json)
_schema_cache: Optional[Dict[str, Dict]] = None


def _load_schema() -> Dict[str, Dict]:
    """Load table schema, returning {table_name: {id_cols, text_col}}."""
    global _schema_cache
    if _schema_cache is not None:
        return _schema_cache

    schema_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "schema_FptEnglish.json")
    if not os.path.exists(schema_path):
        logger.warning(f"Schema file not found: {schema_path}")
        return {}

    with open(schema_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    _schema_cache = {}
    for t in data.get("tables", []):
        _schema_cache[t["table_name"]] = {
            "id_cols": t.get("identifier_columns", []),
            "text_col": t.get("prediction_text_column", "Text"),
        }
    return _schema_cache


# ── In-Memory Session Cache ──────────────────────────────────────────────────
_sessions: Dict[str, "_SessionEntry"] = {}
_sessions_lock = threading.Lock()


class _SessionEntry:
    """Holds the in-memory state for one chat session."""
    __slots__ = (
        "docs", "vectors", "index", "doc_count", "created_at",
        "last_accessed",       # Timestamp of last query (for TTL)
        "conversation_buffer", # Last ≤5 Q/A pairs (temporary, not in pkl)
        "summary",             # Rolling conversation summary (persisted in pkl)
        "summary_query_count", # Total queries covered in current summary
        "total_interactions",  # Total interactions since session creation
    )

    def __init__(self, docs: List[Dict[str, Any]], vectors: np.ndarray,
                 summary: str = "", summary_query_count: int = 0):
        self.docs = docs
        self.vectors = vectors
        self.index = None
        self.doc_count = len(docs)
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.conversation_buffer: List[Dict[str, str]] = []  # [{query, response}]
        self.summary = summary
        self.summary_query_count = summary_query_count
        self.total_interactions = 0
        self._build_index()

    def _build_index(self):
        """Build a FAISS inner-product (cosine) index from the vectors."""
        if self.doc_count == 0:
            return
        try:
            import faiss
            dim = self.vectors.shape[1]
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normalized = (self.vectors / norms).astype(np.float32)
            self.vectors = normalized
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(normalized)
        except ImportError:
            logger.warning("FAISS not installed. Falling back to NumPy brute-force search.")
            self.index = None

    def search(self, query_vec: np.ndarray, k: int = 10) -> List[tuple]:
        """Return list of (doc_index, score) sorted by relevance (highest first)."""
        if self.doc_count == 0:
            return []
        q = query_vec.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        if self.index is not None:
            scores, indices = self.index.search(q, min(k, self.doc_count))
            return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]
        else:
            similarities = (self.vectors @ q.T).flatten()
            top_k_idx = np.argsort(similarities)[::-1][:k]
            return [(int(idx), float(similarities[idx])) for idx in top_k_idx]

    def touch(self):
        """Update last-accessed timestamp."""
        self.last_accessed = time.time()


# ── Persistence ──────────────────────────────────────────────────────────────

def _get_pkl_path(session_id: str) -> str:
    """Get the .pkl file path for a given session."""
    return os.path.join(SESSION_DIR, f"{session_id}.pkl")


def _save_session_pkl(session_id: str, docs: list, vectors: np.ndarray,
                      summary: str = "", summary_query_count: int = 0):
    """Persist session data to a .pkl file (includes summary)."""
    pkl_path = _get_pkl_path(session_id)
    data = {
        "docs": docs,
        "vectors": vectors,
        "summary": summary,
        "summary_query_count": summary_query_count,
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.debug(f"Session {session_id} saved to {pkl_path} ({len(docs)} docs, summary_queries={summary_query_count})")


def _save_session_summary(session_id: str):
    """Update only the summary fields in the existing .pkl (quick partial save)."""
    if session_id not in _sessions:
        return
    entry = _sessions[session_id]
    _save_session_pkl(session_id, entry.docs, entry.vectors, entry.summary, entry.summary_query_count)


def _load_session_pkl(session_id: str) -> bool:
    """Load a session from its .pkl file into memory. Returns True if loaded."""
    pkl_path = _get_pkl_path(session_id)
    if not os.path.exists(pkl_path):
        return False
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        entry = _SessionEntry(
            data["docs"], data["vectors"],
            summary=data.get("summary", ""),
            summary_query_count=data.get("summary_query_count", 0)
        )
        with _sessions_lock:
            _sessions[session_id] = entry
        logger.info(f"Session {session_id} loaded from pkl ({len(data['docs'])} docs)")
        return True
    except Exception as e:
        logger.error(f"Failed to load session pkl {session_id}: {e}")
        return False


# ── TTL / Inactivity Eviction ────────────────────────────────────────────────

_ttl_thread_started = False
_ttl_thread_lock = threading.Lock()


def _start_ttl_thread():
    """Start a background daemon thread that flushes inactive sessions every 30s."""
    global _ttl_thread_started
    with _ttl_thread_lock:
        if _ttl_thread_started:
            return
        _ttl_thread_started = True

    def _eviction_loop():
        while True:
            time.sleep(30)  # Check every 30 seconds
            _flush_inactive_sessions()

    t = threading.Thread(target=_eviction_loop, daemon=True, name="session-ttl-evictor")
    t.start()
    logger.info("Session TTL eviction thread started (check every 30s, TTL=%ds)", SESSION_TTL_SECONDS)


def _flush_inactive_sessions():
    """Remove sessions from RAM that haven't been accessed within TTL."""
    now = time.time()
    to_flush = []
    with _sessions_lock:
        for sid, entry in _sessions.items():
            if (now - entry.last_accessed) > SESSION_TTL_SECONDS:
                to_flush.append(sid)

    for sid in to_flush:
        with _sessions_lock:
            entry = _sessions.pop(sid, None)
        if entry:
            # Persist summary before evicting
            _save_session_pkl(sid, entry.docs, entry.vectors, entry.summary, entry.summary_query_count)
            logger.info(f"TTL eviction: flushed session {sid} from RAM (idle {now - entry.last_accessed:.0f}s)")


# ── Conversation Buffer & Summarization ──────────────────────────────────────

def record_interaction(session_id: str, query: str, response: str):
    """
    Record a Q/A interaction into the session's conversation buffer.
    Called AFTER the response is sent to the user.
    """
    if session_id not in _sessions:
        return

    entry = _sessions[session_id]
    entry.touch()
    entry.total_interactions += 1
    entry.conversation_buffer.append({
        "query": query,
        "response": response,
        "timestamp": time.time()
    })
    logger.debug(f"Session {session_id}: recorded interaction #{entry.total_interactions}, buffer size={len(entry.conversation_buffer)}")


def get_conversation_context(session_id: str) -> Dict[str, Any]:
    """
    Get conversation context to send with LLM calls.
    Returns:
        {
            "summary": str,           # Rolling summary of past conversations
            "recent_exchanges": list,  # Last ≤5 Q/A pairs
        }
    """
    if session_id not in _sessions:
        if not _load_session_pkl(session_id):
            return {"summary": "", "recent_exchanges": []}

    entry = _sessions[session_id]
    entry.touch()

    return {
        "summary": entry.summary,
        "recent_exchanges": [
            {"query": item["query"], "response": item["response"]}
            for item in entry.conversation_buffer[-SUMMARIZE_EVERY:]
        ]
    }


def should_summarize(session_id: str) -> bool:
    """Check if summarization should be triggered (every 5 interactions)."""
    if session_id not in _sessions:
        return False
    entry = _sessions[session_id]
    return len(entry.conversation_buffer) >= SUMMARIZE_EVERY


def trigger_summarization(session_id: str, llm_model: str = None):
    """
    Run summarization asynchronously in a background thread.
    Called ONLY after the response has been sent to the user.
    """
    if not should_summarize(session_id):
        return

    def _do_summarize():
        try:
            _run_summarization(session_id, llm_model)
        except Exception as e:
            logger.error(f"Summarization failed for {session_id}: {e}")

    t = threading.Thread(target=_do_summarize, daemon=True, name=f"summarize-{session_id}")
    t.start()


def _run_summarization(session_id: str, llm_model: str = None):
    """
    Summarize the conversation buffer and update the session's rolling summary.
    
    Rules:
        - Summarizes the last 5 Q/A pairs
        - Merges into existing summary
        - Rolling window: max 20 queries
        - Beyond 20: discard old summary, start fresh with latest 20
    """
    if session_id not in _sessions:
        return

    entry = _sessions[session_id]
    buffer = entry.conversation_buffer[:SUMMARIZE_EVERY]

    if len(buffer) < SUMMARIZE_EVERY:
        return

    # Build the text to summarize
    exchanges_text = "\n".join([
        f"Q: {item['query']}\nA: {item['response'][:300]}"  # Truncate responses for summarization
        for item in buffer
    ])

    existing_summary = entry.summary if entry.summary else ""
    new_query_count = entry.summary_query_count + SUMMARIZE_EVERY

    # Check rotation: if we'd exceed 20 queries, reset
    if new_query_count > MAX_SUMMARY_QUERIES:
        existing_summary = ""
        new_query_count = SUMMARIZE_EVERY
        logger.info(f"Session {session_id}: summary rotation triggered (exceeded {MAX_SUMMARY_QUERIES} queries)")

    # Build summarization prompt
    if existing_summary:
        prompt = (
            f"You are a conversation summarizer for an astrology chatbot.\n\n"
            f"EXISTING SUMMARY (covering {entry.summary_query_count} previous queries):\n"
            f"{existing_summary}\n\n"
            f"NEW EXCHANGES (5 queries):\n"
            f"{exchanges_text}\n\n"
            f"TASK: Merge the new exchanges into the existing summary. "
            f"Keep it concise (150-250 words max). Focus on:\n"
            f"- Key astrological topics discussed\n"
            f"- Important insights revealed\n"
            f"- User's areas of interest/concern\n"
            f"- Any contradictions or nuances explored\n"
            f"Return ONLY the merged summary, no preamble."
        )
    else:
        prompt = (
            f"You are a conversation summarizer for an astrology chatbot.\n\n"
            f"EXCHANGES (5 queries):\n"
            f"{exchanges_text}\n\n"
            f"TASK: Create a concise summary (100-150 words). Focus on:\n"
            f"- Key astrological topics discussed\n"
            f"- Important insights revealed\n"
            f"- User's areas of interest/concern\n"
            f"Return ONLY the summary, no preamble."
        )

    try:
        from backend.common.llm import create_llm, strip_think_tags
        from backend.common.model_registry import ModelTask
        llm = create_llm(task=ModelTask.SUMMARIZATION, model_override=llm_model)
        response = llm.invoke([{"role": "user", "content": prompt}])
        raw = str(response.content)
        summary = strip_think_tags(raw).strip()

        # Update session
        entry.summary = summary
        entry.summary_query_count = new_query_count
        # Clear the consumed buffer items
        entry.conversation_buffer = entry.conversation_buffer[SUMMARIZE_EVERY:]

        # Persist to disk
        _save_session_summary(session_id)

        logger.info(f"Session {session_id}: summarization complete "
                    f"(covers {new_query_count} queries, summary={len(summary)} chars)")
    except Exception as e:
        logger.error(f"Summarization LLM call failed for {session_id}: {e}")


# ── FptEnglish.db Query ─────────────────────────────────────────────────────

def _query_fpt_english(
    dataset_refs: List[Dict[str, Any]],
    source_db_path: str,
) -> List[Dict[str, Any]]:
    """
    Query FptEnglish.db for the rows matching dataset_refs.
    Returns list of {table, content, metadata, embed_text}.
    """
    schema = _load_schema()
    conn = sqlite3.connect(source_db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row

    results = []
    for ref in dataset_refs:
        tname = ref.get("table", "")
        identifiers = ref.get("identifiers", {})

        tinfo = schema.get(tname)
        if not tinfo:
            logger.debug(f"Table {tname} not in schema, skipping")
            continue

        text_col = tinfo["text_col"]
        id_cols = tinfo["id_cols"]

        # Build WHERE clause from identifiers
        clauses = []
        params = []
        for col in id_cols:
            if col in identifiers:
                clauses.append(f'"{col}" = ?')
                params.append(identifiers[col])

        if not clauses:
            continue

        sql = f'SELECT * FROM "{tname}" WHERE {" AND ".join(clauses)}'
        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as e:
            logger.warning(f"SQL error for {tname}: {e}")
            continue

        for row in rows:
            row_dict = dict(row)
            text = row_dict.get(text_col, "")
            if not text or not str(text).strip():
                continue
            text = str(text).strip()

            # Build context-injected text for embedding
            context_parts = [f"Table: {tname}"]
            for col in id_cols:
                v = row_dict.get(col)
                if v is not None:
                    context_parts.append(f"{col}: {v}")
            embed_text = DOC_PREFIX + ", ".join(context_parts) + f". {text}"

            # Build LLM-friendly content
            header_parts = [f"[{tname}]"] + [f"{c}={row_dict.get(c)}" for c in id_cols if row_dict.get(c) is not None]
            llm_content = " | ".join(header_parts) + f": {text}"

            results.append({
                "table": tname,
                "content": llm_content,
                "embed_text": embed_text,
                "metadata": {c: row_dict[c] for c in id_cols if c in row_dict and row_dict[c] is not None},
                "is_prediction": False,
            })

    conn.close()
    logger.info(f"Queried FptEnglish.db: {len(results)} rows from {len(dataset_refs)} refs")
    return results


# ── Session Lifecycle ────────────────────────────────────────────────────────

def hydrate_session(
    session_id: str,
    dataset_refs: List[Dict[str, Any]],
    source_db_path: str,
    embedding,
    dim: int = 384,
    predictions: List[Dict[str, Any]] = None
) -> int:
    """
    Create a session-specific .pkl by querying FptEnglish.db for matching rows,
    embedding them on-the-fly, and building a FAISS index.

    Used for: Kundali sessions → session_id.pkl
    """
    # Start TTL eviction thread on first session creation
    _start_ttl_thread()

    # Check if .pkl already exists
    if _load_session_pkl(session_id):
        return _sessions[session_id].doc_count

    # 1. Query FptEnglish.db for matching rows
    rows = _query_fpt_english(dataset_refs, source_db_path)
    if not rows:
        logger.warning(f"No rows found for session {session_id}")
        with _sessions_lock:
            _sessions[session_id] = _SessionEntry([], np.zeros((0, dim), dtype=np.float32))
        return 0

    docs = [{k: v for k, v in r.items() if k != "embed_text"} for r in rows]
    embed_texts = [r["embed_text"] for r in rows]
    
    # 2. Add API Predictions directly into the VDB context
    for item in (predictions or []):
        if isinstance(item, dict):
            tname = item.get("table", "Prediction")
            p_keys = item.get("keys", [])
            for p_text in item.get("predictions", []):
                p_text = str(p_text).strip()
                if not p_text: continue
                docs.append({
                    "content": p_text,
                    "table": tname,
                    "metadata": {"source": "API_PREDICTION", "table": tname, "keys": p_keys},
                    "is_prediction": True,
                })
                embed_texts.append(DOC_PREFIX + f"Table: {tname}. {p_text}")
        else:
            p_text = str(item).strip()
            if p_text:
                docs.append({
                    "content": p_text,
                    "table": "Prediction",
                    "metadata": {"source": "API_PREDICTION"},
                    "is_prediction": True,
                })
                embed_texts.append(DOC_PREFIX + p_text)

    if not embed_texts:
        with _sessions_lock:
            _sessions[session_id] = _SessionEntry([], np.zeros((0, dim), dtype=np.float32))
        return 0

    # 3. Embed on-the-fly
    logger.info(f"Embedding {len(embed_texts)} texts for session {session_id}...")
    t0 = time.time()
    vectors = np.array(embedding.embed_documents(embed_texts), dtype=np.float32)
    logger.info(f"Embedding done in {time.time()-t0:.1f}s")

    # (Already completed above)

    # 4. Build session + persist .pkl
    with _sessions_lock:
        _sessions[session_id] = _SessionEntry(docs, vectors)
    _save_session_pkl(session_id, docs, vectors)

    return len(docs)


def hydrate_remedies_session(
    session_id: str,
    remedies_refs: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    embedding,
    source_db_path: str,
    dim: int = 384
) -> int:
    """
    Remedies-specific hydration → session_id_rm.pkl
    1. Queries FptEnglish.db for remedy rows
    2. Embeds plain-text predictions on-the-fly
    3. Combines and persists
    """
    _start_ttl_thread()
    remedies_sid = session_id if session_id.endswith("_rm") else session_id

    # Check if already exists
    if _load_session_pkl(remedies_sid):
        return _sessions[remedies_sid].doc_count

    # 1. Query FptEnglish.db for remedy rules
    rows = _query_fpt_english(remedies_refs, source_db_path)
    embed_texts = [r["embed_text"] for r in rows]
    docs = [{k: v for k, v in r.items() if k != "embed_text"} for r in rows]

    # 2. Add predictions
    for item in (predictions or []):
        if isinstance(item, dict):
            tname = item.get("table", "Prediction")
            p_keys = item.get("keys", [])
            for p_text in item.get("predictions", []):
                p_text = str(p_text).strip()
                if not p_text:
                    continue
                docs.append({
                    "content": p_text,
                    "table": tname,
                    "metadata": {"source": "API_PREDICTION", "table": tname, "keys": p_keys},
                    "is_prediction": True,
                })
                embed_texts.append(DOC_PREFIX + f"Table: {tname}. {p_text}")
        else:
            p_text = str(item).strip()
            if not p_text:
                continue
            docs.append({
                "content": p_text,
                "table": "Prediction",
                "metadata": {"source": "API_PREDICTION"},
                "is_prediction": True,
            })
            embed_texts.append(DOC_PREFIX + p_text)

    if not embed_texts:
        with _sessions_lock:
            _sessions[remedies_sid] = _SessionEntry([], np.zeros((0, dim), dtype=np.float32))
        return 0

    # 3. Embed all at once
    logger.info(f"Embedding {len(embed_texts)} texts for remedies session {remedies_sid}...")
    vectors = np.array(embedding.embed_documents(embed_texts), dtype=np.float32)

    # 4. Build + persist
    with _sessions_lock:
        _sessions[remedies_sid] = _SessionEntry(docs, vectors)
    _save_session_pkl(remedies_sid, docs, vectors)

    logger.info(f"Remedies session {remedies_sid}: {len(rows)} rules + {len(docs)-len(rows)} predictions = {len(docs)} total")
    return len(docs)


def create_session(
    session_id: str,
    documents: List[Any],
    embedding,
    dim: int = 1024,
    append: bool = False
) -> int:
    """Dynamic embedding for arbitrary documents (used by matching predictions)."""
    _start_ttl_thread()
    texts = [d.page_content for d in documents]
    if not texts:
        if not append:
            with _sessions_lock:
                _sessions[session_id] = _SessionEntry([], np.zeros((0, dim), dtype=np.float32))
        return 0

    vecs = np.array(embedding.embed_documents(texts), dtype=np.float32)
    docs = [{
        "content": d.page_content,
        "table": d.metadata.get("table", ""),
        "metadata": d.metadata,
        "is_prediction": d.metadata.get("is_prediction", False)
    } for d in documents]

    if append and session_id in _sessions:
        entry = _sessions[session_id]
        docs = list(entry.docs) + docs
        vecs = np.vstack([entry.vectors, vecs])

    with _sessions_lock:
        _sessions[session_id] = _SessionEntry(docs, vecs)
    _save_session_pkl(session_id, docs, vecs)
    return len(docs)


# ── Matching Triple Store ────────────────────────────────────────────────────

def hydrate_matching_session(
    session_id: str,
    dataset_refs: List[Dict[str, Any]],
    source_db_path: str,
    embedding,
    dim: int = 384,
    predictions: List[Dict[str, Any]] = None
) -> int:
    """Matching sub-store hydration → session_id_mr.pkl (or _r1, _r2)."""
    return hydrate_session(session_id, dataset_refs, source_db_path, embedding, dim, predictions)


# ── Query / Retrieval ────────────────────────────────────────────────────────

def query_session(
    session_id: str,
    query: str,
    embedding,
    k: int = 10,
    topic: str = None,
    allowed_tables: List[str] = None
) -> List[Any]:
    """
    Hybrid retrieval: entity match + vector similarity search.
    Auto-loads from .pkl if session not in memory.
    """
    from langchain_core.documents import Document

    # Auto-load from pkl if not in memory
    if session_id not in _sessions:
        if not _load_session_pkl(session_id):
            logger.warning(f"Session {session_id} not found in memory or on disk")
            return []

    entry = _sessions[session_id]
    entry.touch()  # Reset TTL timer

    if entry.doc_count == 0:
        return []

    if topic == "RESTRICTED" or not query:
        return []

    # 1. Entity-based exact match
    entity_docs = []
    try:
        from backend.standard.entity_extractor import extract_entities
        entities = extract_entities(query)
        if entities.get("tables") or entities.get("planets") or entities.get("houses"):
            for i, doc in enumerate(entry.docs):
                tbl = doc.get("table", "").lower()
                meta = doc.get("metadata", {})

                if entities.get("tables"):
                    if not any(et.lower() in tbl for et in entities["tables"]):
                        continue

                planet_match = True
                if entities.get("planets"):
                    doc_planet = str(meta.get("Planet", ""))
                    planet_match = any(str(p) == doc_planet for p in entities["planets"])

                house_match = True
                if entities.get("houses"):
                    doc_house = str(meta.get("House", meta.get("InHouse", "")))
                    house_match = any(str(h) == doc_house for h in entities["houses"])

                if planet_match and house_match:
                    entity_docs.append((i, 1.0))
    except ImportError:
        pass

    # 2. Vector similarity search
    q_vec = np.array(embedding.embed_query(query), dtype=np.float32)
    vector_results = entry.search(q_vec, k=k * 2)

    # 3. Merge & deduplicate
    seen = set()
    merged = []
    for idx, score in entity_docs:
        if idx not in seen:
            seen.add(idx)
            merged.append((idx, score + 0.5))
    for idx, score in vector_results:
        if idx not in seen:
            seen.add(idx)
            merged.append((idx, score))

    # 4. Hard-bind tables: ALWAYS include these core readings regardless of similarity
    HARD_BIND_TABLES = {"planet_in_house_reading", "navamsa_reading", "num_redical"}
    hardbind_indices = []
    for i, doc in enumerate(entry.docs):
        tbl = doc.get("table", "").lower()
        if tbl in HARD_BIND_TABLES and i not in seen and not doc.get("is_prediction"):
            hardbind_indices.append(i)
            seen.add(i)

    # 5. Table filtering
    if allowed_tables:
        import fnmatch
        merged = [(idx, s) for idx, s in merged
                  if any(fnmatch.fnmatch(entry.docs[idx].get("table", "").lower(), p.replace("%", "*").lower()) for p in allowed_tables)]

    merged.sort(key=lambda x: x[1], reverse=True)

    # 6. Append plain-text predictions (Always include API-provided text, ignoring DB table filters)
    prediction_indices = []
    for i, doc in enumerate(entry.docs):
        if doc.get("is_prediction") and i not in seen:
            # We explicitly bypass allowed_tables here so that plain text "Prediction"
            # blocks injected by the API are not accidentally filtered out.
            prediction_indices.append(i)

    # Convert to LangChain Documents
    results = []

    # Hard-bind docs FIRST (always present as foundational context)
    for idx in hardbind_indices:
        doc = entry.docs[idx]
        orig_meta = doc.get("metadata", {})
        results.append(Document(
            page_content=doc.get("content", ""),
            metadata={**orig_meta, "table": doc.get("table", ""), "score": 0.95, "json_meta": str(orig_meta), "hardbind": True}
        ))

    # Then ranked results from entity+vector search
    for idx, score in merged[:k]:
        doc = entry.docs[idx]
        orig_meta = doc.get("metadata", {})
        results.append(Document(
            page_content=doc.get("content", ""),
            metadata={**orig_meta, "table": doc.get("table", ""), "score": score, "json_meta": str(orig_meta)}
        ))

    # Then predictions
    for idx in prediction_indices[:3]:
        doc = entry.docs[idx]
        results.append(Document(
            page_content=doc.get("content", ""),
            metadata={"table": doc.get("table", ""), "score": 1.0, "source": "prediction"}
        ))

    if results:
        tables = list(set(d.metadata.get("table") for d in results))
        logger.info(f"VDB[retrieve]: session={session_id} k={k} results={len(results)} tables={tables} hardbind={len(hardbind_indices)}")

    return results


def get_all_predictions(session_id: str, allowed_tables: List[str] = None) -> List[Any]:
    """Retrieve all plain-text prediction rows for a session."""
    from langchain_core.documents import Document

    if session_id not in _sessions:
        if not _load_session_pkl(session_id):
            return []

    entry = _sessions[session_id]
    entry.touch()
    docs = []
    for doc in entry.docs:
        if not doc.get("is_prediction"):
            continue
        tbl = doc.get("table", "")
        if allowed_tables:
            import fnmatch
            if not any(fnmatch.fnmatch(tbl.lower(), p.replace("%", "*").lower()) for p in allowed_tables):
                continue
        docs.append(Document(
            page_content=doc.get("content", ""),
            metadata={"table": tbl, "source": "prediction", "score": 0.0}
        ))
    return docs


def destroy_session(session_id: str):
    """Remove a session from memory and delete its .pkl file."""
    with _sessions_lock:
        _sessions.pop(session_id, None)
    pkl_path = _get_pkl_path(session_id)
    if os.path.exists(pkl_path):
        try:
            os.remove(pkl_path)
        except OSError:
            pass


def session_exists(session_id: str) -> bool:
    """Check if a session exists in memory or on disk."""
    return session_id in _sessions or os.path.exists(_get_pkl_path(session_id))
