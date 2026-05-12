import logging
import sqlite3
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
import json
import re

from backend.config import settings, EMBEDDING_CONFIGS
from backend.common.state import AstroState
from backend.common.utils import get_embedding, collect_user_profile, resolve_rows_from_remote
from backend.standard.session_store_v2 import (
    hydrate_session, query_session, destroy_session,
    get_conversation_context, record_interaction, trigger_summarization
)

logger = logging.getLogger(__name__)

# ── Legacy Session Build (kept for backward compatibility) ────────────────────

def fetch_and_build_session_standard(state: AstroState) -> dict:
    """
    Query FptEnglish.db for matching rows, embed on-the-fly with bge-large,
    and build a per-session FAISS index saved as {session_id}.pkl.
    """
    session_id = state["session_id"]
    dataset_refs = state.get("selected_row_refs", [])

    if not dataset_refs:
        if "_" not in session_id:
            return {"session_documents": []}
        logger.warning(f"Node[fetch_build_standard]: no dataset refs for session {session_id}")
        return {"session_documents": []}

    logger.info(f"Node[fetch_build_standard]: loading {len(dataset_refs)} dataset refs for session {session_id}")

    emb_model = settings.EMBEDDING_MODEL
    emb_cfg = EMBEDDING_CONFIGS.get(emb_model, {"dim": settings.EMBEDDING_DIM})
    emb_dim = emb_cfg["dim"]
    embedding = get_embedding(emb_model)

    loaded = hydrate_session(
        session_id=session_id,
        dataset_refs=dataset_refs,
        source_db_path=settings.SOURCE_DB_PATH,
        embedding=embedding,
        dim=emb_dim,
        predictions=state.get("predictions")
    )

    logger.info("Node[fetch_build_standard]: session '%s' ready (%d docs, embedded on-the-fly)", session_id, loaded)
    return {"session_documents": []}


# ── NEW: RAG Pipeline Nodes ───────────────────────────────────────────────────

def build_rag_session_node(state: AstroState) -> dict:
    """
    Build 3-domain FAISS indices for the user session.
    Uses the new astro_rag pipeline: SQL → Chunk → Label → Route → Embed → Index.
    """
    from backend.astro_rag.pipeline.sql_runner import run_sql
    from backend.astro_rag.pipeline.semantic_chunker import chunk_rows
    from backend.astro_rag.pipeline.labeler import label_chunks
    from backend.astro_rag.pipeline.domain_router import route_chunks
    from backend.astro_rag.pipeline.embedder import embed_domain_chunks
    from backend.astro_rag.pipeline.index_builder import build_all_indices
    from backend.astro_rag.session.rag_session_manager import RAGSessionManager

    session_id = state["session_id"]
    dataset_refs = state.get("selected_row_refs", [])
    timing_records = state.get("timing_records") or []
    raw_texts = state.get("raw_texts") or []

    if not dataset_refs and not timing_records and not raw_texts:
        logger.warning(f"[RAGBuild] No dataset refs, timing records, or raw texts for session {session_id}")
        return {"session_documents": []}

    # Get shared models from app globals
    try:
        from backend.app import RAG_EMBED_MODEL, RAG_LABEL_VECS, rag_session_manager, TIMING_ONLY_MODE
        embed_model = RAG_EMBED_MODEL
        label_vecs = RAG_LABEL_VECS
    except (ImportError, AttributeError):
        logger.error("[RAGBuild] RAG models not loaded. Falling back to legacy flow.")
        return fetch_and_build_session_standard(state)

    # Check if session already exists
    existing = rag_session_manager.get_session(session_id)
    if existing:
        logger.info(f"[RAGBuild] Session {session_id} already exists, reusing")
        return {"session_documents": []}

    # ── 7-stage pipeline ──
    # 1. SQL: query all tables from structured records
    rows = []
    if dataset_refs:
        rows = run_sql(dataset_refs, settings.SOURCE_DB_PATH)
    
    # 1b. SQL: query timing_records separately (dedicated timing tables)
    if timing_records:
        from backend.common.parser import parse_compressed_records
        timing_refs = parse_compressed_records(timing_records)
        timing_rows = run_sql(timing_refs, settings.SOURCE_DB_PATH)
        rows.extend(timing_rows)
        logger.info(f"[RAGBuild] Added {len(timing_rows)} timing rows from timing_records")

    # 2. Chunk: semantic splitting
    chunks = chunk_rows(rows, embed_model=embed_model)

    # 2b. Chunk raw texts (if any)
    if raw_texts:
        for rt in raw_texts:
            label_name = rt.get("label", "Custom Text")
            text = rt.get("text", "")
            forced_domain = rt.get("domain")
            if not text.strip():
                continue
            # Create a synthetic row for the chunker
            synthetic_row = {
                "table": f"RawText_{label_name}",
                "text": text,
                "keys": {},
            }
            raw_chunks = chunk_rows([synthetic_row], embed_model=embed_model)
            # Override domain if specified
            if forced_domain:
                for c in raw_chunks:
                    c["_forced_domain"] = forced_domain
            chunks.extend(raw_chunks)
        logger.info(f"[RAGBuild] Added {len(raw_texts)} raw text sources as chunks")

    # 3. Label: 3-tier category assignment
    chunks = label_chunks(chunks, embed_model=embed_model, label_vecs=label_vecs)

    # 4. Route: sort into 3 domain buckets
    domain_buckets = route_chunks(chunks)

    # 4b. Filter to timing-only if TIMING_ONLY_MODE
    if TIMING_ONLY_MODE:
        domain_buckets = {k: v for k, v in domain_buckets.items() if k == "timing"}
        logger.info("[RAGBuild] TIMING_ONLY_MODE: filtered to timing domain only")

    # 5. Embed: GPU batch embedding per domain
    domain_vectors = {}
    for domain, domain_chunks in domain_buckets.items():
        domain_vectors[domain] = embed_domain_chunks(domain, domain_chunks, embed_model)

    # 6. Build: FAISS indices
    indices = build_all_indices(domain_vectors)

    # 7. Register session
    session = rag_session_manager.create_session(session_id)
    for domain in domain_buckets:
        session.set_domain(domain, indices[domain], domain_buckets[domain])

    total_chunks = sum(len(v) for v in domain_buckets.values())
    
    # 8. Persist to disk
    rag_session_manager.persist_session(session_id)
    
    logger.info(f"[RAGBuild] Session {session_id} ready: {total_chunks} chunks across {len(domain_buckets)} domain(s)")
    return {"session_documents": []}


# ── Graph Builders ────────────────────────────────────────────────────────────

def build_setup_graph_standard():
    graph = StateGraph(AstroState)
    graph.add_node("collect_user_profile", collect_user_profile)
    graph.add_node("resolve_rows_from_remote", resolve_rows_from_remote)
    graph.add_node("build_rag_session", build_rag_session_node)
    graph.add_edge(START, "collect_user_profile")
    graph.add_edge("collect_user_profile", "resolve_rows_from_remote")
    graph.add_edge("resolve_rows_from_remote", "build_rag_session")
    graph.add_edge("build_rag_session", END)
    return graph.compile()
