"""
semantic_chunker.py – GPU-accelerated semantic chunking for long text rows.

Strategy:
- Short rows (<80 words): keep as single chunk
- Long rows (80+ words): sentence-split → batch embed → cosine similarity →
  split at topic boundaries using calibrated threshold
- Merge fragments <40 words with adjacent chunk
- GPU/CPU adaptive: uses GPU if available, falls back to CPU
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Load thresholds config
_THRESHOLDS_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "thresholds.json")
_thresholds: Optional[Dict] = None


def _load_thresholds() -> Dict:
    global _thresholds
    if _thresholds is not None:
        return _thresholds
    try:
        with open(_THRESHOLDS_PATH, "r") as f:
            _thresholds = json.load(f)
    except Exception:
        _thresholds = {
            "semantic_split_threshold": 0.32,
            "min_chunk_words": 40,
            "max_chunk_words": 500,
            "short_row_threshold": 80,
        }
    return _thresholds


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using simple regex."""
    # Split on period, exclamation, question mark followed by space or end
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _word_count(text: str) -> int:
    return len(text.split())


def chunk_rows(
    rows: List[Dict[str, Any]],
    embed_model=None,
) -> List[Dict[str, Any]]:
    """
    Semantically chunk a list of row dicts.

    Args:
        rows: Output of sql_runner.run_sql()
        embed_model: SentenceTransformer model (optional, needed for long rows)

    Returns:
        List of chunk dicts:
        [{
            "table": str,
            "keys": dict,
            "text": str,           # The chunk text
            "chunk_index": int,    # 0 for single-chunk rows
            "house": int | None,   # From row's House column if present
        }]
    """
    from backend.astro_rag.config.domain_mapping import TABLES_WITH_HOUSE_COL

    thresholds = _load_thresholds()
    short_threshold = thresholds["short_row_threshold"]
    min_words = thresholds["min_chunk_words"]
    split_threshold = thresholds["semantic_split_threshold"]

    from backend.astro_rag.config.label_keywords import TIMING_CATEGORY_COLUMNS

    chunks = []

    for row in rows:
        table = row["table"]
        keys = row["keys"]
        text = row["text"]
        full_row = row.get("row", {})

        # Extract house metadata if table has a House column
        house = None
        if table in TABLES_WITH_HOUSE_COL:
            house_val = full_row.get("House") or keys.get("House")
            if house_val is not None:
                try:
                    house = int(house_val)
                except (ValueError, TypeError):
                    pass

        # ── Layer 1: Per-category-column chunking for multi-column tables ──
        # Tables like MahaDasa and YearlyPredictionNew have separate columns
        # for Health, Career, Family, etc. We split these into individual
        # chunks with pre-assigned categories for precise filtering.
        if table in TIMING_CATEGORY_COLUMNS:
            col_map = TIMING_CATEGORY_COLUMNS[table]
            created_subcol = False
            for category, col_name in col_map.items():
                col_text = full_row.get(col_name, "")
                if col_text and str(col_text).strip():
                    chunks.append({
                        "table": table,
                        "keys": keys,
                        "text": str(col_text).strip(),
                        "chunk_index": 0,
                        "house": house,
                        "category": category,  # Pre-assigned — labeler will skip
                    })
                    created_subcol = True
            if created_subcol:
                continue  # Skip the default single-chunk path for this row

        wc = _word_count(text)

        if wc < short_threshold or embed_model is None:
            # Short row or no model: keep as single chunk
            chunks.append({
                "table": table,
                "keys": keys,
                "text": text,
                "chunk_index": 0,
                "house": house,
            })
            continue

        # Long row: sentence-level semantic splitting
        sentences = _split_sentences(text)
        if len(sentences) <= 1:
            chunks.append({
                "table": table,
                "keys": keys,
                "text": text,
                "chunk_index": 0,
                "house": house,
            })
            continue

        # Batch embed all sentences
        try:
            device = "cuda" if _has_gpu() else "cpu"
            vecs = embed_model.encode(
                sentences,
                batch_size=64,
                normalize_embeddings=True,
                device=device,
                show_progress_bar=False,
            )
        except Exception as e:
            logger.warning(f"[Chunker] Embedding failed for {table}, keeping as single chunk: {e}")
            chunks.append({
                "table": table, "keys": keys, "text": text,
                "chunk_index": 0, "house": house,
            })
            continue

        # Compute pairwise cosine similarity between adjacent sentences
        split_points = [0]
        for i in range(1, len(vecs)):
            sim = float(np.dot(vecs[i - 1], vecs[i]))
            if sim < split_threshold:
                split_points.append(i)
        split_points.append(len(sentences))

        # Build chunks from split ranges, merge short fragments
        raw_chunks = []
        for j in range(len(split_points) - 1):
            segment = " ".join(sentences[split_points[j]:split_points[j + 1]])
            raw_chunks.append(segment)

        # Merge fragments shorter than min_words with previous chunk
        merged = []
        for seg in raw_chunks:
            if merged and _word_count(merged[-1]) < min_words:
                merged[-1] = merged[-1] + " " + seg
            else:
                merged.append(seg)

        # Final merge: if last chunk is too short, merge with previous
        if len(merged) > 1 and _word_count(merged[-1]) < min_words:
            merged[-2] = merged[-2] + " " + merged[-1]
            merged.pop()

        for ci, chunk_text in enumerate(merged):
            chunks.append({
                "table": table,
                "keys": keys,
                "text": chunk_text,
                "chunk_index": ci,
                "house": house,
            })

    logger.info(f"[Chunker] {len(rows)} rows → {len(chunks)} chunks")
    return chunks


def _has_gpu() -> bool:
    """Check if GPU is available using the global flag."""
    from backend.config import GPU_AVAILABLE
    return GPU_AVAILABLE
