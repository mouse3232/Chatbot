"""
reranker.py – Cross-encoder reranking using bge-reranker-large.

Takes top-50 FAISS results and re-scores them with a cross-encoder model.
Returns top-12 highest-scoring chunks.
"""

import logging
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Default reranker output size
DEFAULT_TOP_K = 12


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    reranker_model,
    top_k: int = DEFAULT_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Re-score candidates using cross-encoder and return top-K.

    Args:
        query: Original user query text
        candidates: List of {"chunk": dict, "score": float, ...}
        reranker_model: CrossEncoder model instance
        top_k: Number of top results to return

    Returns:
        Top-K candidates re-scored, sorted by reranker score descending.
        Each result gets an additional "rerank_score" field.
    """
    if not candidates:
        return []

    if len(candidates) <= top_k:
        # Too few candidates, no need to rerank
        for c in candidates:
            c["rerank_score"] = c.get("score", 0.0)
        return candidates

    # Build cross-encoder input pairs
    pairs = [
        [query, c["chunk"]["text"]]
        for c in candidates
    ]

    t0 = time.time()
    try:
        scores = reranker_model.predict(pairs, show_progress_bar=False)
    except Exception as e:
        logger.error(f"[Reranker] Failed: {e}")
        # Fallback: return original FAISS-sorted top-K
        for c in candidates[:top_k]:
            c["rerank_score"] = c.get("score", 0.0)
        return candidates[:top_k]

    elapsed = time.time() - t0

    # Attach rerank scores
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    # Sort by rerank score descending
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    result = candidates[:top_k]
    logger.info(
        f"[Reranker] {len(pairs)} pairs → top {top_k} in {elapsed:.3f}s "
        f"(best={result[0]['rerank_score']:.3f})"
    )
    return result
