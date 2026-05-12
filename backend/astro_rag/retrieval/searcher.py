"""
searcher.py – Filtered FAISS search with category + house pre-filtering.

1. Pre-filter chunks by category + house from decomposer
2. If filter too strict (<5 results), fallback to full domain index
3. FAISS inner-product search → top-K
4. Return ranked candidates
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from backend.astro_rag.session.rag_session import RAGSession

logger = logging.getLogger(__name__)

# BGE query prefix (matches embedder.py DOC_PREFIX format for queries)
QUERY_PREFIX = "Represent this astrological question for retrieval: "


def build_filter_fn(
    category_filter: List[str],
    house_filter: List[int],
) -> Optional[Callable]:
    """
    Build a chunk filter function from decomposer fields.

    Returns None if no filters are active.
    """
    has_cat = bool(category_filter)
    has_house = bool(house_filter)

    if not has_cat and not has_house:
        return None

    def filter_fn(chunk: Dict) -> bool:
        # Category filter
        if has_cat and chunk.get("category") not in category_filter:
            return False
        # House filter
        if has_house:
            chunk_house = chunk.get("house")
            if chunk_house is not None and chunk_house not in house_filter:
                return False
        return True

    return filter_fn


def search_domain(
    session: RAGSession,
    domain: str,
    query_text: str,
    embed_model,
    category_filter: List[str] = None,
    house_filter: List[int] = None,
    top_k: int = 50,
) -> List[Dict[str, Any]]:
    """
    Run filtered FAISS search on a single domain.

    Args:
        session: Active RAGSession
        domain: Domain to search (astrology/remedies/timing)
        query_text: Semantic query text (from decomposer's `sem` field)
        embed_model: SentenceTransformer model
        category_filter: Category labels to filter by
        house_filter: House numbers to filter by
        top_k: Max results to return

    Returns:
        List of {"chunk": dict, "score": float}
    """
    # Embed query with BGE prefix
    from backend.common.utils import get_embedding_device
    device = get_embedding_device()

    query_vec = embed_model.encode(
        [f"{QUERY_PREFIX}{query_text}"],
        normalize_embeddings=True,
        device=device,
        show_progress_bar=False,
    ).astype(np.float32)

    # Build filter
    filter_fn = build_filter_fn(
        category_filter=category_filter or [],
        house_filter=house_filter or [],
    )

    # Run search
    results = session.search(domain, query_vec, k=top_k, filter_fn=filter_fn)

    logger.debug(
        f"[Searcher] Domain '{domain}': query='{query_text[:50]}...' → {len(results)} results"
    )
    return results


def search_multi_domain(
    session: RAGSession,
    search_requests: List[Dict[str, Any]],
    embed_model,
    top_k_per_domain: int = 25,
) -> List[Dict[str, Any]]:
    """
    Execute multiple search requests across domains and merge results.

    Args:
        session: Active RAGSession
        search_requests: Output of decomposer_bridge.parse_decomposer_output()
        embed_model: SentenceTransformer model
        top_k_per_domain: Max results per domain search

    Returns:
        Combined list of {"chunk": dict, "score": float, "domain": str}
    """
    all_results = []

    for req in search_requests:
        domain = req["domain"]
        results = search_domain(
            session=session,
            domain=domain,
            query_text=req["query"],
            embed_model=embed_model,
            category_filter=req.get("category_filter"),
            house_filter=req.get("house_filter"),
            top_k=top_k_per_domain,
        )

        for r in results:
            r["domain"] = domain
        all_results.extend(results)

    # Sort by score descending
    all_results.sort(key=lambda x: x["score"], reverse=True)
    logger.debug(f"[Searcher] Multi-domain: {len(all_results)} total results")
    return all_results
