"""
index_builder.py – Build FAISS IndexFlatIP indices for each domain.

GPU-first: uses faiss-gpu if available, falls back to CPU FAISS.
Builds one inner-product (cosine) index per domain.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def build_faiss_index(vectors: np.ndarray, domain: str = ""):
    """
    Build a FAISS inner-product (cosine) index from pre-normalized vectors.

    Args:
        vectors: np.ndarray of shape (n, dim), L2-normalized
        domain: Domain name for logging

    Returns:
        FAISS index (GPU or CPU)
    """
    import faiss

    if vectors.shape[0] == 0:
        logger.warning(f"[IndexBuilder] Empty vectors for domain '{domain}', returning empty index")
        index = faiss.IndexFlatIP(1024)  # Default dim
        return index

    dim = vectors.shape[1]
    t0 = time.time()

    # Build CPU index first
    index = faiss.IndexFlatIP(dim)
    index.add(vectors.astype(np.float32))

    # Try to move to GPU only if supported
    try:
        from backend.config import GPU_AVAILABLE
        if GPU_AVAILABLE:
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            elapsed = time.time() - t0
            logger.info(
                f"[IndexBuilder] Domain '{domain}': {vectors.shape[0]} vectors "
                f"→ GPU index in {elapsed:.3f}s"
            )
            return gpu_index
        else:
            # Silent fallback to CPU
            return index
    except Exception:
        return index


def build_all_indices(
    domain_vectors: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """
    Build FAISS indices for all domains.

    Args:
        domain_vectors: {"astrology": np.ndarray, "remedies": ..., "timing": ...}

    Returns:
        {"astrology": faiss.Index, "remedies": ..., "timing": ...}
    """
    indices = {}
    for domain, vecs in domain_vectors.items():
        indices[domain] = build_faiss_index(vecs, domain)
    return indices
