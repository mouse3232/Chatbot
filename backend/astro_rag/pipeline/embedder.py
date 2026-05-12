"""
embedder.py – GPU batch embedding for domain chunks using bge-large-en-v1.5.

Generates context-injected text with [table] [category] prefix per BGE format.
Single model.encode() call per domain for efficiency.
"""

import logging
import time
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# BGE document prefix for retrieval-optimized embedding
DOC_PREFIX = "Represent this astrological reading for retrieval: "


def _build_embed_text(chunk: Dict[str, Any]) -> str:
    """Build context-injected text for embedding."""
    table = chunk.get("table", "")
    category = chunk.get("category", "")
    text = chunk.get("text", "")
    return f"{DOC_PREFIX}[{table}] [{category}] {text}"


def embed_domain_chunks(
    domain: str,
    chunks: List[Dict[str, Any]],
    model,
) -> np.ndarray:
    """
    Batch embed all chunks for a single domain.

    Args:
        domain: Domain name (for logging)
        chunks: List of chunk dicts
        model: SentenceTransformer model instance

    Returns:
        np.ndarray of shape (len(chunks), dim), L2-normalized.
    """
    if not chunks:
        return np.zeros((0, 1024), dtype=np.float32)

    texts = [_build_embed_text(c) for c in chunks]

    device = "cuda"
    try:
        import torch
        if not torch.cuda.is_available():
            device = "cpu"
    except ImportError:
        device = "cpu"

    t0 = time.time()
    vectors = model.encode(
        texts,
        batch_size=256,
        normalize_embeddings=True,
        device=device,
        show_progress_bar=False,
    )
    vectors = np.array(vectors, dtype=np.float32)
    elapsed = time.time() - t0

    logger.info(
        f"[Embedder] Domain '{domain}': {len(chunks)} chunks → "
        f"({vectors.shape[0]}, {vectors.shape[1]}) in {elapsed:.2f}s on {device}"
    )
    return vectors
