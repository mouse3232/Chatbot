"""
labeler.py – 3-tier category label assignment for chunks.

Tier 1 (Fixed):    Table-level label lookup — no model needed.
Tier 2 (Keyword):  Score text against keyword lists — accept if clear winner.
Tier 3 (Zero-shot): Embed chunk vs. label description vectors — model fallback.
Fallback:           "general" (never leave unlabeled, but should be rare).
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from backend.astro_rag.config.label_keywords import (
    ALL_LABELS,
    FIXED_TABLE_LABELS,
    LABEL_KEYWORDS,
)

logger = logging.getLogger(__name__)


def _keyword_score(text: str) -> Optional[str]:
    """
    Score text against all keyword lists.
    Returns the label if there's a clear winner (≥2 hits and >1.5× runner-up).
    Returns None if ambiguous or no hits.
    """
    text_lower = text.lower()
    scores = {}
    for label, keywords in LABEL_KEYWORDS.items():
        if not keywords:  # Skip "general" (empty list)
            continue
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > 0:
            scores[label] = count

    if not scores:
        return None

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_label, best_count = sorted_scores[0]

    if best_count < 2:
        return None

    # Check if clear winner (>1.5× runner-up)
    if len(sorted_scores) > 1:
        runner_up_count = sorted_scores[1][1]
        if best_count <= runner_up_count * 1.5:
            return None  # Too ambiguous

    return best_label


def _zero_shot_label(chunk_text: str, embed_model, label_vecs: np.ndarray) -> str:
    """
    Embed chunk and compare against pre-computed label description vectors.
    Returns the most similar label.
    """
    try:
        from backend.common.utils import get_embedding_device
        device = get_embedding_device()

        chunk_vec = embed_model.encode(
            [chunk_text], normalize_embeddings=True, device=device, show_progress_bar=False
        )
        sims = np.dot(chunk_vec, label_vecs.T).flatten()
        best_idx = int(np.argmax(sims))
        # Only accept if similarity is meaningful (> 0.3)
        if sims[best_idx] > 0.3:
            return ALL_LABELS[best_idx]
    except Exception as e:
        logger.warning(f"[Labeler] Zero-shot failed: {e}")

    return "general"


def label_chunks(
    chunks: List[Dict[str, Any]],
    embed_model=None,
    label_vecs: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    Assign a category label to each chunk using the 3-tier strategy.

    Mutates chunks in-place by adding chunk["category"] = str.

    Args:
        chunks: Output of semantic_chunker.chunk_rows()
        embed_model: SentenceTransformer model (optional, for tier 3)
        label_vecs: Pre-computed label description vectors (optional, for tier 3)

    Returns:
        Same list with "category" field added to each chunk.
    """
    tier_counts = {"fixed": 0, "keyword": 0, "zeroshot": 0, "fallback": 0}

    for chunk in chunks:
        # Skip chunks with pre-assigned categories (from per-column chunking)
        if chunk.get("category"):
            tier_counts["fixed"] += 1
            continue

        table = chunk["table"]

        # Tier 1: Fixed table label
        if table in FIXED_TABLE_LABELS:
            chunk["category"] = FIXED_TABLE_LABELS[table]
            tier_counts["fixed"] += 1
            continue

        # Tier 2: Keyword scoring
        kw_label = _keyword_score(chunk["text"])
        if kw_label:
            chunk["category"] = kw_label
            tier_counts["keyword"] += 1
            continue

        # Tier 3: Zero-shot embedding (if model available)
        if embed_model is not None and label_vecs is not None:
            chunk["category"] = _zero_shot_label(chunk["text"], embed_model, label_vecs)
            tier_counts["zeroshot"] += 1
            continue

        # Fallback
        chunk["category"] = "general"
        tier_counts["fallback"] += 1

    logger.info(
        f"[Labeler] {len(chunks)} chunks labeled: "
        f"fixed={tier_counts['fixed']}, keyword={tier_counts['keyword']}, "
        f"zeroshot={tier_counts['zeroshot']}, fallback={tier_counts['fallback']}"
    )
    return chunks


def build_label_description_vectors(embed_model) -> np.ndarray:
    """
    Pre-compute embedding vectors for label descriptions.
    Called once at server startup.

    Returns:
        np.ndarray of shape (len(ALL_LABELS), dim)
    """
    descriptions = [
        "health and medical conditions, diseases, physical constitution, longevity",
        "career advancement, job opportunities, professional growth, government service",
        "financial wealth, money, income, property, investments, gains and losses",
        "marriage, spouse, romantic relationships, conjugal life, compatibility",
        "professional skills, trade, vocation, craft, occupation, business expertise",
        "education, academic pursuits, knowledge, intellectual growth, examinations",
        "personal character, temperament, nature, appearance, personality traits",
        "general astrological reading, miscellaneous predictions",
    ]
    assert len(descriptions) == len(ALL_LABELS), \
        f"Mismatch: {len(descriptions)} descriptions vs {len(ALL_LABELS)} labels"

    from backend.common.utils import get_embedding_device
    device = get_embedding_device()

    vecs = embed_model.encode(
        descriptions, normalize_embeddings=True, device=device, show_progress_bar=False
    )
    logger.info(f"[Labeler] Built {len(descriptions)} label description vectors (dim={vecs.shape[1]})")
    return vecs
