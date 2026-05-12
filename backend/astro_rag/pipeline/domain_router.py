"""
domain_router.py – Sort labeled chunks into 3 domain buckets.

O(1) dict lookup per chunk: TABLE_TO_DOMAIN[chunk["table"]] → domain.
Chunks from unknown tables are dropped with a warning.
"""

import logging
from typing import Any, Dict, List

from backend.astro_rag.config.domain_mapping import DOMAINS, TABLE_TO_DOMAIN

logger = logging.getLogger(__name__)


def route_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Sort chunks into domain buckets.

    Args:
        chunks: Labeled chunks from labeler.label_chunks()

    Returns:
        {"astrology": [...], "remedies": [...], "timing": [...]}
    """
    buckets: Dict[str, List[Dict[str, Any]]] = {d: [] for d in DOMAINS}
    dropped = 0

    for chunk in chunks:
        # Support forced domain override from raw_texts
        forced = chunk.pop("_forced_domain", None)
        if forced and forced in DOMAINS:
            buckets[forced].append(chunk)
            continue

        table = chunk["table"]
        domain = TABLE_TO_DOMAIN.get(table)
        # For synthetic RawText_ tables, auto-detect via label or default to astrology
        if domain is None and table.startswith("RawText_"):
            domain = "astrology"  # Default for raw text without forced domain
        if domain is None:
            logger.debug(f"[Router] Unknown table '{table}', dropping chunk")
            dropped += 1
            continue
        buckets[domain].append(chunk)

    counts = {d: len(v) for d, v in buckets.items()}
    logger.info(f"[Router] Routed {len(chunks)} chunks: {counts} (dropped {dropped})")
    return buckets
