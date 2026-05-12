"""
decomposer_bridge.py – Bridge between Gemini Decomposer JSON and RAG search.

Converts structured decomposer output into search instructions.
Routes Timing queries into Horary (≤1mo) or Timing-only (>1mo) flow.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from backend.astro_rag.config.domain_mapping import HORARY_MAX_DAYS
from backend.astro_rag.config.label_keywords import CAT_TO_LABELS

logger = logging.getLogger(__name__)

# ── Intent → Domain mapping ──────────────────────────────────────────────────
INTENT_TO_DOMAIN = {
    "Kundali":  "astrology",
    "Remedies": "remedies",
    "Timing":   "timing",
}


def _resolve_duration_days(tp: Optional[Dict]) -> int:
    """
    Convert tp.v to approximate number of days.

    Examples:
        "Today"     → 1
        "3 days"    → 3
        "2 weeks"   → 14
        "1 month"   → 30
        "Yearly"    → 365
        "2026"      → 365
        "2026-2028" → 730
        "Unknown"   → 0
    """
    if not tp:
        return 0

    v = str(tp.get("v", "")).strip().lower()
    if not v or v == "unknown":
        return 0

    # "today" / "tonight"
    if "today" in v or "tonight" in v:
        return 1

    # Extract numeric part
    nums = re.findall(r'\d+', v)
    n = int(nums[0]) if nums else 1

    if "day" in v:
        return n
    if "week" in v:
        return n * 7
    if "month" in v:
        return n * 30
    if "year" in v:
        return n * 365

    # Raw year (e.g., "2026")
    if len(nums) == 1 and n > 2000:
        return 365
    # Year range (e.g., "2026-2028")
    if len(nums) == 2:
        try:
            y1, y2 = int(nums[0]), int(nums[1])
            if y1 > 2000 and y2 > 2000:
                return max(1, (y2 - y1)) * 365
        except ValueError:
            pass

    return 0


def parse_decomposer_output(decomposer_json: List[Dict]) -> List[Dict[str, Any]]:
    """
    Convert Gemini decomposer JSON into search instructions.

    Routes Timing queries into:
    - "horary" flow (≤1 month)   → DailyPrediction/WeeklyPrediction/MonthlyPrediction + Horary remote
    - "timing_only" flow (>1 month) → YearlyPrediction/MahaDasa + Timing remote

    Args:
        decomposer_json: List of decomposer output dicts

    Returns:
        List of search request dicts:
        [{
            "domain": "astrology" | "remedies" | "timing",
            "query": str,                  # sem field → embedded for FAISS search
            "category_filter": [str],      # Mapped from cat → label list
            "house_filter": [int],         # From hd.h
            "time_period": dict | None,    # Raw tp object
            "flow": "horary" | "timing_only" | None,
            "duration_days": int,
            "horary_data": dict | None,    # hd object (only for horary flow)
        }]
    """
    requests = []

    for item in decomposer_json:
        intent = item.get("int", "Kundali")
        domain = INTENT_TO_DOMAIN.get(intent, "astrology")
        cat = item.get("cat", "")
        cat_labels = CAT_TO_LABELS.get(cat, [])
        hd = item.get("hd") or {}
        tp = item.get("tp")
        duration_days = _resolve_duration_days(tp)

        # Determine flow type for Timing domain
        flow = None
        horary_data = None
        if domain == "timing":
            if 0 < duration_days <= HORARY_MAX_DAYS:
                flow = "horary"
                horary_data = hd
            else:
                flow = "timing_only"

        requests.append({
            "domain": domain,
            "query": item.get("sem", ""),
            "category_filter": cat_labels,
            "house_filter": hd.get("h", []),
            "time_period": tp,
            "flow": flow,
            "duration_days": duration_days,
            "horary_data": horary_data,
        })

    logger.info(
        f"[DecomposerBridge] Parsed {len(decomposer_json)} intents → "
        f"{len(requests)} search requests"
    )
    return requests
