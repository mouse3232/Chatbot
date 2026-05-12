"""
entity_extractor.py – Lightweight keyword-to-entity mapper for hybrid retrieval.

Extracts astrological entities (planets, houses, tables) from user queries
using pure dictionary lookup. No LLM call needed.
"""

import re
from typing import Dict, List, Any

# ── Planet Name → ID Mapping ─────────────────────────────────────────────────
PLANET_NAMES = {
    "sun": 1, "surya": 1,
    "moon": 2, "chandra": 2,
    "mercury": 3, "budh": 3, "budha": 3,
    "mars": 4, "mangal": 4, "kuja": 4,
    "jupiter": 5, "guru": 5, "brihaspati": 5,
    "venus": 6, "shukra": 6,
    "saturn": 7, "shani": 7,
    "rahu": 8,
    "ketu": 9,
}

# ── House Number Patterns ────────────────────────────────────────────────────
HOUSE_PATTERN = re.compile(
    r"(\d{1,2})(?:st|nd|rd|th)\s*house",
    re.IGNORECASE
)
HOUSE_WORD_MAP = {
    "first": 1, "second": 2, "third": 3, "fourth": 4,
    "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
    "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12,
    "lagna": 1, "ascendant": 1,
}

# ── Table Name Keyword Mapping ───────────────────────────────────────────────
TABLE_KEYWORDS = {
    "lal kitab": ["Lalkitab", "LKUpay"],
    "lalkitab": ["Lalkitab", "LKUpay"],
    "lk upay": ["LKUpay"],
    "remedy": ["LKUpay", "Num_Remedies", "Rudraksh", "Ratna_Vichar_House"],
    "upay": ["LKUpay", "Num_Remedies"],
    "stone": ["Ratna_Vichar_House", "Ratna_Vichar_Lagna"],
    "gem": ["Ratna_Vichar_House", "Ratna_Vichar_Lagna"],
    "ratna": ["Ratna_Vichar_House", "Ratna_Vichar_Lagna"],
    "rudraksh": ["Rudraksh"],
    "kavach": ["Rudraksh"],
    "kaal sarp": ["KaalSarp"],
    "kaalsarp": ["KaalSarp"],
    "manglik": ["Mars_Reading"],
    "navamsa": ["Navamsa_Reading"],
    "nakshatra": ["Nakshatra_Reading", "BirthOnNakshatra"],
    "shloka": ["Shlokas_on_Gana", "Shlokas_on_Nakshatra", "Shlokas_on_Rashi", "Shlokas_on_Yoni"],
    "numerology": ["Num_Bhagyank", "Num_Lucky", "Num_Redical", "Num_Remedies"],
    "bhagyank": ["Num_Bhagyank"],
    "lucky number": ["Num_Lucky"],
    "radical": ["Num_Redical"],
    "matching": ["Matching1", "Matching2", "Matching3", "Matching4", "Matching5"],
    "compatibility": ["Matching1", "Matching2", "Matching3", "Matching4", "Matching5"],
    "guna": ["Bhakoot", "Gana", "Nadi", "Tara", "Varna", "Vashya", "GrahMaitri"],
    "nadi": ["Nadi"],
    "bhakoot": ["Bhakoot"],
    "gana": ["Gana"],
    "pitra": ["KaalSarp"],
    "paya": ["Paya_Vichar"],
}


def extract_entities(query: str) -> Dict[str, List[Any]]:
    """
    Extract astrological entities from a user query.

    Returns:
        {
            "planets": [4, 7],           # planet IDs
            "houses": [7, 10],           # house numbers
            "tables": ["Lalkitab"],      # specific table names
        }
    """
    q_lower = query.lower()
    result: Dict[str, List[Any]] = {
        "planets": [],
        "houses": [],
        "tables": [],
    }

    # 1. Planet extraction
    for name, pid in PLANET_NAMES.items():
        # Word boundary check to avoid partial matches like "marshmallow"
        if re.search(rf"\b{re.escape(name)}\b", q_lower):
            if pid not in result["planets"]:
                result["planets"].append(pid)

    # 2. House extraction (numeric: "7th house")
    for match in HOUSE_PATTERN.finditer(q_lower):
        h = int(match.group(1))
        if 1 <= h <= 12 and h not in result["houses"]:
            result["houses"].append(h)

    # House extraction (word: "seventh house")
    for word, h in HOUSE_WORD_MAP.items():
        if word in q_lower and h not in result["houses"]:
            result["houses"].append(h)

    # 3. Table extraction
    seen_tables = set()
    for keyword, tables in TABLE_KEYWORDS.items():
        if keyword in q_lower:
            for t in tables:
                if t not in seen_tables:
                    result["tables"].append(t)
                    seen_tables.add(t)

    return result
