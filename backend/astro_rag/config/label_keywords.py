"""
label_keywords.py – 8-category taxonomy for chunk labeling.

Used by the labeler module to assign a single category label to each chunk.
Labels are matched against chunk text via keyword hits, fixed table mappings,
or zero-shot embedding fallback.
"""

# ── 8-Category Taxonomy ──────────────────────────────────────────────────────
# "general" is the fallback — should be rarely assigned.
LABEL_KEYWORDS = {
    "health": [
        "disease", "body", "constitution", "illness", "diet", "injury",
        "nervous", "medical", "vitality", "physical", "ailment", "hospital",
        "surgery", "blood", "fever", "chronic", "longevity",
    ],
    "career": [
        "career", "job", "profession", "business", "executive", "service",
        "work", "employment", "promotion", "authority", "government",
        "position", "rank", "boss", "office", "corporation",
    ],
    "finance": [
        "wealth", "money", "income", "gains", "losses", "property", "debt",
        "financial", "fortune", "profit", "inheritance", "investment",
        "expenditure", "assets", "bank", "loan",
    ],
    "marriage": [
        "marriage", "spouse", "conjugal", "partner", "husband", "wife",
        "relationship", "marital", "wedding", "bride", "groom", "divorce",
        "compatibility", "love", "romance",
    ],
    "profession": [
        "trade", "vocation", "skill", "craft", "occupation", "expertise",
        "livelihood", "enterprise", "industry", "labor", "merchant",
    ],
    "education": [
        "education", "knowledge", "study", "learning", "intellect", "wisdom",
        "academic", "school", "university", "degree", "scholarship", "exam",
        "research", "teacher", "student",
    ],
    "personal": [
        "nature", "character", "appearance", "temperament", "personality",
        "self", "behavior", "attitude", "mind", "mental", "disposition",
        "habits", "complexion", "physique",
    ],
    "general": [],  # No keywords — assigned only when no other label matches
}

# All valid category labels
ALL_LABELS = list(LABEL_KEYWORDS.keys())

# ── Fixed Table Labels ────────────────────────────────────────────────────────
# Tables where the category is deterministic (no model/keyword needed)
FIXED_TABLE_LABELS = {
    "Rudraksh":          "remedies",
    "Ratna_Vichar_House": "remedies",
    "Ratna_Vichar_Lagna": "remedies",
    "LKUpay":            "remedies",
    "Lalkitab":          "remedies",
    "Num_Remedies":      "remedies",
    "Num_Bhagyank":      "personal",
    "Num_Redical":       "personal",
    "Num_Lucky":         "personal",
    "BirthOnDayTime":    "personal",
    "BirthOnNakshatra":  "personal",
    "BirthOnRasiLagna":  "personal",
    "WeeklyPredictionText": "general",
    "DailyPredictionText":  "general",
}

# ── Decomposer Category → Label Mapping ──────────────────────────────────────
# Maps the Gemini decomposer's `cat` field to chunk label(s) for filtering.
CAT_TO_LABELS = {
    "Personal": ["health", "personal"],
    "Wealth":   ["finance"],
    "Career":   ["career", "profession"],
    "Family":   ["marriage", "personal"],
    "Travel":   ["education", "personal"],
    "General":  ["general"],  # Rarely chosen by decomposer
}

# ── MahaDasa & YearlyPredictionNew per-category column mapping ────────────────
# These tables have multiple prediction columns, one per life-area.
TIMING_CATEGORY_COLUMNS = {
    "MahaDasa": {
        "health":      "Health",
        "career":      "Profession",
        "education":   "Education_and_Learning",
        "family":      "Family_Life",
        "travel":      "Vehicle_Journey_and_Property",
        "general":     "Text",
        "suggestions": "Suggestions",
        "antardasha":  "Antardasha",
    },
    "YearlyPredictionNew": {
        "career":    "Profession",
        "finance":   "Wealth",
        "property":  "Property",
        "family":    "Family and Society",
        "children":  "Children",
        "health":    "Health",
        "education": "Career and Competition",
        "travel":    "Travel and Transfer",
        "spiritual": "Religious Deeds and Propitiation of Planets",
        "general":   "Yearly Prediction",
    },
}
