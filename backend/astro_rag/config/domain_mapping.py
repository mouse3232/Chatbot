"""
domain_mapping.py – Static table-to-domain mapping for the 3-domain RAG pipeline.

Maps every table in FptEnglish.db to one of: astrology, remedies, timing.
Tables not listed here are ignored during session hydration.
"""

# ── Table → Domain ────────────────────────────────────────────────────────────
TABLE_TO_DOMAIN = {
    # ── Kundali (Astrology) — chart analysis, doshas, personality, birth readings ──
    "Planet_in_House_Reading": "astrology",
    "Nakshatra_Reading":       "astrology",
    "PlanetInHouse":           "astrology",
    "PlanetInRasi":            "astrology",
    "RasiInHouse":             "astrology",
    "LordOfHouseInHouse":      "astrology",
    "PlanetFromMoon":          "astrology",
    "PlanetWithPlanets":       "astrology",
    "KaalSarp":                "astrology",
    "Mars_Reading":            "astrology",
    "Moon_Reading":            "astrology",
    "Sun_Reading":             "astrology",
    "Navamsa_Reading":         "astrology",
    "Num_Redical":             "astrology",
    "Num_Bhagyank":            "astrology",
    "Num_Lucky":               "astrology",
    "BirthOnDayTime":          "astrology",
    "BirthOnNakshatra":        "astrology",
    "BirthOnRasiLagna":        "astrology",
    "Shlokas_on_Nakshatra":    "astrology",
    "Shlokas_on_Rashi":        "astrology",
    "Shlokas_on_Gana":         "astrology",
    "Shlokas_on_Yoni":         "astrology",
    "Paya_Vichar":             "astrology",
    "DasaParagraph":           "astrology",
    "DasaPrediction":          "astrology",

    # ── Remedies ──
    "Rudraksh":           "remedies",
    "Ratna_Vichar_House": "remedies",
    "Ratna_Vichar_Lagna": "remedies",
    "LKUpay":             "remedies",
    "Lalkitab":           "remedies",
    "Num_Remedies":       "remedies",

    # ── Timing — prediction tables ──
    "AntarDasaSmall":       "timing",
    "MahaDasa":             "timing",
    "YearlyPredictionNew":  "timing",
    "Monthly_Prediction":   "timing",
    "WeeklyPredictionText": "timing",
    "DailyPredictionText":  "timing",
    "Varsh_Lagnesh":        "timing",
    "Varshesh":             "timing",
}

# Reverse lookup: domain → list of tables
DOMAIN_TABLES = {}
for _tbl, _dom in TABLE_TO_DOMAIN.items():
    DOMAIN_TABLES.setdefault(_dom, []).append(_tbl)

# All known domains
DOMAINS = ("astrology", "remedies", "timing")

# ── Tables that have a House column (used for chunk metadata) ─────────────────
TABLES_WITH_HOUSE_COL = {
    "Planet_in_House_Reading", "PlanetInHouse", "PlanetFromMoon",
    "LordOfHouseInHouse", "KaalSarp", "Mars_Reading", "Moon_Reading",
    "Sun_Reading", "Paya_Vichar", "DasaPrediction",
    "Rudraksh", "Ratna_Vichar_House", "LKUpay", "Lalkitab",
    "RasiInHouse", "AntarDasaSmall", "Monthly_Prediction",
}

# ── Timing flow thresholds ────────────────────────────────────────────────────
# Queries ≤ HORARY_MAX_DAYS → Horary-enabled flow (short-term)
# Queries >  HORARY_MAX_DAYS → Timing-only flow (long-term)
HORARY_MAX_DAYS = 30

# ── Timing table → prediction_type mapping ────────────────────────────────────
TIMING_PREDICTION_TYPE = {
    "AntarDasaSmall":       ["yearly", "monthly"],
    "MahaDasa":             ["yearly", "monthly"],
    "YearlyPredictionNew":  ["yearly"],
    "Monthly_Prediction":   ["monthly"],
    "WeeklyPredictionText": ["weekly"],
    "DailyPredictionText":  ["daily"],
    "Varsh_Lagnesh":        ["yearly"],
    "Varshesh":             ["yearly"],
}

# ── Feature flag: extra metadata on timing chunks ─────────────────────────────
# When True, each timing chunk gets prediction_type + category metadata.
# Default OFF — turn on after calibration.
ENABLE_TIMING_METADATA = False
