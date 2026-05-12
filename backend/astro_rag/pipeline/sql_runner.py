"""
sql_runner.py – Query FptEnglish.db for all rows matching user chart keys.

Refactored from session_store_v2._query_fpt_english().
Key difference: queries ALL tables at once, returns ALL rows with no domain filtering.
Domain routing happens downstream in domain_router.py.
"""

import json
import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Schema cache (loaded once)
_schema_cache: Optional[Dict[str, Dict]] = None
_SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "schema_FptEnglish.json",
)


def _load_schema() -> Dict[str, Dict]:
    """Load table schema → {table_name: {id_cols: [...], text_col: str}}."""
    global _schema_cache
    if _schema_cache is not None:
        return _schema_cache

    if not os.path.exists(_SCHEMA_PATH):
        logger.warning(f"Schema file not found: {_SCHEMA_PATH}")
        return {}

    with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    _schema_cache = {}
    for t in data.get("tables", []):
        _schema_cache[t["table_name"]] = {
            "id_cols": t.get("identifier_columns", []),
            "text_col": t.get("prediction_text_column", "Text"),
        }
    return _schema_cache


def run_sql(
    dataset_refs: List[Dict[str, Any]],
    source_db_path: str,
) -> List[Dict[str, Any]]:
    """
    Query FptEnglish.db for rows matching dataset_refs.

    Args:
        dataset_refs: List of {"table": str, "identifiers": {col: val, ...}}
        source_db_path: Path to FptEnglish.db

    Returns:
        List of row dicts:
        [
            {
                "table": "PlanetInHouse",
                "keys": {"Planet": 1, "House": 7},
                "text": "The Sun in the seventh house...",
                "row": { ...full row dict... },
            },
            ...
        ]
    """
    schema = _load_schema()
    if not schema:
        logger.error("No schema loaded, cannot query FptEnglish.db")
        return []

    conn = sqlite3.connect(source_db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row

    results = []
    for ref in dataset_refs:
        tname = ref.get("table", "")
        identifiers = ref.get("identifiers", {})

        tinfo = schema.get(tname)
        if not tinfo:
            logger.debug(f"Table '{tname}' not in schema, skipping")
            continue

        text_col = tinfo["text_col"]
        id_cols = tinfo["id_cols"]

        # Build WHERE clause from identifiers
        clauses = []
        params = []
        for col in id_cols:
            if col in identifiers:
                clauses.append(f'"{col}" = ?')
                params.append(identifiers[col])

        if not clauses:
            continue

        sql = f'SELECT * FROM "{tname}" WHERE {" AND ".join(clauses)}'
        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as e:
            logger.warning(f"SQL error for {tname}: {e}")
            continue

        for row in rows:
            row_dict = dict(row)
            text = row_dict.get(text_col, "")
            if not text or not str(text).strip():
                continue
            text = str(text).strip()

            # Extract identifier key-values for metadata
            keys = {c: row_dict[c] for c in id_cols if c in row_dict and row_dict[c] is not None}

            results.append({
                "table": tname,
                "keys": keys,
                "text": text,
                "row": row_dict,
            })

    conn.close()
    logger.debug(f"[SQLRunner] Queried {len(results)} rows from {len(dataset_refs)} refs")
    return results
