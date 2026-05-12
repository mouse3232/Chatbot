import logging
import sqlite3
import json
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

def parse_compressed_records(records_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Parses the compressed payload format:
    [{"table": "T1", "keys": ["K1", "K2"], "values": [[V1, V2], [V3, V4]]}]
    Returns a unified DatasetReference: [{"table": "T1", "identifiers": {"K1": V1, "K2": V2}}]
    """
    dataset_refs = []
    for item in records_list:
        table_name = item.get("table")
        keys = item.get("keys", [])
        values = item.get("values", [])
        for row_values in values:
            if len(keys) == len(row_values):
                identifiers = dict(zip(keys, row_values))
                dataset_refs.append({
                    "table": table_name,
                    "identifiers": identifiers
                })
    return dataset_refs

def resolve_rows_from_records(
    records_list: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Resolves dataset references (table + identifier-values) from the JSON records list.
    No longer resolves rowids; the session_store handles the SQL transformation.
    """
    refs = parse_compressed_records(records_list)
    logger.info("Resolved %d schema-based dataset references", len(refs))
    return refs

def parse_default_text(content: str) -> List[Dict[str, Any]]:
    """Legacy parser for text-based birth records."""
    results = []
    lines = content.split('\n')
    for line in lines:
        if ':' in line:
            parts = line.split(':', 1)
            table = parts[0].strip()
            results.append({"table": table, "identifiers": {}}) 
    return results

def resolve_rows_from_text(content: str) -> List[Dict[str, Any]]:
    """Resolve dataset references from legacy text content."""
    return parse_default_text(content)
