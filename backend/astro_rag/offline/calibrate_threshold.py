"""
calibrate_threshold.py – Offline tool to calibrate semantic chunking threshold.

Usage:
    python -m backend.astro_rag.offline.calibrate_threshold

Reads FptEnglish.db, samples long text rows, and finds the optimal
cosine similarity threshold for sentence-level semantic splitting.
Outputs the result to config/thresholds.json.
"""

import json
import os
import sqlite3
import statistics
import sys
from typing import List

import numpy as np

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


def _split_sentences(text: str) -> List[str]:
    import re
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def calibrate(db_path: str, sample_tables: List[str] = None, n_samples: int = 50):
    """
    Sample long-text rows from FptEnglish.db and compute inter-sentence
    cosine similarities to find an optimal split threshold.
    """
    from sentence_transformers import SentenceTransformer

    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return

    print(f"📊 Loading embedding model...")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Default sample tables (mix of long-text sources)
    if not sample_tables:
        sample_tables = [
            "Planet_in_House_Reading", "Nakshatra_Reading", "MahaDasa",
            "BirthOnNakshatra", "LKUpay", "DasaParagraph",
        ]

    all_sims = []

    for tbl in sample_tables:
        try:
            rows = conn.execute(f'SELECT * FROM "{tbl}" LIMIT {n_samples}').fetchall()
        except sqlite3.OperationalError:
            print(f"  ⚠️ Table '{tbl}' not found, skipping")
            continue

        for row in rows:
            row_dict = dict(row)
            # Find the longest text column
            text = ""
            for v in row_dict.values():
                if isinstance(v, str) and len(v) > len(text):
                    text = v

            if len(text.split()) < 80:
                continue

            sentences = _split_sentences(text)
            if len(sentences) < 3:
                continue

            # Embed sentences
            vecs = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)

            # Compute adjacent cosine similarities
            for i in range(1, len(vecs)):
                sim = float(np.dot(vecs[i - 1], vecs[i]))
                all_sims.append(sim)

    conn.close()

    if not all_sims:
        print("❌ No suitable samples found")
        return

    # Compute statistics
    mean_sim = statistics.mean(all_sims)
    median_sim = statistics.median(all_sims)
    std_sim = statistics.stdev(all_sims) if len(all_sims) > 1 else 0
    p25 = sorted(all_sims)[len(all_sims) // 4]
    p10 = sorted(all_sims)[len(all_sims) // 10]

    # Threshold = mean - 1 std (topic change boundary)
    threshold = round(mean_sim - std_sim, 3)

    print(f"\n📊 Calibration Results ({len(all_sims)} sentence pairs):")
    print(f"  Mean similarity:   {mean_sim:.4f}")
    print(f"  Median similarity: {median_sim:.4f}")
    print(f"  Std deviation:     {std_sim:.4f}")
    print(f"  25th percentile:   {p25:.4f}")
    print(f"  10th percentile:   {p10:.4f}")
    print(f"  ➡️  Recommended threshold: {threshold}")

    # Write to config
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "thresholds.json"
    )
    config = {
        "semantic_split_threshold": threshold,
        "min_chunk_words": 40,
        "max_chunk_words": 500,
        "short_row_threshold": 80,
        "_calibration": {
            "n_pairs": len(all_sims),
            "mean": round(mean_sim, 4),
            "median": round(median_sim, 4),
            "std": round(std_sim, 4),
        },
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  ✅ Written to {config_path}")


if __name__ == "__main__":
    from backend.config import settings
    calibrate(settings.SOURCE_DB_PATH)
