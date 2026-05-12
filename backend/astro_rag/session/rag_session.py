"""
rag_session.py – RAGSession class holding 3 FAISS indices + chunk metadata.

One RAGSession per user. Created during session init, cached in RAM.
Conversation state lives in Redis (chat_history.py), NOT in this class.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from backend.astro_rag.config.domain_mapping import DOMAINS

logger = logging.getLogger(__name__)


class RAGSession:
    """
    Holds the 3-domain FAISS indices and chunk metadata for a single user session.

    Attributes:
        session_id:     Unique session identifier
        indices:        {domain: faiss.Index}
        chunks:         {domain: list[dict]}  — chunk metadata for each vector
        created_at:     Session creation timestamp
        last_accessed:  Last query timestamp (updated on every search)
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.indices: Dict[str, Any] = {}           # domain → faiss.Index
        self.chunks: Dict[str, List[Dict]] = {}     # domain → chunk list (parallel to index)
        self.created_at = time.time()
        self.last_accessed = time.time()

    def set_domain(self, domain: str, index, chunks: List[Dict]):
        """Store FAISS index + chunks for a domain."""
        self.indices[domain] = index
        self.chunks[domain] = chunks
        logger.debug(f"[RAGSession] {self.session_id}: set {domain} with {len(chunks)} chunks")

    def search(
        self,
        domain: str,
        query_vec: np.ndarray,
        k: int = 50,
        filter_fn=None,
    ) -> List[Dict[str, Any]]:
        """
        Search within a specific domain index.

        Args:
            domain:    Domain name (astrology/remedies/timing)
            query_vec: (1, dim) normalized query vector
            k:         Number of nearest neighbors
            filter_fn: Optional callable(chunk_dict) → bool for pre-filtering

        Returns:
            List of {"chunk": dict, "score": float}, sorted by score desc
        """
        self.last_accessed = time.time()

        if domain not in self.indices or domain not in self.chunks:
            logger.warning(f"[RAGSession] Domain '{domain}' not found in session {self.session_id}")
            return []

        index = self.indices[domain]
        all_chunks = self.chunks[domain]

        if not all_chunks:
            return []

        # Apply pre-filter if provided
        if filter_fn:
            valid_indices = [i for i, c in enumerate(all_chunks) if filter_fn(c)]
            if len(valid_indices) < 2:
                # Filter too restrictive — fallback to full index
                logger.debug(f"[RAGSession] Filter too strict ({len(valid_indices)} results), using full index")
                valid_indices = list(range(len(all_chunks)))
        else:
            valid_indices = list(range(len(all_chunks)))

        # Build subset index for filtered search
        import faiss

        if len(valid_indices) == len(all_chunks):
            # No filtering needed, search full index
            scores, ids = index.search(query_vec.astype(np.float32), min(k, len(all_chunks)))
        else:
            # Create filtered subset index
            full_vecs = self._get_vectors(domain)
            if full_vecs is None:
                scores, ids = index.search(query_vec.astype(np.float32), min(k, len(all_chunks)))
            else:
                subset_vecs = full_vecs[valid_indices]
                subset_index = faiss.IndexFlatIP(subset_vecs.shape[1])
                subset_index.add(subset_vecs)
                scores, ids = subset_index.search(
                    query_vec.astype(np.float32), min(k, len(valid_indices))
                )
                # Map back to original indices
                ids = np.array([[valid_indices[i] if i >= 0 else -1 for i in row] for row in ids])

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            results.append({
                "chunk": all_chunks[idx],
                "score": float(score),
            })

        return results

    def _get_vectors(self, domain: str) -> Optional[np.ndarray]:
        """Reconstruct vectors from FAISS index (CPU only)."""
        try:
            import faiss
            index = self.indices.get(domain)
            if index is None:
                return None
            # Try to get the CPU index if GPU
            try:
                cpu_index = faiss.index_gpu_to_cpu(index)
            except Exception:
                cpu_index = index
            n = cpu_index.ntotal
            if n == 0:
                return None
            return faiss.rev_swig_ptr(cpu_index.get_xb(), n * cpu_index.d).reshape(n, cpu_index.d).copy()
        except Exception as e:
            logger.warning(f"[RAGSession] Failed to reconstruct vectors for {domain}: {e}")
            return None

    def cleanup(self):
        """Free GPU memory on session eviction."""
        for domain, idx in self.indices.items():
            try:
                del idx
            except Exception:
                pass
        self.indices.clear()
        self.chunks.clear()
        logger.info(f"[RAGSession] Cleaned up session {self.session_id}")

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_accessed

    def __repr__(self):
        counts = {d: len(c) for d, c in self.chunks.items()}
        return f"<RAGSession {self.session_id} chunks={counts} idle={self.idle_seconds:.0f}s>"
