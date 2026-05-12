"""
rag_session_manager.py – Session lifecycle: create, cache, evict, persist.

- In-memory cache of RAGSession objects
- TTL eviction (120s idle)
- .pkl persistence for FAISS indices only (conversation state in Redis)
"""

import logging
import os
import pickle
import threading
import time
from typing import Any, Dict, List, Optional

from backend.astro_rag.session.rag_session import RAGSession

logger = logging.getLogger(__name__)

# Session TTL in seconds (idle timeout)
SESSION_TTL = 120
# Background eviction check interval
EVICTION_INTERVAL = 30

# Default persistence directory
_DEFAULT_PKL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "sessions",
)


class RAGSessionManager:
    """
    Manages RAGSession lifecycle: creation, caching, eviction, persistence.
    Thread-safe via lock.
    """

    def __init__(self, pkl_dir: Optional[str] = None):
        self._sessions: Dict[str, RAGSession] = {}
        self._lock = threading.Lock()
        self._pkl_dir = pkl_dir or _DEFAULT_PKL_DIR
        os.makedirs(self._pkl_dir, exist_ok=True)

        # Start background eviction thread
        self._eviction_thread = threading.Thread(
            target=self._eviction_loop, daemon=True, name="rag-eviction"
        )
        self._eviction_thread.start()
        logger.info(f"[SessionMgr] Started with pkl_dir={self._pkl_dir}, TTL={SESSION_TTL}s")

    def get_session(self, session_id: str) -> Optional[RAGSession]:
        """Get an active session. Returns None if not found."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.last_accessed = time.time()
            return session

    def create_session(self, session_id: str) -> RAGSession:
        """Create a new session and register it."""
        session = RAGSession(session_id)
        with self._lock:
            # Evict old session if exists
            old = self._sessions.pop(session_id, None)
            if old:
                old.cleanup()
            self._sessions[session_id] = session
        logger.debug(f"[SessionMgr] Created session {session_id}")
        return session

    def remove_session(self, session_id: str):
        """Remove and cleanup a session."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session:
            session.cleanup()
            logger.debug(f"[SessionMgr] Removed session {session_id}")

    def persist_session(self, session_id: str) -> bool:
        """Save session indices to .pkl file."""
        session = self.get_session(session_id)
        if not session:
            return False

        pkl_path = os.path.join(self._pkl_dir, f"{session_id}.pkl")
        try:
            import faiss

            # Convert GPU indices to CPU for serialization
            save_data = {
                "session_id": session_id,
                "chunks": session.chunks,
                "created_at": session.created_at,
                "indices": {},
            }
            for domain, index in session.indices.items():
                try:
                    cpu_index = faiss.index_gpu_to_cpu(index)
                except Exception:
                    cpu_index = index
                save_data["indices"][domain] = faiss.serialize_index(cpu_index)

            with open(pkl_path, "wb") as f:
                pickle.dump(save_data, f)

            logger.info(f"[SessionMgr] Persisted session {session_id} to {pkl_path}")
            return True
        except Exception as e:
            logger.error(f"[SessionMgr] Failed to persist {session_id}: {e}")
            return False

    def restore_session(self, session_id: str) -> Optional[RAGSession]:
        """Restore session from .pkl file."""
        pkl_path = os.path.join(self._pkl_dir, f"{session_id}.pkl")
        if not os.path.exists(pkl_path):
            return None

        try:
            import faiss

            with open(pkl_path, "rb") as f:
                data = pickle.load(f)

            session = RAGSession(session_id)
            session.chunks = data.get("chunks", {})
            session.created_at = data.get("created_at", time.time())

            for domain, index_bytes in data.get("indices", {}).items():
                cpu_index = faiss.deserialize_index(index_bytes)
                # Try GPU only if supported
                try:
                    from backend.config import GPU_AVAILABLE
                    if GPU_AVAILABLE:
                        res = faiss.StandardGpuResources()
                        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                        session.indices[domain] = gpu_index
                    else:
                        session.indices[domain] = cpu_index
                except Exception:
                    session.indices[domain] = cpu_index

            with self._lock:
                self._sessions[session_id] = session

            logger.info(f"[SessionMgr] Restored session {session_id} from pkl")
            return session
        except Exception as e:
            logger.error(f"[SessionMgr] Failed to restore {session_id}: {e}")
            return None

    def _eviction_loop(self):
        """Background thread to evict idle sessions."""
        while True:
            time.sleep(EVICTION_INTERVAL)
            try:
                self._evict_idle()
            except Exception as e:
                logger.error(f"[SessionMgr] Eviction error: {e}")

    def _evict_idle(self):
        """Evict sessions that have been idle longer than TTL."""
        with self._lock:
            to_evict = [
                sid for sid, s in self._sessions.items()
                if s.idle_seconds > SESSION_TTL
            ]

        for sid in to_evict:
            self.persist_session(sid)
            self.remove_session(sid)

        if to_evict:
            logger.debug(f"[SessionMgr] Evicted {len(to_evict)} idle sessions")

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)
