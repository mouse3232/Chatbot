"""
chat_history.py – Redis-backed sliding window chat history.

- Sliding window of last N Q/A pairs for LLM context
- Summarize every 3 interactions (rolling summary in Redis)
- Evict entries older than 20 from Redis → write to disk logs (never lost)
- TTL matches session idle timeout
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
WINDOW_SIZE = 5         # Last N Q/A pairs sent to LLM as recent context
SUMMARIZE_EVERY = 3     # Summarize after every N new interactions
MAX_HISTORY = 20        # Evict entries older than this from Redis
SESSION_TTL_SECONDS = 120  # Redis key expiry matching session TTL

# Redis connection (lazy-loaded)
_redis_client = None


def _get_redis():
    """Lazy-load Redis connection."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        import redis
        _redis_client = redis.Redis(
            host="localhost", port=6379, db=0, decode_responses=True
        )
        _redis_client.ping()
        logger.info("[ChatHistory] Connected to Redis")
        return _redis_client
    except Exception as e:
        logger.warning(f"[ChatHistory] Redis unavailable, using in-memory fallback: {e}")
        return None


# ── In-memory fallback ────────────────────────────────────────────────────────
_memory_store: Dict[str, Dict] = {}


def _get_fallback(session_id: str) -> Dict:
    """In-memory fallback when Redis is unavailable."""
    if session_id not in _memory_store:
        _memory_store[session_id] = {
            "history": [],
            "summary": "",
            "unsummarized": 0,
        }
    return _memory_store[session_id]


# ── Public API ────────────────────────────────────────────────────────────────

def record_interaction(session_id: str, query: str, response: str):
    """
    Record a Q/A interaction. Trim to MAX_HISTORY, logging evicted entries.
    """
    entry = json.dumps({
        "q": query,
        "r": response[:500],  # Truncate long responses
        "ts": time.time(),
    })

    r = _get_redis()
    if r:
        try:
            key = f"chat:{session_id}:history"
            r.rpush(key, entry)
            r.expire(key, SESSION_TTL_SECONDS)

            # Evict old entries beyond MAX_HISTORY
            overflow = r.llen(key) - MAX_HISTORY
            if overflow > 0:
                for _ in range(overflow):
                    old = r.lpop(key)
                    logger.info(f"[ChatHistory] Evicted from {session_id}: {old}")

            # Increment unsummarized counter
            count_key = f"chat:{session_id}:unsummarized"
            r.incr(count_key)
            r.expire(count_key, SESSION_TTL_SECONDS)
            return
        except Exception as e:
            logger.warning(f"[ChatHistory] Redis write failed: {e}")

    # Fallback: in-memory
    fb = _get_fallback(session_id)
    fb["history"].append(entry)
    if len(fb["history"]) > MAX_HISTORY:
        old = fb["history"].pop(0)
        logger.info(f"[ChatHistory] Evicted (memory) from {session_id}: {old}")
    fb["unsummarized"] += 1


def get_recent_context(session_id: str) -> Dict[str, Any]:
    """
    Return sliding window of last N exchanges + rolling summary.

    Returns:
        {"summary": str, "recent": [{"q": str, "r": str, "ts": float}, ...]}
    """
    r = _get_redis()
    if r:
        try:
            key = f"chat:{session_id}:history"
            summary_key = f"chat:{session_id}:summary"
            entries = r.lrange(key, -WINDOW_SIZE, -1)
            return {
                "summary": r.get(summary_key) or "",
                "recent": [json.loads(e) for e in entries],
            }
        except Exception as e:
            logger.warning(f"[ChatHistory] Redis read failed: {e}")

    # Fallback
    fb = _get_fallback(session_id)
    return {
        "summary": fb["summary"],
        "recent": [json.loads(e) for e in fb["history"][-WINDOW_SIZE:]],
    }


def should_summarize(session_id: str) -> bool:
    """Check if we should trigger summarization (every N interactions)."""
    r = _get_redis()
    if r:
        try:
            count = int(r.get(f"chat:{session_id}:unsummarized") or 0)
            return count >= SUMMARIZE_EVERY
        except Exception:
            pass

    fb = _get_fallback(session_id)
    return fb["unsummarized"] >= SUMMARIZE_EVERY


def save_summary(session_id: str, summary_text: str):
    """Store the rolling summary and reset unsummarized counter."""
    r = _get_redis()
    if r:
        try:
            r.set(f"chat:{session_id}:summary", summary_text)
            r.expire(f"chat:{session_id}:summary", SESSION_TTL_SECONDS)
            r.set(f"chat:{session_id}:unsummarized", 0)
            r.expire(f"chat:{session_id}:unsummarized", SESSION_TTL_SECONDS)
            return
        except Exception as e:
            logger.warning(f"[ChatHistory] Redis summary save failed: {e}")

    fb = _get_fallback(session_id)
    fb["summary"] = summary_text
    fb["unsummarized"] = 0


def get_full_history(session_id: str) -> List[Dict]:
    """Get all history entries (for summarization)."""
    r = _get_redis()
    if r:
        try:
            key = f"chat:{session_id}:history"
            entries = r.lrange(key, 0, -1)
            return [json.loads(e) for e in entries]
        except Exception:
            pass

    fb = _get_fallback(session_id)
    return [json.loads(e) for e in fb["history"]]


def cleanup_session(session_id: str):
    """Remove all Redis keys for a session."""
    r = _get_redis()
    if r:
        try:
            r.delete(
                f"chat:{session_id}:history",
                f"chat:{session_id}:summary",
                f"chat:{session_id}:unsummarized",
            )
        except Exception:
            pass

    _memory_store.pop(session_id, None)
