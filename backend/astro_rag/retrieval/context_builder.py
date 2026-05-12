"""
context_builder.py – Format reranked results into structured XML for LLM.

Builds an XML context block from:
- Remote prediction text (Horary or Timing)
- FAISS-retrieved and reranked chunks
- Chat history summary + recent exchanges
"""

import logging
from typing import Any, Dict, List, Optional
from xml.sax.saxutils import escape

logger = logging.getLogger(__name__)


def build_context(
    reranked_results: List[Dict[str, Any]],
    remote_text: Optional[str] = None,
    remote_type: Optional[str] = None,
    chat_context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build XML context string for LLM prompt injection.

    Args:
        reranked_results: Output of reranker.rerank()
        remote_text: Text from remote prediction server (if Timing flow)
        remote_type: "horary" or "timing" (determines XML tag)
        chat_context: {"summary": str, "recent": [{"q": str, "r": str}]}

    Returns:
        XML-formatted context string
    """
    parts = []

    # ── Conversation context ──────────────────────────────────────────────
    if chat_context:
        summary = chat_context.get("summary", "")
        recent = chat_context.get("recent", [])

        if summary:
            parts.append(f"<Conversation_Summary>{escape(summary)}</Conversation_Summary>")

        if recent:
            recent_parts = []
            for ex in recent[-5:]:
                q = escape(str(ex.get("q", "")))
                r = escape(str(ex.get("r", "")))
                recent_parts.append(f'  <Exchange question="{q}">{r}</Exchange>')
            parts.append(
                "<Recent_Exchanges>\n" + "\n".join(recent_parts) + "\n</Recent_Exchanges>"
            )

    # ── Remote prediction ─────────────────────────────────────────────────
    if remote_text and remote_text.strip():
        if remote_type == "horary":
            parts.append(
                f"<HoraryInterpretation>{escape(remote_text)}</HoraryInterpretation>"
            )
        elif remote_type == "timing":
            parts.append(
                f"<TimingPrediction>{escape(remote_text)}</TimingPrediction>"
            )
        else:
            parts.append(
                f"<RemotePrediction>{escape(remote_text)}</RemotePrediction>"
            )

    # ── FAISS-retrieved readings ──────────────────────────────────────────
    if reranked_results:
        reading_parts = []
        for r in reranked_results:
            chunk = r.get("chunk", {})
            table = chunk.get("table", "")
            category = chunk.get("category", "")
            house = chunk.get("house")
            score = r.get("rerank_score", r.get("score", 0.0))
            text = escape(chunk.get("text", ""))

            # Build attributes
            attrs = f'table="{table}" category="{category}" score="{score:.3f}"'
            if house is not None:
                attrs += f' house="{house}"'

            reading_parts.append(f"  <Reading {attrs}>{text}</Reading>")

        parts.append(
            "<Astrological_Context>\n" + "\n".join(reading_parts) + "\n</Astrological_Context>"
        )

    return "\n\n".join(parts)
