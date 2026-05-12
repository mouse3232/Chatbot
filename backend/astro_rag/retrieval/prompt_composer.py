"""
prompt_composer.py – Master Prompt + Skill Prompt dynamic composition.

Architecture:
- MASTER_PROMPT: Controls orchestration, multi-domain reasoning, unified response
- Skill Prompts: KUNDALI, REMEDIES, TIMING, HORARY — each domain-specific
- Composer: Dynamically assembles the prompt based on active intents from the decomposer
- Single LLM call handles all domains

Context Flow:
- RAG-retrieved text → injected into relevant skill prompt sections
- No raw DB logic or retrieval instructions exposed to LLM
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── COMPOSER ──────────────────────────────────────────────────────────────────

def compose_prompt(
    active_domains: List[Dict[str, Any]],
    user_question: str,
    user_name: str = "Dear one",
    age: int = 25,
    gender: str = "Male",
    birth_info: str = "",
    answer_length: str = "Short",
    language: str = "English",
    conversation_summary: str = "",
) -> str:
    """
    Dynamically route to the appropriate Astro Skill module based on active domains.
    """
    from backend.common.prompts import ANSWER_LENGTHS
    import importlib

    min_w, max_w = ANSWER_LENGTHS.get(answer_length, (30, 50))

    # Build contexts dict and extract active domain names
    contexts = {}
    domain_names = []

    for domain_info in active_domains:
        domain = domain_info["domain"]
        domain_names.append(domain)
        
        context_text = domain_info.get("context", "No context available.")
        time_period = domain_info.get("time_period")
        remote_text = domain_info.get("remote_prediction")
        
        # Assemble domain context string
        parts = []
        if time_period:
            parts.append(f"PREDICTION PERIOD: {time_period}")
        if remote_text:
            tag = "HORARY SERVER INTERPRETATION" if domain == "horary" else "TIMING SERVER PREDICTION"
            parts.append(f"{tag}:\\n{remote_text}")
        
        parts.append(f"RAG CONTEXT:\\n{context_text}")
        contexts[domain] = "\\n\\n".join(parts)

    # Sort domains alphabetically to match module signatures (e.g. 'horary_kundali_remedies_timing')
    domain_names = sorted(list(set(domain_names)))
    
    # Construct signature
    signature = "_".join(domain_names)
    if not signature:
        signature = "kundali_only"
    elif len(domain_names) == 1:
        signature = f"{signature}_only"

    # Profile dictionary
    user_profile = {
        "name": user_name,
        "age": age,
        "gender": gender,
        "birth_info": birth_info or "Not provided"
    }

    try:
        # Dynamically import the skill module
        skill_module = importlib.import_module(f"backend.astro_skills.{signature}")
        prompt = skill_module.build_prompt(
            user_profile=user_profile,
            contexts=contexts,
            answer_length_words=(min_w, max_w),
            conversation_summary=conversation_summary,
            user_question=user_question
        )
        
        # Prepend the dynamic language instruction
        prompt = f"System Instruction: Please respond in {language}.\\n\\n" + prompt
        
        logger.info(f"[PromptComposer] Routed to {signature} skill module. Length: {min_w}-{max_w} words.")
        return prompt
        
    except Exception as e:
        logger.error(f"[PromptComposer] Failed to load skill module '{signature}': {e}. Falling back to kundali_only.")
        try:
            fallback_module = importlib.import_module("backend.astro_skills.kundali_only")
            return fallback_module.build_prompt(user_profile, contexts, (min_w, max_w), conversation_summary, user_question)
        except Exception as fallback_e:
            logger.critical(f"Fallback also failed: {fallback_e}")
            return f"RESPOND TO THIS QUESTION: {user_question}"


def build_domain_list_from_search(
    search_requests: List[Dict[str, Any]],
    reranked_results: List[Dict[str, Any]],
    remote_text: Optional[str] = None,
    remote_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Convert search requests + reranked FAISS results into active_domains list
    for compose_prompt().

    Groups reranked results by domain, builds context text per domain.
    """
    from backend.astro_rag.retrieval.context_builder import build_context
    from xml.sax.saxutils import escape

    # Group results by domain
    domain_results: Dict[str, List] = {}
    for r in reranked_results:
        d = r.get("domain", "kundali")
        # Map internal domain names to skill names
        if d == "astrology":
            d = "kundali"
        domain_results.setdefault(d, []).append(r)

    # Determine which flows are active
    active_flows = {}
    for req in search_requests:
        domain = req["domain"]
        if domain == "astrology":
            domain = "kundali"
        flow = req.get("flow")
        if flow == "horary":
            domain = "horary"  # Override timing → horary for short-term
        active_flows[domain] = {
            "time_period": req.get("time_period"),
            "flow": flow,
        }

    # Build active domains
    active_domains = []

    for domain, results in domain_results.items():
        # Build plain text context (no XML, just readings)
        context_parts = []
        for r in results:
            chunk = r.get("chunk", {})
            table = chunk.get("table", "")
            category = chunk.get("category", "")
            score = r.get("rerank_score", r.get("score", 0.0))
            text = chunk.get("text", "")
            house = chunk.get("house")

            header = f"[{table}"
            if category:
                header += f" | {category}"
            if house is not None:
                header += f" | House {house}"
            header += f" | score={score:.2f}]"

            context_parts.append(f"{header}\n{text}")

        context_text = "\n\n---\n\n".join(context_parts) if context_parts else "No specific context found."

        # Get flow metadata
        flow_info = active_flows.get(domain, {})
        tp = flow_info.get("time_period")
        time_period_str = None
        if tp:
            time_period_str = f"{tp.get('t', '')}: {tp.get('v', '')}"

        # Remote prediction
        domain_remote = None
        if domain == "horary" and remote_type == "horary":
            domain_remote = remote_text
        elif domain == "timing" and remote_type == "timing":
            domain_remote = remote_text

        active_domains.append({
            "domain": domain,
            "context": context_text,
            "time_period": time_period_str,
            "remote_prediction": domain_remote,
        })

    # Ensure at least one domain (fallback to kundali)
    if not active_domains:
        active_domains.append({
            "domain": "kundali",
            "context": "No specific context retrieved.",
            "time_period": None,
            "remote_prediction": None,
        })

    return active_domains
