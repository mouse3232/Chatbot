import logging
import asyncio
import re
import os
import json
from typing import Dict, Any, List, Optional
from backend.common.llm import create_llm, strip_think_tags

from backend.standard.session_store_v2 import SESSION_DIR
from backend.common.model_registry import ModelTask
from backend.config import settings
from backend.common.utils import get_embedding
from backend.standard.suggestion_pool import SUGGESTION_POOL

logger = logging.getLogger(__name__)

# Cache for suggestion embeddings
_suggestion_embeddings_cache = {}

def get_suggestion_embeddings(category: str):
    """Fetch or compute embeddings for a suggestion category pool."""
    if category in _suggestion_embeddings_cache:
        return _suggestion_embeddings_cache[category]
    
    logger.info(f"[Orchestrator] Embedding suggestion pool for category: {category}")
    questions = SUGGESTION_POOL.get(category, [])
    if not questions:
        return None
        
    import torch
    emb_model = get_embedding(settings.EMBEDDING_MODEL)
    # Handle both SentenceTransformer and LangChain-style embedding models
    if hasattr(emb_model, "embed_documents"):
        vectors = emb_model.embed_documents(questions)
    else:
        vectors = emb_model.encode(questions)
        
    tensor = torch.tensor(vectors).float()
    
    _suggestion_embeddings_cache[category] = (questions, tensor)
    return questions, tensor

def get_relevant_suggestions(category: str, reference_text: str, top_k: int = 3) -> List[str]:
    """Retrieve top-K most relevant suggestions from the pool using semantic similarity."""
    try:
        pool_data = get_suggestion_embeddings(category)
        if not pool_data:
            return ["Tell me more about my career.", "What about my finances?", "When is a good time to travel?"]
        
        questions, pool_tensor = pool_data
        
        import torch
        emb_model = get_embedding(settings.EMBEDDING_MODEL)
        
        # Handle both SentenceTransformer (.encode) and LangChain (.embed_query)
        if hasattr(emb_model, "embed_query"):
            query_vector = emb_model.embed_query(reference_text)
        else:
            query_vector = emb_model.encode([reference_text])[0]
            
        query_tensor = torch.tensor(query_vector).unsqueeze(0).float() # [1, Dim]
        
        # Calculate Cosine Similarity (Dot product since vectors are normalized)
        scores = torch.mm(query_tensor, pool_tensor.t()).squeeze(0) # [NumPool]
        
        # Get top indices
        top_indices = torch.topk(scores, k=min(top_k, len(questions))).indices.tolist()
        
        return [questions[i] for i in top_indices]
    except Exception as e:
        logger.error(f"[Orchestrator] Suggestion retrieval failed: {e}")
        return SUGGESTION_POOL.get(category, ["Tell me more?"])[:3]

async def summarize_history(session_id: str, messages: List[Dict[str, str]]) -> str:
    """Agent 0c: Historian. Summarizes the conversation into 2-3 lines every 3 turns."""
    if not messages or len(messages) < 2:
        return ""
        
    llm = create_llm(task=ModelTask.SUMMARIZATION)
    # Take the last 6-8 messages for a rolling summary
    history_str = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages[-8:]])
    
    prompt = f"""
    Summarize the following astrological conversation into exactly 2-3 concise lines.
    Highlight the core topics discussed and any specific user concerns or preferences mentioned.
    This summary will used to help the AI remember the context for future questions.
    
    CONVERSATION:
    {history_str}
    
    CRITICAL: RETURN ONLY THE 2-3 LINE SUMMARY. NO INTRO OR OUTRO.
    """
    
    try:
        summary = (await asyncio.get_event_loop().run_in_executor(None, llm.invoke, prompt)).content.strip()
        logger.info(f"[Orchestrator] Updated Conversation Summary: {summary}")
        return summary
    except Exception as e:
        logger.error(f"[Orchestrator] Summarization Error: {e}")
        return ""

def get_astro_skill(active_domains: List[str]):
    """
    Dynamically imports the correct astrological skill module based on active domains.
    """
    if not active_domains:
        import backend.astro_skills.kundali_only as default_skill
        return default_skill
        
    # Sort for deterministic naming (e.g., kundali_remedies)
    valid_domains = ["horary", "kundali", "remedies", "timing"]
    filtered = sorted([d for d in active_domains if d in valid_domains])
    
    if not filtered:
        import backend.astro_skills.kundali_only as default_skill
        return default_skill

    # Construct module name
    base_name = "_".join(filtered)
    if len(filtered) == 1:
        base_name += "_only"
    
    try:
        import importlib
        module_path = f"backend.astro_skills.{base_name}"
        return importlib.import_module(module_path)
    except Exception as e:
        logger.error(f"[Orchestrator] Failed to load skill {base_name}: {e}. Falling back to kundali_only.")
        import backend.astro_skills.kundali_only as fallback
        return fallback

async def translate_text(text: str, target_language: str, language_code: int = 0) -> str:
    """Agent 0t: Translator. Translates a finalized response into the target language."""
    if not target_language or target_language.lower() == "english":
        return text
        
    # FORCE Sarvam 30B for non-English translations regardless of LOCAL setting
    if language_code != 0:
        logger.info(f"[Orchestrator] Forcing Sarvam 30B for translation (code: {language_code})")
        llm = create_llm(task=ModelTask.MULTILINGUAL, model_override="sarvam/sarvam-30b")
    else:
        llm = create_llm(task=ModelTask.MULTILINGUAL)

    # Specific strictness for Hindi
    hindi_rules = ""
    if str(language_code) == "1" or target_language.lower() == "hindi":
        hindi_rules = """
        STRICT HINDI ENFORCEMENT:
        1. Use ONLY Devanagari script.
        2. ABSOLUTELY NO Hinglish or Latin characters.
        3. Ensure the translation is in pure, formal, yet compassionate Hindi.
        4. Do NOT mix English words (like 'stable', 'career', 'match') into the middle of Hindi sentences. 
        5. Use proper Hindi counterparts for common terms (e.g., use 'स्थिर' for 'stable', 'व्यवसाय' for 'career').
        """

    # Distilled prompt for short snippets (suggestions/headers)
    is_short = len(text.split()) < 20
    if is_short:
        prompt = f"Translate the following astrological question exactly into {target_language}. Return ONLY the translation.\n\nTEXT: {text}\n\n{hindi_rules}"
    else:
        prompt = f"""
        Translate the following text into {target_language}. 
        
        CRITICAL RULES:
        1. NO intro/outro like 'Here is the translation'.
        2. Maintain the compassionate tone.
        3. Translate the text EXACTLY as provided.
        {hindi_rules}

        TEXT:
        {text}
        
        CRITICAL: RETURN ONLY THE TRANSLATED TEXT.
        """
        
    
    try:
        translated = (await asyncio.get_event_loop().run_in_executor(None, llm.invoke, prompt)).content.strip()
        translated = strip_think_tags(translated)
        # Final scrub of model chatter
        translated = re.split(r'Translation:', translated, flags=re.I)[-1].strip()
        translated = re.split(r'CRITICAL:', translated, flags=re.I)[0].strip()
        return translated
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return text


async def generate_last_call_suggestions(answer: str, intent: str = "KUNDALI") -> List[str]:
    """
    Retrieves the top 3 most relevant questions from the predefined suggestion pool.
    This ensures quality and consistency while maintaining strict English output.
    """
    logger.info(f"[Orchestrator] Fetching suggestions from POOL for intent: {intent}")
    return get_relevant_suggestions(intent, answer, top_k=3)

def _clean_suggestion(text: str) -> str:
    """Helper to remove common LLM artifacts from suggestions."""
    text = re.sub(r'^[0-9\.\-\*\s]+', '', text)
    text = re.sub(r'^(Question|Suggestion)\s*\d*:\s*', '', text, flags=re.I)
    return text.strip()


def validate_session_readiness(session_id: str, intent: str) -> bool:
    """
    Checks if the required .pkl files for the given intent physically exist on disk.
    A session is 'expired/invalid' only if these files are missing from the sessions/ directory.
    """
    if intent == "HORARY":
        return True
        
    required_suffixes = []
    if intent == "KUNDALI":
        required_suffixes = ["_records"]
    elif intent == "REMEDIES":
        required_suffixes = ["_records", "_remedies"]
    elif intent == "MATCHING":
        required_suffixes = ["_r1", "_r2", "_mr"]
        
    for suffix in required_suffixes:
        pkl_path = os.path.join(SESSION_DIR, f"{session_id}{suffix}.pkl")
        if not os.path.exists(pkl_path):
            # Check if session exists in memory with 0 chunks
            try:
                from backend.app import rag_session_manager
                sess = rag_session_manager.get_session(f"{session_id}{suffix}")
                if sess and sess.total_chunks == 0:
                    logger.info(f"[Orchestrator] Readiness check passed for empty session: {session_id}{suffix}")
                    continue
            except Exception:
                pass
            logger.warning(f"[Orchestrator] Readiness Check FAILED: Missing {pkl_path}")
            return False
            
    return True

async def orchestrate_astro_query(session_id: str, query: str, user_profile: Dict[str, Any], 
                               language: str = "English", language_code: int = 0, 
                               is_matching: bool = False, intent_override: Optional[str] = None, 
                               stream: bool = False, chart_data: Optional[Dict[str, Any]] = None, 
                               debug: bool = False, parent_id: Optional[str] = None) -> Any:
    """Agent 0: Orchestrator. Coordinates specialized agents and synthesizes final response."""
    # 0a. Dynamic Length Detection
    q_low = query.lower()
    length_preference = "Long" # Default 150-250 words
    if any(w in q_low for w in ["brief", "short", "summary", "quick", "1 sentence"]):
        length_preference = "Short"
    
    # Enforce higher fidelity K=12 for most data-driven queries
    k_value = 12
    
    from backend.app import session_manager
    session_data = session_manager.get_session(session_id) or {"messages": []}
    messages = session_data.get("messages", [])
    current_summary = session_data.get("conversation_summary", "")

    # 1. Stage 1: Intent & Expansion using Gemini Decomposer (replaces BERT + T5)
    name2 = user_profile.get("name2", "")
    q_up = query.upper()
    
    from backend.astro_rag.retrieval.query_decomposer import decompose_query
    from backend.astro_rag.retrieval.decomposer_bridge import parse_decomposer_output
    
    # Run Gemini Decomposer (REST API call)
    raw_json = await asyncio.get_event_loop().run_in_executor(None, decompose_query, query, user_profile)
    try:
        decomposer_output = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
    except (json.JSONDecodeError, TypeError):
        decomposer_output = []
    
    search_requests = parse_decomposer_output(decomposer_output)
    
    # Intent Logic
    intent = "KUNDALI"
    if intent_override:
        intent = intent_override.upper()
        logger.debug(f"[Orchestrator] Using intent override: {intent}")
    else:
        # Keyword Overrides (High Priority)
        COMPATIBILITY_TERMS = ["MANGLIK", "NADI", "ASHTAKOOT", "BHAKOOT", "GUNA", "MILAN", "COMPATIBILITY", "RELATIONSHIP", "MARRIAGE", "MATCH"]
        REMEDY_TERMS = ["UPAY", "REMEDY", "REMEDIES", "GEM", "RATNA", "MANTRA", "RUDRAKSHA", "LAL KITAB UPAY", "LALKITAB UPAY", "STONE", "WEAR"]
        
        if any(term in q_up for term in COMPATIBILITY_TERMS) and (name2 or is_matching):
            intent = "MATCHING"
            logger.debug("[Orchestrator] Compatibility query detected. Forcing MATCHING intent.")
        elif any(term in q_up for term in REMEDY_TERMS):
            intent = "REMEDIES"
            logger.debug("[Orchestrator] Remedy query detected. Forcing REMEDIES intent.")
        else:
            # Use Gemini Decomposer primary intent
            primary = decomposer_output[0] if decomposer_output else {}
            primary_int = primary.get("int", "Kundali")
            
            if primary_int == "Remedies":
                intent = "REMEDIES"
            elif primary_int == "Timing":
                intent = "TIMING"
            else:
                intent = "KUNDALI"
                
    # Expansion Logic — use decomposer semantic intents
    queries = [req["query"] for req in search_requests] if search_requests else [query]
    primary = decomposer_output[0] if decomposer_output else {}
    time_period = primary.get("tp")
    
    # Log the Decomposer results to the terminal
    print("\n" + "="*50)
    print("🤖 [GEMINI DECOMPOSER RESULTS]")
    print(f"  Query        : {query}")
    print(f"  Intents      : {[d.get('int') for d in decomposer_output]}")
    print(f"  Sem Queries  : {queries}")
    print(f"  Time Period  : {time_period}")
    print(f"  Flow Types   : {[r.get('flow') for r in search_requests]}")
    print("="*50 + "\n")
    
    expansion_data = {
        "expanded_queries": queries,
        "detected_topic": "General",
        "search_requests": search_requests,
    }
    
    topic = expansion_data.get('detected_topic', 'General')
    
    logger.info(f"[Orchestrator] Stage 1 Done: Intent={intent} | Period={time_period} | Queries={len(queries)}")
    
    if topic == "RESTRICTED":
         ans = "I apologize, but your query falls outside the scope of my astrological guidance."
         if stream:
             async def restricted_stream(): yield ans
             return restricted_stream()
         return {
            "answer": ans,
            "intent": "RESTRICTED",
            "agent_id": "ORCHESTRATOR"
        }

    # 2. Parallel Stage 2: Specialist Agent & Summarization
    # Check session readiness first
    if not validate_session_readiness(session_id, intent):
        ans = "Incomplete data !!!" if intent == "MATCHING" else "Invalid Session_id"
        if stream:
            async def error_stream(): yield ans
            return error_stream()
        return {
            "answer": ans,
            "intent": "ERROR" if intent == "MATCHING" else "EXPIRED",
            "agent_id": "ORCHESTRATOR"
        }

    # Prepare Specialist Call
    common_context = {
        "is_matching": is_matching,
        "expanded_queries": expansion_data['expanded_queries'],
        "detected_topic": topic,
        "chart_data": chart_data
    }
    
    # Route to specialist
    async def run_specialist():
        # Use unified RAG Searcher for ALL intents
        from backend.astro_rag.retrieval.searcher import search_multi_domain
        from backend.common.utils import get_embedding
        
        # Use pre-loaded embedding model
        embed_model = get_embedding("BAAI/bge-large-en-v1.5")
        
        # Fetch the active RAG session
        from backend.app import rag_session_manager
        
        # Determine the session suffix
        suffix = "_records"
        if intent == "REMEDIES":
            suffix = "_remedies"
        elif intent == "MATCHING":
            suffix = "_mr" # Use matchrecords store for matching intent
        
        session = rag_session_manager.get_session(f"{session_id}{suffix}")
        
        if not session:
            logger.warning(f"[Orchestrator] RAG Session {session_id}{suffix} not found for intent {intent}")
            # If no specific session, return raw chart data if available
            return {"findings": "No specific records found for this query.", "is_raw_data": True}
        
        results = search_multi_domain(
            session=session,
            search_requests=expansion_data["search_requests"],
            embed_model=embed_model,
            top_k_per_domain=k_value
        )
        
        # ── Layer 2: Reranker Gate ──
        # Score each chunk against the original query using the cross-encoder.
        # Drop irrelevant chunks to prevent off-topic context pollution.
        try:
            from backend.app import RAG_RERANKER
            if RAG_RERANKER and results:
                rerank_pairs = [(query, r["chunk"].get("text", "")) for r in results]
                rerank_scores = RAG_RERANKER.predict(rerank_pairs)
                
                # Attach scores and filter
                for i, r in enumerate(results):
                    r["rerank_score"] = float(rerank_scores[i])
                
                # Keep only chunks above relevance threshold (Relaxed to -0.5 for better recall)
                max_score = max(rerank_scores) if len(rerank_scores) > 0 else 0
                results = [r for r in results if r["rerank_score"] > -0.5]
                
                # Re-sort by reranker score (more accurate than FAISS cosine)
                results.sort(key=lambda x: x["rerank_score"], reverse=True)
                # Cap at top 8 to keep context focused
                results = results[:8]
                
                logger.info(f"[Orchestrator] Reranker: {len(rerank_pairs)} candidates → {len(results)} after filtering (Max Score: {max_score:.2f})")
        except ImportError:
            pass  # Reranker not loaded, use FAISS scores as-is
        
        # Consolidate findings
        findings_list = []
        for r in results:
            chunk = r["chunk"]
            findings_list.append(f"[{r['domain'].upper()}] {chunk.get('text', '')}")
        
        findings = "\n\n".join(findings_list)
        return {"findings": findings, "is_raw_data": True, "raw_results": results}

    specialist_task = run_specialist()
    
    # Trigger summarization in parallel if needed
    summary_task = None
    if len(messages) >= 2 and (len(messages) + 1) % 4 == 0:
        logger.info(f"[Orchestrator] Triggering parallel summarization for session {session_id}")
        summary_task = summarize_history(session_id, messages + [{"role": "user", "content": query}])
    else:
        # Dummy task that returns None
        summary_task = asyncio.Future()
        summary_task.set_result(None)

    stage2_results = await asyncio.gather(specialist_task, summary_task)
    agent_res = stage2_results[0]
    new_summary = stage2_results[1]
    
    if new_summary:
        session_manager.update_summary(session_id, new_summary)
        current_summary = new_summary
    
    # --- WORKFLOW: English-Only Synthesis ---
    # Combine specialist findings into a cohesive English narrative.
    import uuid
    provisional_interaction_id = str(uuid.uuid4())
    final_english_answer = ""

    if agent_res.get("answer") and not agent_res.get("is_raw_data"):
        final_english_answer = agent_res["answer"]
    else:
        # 1. DYNAMIC SKILL SELECTION
        active_domains = list(set(r['domain'] for r in agent_res.get('raw_results', [])))
        skill = get_astro_skill(active_domains)
        
        # Build Contexts map for the skill
        contexts = {}
        findings_list = []
        for r in agent_res.get('raw_results', []):
            d = r['domain']
            if d not in contexts: contexts[d] = ""
            text = r['chunk'].get('text', '')
            contexts[d] += f"\n{text}"
            findings_list.append(f"[{d.upper()}] {text}")
            
        findings = "\n\n".join(findings_list)
        # Issue 2: Increased context cap to 8000 chars for richer answers
        findings = findings[:8000] if findings else 'Detailed charts analyzed.'
            
        from backend.common.prompts import ANSWER_LENGTHS
        length_range = ANSWER_LENGTHS.get(length_preference, (350, 600))
        
        # Generate the high-fidelity system prompt from the specialized skill module
        system_prompt = skill.build_prompt(
            user_profile=user_profile,
            contexts=contexts,
            answer_length_words=length_range,
            conversation_summary=current_summary,
            user_question=query
        )

        # 2. Messages with ASSISTANT PREFILL — forces the model to start writing immediately
        raw_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Answer this question using the astrological data provided above: {query}"},
            {"role": "assistant", "content": "**"},
        ]

        # Capture context for short-lived debugging (Audit Trail)
        from backend.common.utils import save_llm_context
        save_llm_context(provisional_interaction_id, {
            "session_id": session_id,
            "interaction_id": provisional_interaction_id,
            "system_prompt": system_prompt,
            "user_query": query,
            "context_findings": findings,
            "rag_metadata": {
                "chunks": agent_res.get("raw_results", [])
            },
            "raw_messages": raw_messages
        })
        
        # Resolve the model string so we can use raw streaming
        from backend.common.model_registry import get_model_for_task
        model_str = get_model_for_task(ModelTask.FINAL_ANSWER, language_code=0)
        is_llamacpp = model_str.startswith("llamacpp@") or model_str.startswith("vllm@")
        
        # LangChain fallback for non-streaming or non-llamacpp providers
        llm_synthesis = create_llm(task=ModelTask.FINAL_ANSWER, language_code=0, max_tokens=2500, streaming=stream)

        # Define the generator for synthesis
        async def stream_generator():
            import re
            full_content = "**"  # We already "sent" the opening ** via prefill
            buffer = "**"       # Pre-seed with the prefill token
            total_chars = 0
            has_started_yield = False
            
            try:
                # ── RAW HTTPX STREAMING (bypasses LangChain entirely) ──
                if is_llamacpp:
                    from backend.common.llm import stream_llamacpp_raw
                    token_source = stream_llamacpp_raw(
                        messages=raw_messages,
                        model_str=model_str,
                        max_tokens=1200,
                        temperature=0.7,
                        stop=["<|im_end|>", "<|endoftext|>"],
                    )
                else:
                    # Fallback: LangChain astream for cloud providers
                    from langchain_core.messages import SystemMessage, HumanMessage
                    lc_messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=f"Answer this question using the astrological data provided above: {query}")
                    ]
                    
                    async def _langchain_stream():
                        async for chunk in llm_synthesis.astream(lc_messages):
                            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                            if content:
                                yield content
                    token_source = _langchain_stream()
                
                async for token in token_source:
                    full_content += token
                    total_chars += len(token)
                    
                    if not has_started_yield:
                        has_started_yield = True
                        # Yield the prefill character plus the first real token
                        yield full_content 
                    else:
                        yield token
                
                # END-OF-STREAM FALLBACK: If we consumed everything but yielded nothing
                if not has_started_yield and full_content:
                    cleaned = strip_think_tags(full_content)
                    if cleaned:
                        yield cleaned
                
                # Note: Interaction saving moved to final_stream_wrapper to capture full context
                stream_generator.full_content = full_content
            except Exception as e:
                logger.error(f"Streaming failed: {e}")
                yield " [Streaming error. Please try again.]"

        if not stream:
            try:
                from langchain_core.messages import SystemMessage, HumanMessage
                lc_messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"Answer this question using the astrological data provided above: {query}")
                ]
                res = (await asyncio.get_event_loop().run_in_executor(None, llm_synthesis.invoke, lc_messages))
                final_english_answer = strip_think_tags(res.content.strip())
            except Exception as e:
                logger.error(f"Synthesis failed: {e}")
                final_english_answer = "Analysis complete. I have updated your chart with localized insights."

    # --- WORKFLOW: Localization (Post-Processing) ---
    final_answer = final_english_answer
    if language.lower() != "english" and language_code != 0:
        # Note: Translation happens on the whole block, which is fine for streaming fallback
        final_answer = await translate_text(final_english_answer, language, language_code=language_code)

    # Flag to signal whether streaming synthesis is available (captured by closure)
    _use_stream_gen = stream and 'findings' in agent_res and not agent_res.get('answer')
    logger.info(f"[Orchestrator] Stream Synthesis Selection: use_gen={_use_stream_gen}, stream={stream}, has_findings={'findings' in agent_res}")

    if stream:
        # UNIFIED WRAPPER: Single exit point for ALL streaming
        async def final_stream_wrapper():
            captured_content = ""
            # 0. Yield debug info FIRST if requested
            if debug:
                debug_data = {
                    "decomposer_output": decomposer_output,
                    "search_requests": [
                        {"domain": r.get("domain"), "query": r.get("query", "")[:100],
                         "flow": r.get("flow"), "category_filter": r.get("category_filter", []),
                         "house_filter": r.get("house_filter", [])}
                        for r in search_requests
                    ],
                    "intent": intent,
                    "findings": findings if 'findings' in locals() else agent_res.get('findings', '')
                }
                yield {"debug": debug_data}

            # 1. Yield the primary content
            if _use_stream_gen:
                async for token in stream_generator():
                    yield token
                captured_content = getattr(stream_generator, 'full_content', '')
            else:
                yield final_answer
                captured_content = final_answer

            # 2. Yield metadata & suggestions ONCE at the absolute end
            try:
                # Save the interaction (Now in DB with IDs)
                from backend.app import session_manager
                meta_ids = session_manager.save_interaction(
                    session_id=session_id,
                    category=intent.lower(),
                    query=query,
                    response=strip_think_tags(captured_content),
                    parent_id=parent_id,
                    interaction_id=provisional_interaction_id
                )
                
                from backend.agents.orchestrator import generate_last_call_suggestions, _clean_suggestion
                raw_sug = await generate_last_call_suggestions(captured_content, intent=intent)
                yield {
                    "suggestions": [_clean_suggestion(s) for s in raw_sug[:3]],
                    "interaction_metadata": meta_ids
                }
            except Exception as e:
                logger.error(f"[Orchestrator] Final metadata/suggestion yield failed: {e}")
                yield {"suggestions": ["Tell me about my career.", "What about my health?", "Any remedies?"]}
        
        return final_stream_wrapper()

    # --- WORKFLOW: Suggestion Last-Call ---
    # Suggestions are now fetched from a predefined pool based on intent and relevance.
    # Suggestions are ALWAYS in English as per system requirements.
    suggestions = await generate_last_call_suggestions(final_english_answer, intent=intent)

    # Clean suggestions numbering/formatting
    suggestions = [_clean_suggestion(s) for s in suggestions]

    # Persistence: [REMOVED]
    # Responsibility moved to app.py to prevent duplication and support single-row storage.

    return {
        "session_id": session_id,
        "answer": final_answer,
        "intent": intent,
        "suggestions": suggestions[:3],
        "interaction_id": provisional_interaction_id
    }
