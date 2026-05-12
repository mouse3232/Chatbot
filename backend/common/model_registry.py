import logging
from typing import Optional
from backend.config import settings, MODEL_MAP_LOCAL, MODEL_MAP_REMOTE

logger = logging.getLogger(__name__)

class ModelTask:
    FINAL_ANSWER = "FINAL_ANSWER"
    SUMMARIZATION = "SUMMARIZATION"
    REASONING = "REASONING"
    MULTILINGUAL = "MULTILINGUAL"
    ULTRA_LIGHT = "ULTRA_LIGHT"
    BACKUP = "BACKUP"

def get_model_for_task(task_key: str, language_code: Optional[int] = 0) -> str:
    """
    Resolves the appropriate model string based on task and environment mode.
    Handles the special case for LOCAL mode with non-English queries.
    """
    is_local = settings.LOCAL
    
    # Force localized multilingual model if language is not English in LOCAL mode
    if is_local and language_code and int(language_code) != 0:
        model = MODEL_MAP_LOCAL.get("MULTILINGUAL", "ollama/gemma4:26b")
        logger.debug(f"[ModelRegistry] LOCAL/NON-ENGLISH detected (LC:{language_code}). Routing all to {model}")
        return model

    # Select base mapping
    mapping = MODEL_MAP_LOCAL if is_local else MODEL_MAP_REMOTE
    
    resolved_model = mapping.get(task_key)
    
    if not resolved_model:
        logger.warning(f"[ModelRegistry] No model mapped for task '{task_key}'. Falling back to REMOVED/BACKUP.")
        resolved_model = mapping.get("BACKUP", "sarvam/sarvam-30b")
        
    return resolved_model
