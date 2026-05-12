"""
config.py – Centralised settings for ChatBot_Real_V2.

Loads from .env file via pydantic-settings.
"""

from __future__ import annotations

import os
import logging
from typing import Optional

# Silence noisy ML and HTTP logs
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Explicitly silence third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# ── Global GPU Detection ──────────────────────────────────────────────────────
try:
    import torch
    import faiss
    TORCH_GPU = torch.cuda.is_available()
    # Check if FAISS has GPU support and at least one GPU is visible
    FAISS_GPU = hasattr(faiss, 'StandardGpuResources') and getattr(faiss, 'get_num_gpus', lambda: 0)() > 0
    GPU_AVAILABLE = TORCH_GPU and FAISS_GPU
except Exception:
    GPU_AVAILABLE = False

if GPU_AVAILABLE:
    print("\n" + "*"*60)
    print("🚀 [STARTUP] GPU Found. Using GPU for all processes.")
    print("*"*60 + "\n")
else:
    print("\n" + "-"*60)
    print("💻 [STARTUP] GPU Not Found. Defaulting to CPU for all processes.")
    print("-"*60 + "\n")

# ── Project Root Discovery ────────────────────────────────────────────────────
_this_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_this_dir)

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Performance & Cache
    HF_OFFLINE: bool = True
    HORARY_AI: bool = False
    HF_TOKEN: Optional[str] = None

    # T5Gemma 2
    T5GEMMA_ENABLED: bool = True
    T5GEMMA_INTENT_MODEL: str = "google/t5gemma-2-270m-270m"
    # T5GEMMA_TRANS_MODEL removed as per optimization requirements

    # LLM
    LLM_PROVIDER: str = "sarvam"
    SARVAM_API_KEY: Optional[str] = None
    SARVAM_MODEL: str = "sarvam-30b"
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-oss-20b"
    OLLAMA_MODEL: str = "qwen3:4b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    GOOGLE_API_KEY: Optional[str] = None
    
    # Dual Mode Control
    LOCAL: bool = False

    # Task Assignments - Remote
    REMOTE_MODEL_FINAL_ANSWER: str = "vllm@192.168.1.205:8000/Qwen/Qwen2.5-7B-Instruct-AWQ"
    REMOTE_MODEL_SUMMARIZATION: str = "sarvam/sarvam-30b"
    REMOTE_MODEL_REASONING: str = "sarvam/sarvam-30b"
    REMOTE_MODEL_MULTILINGUAL: str = "sarvam/sarvam-30b"
    REMOTE_MODEL_ULTRA_LIGHT: str = "groq/llama-3.1-8b-instant"
    REMOTE_MODEL_BACKUP: str = "sarvam/sarvam-30b"

    # Task Assignments - Local (llama.cpp servers)
    LOCAL_MODEL_FINAL_ANSWER: str = "vllm@192.168.1.205:8000/Qwen/Qwen2.5-7B-Instruct-AWQ"
    LOCAL_MODEL_SUMMARIZATION: str = "llamacpp@192.168.1.35:7002"
    LOCAL_MODEL_REASONING: str = "llamacpp@192.168.1.35:7000"
    LOCAL_MODEL_MULTILINGUAL: str = "llamacpp@192.168.1.211:8001/unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL"
    LOCAL_MODEL_ULTRA_LIGHT: str = "llamacpp@192.168.1.35:7002"
    LOCAL_MODEL_BACKUP: str = "llamacpp@192.168.1.35:7002"

    # Paths (Auto-resolved to Project Root)
    SOURCE_DB_PATH: str = os.path.join(PROJECT_ROOT, "FptEnglish.db")
    GLOBAL_LOG_PATH: str = os.path.join(PROJECT_ROOT, "logs", "global_interactions.jsonl")
    SESSIONS_DB_PATH: str = os.path.join(PROJECT_ROOT, "sessions.db")
    SESSION_LOGS_ROOT: str = os.path.join(PROJECT_ROOT, "logs", "session_log")
    HORARY_DB_PATH: str = os.path.join(PROJECT_ROOT, "backend", "horary", "EmbedDB", "vectors")

    # Embedding
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIM: int = 384

    # Look for .env in current dir or parent dir (important for remote deployment)
    model_config = SettingsConfigDict(
        env_file=[".env", "../.env"],
        env_file_encoding="utf-8",
        extra="ignore"
    )
 

settings = Settings()

# Sync Hugging Face offline mode with environment variables
if settings.HF_OFFLINE:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
else:
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

# ── Multi-model configuration ────────────────────────────────────────────────

# Available LLM models
LLM_MODELS = {
    "sarvam/sarvam-30b": "Sarvam 30B",
    "qwen/qwen3-32b": "Qwen3 32B (Reasoning)",
    "unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL": "Qwen 3.5 9B (Local)",
    "openai/gpt-oss-20b": "GPT-OSS 20B",
    "google/gemini-1.5-flash": "Gemini 1.5 Flash",
    "google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash-Lite (Preview)",
    "google/gemini-3-flash-preview": "Gemini 3 Flash (Preview)",
    "vllm@192.168.1.205:8000/Qwen/Qwen2.5-7B-Instruct-AWQ": "Qwen 2.5 7B (vLLM Remote)"
}

# Embedding model configuration (session-level on-demand embedding)
EMBEDDING_CONFIGS = {
    "BAAI/bge-small-en-v1.5": {
        "dim": 384,
        "label": "BGE Small EN v1.5 (Context-Injected)",
    },
}
# ── Task-Based Model Mappings ────────────────────────────────────────────────

MODEL_MAP_REMOTE = {
    "FINAL_ANSWER": settings.REMOTE_MODEL_FINAL_ANSWER,
    "SUMMARIZATION": settings.REMOTE_MODEL_SUMMARIZATION,
    "REASONING": settings.REMOTE_MODEL_REASONING,
    "MULTILINGUAL": settings.REMOTE_MODEL_MULTILINGUAL,
    "ULTRA_LIGHT": settings.REMOTE_MODEL_ULTRA_LIGHT,
    "BACKUP": settings.REMOTE_MODEL_BACKUP
}

MODEL_MAP_LOCAL = {
    "FINAL_ANSWER": settings.LOCAL_MODEL_FINAL_ANSWER,
    "SUMMARIZATION": settings.LOCAL_MODEL_SUMMARIZATION,
    "REASONING": settings.LOCAL_MODEL_REASONING,
    "MULTILINGUAL": settings.LOCAL_MODEL_MULTILINGUAL,
    "ULTRA_LIGHT": settings.LOCAL_MODEL_ULTRA_LIGHT,
    "BACKUP": settings.LOCAL_MODEL_BACKUP
}
