import logging
import re
import json as _json
from typing import Optional, Iterator, AsyncIterator, List, Dict

import httpx

from backend.config import settings

logger = logging.getLogger(__name__)


# ── Raw httpx SSE Streaming (Bypasses LangChain) ─────────────────────────────
async def stream_llamacpp_raw(
    messages: List[Dict[str, str]],
    model_str: str,
    max_tokens: int = 600,
    temperature: float = 0.7,
    stop: List[str] = None,
) -> AsyncIterator[str]:
    """
    Stream tokens directly from a llama.cpp /v1/chat/completions endpoint
    using raw httpx. This bypasses LangChain's ChatOpenAI wrapper which
    buffers the entire response before yielding.
    
    Yields individual text tokens as they arrive from the server.
    """
    # Parse "provider@host:port/model" -> url
    parts = model_str.split('@', 1)[1].split('/', 1)
    host_port = parts[0]
    actual_model = parts[1] if len(parts) > 1 else "local-model"
    url = f"http://{host_port}/v1/chat/completions"
    
    payload = {
        "model": actual_model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
    }
    if stop:
        payload["stop"] = stop

    logger.info("RAW STREAM: POST %s (stream=true, max_tokens=%d)", url, max_tokens)

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]  # strip "data: " prefix
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = _json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        yield token
                except _json.JSONDecodeError:
                    logger.warning(f"Failed to decode chunk: {data_str}")
                    continue

def strip_think_tags(text: str) -> str:
    """Brutal cleanup for reasoning models that repeat the prompt or drafting steps."""
    if not text or not text.strip(): 
        return "[The model returned an empty response. Please check if llama.cpp is running and the model is loaded correctly.]"
    
    # 1. Broadly remove "Analyze the Request", "Drafting Strategy", etc.
    patterns = [
        r"(?:Thinking Process|Reasoning|Analysis|Plan|Step \d+|Analyze the Request|Analyze the Data|Drafting Strategy|Drafting|Step-by-Step):.*?\n\n",
        r"^(?:\d+\.\s+)?\*?\*?(?:Analyze the Request|Analyze the Data|Drafting Strategy|Drafting|Step-by-Step|Analyze the Context):\*?\*?\s*",
        r"^\*?\s*(?:Role|Language|Formatting|Constraints|Data|Core Principles):.*?\n",
        r"^\d+\.\s+\*\*.*?\*\*.*?\n" # Strips "1. **Analyze Context**"
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE | re.MULTILINE)

    # 2. ANCHOR: EVERYTHING before the first TRUE bold header is noise.
    # We ignore headers that look like reasoning (Analyze, Context, etc.)
    noise_words = ["analyze", "context", "reasoning", "thinking", "strategy", "drafting", "step"]
    all_headers = list(re.finditer(r"\*\*[^*]+\*\*", text))
    
    first_true_header_pos = -1
    for m in all_headers:
        header_text = m.group(0).lower()
        if not any(word in header_text for word in noise_words):
            first_true_header_pos = m.start()
            break
            
    if first_true_header_pos != -1:
        text = text[first_true_header_pos:]
    elif len(text) > 400:
        # Fallback for models that output only reasoning or fail to header properly
        if any(word in text[:200].lower() for word in noise_words):
            return "[Error: LLM outputted only reasoning. Please try again.]"

    # 3. Standard Tag Stripping
    cleaned = re.sub(r"<(?:think|reasoning)>.*?</(?:think|reasoning)>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"</?(?:think|reasoning|tool_call)>", "", cleaned, flags=re.IGNORECASE)
    
    return cleaned.strip() if cleaned.strip() else text.strip()



# ── Sarvam SDK Wrapper ────────────────────────────────────────────────────────
class _SarvamMessage:
    """Thin wrapper to mimic LangChain message return."""
    def __init__(self, content: str):
        self.content = content


class SarvamLLM:
    """
    A lightweight LangChain-compatible wrapper around the Sarvam AI SDK.
    Supports both .invoke() (returns message-like object) and .stream() (yields chunks).
    """
    def __init__(self, api_key: str, model: str = "sarvam-30b", temperature: float = 0.3, max_tokens: int = 4096):
        from sarvamai import SarvamAI
        self._client = SarvamAI(api_subscription_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _to_sdk_messages(self, messages) -> list:
        """Convert LangChain message objects or dicts to Sarvam SDK dict format."""
        sdk_msgs = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get('role', 'user')
                content = msg.get('content', '')
            else:
                role = getattr(msg, 'type', 'user')
                # LangChain roles: human -> user, ai -> assistant, system -> system
                role_map = {'human': 'user', 'ai': 'assistant', 'system': 'system'}
                role = role_map.get(role, 'user')
                content = getattr(msg, 'content', '')
            
            sdk_msgs.append({'role': role, 'content': content})
        return sdk_msgs

    def invoke(self, messages):
        """Synchronous invocation, returns a message-like object with .content."""
        # Support passing plain string (used in some nodes)
        if isinstance(messages, str):
            sdk_msgs = [{'role': 'user', 'content': messages}]
        else:
            sdk_msgs = self._to_sdk_messages(messages)

        full_text = ""
        try:
            for chunk in self._client.chat.completions(
                model=self.model,
                messages=sdk_msgs,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            ):
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        full_text += delta.content
        except Exception as e:
            logger.error("Sarvam API error: %s", e)
            raise
        return _SarvamMessage(content=full_text)

    def stream(self, messages) -> Iterator[str]:
        """Streaming invocation, yields text chunks."""
        if isinstance(messages, str):
            sdk_msgs = [{'role': 'user', 'content': messages}]
        else:
            sdk_msgs = self._to_sdk_messages(messages)

        for chunk in self._client.chat.completions(
            model=self.model,
            messages=sdk_msgs,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        ):
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content


# ── LLM Factory ───────────────────────────────────────────────────────────────
def create_llm(task: Optional[str] = None, language_code: Optional[int] = 0, model_override: Optional[str] = None, cached_content: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, thinking_level: Optional[str] = None, streaming: bool = False):
    """
    Create an LLM instance. 
    1. If 'task' is provided, resolve via ModelRegistry (environment & language aware).
    2. If 'model_override' is provided, use it directly (legacy/manual support).
    3. Fall back to settings.LLM_PROVIDER.
    Optional: temperature and max_tokens override defaults per-provider.
    """
    from backend.common.model_registry import get_model_for_task
    
    # Resolve the physical model string
    model_str = ""
    if task:
        model_str = get_model_for_task(task, language_code=language_code)
    elif model_override:
        # Check if the override is actually a task key
        if model_override in ["FINAL_ANSWER", "SUMMARIZATION", "REASONING", "MULTILINGUAL", "ULTRA_LIGHT"]:
            model_str = get_model_for_task(model_override, language_code=language_code)
        else:
            model_str = model_override
    else:
        # Strict fallback: if no task/override, use provided defaults or settings
        model_str = settings.LLM_PROVIDER
        
    provider = ""
    llamacpp_url = None  # Will hold the parsed URL for llamacpp provider

    # 1. Explicit prefix-based detection
    if model_str.startswith("llamacpp@") or model_str.startswith("vllm@"):
        provider = "vllm" if model_str.startswith("vllm@") else "llamacpp"
        # Parse host:port from "provider@host:port"
        llamacpp_url = f"http://{model_str.split('@', 1)[1].split('/', 1)[0]}/v1"
    elif model_str.startswith("google/") or model_str.startswith("gemini/"):
        provider = "google"
    elif model_str.startswith("groq/"):
        provider = "groq"
    elif model_str == "qwen/qwen3-32b":
        provider = "groq"
    elif model_str.startswith("sarvam/"):
        provider = "sarvam"
    elif model_str.startswith("openai/"):
        if settings.OPENAI_API_KEY and "dummy" not in settings.OPENAI_API_KEY.lower():
            provider = "openai"
        else:
            provider = "groq"
    elif model_str.startswith("ollama/"):
        provider = "ollama"
    else:
        # 2. Heuristic: detect well-known model families by name
        if any(m in model_str for m in ["llama-3", "llama3", "mixtral", "gemma", "qwen", "qwq"]):
            provider = "groq"
        elif "gpt-" in model_str:
            provider = "openai"
        else:
            # 3. Fall back to settings
            provider = settings.LLM_PROVIDER.lower() if settings.LLM_PROVIDER else "ollama"

    # 4. Auto-detect from available API keys if still unknown (Backup)
    if not provider or provider == "sarvam" and not settings.SARVAM_API_KEY:
        if settings.SARVAM_API_KEY: provider = "sarvam"
        elif settings.GROQ_API_KEY: provider = "groq"
        else: provider = "ollama"

    if provider in ["llamacpp", "vllm"]:
        from langchain_openai import ChatOpenAI
        # Parse host:port and optional model from "provider@host:port/model"
        parts = model_str.split('@', 1)[1].split('/', 1)
        host_port = parts[0]
        actual_model = parts[1] if len(parts) > 1 else "local-model"
        
        llm_url = f"http://{host_port}/v1"
        logger.info("LLM: %s at %s (Model: %s)", provider.upper(), llm_url, actual_model)
        
        return ChatOpenAI(
            model=actual_model,
            openai_api_key="not-needed",
            openai_api_base=llm_url,
            temperature=temperature if temperature is not None else 0.7,
            max_tokens=max_tokens if max_tokens is not None else 512,
            timeout=60,
            streaming=streaming,
            stop=["<|im_end|>", "<|endoftext|>", "USER PROFILE:", "TASK:"],
            presence_penalty=0.1,
            frequency_penalty=0.2,
            model_kwargs={"stream": streaming}
        )
    elif provider == "sarvam":
        model = model_str.replace("sarvam/", "")
        logger.info("LLM: Sarvam AI (%s)", model)
        return SarvamLLM(
            api_key=settings.SARVAM_API_KEY,
            model=model,
            temperature=0.7,
            max_tokens=4096,
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq
        model = model_str.replace("groq/", "")
        logger.info("LLM: Groq (%s)", model)
        return ChatGroq(
            model=model,
            api_key=settings.GROQ_API_KEY,
            temperature=temperature if temperature is not None else 0.3,
            max_tokens=max_tokens if max_tokens is not None else 4096,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        model = model_str.replace("openai/", "")
        logger.info("LLM: OpenAI (%s)", model)
        return ChatOpenAI(
            model=model,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.3,
            max_tokens=4096,
        )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = model_str.replace("google/", "").replace("gemini/", "")
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=temperature if temperature is not None else 0.3,
            cached_content=cached_content,
            thinking_level=thinking_level,
        )
    else:
        from langchain_ollama import ChatOllama
        model = model_str.replace("ollama/", "")
        logger.info("LLM: Ollama (%s) at %s", model, settings.OLLAMA_BASE_URL)
        return ChatOllama(
            model=model,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.3,
        )
