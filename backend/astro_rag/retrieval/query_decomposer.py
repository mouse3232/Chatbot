import logging
import json
import os
from datetime import datetime
from backend.common.llm import create_llm
from backend.common.model_registry import ModelTask

logger = logging.getLogger(__name__)

def decompose_query(query: str, user_profile: dict = None) -> str:
    """
    Decomposes an astrology query using the specialized prompt and schema
    provided in the standalone decomposer script.
    """
    llm = create_llm(task=ModelTask.ULTRA_LIGHT)
    today_str = datetime.today().strftime("%d %B %Y")
    
    # Extract context from profile
    age = user_profile.get("age", "Unknown") if user_profile else "Unknown"
    gender = user_profile.get("gender", "Unknown") if user_profile else "Unknown"
    name = user_profile.get("name", "User") if user_profile else "User"

    # The exact prompt from the standalone script
    prompt = f"""
[TODAY: {today_str}]
[USER: {name}, Age {age}, {gender}]
Convert query into semantic intent JSON (max 3).
Intents: Kundali (cause), Remedies (upay), Timing (when).

Rules:
1. sem: Broadened thematic query for RAG. Do NOT just repeat the question. Expand it with astrological synonyms and related thematic keywords (e.g., if asked about "job", output "career, profession, job, 10th house, promotion, employment prospects, business"). Use Age and Gender to tailor context.
2. Timing: ONLY if "when/timeframe" asked. 
3. tp (Timing only): t=Horary(<=1mo)/Timing(>1mo). If specified, use exact period (e.g., "10 days", "2 weeks", "1 month"). If vague, use: "Today", "Some days", "Some weeks", "Some months", "Yearly | 2026" (1yr), "2026-2028" (2yr+). Resolve based on [TODAY].
4. cat: Personal, Wealth, Career, Family, Travel. (Strictly one).
5. hd (All intents): h (max 2), p (max 2, numeric: 1 Sun, 2 Moon, 3 Mars, 4 Mercury, 5 Jupiter, 6 Venus, 7 Saturn, 8 Rahu, 9 Ketu). Otherwise null.
6. rel (In hd): Primary relation house: 1=Self, 7=Partner, 5=Love, 11=Friends. Use turned houses if context requires (e.g., Partner's career = 10th from 7th = 4th).
7. JSON list only.
8. Minimal Decomposition: Prefer outputting exactly 1 intent object focused on the primary concern. Only generate multiple objects if the user asks completely distinct questions spanning different categories or intents.

JSON Structure:
[{{ "int": "Kundali/Remedies/Timing", "sem": "...", "cat": "...", "tp": {{"t": "...", "v": "..."}} or null, "hd": {{"h": [int], "p": [int], "rel": int }} }}] 

Query: "{query}"
""".strip()

    try:
        response = llm.invoke([
            {"role": "system", "content": "You are a specialized astrology query decomposer that outputs only valid JSON."},
            {"role": "user", "content": prompt}
        ])
        
        # Handle both string and list-based content (Gemini sometimes returns parts)
        content = response.content
        if isinstance(content, list):
            raw_text = "".join([str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content]).strip()
        else:
            raw_text = str(content).strip()
        
        # Clean potential markdown wrapping
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0].strip()
            
        # Validate that it's a list
        parsed = json.loads(raw_text)
        if not isinstance(parsed, list):
            parsed = [parsed]
            
        return json.dumps(parsed, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Decomposition failed: {e}")
        # Return fallback JSON matching the schema
        fallback = [{
            "int": "Kundali",
            "sem": query,
            "cat": "Personal",
            "tp": None,
            "hd": {"h": [], "p": [], "rel": 1}
        }]
        return json.dumps(fallback)
