import logging
import sys
import os
import httpx
import json
from typing import List, Dict, Any, Union
import chainlit as cl
from chainlit.input_widget import Slider, Select, Switch

logger = logging.getLogger(__name__)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8001/api")

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_PROFILE = {
    "name": "User",
    "age": 25,
    "gender": "Male",
}

# Available Groq Models for the Settings panel
GROQ_MODELS = {
    "qwen/qwen3-32b": "Qwen3 32B (Reasoning)",
    "openai/gpt-oss-20b": "GPT-OSS 20B",
    "google/gemini-1.5-flash": "Gemini 1.5 Flash",
    "google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash-Lite (Preview)",
    "google/gemini-3-flash-preview": "Gemini 3 Flash (Preview)",
}

EMBEDDING_MODELS = {
    "qwen3-embedding:0.6b": "Qwen3 Embedding 0.6B",
    "embeddinggemma:latest": "Embedding Gemma",
    "bge-m3": "BGE-M3",
    "nomic-embed-text-v2-moe": "Nomic Embed Text V2 MoE",
    "qwen3-embedding:4b": "Qwen3 Embedding 4B",
}

async def _send_suggestions(suggestions: Union[list, dict]):
    """Send suggested follow-up questions as clickable action buttons."""
    if not suggestions:
        return
        
    if isinstance(suggestions, dict):
        # Categorized suggestions (start of session)
        for category, q_list in suggestions.items():
            actions = []
            for i, q in enumerate(q_list[:3]):
                # Unique names for suggestions across categories
                action_name = f"sug_{category.lower()}_{i}"
                actions.append(
                    cl.Action(name=action_name, payload={"question": q}, label=f"💡 {q}")
                )
                # Register callback for this specific action name
                @cl.action_callback(action_name)
                async def on_categorized_suggestion_click(action: cl.Action):
                    question = action.payload.get("question", "")
                    if question:
                        msg = cl.Message(content=question, author="User")
                        await on_message(msg)
            
            await cl.Message(content=f"✨ **{category} Guidance**:", actions=actions).send()
    else:
        # Flat list (contextual follow-up)
        actions = []
        for i, q in enumerate(suggestions[:3]):
            actions.append(
                cl.Action(name=f"suggestion_{i}", payload={"question": q}, label=f"💡 {q}")
            )
        await cl.Message(content="**You might also want to ask:**", actions=actions).send()

@cl.action_callback("suggestion_0")
@cl.action_callback("suggestion_1")
@cl.action_callback("suggestion_2")
async def on_suggestion_click(action: cl.Action):
    question = action.payload.get("question", "")
    if question:
        msg = cl.Message(content=question, author="User")
        await on_message(msg)

# ── Dynamic Path for Parser ───────────────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from backend.common.parser import parse_default_text

def _load_default_payload() -> Dict[str, Any]:
    """Loads the pre-compressed example_payload.json if it exists."""
    path = os.path.join(os.path.dirname(__file__), "..", "example_payload.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading example_payload.json: {e}")
    return {}

def _load_payload_file(filename: str) -> Dict[str, Any]:
    """Load a payload JSON file from the project root (Kundali.txt, Timing.txt, etc.)."""
    path = os.path.join(os.path.dirname(__file__), "..", filename)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
    return {}

async def _fetch_session_list() -> Dict[str, str]:
    """Fetch active sessions from backend and return as {id: label} mapping."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{API_BASE_URL}/sessions", timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            sessions = data.get("sessions", [])
            # Map session_id -> "User Name (short_id)"
            return {s["session_id"]: f"{s['user_name']} ({s['session_id'][:8]})" for s in sessions}
    except Exception as e:
        logger.error(f"Failed to fetch sessions: {e}")
        return {}

async def _send_settings_panel():
    """Build and send/refresh the settings panel with current session list."""
    session_map = await _fetch_session_list()
    current_sid = cl.user_session.get("session_id")
    
    # Sort IDs for consistency, ensuring current is first or handled
    session_ids = list(session_map.keys())
    if not session_ids:
        session_ids = ["none"]
        session_map = {"none": "No active sessions"}
    
    await cl.ChatSettings(
        [
            Slider(
                id="k_value",
                label="Search Depth (K-Value)",
                initial=5,
                min=1,
                max=100,
                step=1,
                description="Number of records to retrieve for each query."
            ),
            Select(
                id="llm_model",
                label="Groq LLM Model",
                values=list(GROQ_MODELS.keys()),
                initial="qwen/qwen3-32b",
                description="The model used to generate the final astrological reading."
            ),
            Select(
                id="embedding_model",
                label="Embedding Model (Ollama)",
                values=list(EMBEDDING_MODELS.keys()),
                initial="qwen3-embedding:0.6b",
                description="The model used for searching records. Affects accuracy and speed."
            ),
            Select(
                id="switch_session",
                label="📂 Switch Active Session",
                initial=current_sid if current_sid in session_ids else session_ids[0],
                items=session_map,
                description="Select an existing session to resume chatting."
            ),
            Switch(
                id="show_debug",
                label="📊 Show Source Data (RAG Chunks)",
                initial=False,
                description="When ON, shows the specific astrological records and RAG chunks used to generate the answer."
            ),
            Switch(
                id="refresh_sessions",
                label="🔄 Refresh Session List",
                initial=False,
                description="Toggle to update the session list from the backend."
            ),
        ]
    ).send()

def _compress_records(parsed: list) -> List[Dict[str, Any]]:
    """Compress parsed (table, conditions) pairs into backend-friendly JSON."""
    table_map = {}
    for table, conditions in parsed:
        if table not in table_map:
            table_map[table] = {"keys": list(conditions.keys()), "values": []}
        val_list = [conditions.get(k) for k in table_map[table]["keys"]]
        table_map[table]["values"].append(val_list)

    compressed = []
    for table, info in table_map.items():
        compressed.append({
            "table": table,
            "keys": info["keys"],
            "values": info["values"]
        })
    return compressed

def _load_records_data(filename: str = "Default.json") -> List[Dict[str, Any]]:
    """Read a specific JSON record file and convert it to compressed format."""
    file_path = os.path.join(os.path.dirname(__file__), "..", filename)
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    parsed = parse_default_text(content)
    return _compress_records(parsed)

def _load_matching_data() -> Dict[str, Any]:
    """Load all matching data from example_payload.json as requested."""
    data = _load_default_payload()
    if not data:
        return {}

    return {
        "user_profile": data.get("user_profile", DEFAULT_PROFILE),
        "records": data.get("records", []),
        "records2": data.get("records2", []),
        "matchrecords": data.get("matchrecords", []),
        "predictions": data.get("predictions", [])
    }


async def _run_setup(user_profile: dict, records: List[Dict[str, Any]] = None):
    await cl.Message(
        f"Thank you, **{user_profile['name']}**! Setting up your astrological profile... 🌟\n\n"
        f"*Age: {user_profile['age']} | Gender: {user_profile['gender']}*"
    ).send()

    payload = {
        "user_profile": user_profile,
        "records": records if records is not None else _load_records_data()
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{API_BASE_URL}/sessions/init", json=payload, timeout=120.0)
            resp.raise_for_status()
            data = resp.json()

        session_id = data["session_id"]
        num_docs = data.get("docs_loaded", 0)
        mode = data.get("mode", "standard")
        
        cl.user_session.set("session_id", session_id)
        cl.user_session.set("messages", [])

        if mode == "context_memory":
            await cl.Message("⚡ **Context Memory Mode Active** (Server configuration) - Dataset loaded directly into memory.").send()

        await cl.Message(
            f"✅ **All set, {user_profile['name']}!** ({num_docs} records loaded)\n\n"
            f"🔑 **Session ID:** `{session_id}` *(valid 24 hours)*\n\n"
            f"Ask me anything about your life, career, health, relationships, or destiny!"
        ).send()

        await _send_suggestions(data.get("initial_suggestions", []))

    except Exception as e:
        logger.error("Error during setup graph: %s", str(e), exc_info=True)
        await cl.Message(f"❌ An error occurred connecting to the backend setup API: {e}").send()


@cl.on_settings_update
async def setup_agent_settings(settings):
    """Sync UI settings with backend session state or switch sessions."""
    # 0. Persist debug toggle locally
    cl.user_session.set("show_debug", settings.get("show_debug", False))

    # 1. Handle Refresh Trigger
    if settings.get("refresh_sessions"):
        await _send_settings_panel()
        return

    # 2. Handle Session Switch
    new_sid = settings.get("switch_session")
    current_sid = cl.user_session.get("session_id")
    if new_sid and new_sid != current_sid and new_sid != "none":
        cl.user_session.set("session_id", new_sid)
        cl.user_session.set("messages", [])
        await cl.Message(content=f"🔄 **Switched to Session:** `{new_sid}`").send()
        return

    # 3. Handle Parameter Updates (K-Value, Model)
    if not current_sid:
        return

    payload = {
        "k_value": int(settings.get("k_value", 5)),
        "llm_model": settings.get("llm_model"),
        "embedding_model": settings.get("embedding_model")
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{API_BASE_URL}/sessions/{current_sid}/settings", json=payload)
            resp.raise_for_status()
            data = resp.json()
        
        if data.get("rebuilt"):
            cl.user_session.set("messages", [])
            await cl.Message(content=f"🔄 **Embedding model changed to `{payload['embedding_model']}`. Session restarted and database rebuilt!**").send()
        else:
            await cl.Message(content=f"⚙️ **Settings Updated**: K={payload['k_value']}, LLM={payload['llm_model']}, Emb={payload['embedding_model']}").send()
            
    except Exception as e:
        logger.error(f"Failed to sync settings: {e}")
        await cl.Message(content=f"⚠️ Failed to update settings on backend: {e}").send()


@cl.on_chat_start
async def on_chat_start():
    # Initialize Settings Panel (enables the settings icon in UI)
    await _send_settings_panel()

    res = await cl.AskActionMessage(
        content=(
            "🙏 **Welcome to the Astrology Timing Assistant!**\n\n"
            "I specialize in analyzing your birth chart to provide precise yearly, monthly, and daily predictions.\n\n"
            "> Click **Start Timing Session** to begin, or **Resume** to continue a previous conversation."
        ),
        actions=[
            cl.Action(name="new_timing", payload={"action": "timing"}, label="⏱️ Start Timing Session"),
            cl.Action(name="resume_session", payload={"action": "resume"}, label="🔄 Resume Session"),
        ],
    ).send()

    if not res:
        await cl.Message("No action selected. Please refresh the page.").send()
        return

    action_value = res.get("payload").get("action")

    # Reset session-specific flags
    cl.user_session.set("is_matching_session", False)
    
    if action_value == "resume":
        sid_res = await cl.AskUserMessage(content="Please enter your **Session ID**:", timeout=300).send()
        if sid_res:
            session_id = sid_res["output"].strip()
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"{API_BASE_URL}/sessions/{session_id}")
                    if resp.status_code == 404:
                        await cl.Message(f"❌ **Invalid Session ID:** `{session_id}`. Please check and try again.").send()
                        await on_chat_start() # Return to start
                        return
                    resp.raise_for_status()
                    data = resp.json()
                
                cl.user_session.set("session_id", session_id)
                cl.user_session.set("messages", [])
                
                # Restore History
                history = data.get("messages", [])
                if history:
                    await cl.Message(content="📜 **Restoring your previous conversation...**").send()
                    for msg_data in history:
                        role = msg_data.get("role", "user")
                        content = msg_data.get("content", "")
                        # Map backend roles to UI authors
                        author = "Assistant" if role == "assistant" else "User"
                        await cl.Message(content=content, author=author).send()
                
                await cl.Message(f"✅ **Session restored successfully!**").send()
                await _send_settings_panel()
                
            except Exception as e:
                logger.error(f"Error during resume: {e}")
                await cl.Message(f"❌ Error connecting to backend: {e}").send()
        else:
            await cl.Message("No Session ID provided. Please refresh.").send()

    elif action_value == "matching":
        await cl.Message("💑 **Starting AI Matching Session...** Loading datasets for both profiles...").send()
        matching_data = _load_matching_data()
        if matching_data and matching_data["user_profile"]:
            await _run_matching_setup(
                user_profile=matching_data["user_profile"],
                records=matching_data["records"],
                records2=matching_data["records2"],
                matchrecords=matching_data["matchrecords"],
                predictions=matching_data["predictions"]
            )
        else:
            await cl.Message("❌ Failed to load dynamic matching data.").send()

    elif action_value == "remedies":
        # Load Remedies.txt and init
        remedies_data = _load_payload_file("Remedies.txt")
        if remedies_data and remedies_data.get("user_profile"):
            await _run_setup(remedies_data["user_profile"], remedies_data.get("records"))
        else:
            await cl.Message("❌ Failed to load Remedies.txt").send()

    elif action_value == "timing":
        # Load Timing.txt and init with timing_records
        timing_data = _load_payload_file("Timing.txt")
        if timing_data and timing_data.get("user_profile"):
            payload = {
                "user_profile": timing_data["user_profile"],
                "timing_records": timing_data.get("timing_records", []),
            }
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(f"{API_BASE_URL}/sessions/init", json=payload, timeout=120.0)
                    resp.raise_for_status()
                    data = resp.json()
                session_id = data["session_id"]
                cl.user_session.set("session_id", session_id)
                cl.user_session.set("messages", [])
                await cl.Message(
                    f"⏱️ **Timing Session Ready!** ({len(timing_data.get('timing_records', []))} timing tables loaded)\n\n"
                    f"🔑 **Session ID:** `{session_id}`\n\n"
                    f"Ask me about your yearly, monthly, weekly, or daily predictions!"
                ).send()
                await _send_suggestions(data.get("initial_suggestions", []))
            except Exception as e:
                await cl.Message(f"❌ Timing session init failed: {e}").send()
        else:
            await cl.Message("❌ Failed to load Timing.txt").send()

    elif action_value == "new":
        details_res = await cl.AskActionMessage(
            content=(
                "**Step 1:** 📝 **Enter your basic details** in one message:\n\n"
                "**Format:** `Name, Age, Gender`\n"
                "*Example: Arun, 68, Male*\n\n"
                "Or click **Skip** to use default values."
            ),
            actions=[
                cl.Action(name="skip_birth", payload={"action": "skip"}, label="⏩ Skip (Use Default)"),
            ],
        ).send()

        if details_res and details_res.get("payload", {}).get("action") == "skip":
            default_data = _load_default_payload()
            if default_data and "user_profile" in default_data and "records" in default_data:
                await _run_setup(default_data["user_profile"], default_data["records"])
            else:
                await _run_setup(DEFAULT_PROFILE)
        else:
            birth_input = await cl.AskUserMessage(
                content="Please type your profile details now:\n`Name, Age, Gender`",
                timeout=600,
            ).send()

            if not birth_input:
                return

            raw = birth_input["output"].strip()
            parts = [p.strip() for p in raw.split(",")]

            if len(parts) < 3:
                await cl.Message("❌ I need 3 values: **Name, Age, Gender** separated by commas.").send()
                return

            try:
                age = int(parts[1])
            except ValueError:
                age = 25

            user_profile = {
                "name": parts[0],
                "age": age,
                "gender": parts[2],
            }
            await _run_setup(user_profile)


async def _run_matching_setup(user_profile: dict, records=None, records2=None, matchrecords=None, predictions=None):
    """Initialize a triple-store matching session with data from example_payload.json."""
    payload = {
        "user_profile": user_profile,
        "records": records,
        "records2": records2,
        "matchrecords": matchrecords,
        "predictions": predictions
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{API_BASE_URL}/matching/init", json=payload, timeout=120.0)
            resp.raise_for_status()
            data = resp.json()

        session_id = data["session_id"]
        stats = data.get("loaded_stats", {})
        
        # KEY FIX: Set the session ID and MATCHING FLAG in context
        cl.user_session.set("session_id", session_id)
        cl.user_session.set("is_matching_session", True)
        cl.user_session.set("messages", [])

        msg = f"✅ **Matching Session Activated!**\n\n"
        msg += f"🔑 **Session ID:** `{session_id}`\n"
        msg += f"📊 **Triple-Dataset Stats:**\n"
        msg += f"* 👨 Profile 1 (Male): {stats.get('records', 0)} tables\n"
        msg += f"* 👩 Profile 2 (Female): {stats.get('records2', 0)} tables\n"
        msg += f"* 💑 Match Records: {stats.get('matchrecords', 0)} tables\n\n"
        msg += "--- \n"
        msg += "✨ **Automatic Matching Mode Active!**\n"
        msg += "You can now ask any question directly about your compatibility, and I will analyze all three datasets automatically."
        await cl.Message(content=msg).send()

    except Exception as e:
        logger.error(f"Matching Setup Error: {e}")
        await cl.Message(f"❌ Failed to init matching: {e}").send()

@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    if not session_id:
        await cl.Message("No active session. Please refresh.").send()
        return

    is_matching = cl.user_session.get("is_matching_session", False)
    msg = cl.Message(content="")
    await msg.send()

    # --- 1. MATCHING INTERCEPT (Priority) ---
    if is_matching or message.content.strip().lower().startswith("/matching "):
        query = message.content
        if query.lower().startswith("/matching "):
            query = query[len("/matching "):].strip()
            
        payload = {"session_id": session_id, "query": query}
        try:
            full_answer = ""
            suggestions = []
            
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", f"{API_BASE_URL}/matching/query", json=payload, timeout=120.0) as resp:
                    resp.raise_for_status()
                    
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        
                        try:
                            data = json.loads(line.replace("data: ", ""))
                            if "answer" in data:
                                full_answer += data["answer"]
                                msg.content = f"💑 **AI Matching Insight**\n\n{full_answer}"
                                await msg.update()
                            elif "suggestions" in data:
                                suggestions = data["suggestions"]
                        except:
                            continue
            
            # Final actions if any
            if suggestions:
                await _send_suggestions(suggestions)
                
            return # CRITICAL: STOP HERE FOR MATCHING
        except Exception as e:
            msg.content = f"❌ Matching Query failed: {e}"
            await msg.update()
            return

    # --- 2. HORARY CLASSIFIER INTERCEPT ---

    # --- HORARY CLASSIFIER INTERCEPT ---
    if message.content.strip().lower().startswith("/horary "):
        query = message.content[len("/horary "):].strip()
        payload = {"query": query, "session_id": session_id}
        
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(f"{API_BASE_URL}/horary/classify", json=payload, timeout=60.0)
                resp.raise_for_status()
                data = resp.json()
                
            houses = data.get("houses", [])
            planets_nums = data.get("planets", [])
            rag_trace = data.get("rag_trace", [])
            
            # Map numbers to names for UI display
            PLANET_MAP = {1:"Sun", 2:"Moon", 3:"Mars", 4:"Mercury", 5:"Jupiter", 6:"Venus", 7:"Saturn", 8:"Rahu", 9:"Ketu"}
            planets_formatted = [f"{PLANET_MAP.get(p, 'Unknown')} ({p})" for p in planets_nums]
            
            content = f"🛠️ **Developer Discovery View: Horary RAG**\n\n"
            content += f"**Final Selection:**\n"
            content += f"* 🏠 Houses: `{', '.join(map(str, houses))}`\n"
            content += f"* 🪐 Planets: `{', '.join(planets_formatted)}`\n\n"
            content += "--- \n"
            
            if rag_trace:
                content += "### 🔍 Scoring Calculation Details\n"
                for i, trace in enumerate(rag_trace):
                    sub_q = trace.get('sub_query', '')
                    terms = trace.get('search_terms', [])
                    
                    # Distinguish between sub-query and extracted keywords
                    keywords = [t for t in terms if t != sub_q]
                    
                    content += f"**[Intent {i+1}]** Sub-query: *\"{sub_q}\"*  \n"
                    reasoning = trace.get('reasoning')
                    if reasoning:
                        content += f"**🧠 Reasoning:** *{reasoning}*  \n"
                    if keywords:
                        content += f"**🔑 Extracted Keywords:** `{', '.join(keywords)}`  \n"
                    else:
                        content += f"**⚠️ No Keywords extracted!**  \n"
                    content += f"**🔎 Full Search List:** `{', '.join(terms)}`  \n"
                    
                    retrieved = trace.get("retrieved", [])
                    if retrieved:
                        h_matches = [r for r in retrieved if r['type'] == 'house']
                        p_matches = [r for r in retrieved if r['type'] == 'planet']
                        
                        if h_matches:
                            content += "  *🏠 House Matches:*  \n"
                            for r in h_matches:
                                similarity = r.get('similarity', 0)
                                weight = r.get('weight', 0)
                                calc = r.get('calc_score', 0)
                                content += f"  > H{r['id']}: \"{r['sentence']}\" *( (1-{r.get('distance', '?')}) * {weight} = **+{calc}** )*  \n"
                        
                        if p_matches:
                            content += "  *🪐 Planet Matches:*  \n"
                            for r in p_matches:
                                similarity = r.get('similarity', 0)
                                weight = r.get('weight', 0)
                                calc = r.get('calc_score', 0)
                                content += f"  > P{r['id']}: \"{r['sentence']}\" *( (1-{r.get('distance', '?')}) * {weight} = **+{calc}** )*  \n"
                    content += "\n"

            # Show ALL house scores that are > 0
            house_scores = data.get("house_scores", {})
            if house_scores:
                content += "### 📊 Final House Scores (All matched)\n"
                sorted_h = sorted(house_scores.items(), key=lambda x: x[1], reverse=True)
                for h, s in sorted_h:
                    if float(s) > 0:
                        content += f"* **House {h}**: {s} {' (PICKED)' if int(h) in houses else ''}  \n"
                content += "\n"

            # Show ALL planet scores that are > 0
            planet_scores = data.get("planet_scores", {})
            if planet_scores:
                content += "### 📊 Final Planet Scores (All matched)\n"
                sorted_p = sorted(planet_scores.items(), key=lambda x: x[1], reverse=True)
                for p, s in sorted_p:
                    if float(s) > 0:
                        content += f"* **Planet {p}**: {s} {' (PICKED)' if int(p) in planets_nums else ''}  \n"

            msg.content = content
            await msg.update()
            return

        except Exception as e:
            logger.error(f"Horary api failed: {e}")
            msg.content = f"❌ Horary Classification failed: {e}"
            await msg.update()
            return
    # -----------------------------------

    show_debug = cl.user_session.get("show_debug", False)
    payload = {
        "session_id": session_id,
        "message": message.content,
        "debug": show_debug,
    }

    try:
        full_answer = ""
        suggestions = []
        debug_info = None
        intent = "KUNDALI"

        async with httpx.AsyncClient() as client:
            async with client.stream("POST", f"{API_BASE_URL}/chat", json=payload, timeout=120.0) as resp:
                if resp.status_code == 404:
                    msg.content = "❌ Session not found or expired on the backend API. Please refresh."
                    await msg.update()
                    return
                resp.raise_for_status()

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    
                    try:
                        data = json.loads(line.replace("data: ", ""))
                        if "answer" in data:
                            token = data["answer"]
                            full_answer += token
                            # Update UI live
                            msg.content = f"✨ **Astrological Reading**\n\n{full_answer}"
                            await msg.update()
                        elif "suggestions" in data:
                            suggestions = data["suggestions"]
                        elif "debug" in data:
                            debug_info = data["debug"]
                    except:
                        continue

        # Final setup after stream finishes
        msg.actions = [
            cl.Action(name="feedback", payload={"value": "good"}, label="👍"),
            cl.Action(name="feedback", payload={"value": "bad"}, label="👎")
        ]
        await msg.update()
        
        # Save for feedback reference
        cl.user_session.set("last_qa", {"question": message.content, "answer": full_answer})
        
        # Show debug info if available
        if debug_info and show_debug:
            debug_content = "🔍 **Debug: Gemini Decomposer Output**\n\n"
            decomp = debug_info.get("decomposer_output", [])
            if decomp:
                debug_content += "```json\n" + json.dumps(decomp, indent=2) + "\n```\n\n"
            
            reqs = debug_info.get("search_requests", [])
            if reqs:
                debug_content += "**Search Requests:**\n"
                for i, r in enumerate(reqs):
                    debug_content += f"* [{i+1}] Domain: `{r.get('domain')}` | Flow: `{r.get('flow')}` | Query: *{r.get('query', '')}*\n"
                    if r.get("category_filter"):
                        debug_content += f"  → Categories: `{', '.join(r['category_filter'])}`\n"
                    if r.get("house_filter"):
                        debug_content += f"  → Houses: `{', '.join(map(str, r['house_filter']))}`\n"
            
            debug_content += f"\n**Detected Intent:** `{debug_info.get('intent', 'N/A')}`\n\n"
            
            findings = debug_info.get("findings")
            if findings:
                debug_content += "### 📚 RAG Findings (Chunks Used)\n"
                debug_content += f"{findings}\n"
            
            await cl.Message(content=debug_content).send()
        
        if suggestions:
            await _send_suggestions(suggestions)

    except Exception as e:
        logger.error("Error communicating with backend API: %s", str(e), exc_info=True)
        msg.content = f"❌ An error occurred connecting to the backend API: {e}"
        await msg.update()

@cl.action_callback("feedback")
async def on_feedback(action: cl.Action):
    """Handle thumbs up/down feedback."""
    session_id = cl.user_session.get("session_id")
    last_qa = cl.user_session.get("last_qa")
    if not session_id or not last_qa:
        return

    feedback_value = action.payload.get("value")
    payload = {
        "session_id": session_id,
        "question": last_qa["question"],
        "answer": last_qa["answer"],
        "feedback": feedback_value
    }
    
    try:
        async with httpx.AsyncClient() as client:
            await client.post(f"{API_BASE_URL}/feedback", json=payload, timeout=5.0)

        # Remove actions from the message to prevent double voting
        action.for_id = None # Clear id to prevent further clicks
        await cl.Message(content=f"Thank you for your feedback! (Rated: {'👍 Good' if feedback_value == 'good' else '👎 Bad'})").send()
        
    except Exception as e:
        logger.error(f"Failed to send feedback to backend: {e}")
