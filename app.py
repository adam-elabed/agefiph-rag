# app.py (Streamlit Cloud-ready) — full code
import os
import re
import base64
import json
import asyncio
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# Load .env only for local runs (Streamlit Cloud uses st.secrets)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextEmbedding

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery


# ==============================
# Settings: st.secrets first, then env
# ==============================
def get_setting(name: str, default: Optional[str] = None) -> Optional[str]:
    # Streamlit secrets (Cloud or local .streamlit/secrets.toml)
    try:
        return str(st.secrets[name]) if name in st.secrets else os.getenv(name, default)
    except Exception:
        # If secrets are not configured locally, fall back to env vars
        return os.getenv(name, default)

def must(name: str) -> str:
    v = get_setting(name)
    if not v:
        raise RuntimeError(f"Missing setting: {name} (add it to Streamlit Secrets or env vars)")
    return v

def env_int(name: str, default: int) -> int:
    v = get_setting(name, str(default))
    try:
        return int(v)  # type: ignore[arg-type]
    except Exception:
        return default

def env_float(name: str, default: float) -> float:
    v = get_setting(name, str(default))
    try:
        return float(v)  # type: ignore[arg-type]
    except Exception:
        return default


# Default hidden params (override via env/secrets)
DEFAULT_TOP_K = env_int("APP_TOP_K", 10)
DEFAULT_GATE = env_float("APP_GATE", 0.30)

# Optional debug (off by default for managers)
DEBUG_INTENT = (get_setting("APP_DEBUG_INTENT", "false") or "false").lower() == "true"
DEBUG_RETRIEVAL = (get_setting("APP_DEBUG_RETRIEVAL", "false") or "false").lower() == "true"


# ==============================
# Async runner (robust in Streamlit)
# ==============================
def run_async(coro):
    """
    Streamlit can run in an environment where an event loop already exists.
    This helper runs async code robustly.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()
    return asyncio.run(coro)


# ==============================
# Robust JSON extraction (handles ```json fences)
# ==============================
def extract_json(raw: str) -> Optional[dict]:
    if not raw:
        return None

    s = raw.strip()

    # Remove code fences if present
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)

    # Try direct parse
    try:
        return json.loads(s)
    except Exception:
        pass

    # Fallback: find first JSON object
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    return None


# ==============================
# Utils
# ==============================
def decode_parent_id(parent_id: str) -> str:
    if not parent_id:
        return ""
    try:
        s = parent_id.strip()
        s += "=" * (-len(s) % 4)
        url = base64.urlsafe_b64decode(s.encode("utf-8")).decode("utf-8")
        url = re.sub(r"\.pdf\d+\b", ".pdf", url)
        url = re.sub(r"\.pdf\d+_", ".pdf_", url)
        return url
    except Exception:
        return parent_id

def extract_page_from_title(title: str) -> Optional[int]:
    m = re.search(r"input-(\d+)\.pdf", title or "")
    return int(m.group(1)) if m else None

def clean_chunk(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())

def history_to_text(messages: List[Dict[str, Any]], max_turns: int = 10) -> str:
    tail = messages[-max_turns * 2:] if len(messages) > max_turns * 2 else messages
    out = []
    for m in tail:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        out.append(f"{'User' if role=='user' else 'Assistant'}: {content}")
    return "\n".join(out)


# ==============================
# Kernel + services
# ==============================
@st.cache_resource(show_spinner=False)
def build_kernel_and_embedder():
    kernel = sk.Kernel()

    chat = AzureChatCompletion(
        service_id="chat",
        deployment_name=must("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        endpoint=must("AZURE_OPENAI_ENDPOINT"),
        api_key=must("AZURE_OPENAI_API_KEY"),
        api_version=must("AZURE_OPENAI_API_VERSION"),
    )
    embedder = AzureTextEmbedding(
        service_id="embed",
        deployment_name=must("AZURE_OPENAI_EMBED_DEPLOYMENT"),
        endpoint=must("AZURE_OPENAI_ENDPOINT"),
        api_key=must("AZURE_OPENAI_API_KEY"),
        api_version=must("AZURE_OPENAI_EMBED_API_VERSION"),
    )

    kernel.add_service(chat)
    kernel.add_service(embedder)
    return kernel, embedder


# ==============================
# Intent router
# ==============================
INTENTS = ("smalltalk", "definition", "overview", "specific")

async def classify_intent(message: str) -> str:
    kernel, _ = build_kernel_and_embedder()
    prompt = f"""
Return ONLY one token:
smalltalk | definition | overview | specific

- smalltalk: greetings, thanks, casual chat, "help", "what can you do"
- definition: "what is Agefiph", "c'est quoi Agefiph", "rôle de l'Agefiph"
- overview: asks for a list/summary of all aids/services
- specific: asks about a specific aid, amount, eligibility, steps, conditions

Message:
{message}
""".strip()

    out = str(await kernel.invoke_prompt(prompt, service_id="chat")).strip().lower()
    for intent in INTENTS:
        if intent in out:
            return intent
    return "specific"

def override_intent_heuristics(q: str, intent: str) -> str:
    q_low = (q or "").lower()

    overview_triggers = [
        "quelles aides", "quelles sont les aides", "liste des aides", "toutes les aides",
        "aides proposées", "aides agefiph", "quels dispositifs", "dispositifs", "panorama",
        "qu'est ce que vous proposez", "que propose agefiph", "quels types d'aides"
    ]
    if any(t in q_low for t in overview_triggers):
        return "overview"

    definition_triggers = [
        "c'est quoi agefiph", "c quoi agefiph", "qu'est-ce que l'agefiph",
        "rôle de l'agefiph", "mission de l'agefiph", "definition agefiph"
    ]
    if any(t in q_low for t in definition_triggers):
        return "definition"

    return intent

async def generate_smalltalk(message: str, history_text: str) -> str:
    kernel, _ = build_kernel_and_embedder()
    prompt = f"""
You are a helpful assistant specialized in Agefiph.

Reply politely and briefly.
If it is a greeting/help request, introduce that you can answer questions about Agefiph aids/services and ask what the user needs.
Do NOT mention internal tools.

Conversation:
{history_text}

User message:
{message}

Assistant reply:
""".strip()
    return str(await kernel.invoke_prompt(prompt, service_id="chat")).strip()


# ==============================
# Retrieval (Azure AI Search hybrid)
# ==============================
async def retrieve_contexts(query: str, k: int, gate_threshold: float) -> List[Dict[str, Any]]:
    _, embedder = build_kernel_and_embedder()
    qvec = (await embedder.generate_embeddings([query]))[0]
    vq = VectorizedQuery(vector=qvec, k=k, fields="text_vector")

    search_client = SearchClient(
        endpoint=must("AZURE_AI_SEARCH_ENDPOINT"),
        index_name=must("AZURE_AI_SEARCH_INDEX"),
        credential=AzureKeyCredential(must("AZURE_AI_SEARCH_API_KEY")),
    )

    try:
        results = await search_client.search(
            search_text=query,
            vector_queries=[vq],
            select=["chunk", "title", "parent_id", "chunk_id"],
            top=k,
        )

        rows: List[Dict[str, Any]] = []
        async for r in results:
            chunk = clean_chunk(r.get("chunk"))
            if not chunk:
                continue
            if "SOMMAIRE" in chunk:
                continue

            title = r.get("title") or ""
            parent_id = r.get("parent_id") or ""
            chunk_id = r.get("chunk_id") or ""
            score = float(r.get("@search.score", 0.0))

            rows.append({
                "text": chunk,
                "title": title,
                "page": extract_page_from_title(title),
                "source_url": decode_parent_id(parent_id),  # kept internal
                "chunk_id": chunk_id,
                "score": score,
            })

        rows.sort(key=lambda x: x["score"], reverse=True)

        # save debug rows (optional)
        st.session_state["last_debug_rows"] = rows[:10]

        if gate_threshold > 0 and (not rows or rows[0]["score"] < gate_threshold):
            return []

        # dedup + cap
        seen = set()
        out = []
        for row in rows:
            if row["text"] in seen:
                continue
            seen.add(row["text"])
            out.append(row)
            if len(out) >= 12:
                break
        return out

    finally:
        await search_client.close()


def retrieval_params_for_intent(intent: str, base_top_k: int, base_gate: float) -> Tuple[int, float]:
    if intent == "overview":
        return max(base_top_k, 30), 0.0
    if intent == "definition":
        return max(base_top_k, 12), 0.0
    return base_top_k, base_gate


# ==============================
# Clarification (1 question max)
# ==============================
async def need_clarification(intent: str, contexts: List[Dict[str, Any]]) -> bool:
    if intent in ("overview", "definition"):
        return False
    if not contexts and intent == "specific":
        return True
    return False

async def generate_clarifying_question(question: str, history_text: str) -> str:
    kernel, _ = build_kernel_and_embedder()
    prompt = f"""
Ask EXACTLY ONE concise clarifying question.
Prefer the single most informative question.
Do not mention documents or internal tools.
Use the same language as the user.

Conversation:
{history_text}

User question:
{question}

One clarifying question:
""".strip()
    return str(await kernel.invoke_prompt(prompt, service_id="chat")).strip()


# ==============================
# Answer generation (no sources shown)
# ==============================
async def generate_rag_answer(question: str, intent: str, contexts: List[Dict[str, Any]], history_text: str) -> str:
    kernel, _ = build_kernel_and_embedder()

    if not contexts:
        if intent == "specific":
            return await generate_clarifying_question(question, history_text)

        prompt = f"""
You are an Agefiph assistant.
Write a short helpful reply asking the user to be more specific (goal/situation),
without mentioning internal tools.

User question:
{question}

Reply:
""".strip()
        return str(await kernel.invoke_prompt(prompt, service_id="chat")).strip()

    context_text = "\n\n---\n\n".join(
        [f"[title={c['title']} page={c.get('page')} chunk_id={c['chunk_id']}]\n{c['text']}" for c in contexts]
    )

    # Definition: never ask clarification
    if intent == "definition":
        prompt = f"""
You are an Agefiph assistant.
Answer ONLY using the context. No external knowledge.
Do NOT ask clarifying questions for definition requests.

Write a clear definition (3-6 lines) + up to 2 short bullets (missions / who it helps) if present.

Conversation:
{history_text}

Question:
{question}

Context:
{context_text}

Answer:
""".strip()
        return str(await kernel.invoke_prompt(prompt, service_id="chat")).strip()

    # Overview/specific use strict JSON schema internally
    if intent == "overview":
        instruction = """
Task (overview):
- Output 6 to 10 distinct aids/services MAX.
- Each bullet: name + one short description (<= 1 sentence) if present in context.
- End with ONE short question to narrow down the need.
- If list might be incomplete, set "partial" to true.
"""
    else:
        instruction = """
Task (specific):
- Answer using ONLY the context.
- If key details are missing to answer correctly, set "needs_clarification" to true and ask ONE question in "clarifying_questions".
- Do NOT add disclaimers about information the user did not ask for.
"""

    prompt = f"""
You are an Agefiph RAG assistant.

Hard rules:
- Use ONLY the context. No external knowledge.
- If you cannot answer from context OR you need user-specific details, do NOT guess.
- For clarifications: ask ONLY ONE question.

Return STRICT JSON only with this schema:
{{
  "needs_clarification": boolean,
  "clarifying_questions": string[],
  "answer": string,
  "partial": boolean
}}

Conversation:
{history_text}

User question:
{question}

Context:
{context_text}

{instruction}

JSON:
""".strip()

    raw = str(await kernel.invoke_prompt(prompt, service_id="chat")).strip()

    data = extract_json(raw)
    if data is None:
        # Fallback: do not expose raw JSON to end users
        return "Je n’ai pas pu formater correctement la réponse. Pouvez-vous reformuler votre question ?"

    if data.get("needs_clarification"):
        qs = data.get("clarifying_questions") or []
        if qs:
            return str(qs[0]).strip()
        return await generate_clarifying_question(question, history_text)

    answer = (data.get("answer") or "").strip()
    if not answer:
        return "Je ne sais pas d'après les documents."

    if data.get("partial") is True and intent == "overview":
        answer += "\n\n(Liste partielle selon les documents.)"

    return answer


# ==============================
# Streamlit UI (manager-friendly)
# ==============================
st.set_page_config(page_title="Agefiph RAG POC", layout="centered")
st.title("Agefiph Chatbot POC")

col1, col2 = st.columns([1, 1])
with col1:
    st.caption("Assistant basé sur les documents Agefiph (POC).")
with col2:
    if st.button("Nouvelle conversation"):
        st.session_state.pop("messages", None)
        st.session_state.pop("last_debug_rows", None)
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_debug_rows" not in st.session_state:
    st.session_state.last_debug_rows = []

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

q = st.chat_input("Pose une question sur l’Agefiph…")

if q:
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            history_text = history_to_text(st.session_state.messages, max_turns=10)

            intent = run_async(classify_intent(q))
            intent = override_intent_heuristics(q, intent)

            if intent == "smalltalk":
                contexts: List[Dict[str, Any]] = []
                answer = run_async(generate_smalltalk(q, history_text))
            else:
                k1, g1 = retrieval_params_for_intent(intent, base_top_k=DEFAULT_TOP_K, base_gate=DEFAULT_GATE)

                retrieval_query = q
                if intent == "definition":
                    retrieval_query = q + " mission objectif rôle Agefiph"

                contexts = run_async(retrieve_contexts(retrieval_query, k=k1, gate_threshold=g1))
                if not contexts:
                    contexts = run_async(retrieve_contexts(retrieval_query, k=max(k1, 25), gate_threshold=0.0))

                ask = run_async(need_clarification(intent, contexts))
                if ask:
                    answer = run_async(generate_clarifying_question(q, history_text))
                else:
                    answer = run_async(generate_rag_answer(q, intent, contexts, history_text))

        st.markdown(answer)

        # optional dev debug (hidden by default)
        if DEBUG_INTENT:
            st.caption(f"intent: `{intent}`")
        if DEBUG_RETRIEVAL:
            with st.expander("Debug retrieval (raw top rows)", expanded=False):
                rows = st.session_state.get("last_debug_rows", [])
                for s in rows:
                    st.write({
                        "score": s.get("score"),
                        "title": s.get("title"),
                        "page": s.get("page"),
                        "chunk_id": s.get("chunk_id"),
                        "preview": (s.get("text") or "")[:160],
                    })

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "intent": intent,
    })