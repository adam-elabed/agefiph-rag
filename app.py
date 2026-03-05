# app.py — Streamlit Cloud-ready (Generic “Answer-first” RAG, no numbered choices, no sources)
#
# Goals (generic, not case-by-case):
# ✅ Answer-first: respond with whatever is possible from context, THEN ask 1 follow-up.
# ✅ Ask clarification ONLY when strictly necessary (otherwise it feels bureaucratic).
# ✅ No numbered choices ("1=..., 2=...") anywhere.
# ✅ Multi-query retrieval (3 queries) to improve recall and reduce “wrong audience” chunks.
# ✅ No sources displayed.
# ✅ Health mentions (pain/symptoms): 1 short safe sentence, no medical advice.
# ✅ Concise + precise: 4–8 sentences, grounded only in indexed docs (no invented amounts/conditions).
#
# Required secrets/env:
#   AZURE_OPENAI_ENDPOINT
#   AZURE_OPENAI_API_KEY
#   AZURE_OPENAI_API_VERSION
#   AZURE_OPENAI_CHAT_DEPLOYMENT
#   AZURE_OPENAI_EMBED_DEPLOYMENT
#   AZURE_OPENAI_EMBED_API_VERSION
#   AZURE_AI_SEARCH_ENDPOINT
#   AZURE_AI_SEARCH_INDEX
#   AZURE_AI_SEARCH_API_KEY
#
# Optional (semantic ranker):
#   AZURE_AI_SEARCH_SEMANTIC_CONFIG = semantic-config-agefiph (or your config name)
#
# Optional tuning:
#   APP_TOP_K=18
#   APP_MAX_CTX=8
#   APP_DEBUG_CONTROLLER=true
#   APP_DEBUG_RETRIEVAL=true

import os
import re
import json
import base64
import asyncio
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

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
# Settings helpers
# ==============================
def get_setting(name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)


def must(name: str) -> str:
    v = get_setting(name)
    if not v:
        raise RuntimeError(f"Missing setting: {name} (Streamlit Secrets or env vars)")
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
        return float(v)
    except Exception:
        return default


DEFAULT_TOP_K = env_int("APP_TOP_K", 18)
DEFAULT_MAX_CONTEXTS = env_int("APP_MAX_CTX", 8)
DEFAULT_GATE = env_float("APP_GATE", 0.0)

DEBUG_CONTROLLER = (get_setting("APP_DEBUG_CONTROLLER", "false") or "false").lower() == "true"
DEBUG_RETRIEVAL = (get_setting("APP_DEBUG_RETRIEVAL", "false") or "false").lower() == "true"


# ==============================
# Async runner
# ==============================
def run_async(coro):
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
# Text utils
# ==============================
def clean_ws(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())


def history_to_text(messages: List[Dict[str, Any]], max_turns: int = 10) -> str:
    tail = messages[-max_turns * 2:] if len(messages) > max_turns * 2 else messages
    out = []
    for m in tail:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        out.append(f"{'Utilisateur' if role=='user' else 'Assistant'}: {content}")
    return "\n".join(out)


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def decode_parent_id(parent_id: str) -> str:
    # Not shown to user; kept only for optional debug.
    if not parent_id:
        return ""
    try:
        s = parent_id.strip()
        s += "=" * (-len(s) % 4)
        url = base64.urlsafe_b64decode(s.encode("utf-8")).decode("utf-8")
        url = re.sub(r"\.pdf\d+\b", ".pdf", url)
        url = re.sub(r"\.pdf\d+_", ".pdf_", url)
        url = re.sub(r"\.html\d+\b", ".html", url)
        url = re.sub(r"\.html\d+_", ".html_", url)
        return url
    except Exception:
        return parent_id


def extract_page_from_title(title: str) -> Optional[int]:
    m = re.search(r"input-(\d+)\.pdf", title or "")
    return int(m.group(1)) if m else None


def normalize_intent(x: str) -> str:
    x = (x or "").strip().lower()
    return x if x in {"smalltalk", "definition", "overview", "specific", "out_of_scope"} else "specific"


def normalize_audience(x: str) -> str:
    x = (x or "").strip().lower()
    return x if x in {"personne", "employeur", "formation", "conseiller", "unknown"} else "unknown"


def is_short_yes(text: str) -> bool:
    t = clean_ws(text).lower()
    return t in {
        "oui", "yes", "ok", "okay", "d'accord", "dac", "merci",
        "c'est bon", "parfait", "super", "top", "nickel",
        "ça répond", "ca repond", "c'est clair", "cest clair"
    }


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
# Generic “Answer-first” Controller + Resolver (NO numbers)
# ==============================
STATUS_QUESTION = (
    "Pour vous orienter au mieux, vous êtes plutôt une personne en situation de handicap, "
    "un employeur, un organisme de formation, ou un conseiller ?"
)

CONTROLLER_PROMPT = """
Vous êtes le contrôleur d’un chatbot Agefiph.

Objectif :
Décider quoi faire AVANT de répondre (intention, audience, besoin de clarification, requête de recherche).

Principe générique “Answer-first” :
- Si une première réponse utile est possible sans information supplémentaire, needs_clarification=false.
- needs_clarification=true uniquement si une clarification est indispensable pour éviter une réponse vide, trompeuse ou inutile.
- Ne posez pas une question “de tri” si vous pouvez déjà répondre partiellement.

Contraintes :
- Périmètre : handicap + emploi + formation + aides/services Agefiph (et partenaires cités).
- Hors périmètre => intent=out_of_scope.
- Une seule question de clarification max.
- IMPORTANT : ne jamais proposer de choix numérotés.

Intent :
- “c’est quoi Agefiph ?” => definition
- “toutes les aides / panorama / toutes les infos” => overview
- salutations => smalltalk
- sinon => specific

Audience (si détectable) :
- personne : PSH/RQTH/demandeur d’emploi handicap/salarié handicap
- employeur : entreprise/RH/recrutement/maintien
- formation : organisme/financement/accessibilité formation
- conseiller : prescripteur/accompagnant
- unknown : sinon

Retour JSON strict :
{{
  "intent": "smalltalk|definition|overview|specific|out_of_scope",
  "audience": "personne|employeur|formation|conseiller|unknown",
  "needs_clarification": true|false,
  "clarifying_question": "string",
  "retrieval_query": "string"
}}

Règles :
- retrieval_query : reformulez la demande pour la recherche documentaire (corrigez les fautes).
- Si needs_clarification=false => clarifying_question="".
- Si needs_clarification=true => clarifying_question = question courte, naturelle, sans numéros.

Historique récent :
{history}

Message utilisateur :
{message}
""".strip()


async def controller_decide(message: str, history_text: str) -> Dict[str, Any]:
    kernel, _ = build_kernel_and_embedder()
    prompt = CONTROLLER_PROMPT.format(history=history_text, message=message)
    raw = str(await kernel.invoke_prompt(prompt, service_id="chat")).strip()

    data = safe_json_loads(raw)
    if not data:
        # Be permissive: answer-first even if parsing fails
        return {
            "intent": "specific",
            "audience": "unknown",
            "needs_clarification": False,
            "clarifying_question": "",
            "retrieval_query": message,
            "raw": raw,
        }

    intent = normalize_intent(str(data.get("intent", "specific")))
    audience = normalize_audience(str(data.get("audience", "unknown")))
    needs = bool(data.get("needs_clarification", False))
    cq = str(data.get("clarifying_question", "") or "").strip()
    rq = str(data.get("retrieval_query", "") or "").strip()

    # Hard rule: no numbered choices
    if re.search(r"\b\d+\s*=", cq) or re.search(r"\b(1|2|3|4)\b\s*[:=-]", cq):
        cq = STATUS_QUESTION

    if needs and not cq:
        cq = STATUS_QUESTION
    if (not needs) and cq:
        cq = ""
    if not rq:
        rq = message

    return {
        "intent": intent,
        "audience": audience,
        "needs_clarification": needs,
        "clarifying_question": cq,
        "retrieval_query": rq,
        "raw": raw,
    }


RESOLVER_PROMPT = """
Vous résolvez une clarification déjà posée.

Objectif :
- Interpréter la réponse libre de l’utilisateur par rapport à la question.
- Si la réponse permet de décider => resolved=true et produire intent/audience/retrieval_query.
- Si la réponse demande un panorama (“tout”, “toutes les aides”, “panorama”, “toutes les infos”) => resolved=true, intent=overview.
- Si la réponse est insuffisante (“oui”, “ok”, “d'accord”...) => resolved=false et poser UNE question courte, sans numéros.

Contraintes :
- IMPORTANT : ne jamais proposer de choix numérotés.

Retour JSON strict :
{{
  "resolved": true|false,
  "intent": "definition|overview|specific|out_of_scope",
  "audience": "personne|employeur|formation|conseiller|unknown",
  "retrieval_query": "string",
  "next_question": "string"
}}

Question précédente :
{clarifying_question}

Réponse utilisateur :
{user_answer}
""".strip()


async def resolve_clarification(clarifying_question: str, user_answer: str) -> Dict[str, Any]:
    kernel, _ = build_kernel_and_embedder()
    prompt = RESOLVER_PROMPT.format(
        clarifying_question=clarifying_question,
        user_answer=user_answer,
    )
    raw = str(await kernel.invoke_prompt(prompt, service_id="chat")).strip()

    data = safe_json_loads(raw) or {}
    resolved = bool(data.get("resolved", False))
    intent = normalize_intent(str(data.get("intent", "specific")))
    audience = normalize_audience(str(data.get("audience", "unknown")))
    rq = str(data.get("retrieval_query", "") or "").strip()
    nq = str(data.get("next_question", "") or "").strip()

    fallback_q = "Pouvez-vous préciser en une phrase ce que vous cherchez exactement (emploi, formation, maintien, adaptation du poste, ou autre) ?"

    if resolved and not rq:
        resolved = False
        nq = fallback_q

    if (not resolved) and not nq:
        nq = fallback_q

    if re.search(r"\b\d+\b", nq) and ("choisissez" in nq.lower() or "répondez" in nq.lower()):
        nq = fallback_q

    return {
        "resolved": resolved,
        "intent": intent,
        "audience": audience,
        "retrieval_query": rq,
        "next_question": nq,
        "raw": raw,
    }


# ==============================
# Multi-query expansion (LLM) + caching
# ==============================
MULTI_QUERY_PROMPT = """
Vous générez des requêtes de recherche pour retrouver des passages dans des documents Agefiph.

Objectif :
Donner 3 requêtes courtes et différentes couvrant la même intention, en français.
- Corrigez les fautes.
- Ajoutez des synonymes utiles.
- Gardez les requêtes concises.
- Pas de numéros.

Retour JSON strict :
{{
  "queries": ["...", "...", "..."]
}}

Entrée :
- audience: {audience}
- intent: {intent}
- question utilisateur: {question}
- requête de base: {base_query}
""".strip()


@st.cache_data(show_spinner=False, ttl=3600)
def cached_multi_queries(audience: str, intent: str, question: str, base_query: str) -> List[str]:
    return run_async(expand_queries(audience, intent, question, base_query))


async def expand_queries(audience: str, intent: str, question: str, base_query: str) -> List[str]:
    kernel, _ = build_kernel_and_embedder()

    if intent in {"smalltalk", "out_of_scope"}:
        return [clean_ws(base_query) or clean_ws(question)]

    prompt = MULTI_QUERY_PROMPT.format(
        audience=audience,
        intent=intent,
        question=question,
        base_query=base_query,
    )
    raw = str(await kernel.invoke_prompt(prompt, service_id="chat")).strip()
    data = safe_json_loads(raw) or {}
    queries = data.get("queries", [])

    out: List[str] = []
    if isinstance(queries, list):
        for q in queries:
            qs = clean_ws(str(q))
            if qs:
                out.append(qs)

    # Always include base query first
    base = clean_ws(base_query) or clean_ws(question)
    merged = [base] + [q for q in out if q.lower() != base.lower()]
    merged = merged[:3] if len(merged) >= 3 else merged
    return merged if merged else [base]


# ==============================
# Retrieval (Hybrid + optional Semantic Ranker)
# ==============================
def get_semantic_config_name() -> str:
    return (get_setting("AZURE_AI_SEARCH_SEMANTIC_CONFIG", "") or "").strip()


async def retrieve_once(query: str, k: int) -> List[Dict[str, Any]]:
    _, embedder = build_kernel_and_embedder()
    qvec = (await embedder.generate_embeddings([query]))[0]
    vq = VectorizedQuery(vector=qvec, k=k, fields="text_vector")

    search_client = SearchClient(
        endpoint=must("AZURE_AI_SEARCH_ENDPOINT"),
        index_name=must("AZURE_AI_SEARCH_INDEX"),
        credential=AzureKeyCredential(must("AZURE_AI_SEARCH_API_KEY")),
    )

    semantic_cfg = get_semantic_config_name()
    use_semantic = bool(semantic_cfg)

    try:
        search_kwargs: Dict[str, Any] = dict(
            search_text=query,
            vector_queries=[vq],
            select=["chunk", "title", "parent_id", "chunk_id"],
            top=k,
        )

        if use_semantic:
            search_kwargs.update(
                {
                    "query_type": "semantic",
                    "semantic_configuration_name": semantic_cfg,
                    "query_caption": "extractive",
                    "query_answer": "extractive",
                }
            )

        results = await search_client.search(**search_kwargs)

        rows: List[Dict[str, Any]] = []
        async for r in results:
            chunk = clean_ws(r.get("chunk"))
            if not chunk:
                continue

            title = r.get("title") or ""
            parent_id = r.get("parent_id") or ""
            chunk_id = r.get("chunk_id") or ""

            score = float(r.get("@search.rerankerScore", r.get("@search.score", 0.0)))

            rows.append(
                {
                    "text": chunk,
                    "title": title,
                    "page": extract_page_from_title(title),
                    "source_url": decode_parent_id(parent_id),
                    "chunk_id": chunk_id,
                    "score": score,
                    "query": query,
                }
            )

        rows.sort(key=lambda x: x["score"], reverse=True)
        return rows

    finally:
        await search_client.close()


async def retrieve_contexts_multi(queries: List[str], k_total: int, gate_threshold: float) -> List[Dict[str, Any]]:
    if not queries:
        return []
    per_q = max(6, k_total // max(1, len(queries)))

    all_rows: List[Dict[str, Any]] = []
    for q in queries:
        all_rows.extend(await retrieve_once(q, k=per_q))

    all_rows.sort(key=lambda x: x["score"], reverse=True)
    st.session_state["last_debug_rows"] = all_rows[:10]

    if gate_threshold > 0 and len(all_rows) < 2:
        return []

    # Dedup by chunk_id (preferred) or by text
    best: Dict[str, Dict[str, Any]] = {}
    for r in all_rows:
        key = (r.get("chunk_id") or "").strip() or r["text"]
        prev = best.get(key)
        if (prev is None) or (r["score"] > float(prev.get("score", 0.0))):
            best[key] = r

    merged = list(best.values())
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:DEFAULT_MAX_CONTEXTS]


def retrieval_params(intent: str) -> Tuple[int, float]:
    if intent == "overview":
        return max(DEFAULT_TOP_K, 24), 0.0
    if intent == "definition":
        return max(DEFAULT_TOP_K, 14), 0.0
    return DEFAULT_TOP_K, DEFAULT_GATE


# ==============================
# Answer generation (grounded + concise + generic follow-up)
# ==============================
SYSTEM_RULES = """
Vous êtes l’assistant virtuel de l’Agefiph.

Règles (génériques) :
- Répondez de façon naturelle, claire et concise (4 à 8 phrases maximum).
- Utilisez UNIQUEMENT les informations présentes dans le CONTEXTE.
- Ne jamais inventer : montants, numéros, délais, conditions, procédures, contacts.
- Si le CONTEXTE ne suffit pas, posez UNE seule question courte pour préciser.
- Si l’utilisateur mentionne une douleur ou un symptôme : ne donnez pas d’avis médical.
  Ajoutez une seule phrase courte : "Je ne peux pas évaluer médicalement, mais je peux vous orienter sur les aides liées au travail/handicap."
- Terminez par UNE question de suivi utile (pour affiner), sans demander de “choisir une catégorie”.
""".strip()


def audience_guard(audience: str) -> str:
    # Generic guard to reduce leakage between audiences without per-case logic.
    if audience == "personne":
        return (
            "Audience = personne. Ne mettez pas l'accent sur les aides exclusivement 'employeur'. "
            "Si le contexte est surtout employeur, reformulez en termes utiles pour la personne (accompagnement, accès à l'emploi, maintien, adaptation)."
        )
    if audience == "employeur":
        return "Audience = employeur. Concentrez-vous sur recrutement, intégration, maintien côté employeur."
    if audience == "formation":
        return "Audience = formation. Concentrez-vous sur aides/dispositifs liés à la formation et accessibilité."
    if audience == "conseiller":
        return "Audience = conseiller. Concentrez-vous sur orientation, prescription, articulation des dispositifs."
    return "Audience inconnue : répondez de manière générale et posez une question de suivi qui précise la situation."


async def generate_smalltalk() -> str:
    return "Bonjour 👋 Dites-moi votre question sur l’Agefiph (emploi, formation, aides…), et je vous réponds."


async def generate_rag_answer(question: str, intent: str, audience: str, contexts: List[Dict[str, Any]]) -> str:
    kernel, _ = build_kernel_and_embedder()

    if not contexts:
        return (
            "Je n’ai pas trouvé l’information nécessaire dans les documents actuellement indexés. "
            "Pouvez-vous préciser votre besoin en une phrase (par exemple : trouver un emploi, garder votre emploi, adapter un poste, ou financer une formation) ?"
        )

    context_text = "\n\n---\n\n".join(
        [f"[title={c['title']} chunk_id={c['chunk_id']}]\n{c['text']}" for c in contexts]
    )

    guard = audience_guard(audience)

    if intent == "definition":
        prompt = f"""
{SYSTEM_RULES}

Garde-fou audience :
{guard}

Tâche :
Donnez une définition courte, fidèle au contexte, puis posez UNE question de suivi utile.

Question :
{question}

CONTEXTE :
{context_text}

Réponse :
""".strip()
        return str(await kernel.invoke_prompt(prompt, service_id="chat")).strip()

    if intent == "overview":
        prompt = f"""
{SYSTEM_RULES}

Garde-fou audience :
{guard}

Tâche :
Faites un panorama court (6 à 10 puces maximum) uniquement à partir du contexte.
Puis posez UNE question de suivi utile.

Question :
{question}

CONTEXTE :
{context_text}

Réponse :
""".strip()
        return str(await kernel.invoke_prompt(prompt, service_id="chat")).strip()

    prompt = f"""
{SYSTEM_RULES}

Garde-fou audience :
{guard}

Règle :
Si la réponse n’est pas clairement dans le contexte, posez UNE seule question courte pour préciser (et rien d’autre).

Question :
{question}

CONTEXTE :
{context_text}

Réponse :
""".strip()
    return str(await kernel.invoke_prompt(prompt, service_id="chat")).strip()


# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Agefiph Chatbot POC", layout="centered")
st.title("Agefiph Chatbot POC")
st.caption("Assistant basé sur des documents Agefiph indexés (POC).")

col1, col2 = st.columns([1, 1])
with col2:
    if st.button("Nouvelle conversation"):
        st.session_state.pop("messages", None)
        st.session_state.pop("last_debug_rows", None)
        st.session_state.pop("pending", None)
        st.session_state.pop("pending_type", None)
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_debug_rows" not in st.session_state:
    st.session_state.last_debug_rows = []
if "pending" not in st.session_state:
    st.session_state.pending = None  # {"question": "..."}
if "pending_type" not in st.session_state:
    st.session_state.pending_type = None  # "clarification" or "followup"

# Render chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

q = st.chat_input("Posez une question sur l’Agefiph…")

if q:
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    # If user says yes/ok after a follow-up, close politely (no loops)
    if st.session_state.pending_type == "followup" and is_short_yes(q):
        answer = "Parfait. Si vous avez une autre question, dites-moi ce que vous cherchez et je vous aide."
        st.session_state.pending = None
        st.session_state.pending_type = None
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            history_text = history_to_text(st.session_state.messages, max_turns=10)

            intent = "specific"
            audience = "unknown"
            contexts: List[Dict[str, Any]] = []

            # CASE A: resolve a clarification
            if st.session_state.pending is not None and st.session_state.pending_type == "clarification":
                pending_q = st.session_state.pending.get("question", "")
                res = run_async(resolve_clarification(pending_q, q))

                if res["resolved"]:
                    st.session_state.pending = None
                    st.session_state.pending_type = None
                    intent = res["intent"]
                    audience = res["audience"]
                    retrieval_query = res["retrieval_query"]

                    if intent == "out_of_scope":
                        answer = "Je peux répondre uniquement aux questions liées au handicap et à l’emploi (aides et services de l’Agefiph)."
                    else:
                        k, gate = retrieval_params(intent)
                        queries = cached_multi_queries(audience, intent, q, retrieval_query)
                        contexts = run_async(retrieve_contexts_multi(queries, k_total=k, gate_threshold=gate))
                        if not contexts:
                            contexts = run_async(retrieve_contexts_multi(queries, k_total=max(k, 24), gate_threshold=0.0))

                        answer = run_async(generate_rag_answer(q, intent, audience, contexts))
                        st.session_state.pending_type = "followup"

                else:
                    answer = res["next_question"]
                    st.session_state.pending = {"question": answer}
                    st.session_state.pending_type = "clarification"

                if DEBUG_CONTROLLER:
                    with st.expander("Debug resolver", expanded=False):
                        st.write(
                            {
                                "pending_question": pending_q,
                                "resolved": res["resolved"],
                                "intent": res["intent"],
                                "audience": res["audience"],
                                "retrieval_query": res["retrieval_query"],
                                "next_question": res["next_question"],
                                "raw": res.get("raw", "")[:900],
                            }
                        )

            # CASE B: normal controller flow
            else:
                decision = run_async(controller_decide(q, history_text))
                intent = decision["intent"]
                audience = decision["audience"]
                needs_clarification = decision["needs_clarification"]
                clarifying_question = decision["clarifying_question"]
                retrieval_query = decision["retrieval_query"]

                if intent == "out_of_scope":
                    answer = "Je peux répondre uniquement aux questions liées au handicap et à l’emploi (aides et services de l’Agefiph)."
                    st.session_state.pending = None
                    st.session_state.pending_type = None

                elif intent == "smalltalk":
                    answer = run_async(generate_smalltalk())
                    st.session_state.pending = None
                    st.session_state.pending_type = None

                elif needs_clarification:
                    answer = (clarifying_question.strip() or STATUS_QUESTION)
                    st.session_state.pending = {"question": answer}
                    st.session_state.pending_type = "clarification"

                else:
                    k, gate = retrieval_params(intent)
                    queries = cached_multi_queries(audience, intent, q, retrieval_query)
                    contexts = run_async(retrieve_contexts_multi(queries, k_total=k, gate_threshold=gate))
                    if not contexts:
                        contexts = run_async(retrieve_contexts_multi(queries, k_total=max(k, 24), gate_threshold=0.0))

                    answer = run_async(generate_rag_answer(q, intent, audience, contexts))
                    st.session_state.pending = None
                    st.session_state.pending_type = "followup"

                if DEBUG_CONTROLLER:
                    with st.expander("Debug controller", expanded=False):
                        dbg_queries = []
                        if intent not in {"smalltalk", "out_of_scope"}:
                            dbg_queries = cached_multi_queries(audience, intent, q, retrieval_query)
                        st.write(
                            {
                                "intent": intent,
                                "audience": audience,
                                "needs_clarification": needs_clarification,
                                "clarifying_question": clarifying_question,
                                "retrieval_query": retrieval_query,
                                "multi_queries": dbg_queries,
                                "raw": decision.get("raw", "")[:900],
                            }
                        )

            if DEBUG_RETRIEVAL:
                with st.expander("Debug retrieval (top rows)", expanded=False):
                    rows = st.session_state.get("last_debug_rows", [])
                    for s in rows:
                        st.write(
                            {
                                "score": s.get("score"),
                                "title": s.get("title"),
                                "chunk_id": s.get("chunk_id"),
                                "query": s.get("query"),
                                "preview": (s.get("text") or "")[:240],
                            }
                        )

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})