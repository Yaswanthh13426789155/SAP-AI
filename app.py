from collections import Counter
from functools import lru_cache
import os
from pathlib import Path
import re

from dotenv import load_dotenv

from sap_ticket_catalog import TICKET_CATALOG


BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".cache" / "huggingface"
OPENAI_FAILURE_NOTICE = None
OLLAMA_FAILURE_NOTICE = None
OPENAI_COMPATIBLE_FAILURE_NOTICE = None
HF_LOCAL_FAILURE_NOTICE = None

load_dotenv(BASE_DIR / ".env")


DATA_FILES = [
    BASE_DIR / "sap_tickets.txt",
    BASE_DIR / "sap_dataset.txt",
    BASE_DIR / "sap_web_data.txt",
]
INDEX_PATH = BASE_DIR / "sap_index"
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_/.-]+")
TCODE_PATTERN = re.compile(r"\b[A-Z]{2,5}\d{1,4}[A-Z]?\b")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "because",
    "can",
    "cannot",
    "do",
    "error",
    "errors",
    "failed",
    "failure",
    "for",
    "from",
    "how",
    "i",
    "in",
    "issue",
    "issues",
    "is",
    "it",
    "missing",
    "my",
    "not",
    "of",
    "on",
    "s4hana",
    "sap",
    "system",
    "the",
    "this",
    "to",
    "what",
    "when",
    "with",
}
TOKEN_ALIASES = {
    "authorisation": "authorization",
    "auth": "authorization",
    "development": "dev",
    "logon": "login",
    "prd": "prod",
    "production": "prod",
    "quality": "qa",
    "qas": "qa",
    "slowly": "slow",
    "testing": "test",
    "tst": "test",
    "uat": "test",
    "transports": "transport",
    "updates": "update",
    "queues": "queue",
    "ids": "idoc",
}
ENVIRONMENT_PROFILES = {
    "DEV": {
        "aliases": {"dev", "development"},
        "label": "Development",
        "guidance": [
            "Reproduce the issue with representative non-production data.",
            "Use traces, dumps, and debug tools to confirm the technical root cause.",
            "Fix configuration, code, or master data in a controlled way before releasing a transport.",
        ],
        "guardrails": [
            "Document the exact objects changed so they can move cleanly through the landscape.",
            "Retest the failed step and one nearby happy-path scenario before transport release.",
        ],
    },
    "QA": {
        "aliases": {"qa", "quality"},
        "label": "Quality Assurance",
        "guidance": [
            "Validate the fix after transport import and confirm the issue is no longer reproducible.",
            "Run regression checks on related interfaces, authorizations, and dependent jobs.",
            "Compare results with DEV if the same transport behaves differently.",
        ],
        "guardrails": [
            "Do not approve transport promotion until both functional and technical checks pass.",
            "Capture evidence for failed and successful retest cycles.",
        ],
    },
    "TEST": {
        "aliases": {"test", "testing", "uat", "tst"},
        "label": "Testing / UAT",
        "guidance": [
            "Retest the end-to-end business process with realistic business data.",
            "Validate user roles, integrations, approvals, and output documents before sign-off.",
            "Confirm the business team accepts the fix and no downstream steps break.",
        ],
        "guardrails": [
            "Use business sign-off before production deployment for high-impact tickets.",
            "Record any failing test cases so they are not carried into production.",
        ],
    },
    "PROD": {
        "aliases": {"prod", "prd", "production"},
        "label": "Production",
        "guidance": [
            "Confirm business impact, user scope, and urgency before making changes.",
            "Use the safest corrective action first and follow approved change controls.",
            "Validate the fix with business users and monitor for recurrence after implementation.",
        ],
        "guardrails": [
            "Avoid risky direct changes, lock deletions, or mass reprocessing without impact review and approval.",
            "Coordinate communications, rollback planning, and timing for any transport or configuration change.",
        ],
    },
}


@lru_cache(maxsize=None)
def get_streamlit_secret(name):
    try:
        import streamlit as st
    except Exception:
        return None

    try:
        value = st.secrets.get(name)
    except Exception:
        return None

    if value in (None, ""):
        return None

    return str(value)


def get_config(name, default=None):
    value = os.getenv(name)
    if value not in (None, ""):
        return value

    secret_value = get_streamlit_secret(name)
    if secret_value not in (None, ""):
        return secret_value

    return default


def get_openai_model():
    return get_config("OPENAI_MODEL", "gpt-4.1-mini")


def get_ollama_base_url():
    return get_config("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")


def get_ollama_model():
    return get_config("OLLAMA_MODEL", "llama3")


def get_open_source_backend():
    return normalize_open_source_backend(get_config("OPEN_SOURCE_BACKEND", "auto"))


def get_openai_compatible_base_url():
    return str(get_config("OPEN_SOURCE_API_BASE_URL", "")).strip().rstrip("/")


def get_openai_compatible_model():
    return str(get_config("OPEN_SOURCE_API_MODEL", "")).strip()


def get_openai_compatible_api_key():
    return str(get_config("OPEN_SOURCE_API_KEY", "open-source-local")).strip()


def get_hf_local_model():
    return str(get_config("HF_LOCAL_MODEL", "")).strip()


def is_vector_context_enabled():
    value = str(get_config("ENABLE_VECTOR_CONTEXT", "0")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def normalize_token(token):
    cleaned = token.lower().strip(".,:;()[]{}")
    return TOKEN_ALIASES.get(cleaned, cleaned)


def normalize_open_source_backend(value):
    normalized = str(value or "").strip().lower()
    aliases = {
        "": "auto",
        "auto": "auto",
        "open_source": "auto",
        "oss": "auto",
        "ollama": "ollama",
        "local_api": "openai_compatible",
        "open_source_api": "openai_compatible",
        "openai_compatible": "openai_compatible",
        "compatible": "openai_compatible",
        "hf": "hf_local",
        "transformers": "hf_local",
        "hf_local": "hf_local",
    }
    return aliases.get(normalized, "auto")


def tokenize(text):
    return [normalize_token(token) for token in TOKEN_PATTERN.findall(text.lower())]


def normalize_text(text):
    return " ".join(tokenize(text))


def extract_tcodes(text):
    return {match.upper() for match in TCODE_PATTERN.findall(text.upper())}


def openai_is_configured():
    return bool(get_config("OPENAI_API_KEY"))


@lru_cache(maxsize=1)
def load_openai_client():
    api_key = get_config("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    kwargs = {
        "api_key": api_key,
        "max_retries": 0,
        "timeout": float(get_config("OPENAI_TIMEOUT_SECONDS", "12")),
    }
    organization = get_config("OPENAI_ORGANIZATION")
    project = get_config("OPENAI_PROJECT")
    if organization:
        kwargs["organization"] = organization
    if project:
        kwargs["project"] = project

    return OpenAI(**kwargs)


def runtime_status():
    open_source_backends = get_available_open_source_backends()
    return {
        "openai_configured": openai_is_configured(),
        "openai_model": get_openai_model(),
        "openai_last_error": get_openai_failure_notice(),
        "open_source_ready": bool(open_source_backends),
        "open_source_backend": get_open_source_backend(),
        "open_source_backends": ", ".join(open_source_backends) if open_source_backends else "None",
        "ollama_available": ollama_is_available(),
        "ollama_model": get_ollama_model(),
        "ollama_last_error": get_ollama_failure_notice(),
        "openai_compatible_available": openai_compatible_is_available(),
        "openai_compatible_model": get_openai_compatible_model() or "Not set",
        "openai_compatible_last_error": get_openai_compatible_failure_notice(),
        "hf_local_available": hf_local_is_available(),
        "hf_local_model": get_hf_local_model() or "Not set",
        "hf_local_last_error": get_hf_local_failure_notice(),
        "vector_index_present": INDEX_PATH.exists(),
        "vector_context_enabled": is_vector_context_enabled(),
        "sap_web_data_present": (BASE_DIR / "sap_web_data.txt").exists(),
        "sap_dataset_present": (BASE_DIR / "sap_dataset.txt").exists(),
    }


def calculate_note_score(query_terms, query_text, note):
    note_terms = Counter(token for token in tokenize(note) if token not in STOPWORDS)
    overlap = sum(min(query_terms[token], note_terms[token]) for token in query_terms)
    return overlap


@lru_cache(maxsize=1)
def load_local_notes():
    notes = []

    for filename in DATA_FILES:
        if not filename.exists():
            continue

        chunks = [chunk.strip() for chunk in filename.read_text(encoding="utf-8").split("\n\n")]
        notes.extend(chunk for chunk in chunks if chunk)

    return notes


def get_openai_failure_notice():
    return OPENAI_FAILURE_NOTICE


def set_openai_failure_notice(message):
    global OPENAI_FAILURE_NOTICE
    OPENAI_FAILURE_NOTICE = message


def clear_openai_failure_notice():
    global OPENAI_FAILURE_NOTICE
    OPENAI_FAILURE_NOTICE = None


def get_ollama_failure_notice():
    return OLLAMA_FAILURE_NOTICE


def set_ollama_failure_notice(message):
    global OLLAMA_FAILURE_NOTICE
    OLLAMA_FAILURE_NOTICE = message


def clear_ollama_failure_notice():
    global OLLAMA_FAILURE_NOTICE
    OLLAMA_FAILURE_NOTICE = None


def get_openai_compatible_failure_notice():
    return OPENAI_COMPATIBLE_FAILURE_NOTICE


def set_openai_compatible_failure_notice(message):
    global OPENAI_COMPATIBLE_FAILURE_NOTICE
    OPENAI_COMPATIBLE_FAILURE_NOTICE = message


def clear_openai_compatible_failure_notice():
    global OPENAI_COMPATIBLE_FAILURE_NOTICE
    OPENAI_COMPATIBLE_FAILURE_NOTICE = None


def get_hf_local_failure_notice():
    return HF_LOCAL_FAILURE_NOTICE


def set_hf_local_failure_notice(message):
    global HF_LOCAL_FAILURE_NOTICE
    HF_LOCAL_FAILURE_NOTICE = message


def clear_hf_local_failure_notice():
    global HF_LOCAL_FAILURE_NOTICE
    HF_LOCAL_FAILURE_NOTICE = None


@lru_cache(maxsize=1)
def fetch_ollama_tags():
    try:
        import requests
    except ImportError:
        return None

    try:
        response = requests.get(
            f"{get_ollama_base_url()}/api/tags",
            timeout=float(get_config("OLLAMA_TIMEOUT_SECONDS", "60")),
        )
        response.raise_for_status()
    except Exception:
        return None

    try:
        payload = response.json()
    except Exception:
        return None

    return payload.get("models", [])


def ollama_is_available():
    models = fetch_ollama_tags()
    if models is None:
        return False

    configured_model = get_ollama_model()
    if not configured_model:
        return False

    names = {
        model.get("name", "")
        for model in models
        if isinstance(model, dict)
    }
    bare_names = {name.split(":")[0] for name in names if name}
    return configured_model in names or configured_model in bare_names


def openai_compatible_is_configured():
    return bool(get_openai_compatible_base_url() and get_openai_compatible_model())


@lru_cache(maxsize=1)
def fetch_openai_compatible_models():
    if not openai_compatible_is_configured():
        return None

    try:
        import requests
    except ImportError:
        return None

    headers = {}
    api_key = get_openai_compatible_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = requests.get(
            f"{get_openai_compatible_base_url()}/models",
            headers=headers,
            timeout=float(get_config("OPEN_SOURCE_API_TIMEOUT_SECONDS", "5")),
        )
        response.raise_for_status()
    except Exception:
        return None

    try:
        payload = response.json()
    except Exception:
        return None

    return payload.get("data", [])


def openai_compatible_is_available():
    models = fetch_openai_compatible_models()
    if models is None:
        return False

    configured_model = get_openai_compatible_model()
    names = {
        str(model.get("id", "")).strip()
        for model in models
        if isinstance(model, dict)
    }
    return configured_model in names


@lru_cache(maxsize=1)
def load_openai_compatible_client():
    if not openai_compatible_is_configured():
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    return OpenAI(
        api_key=get_openai_compatible_api_key() or "open-source-local",
        base_url=get_openai_compatible_base_url(),
        max_retries=0,
        timeout=float(get_config("OPEN_SOURCE_API_TIMEOUT_SECONDS", "25")),
    )


def hf_local_is_configured():
    return bool(get_hf_local_model())


def hf_local_is_available():
    if not hf_local_is_configured():
        return False

    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False

    return True


@lru_cache(maxsize=1)
def load_hf_local_pipeline():
    if not hf_local_is_configured():
        return None

    from transformers import pipeline

    device_preference = str(get_config("HF_LOCAL_DEVICE", "cpu")).strip().lower()
    pipeline_kwargs = {
        "task": "text-generation",
        "model": get_hf_local_model(),
        "tokenizer": get_hf_local_model(),
    }

    if device_preference == "auto":
        pipeline_kwargs["device_map"] = "auto"
    elif device_preference == "cpu":
        pipeline_kwargs["device"] = -1
    elif device_preference.lstrip("-").isdigit():
        pipeline_kwargs["device"] = int(device_preference)

    return pipeline(**pipeline_kwargs)


def get_available_open_source_backends():
    backends = []
    if ollama_is_available():
        backends.append("ollama")
    if openai_compatible_is_available():
        backends.append("openai_compatible")
    if hf_local_is_available():
        backends.append("hf_local")
    return backends


def open_source_is_available():
    return bool(get_available_open_source_backends())


@lru_cache(maxsize=1)
def load_embeddings():
    embedding_cls = None

    try:
        from langchain_huggingface import HuggingFaceEmbeddings as NewEmbeddings

        embedding_cls = NewEmbeddings
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings as CommunityEmbeddings

            embedding_cls = CommunityEmbeddings
        except ImportError:
            return None

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        return embedding_cls(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            cache_folder=str(CACHE_DIR),
        )
    except TypeError:
        try:
            return embedding_cls(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
            )
        except Exception:
            return None
    except Exception:
        return None


@lru_cache(maxsize=1)
def load_vector_db():
    if not INDEX_PATH.exists():
        return None

    try:
        from langchain_community.vectorstores import FAISS
    except ImportError:
        return None

    embeddings = load_embeddings()
    if embeddings is None:
        return None

    try:
        return FAISS.load_local(
            str(INDEX_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except TypeError:
        try:
            return FAISS.load_local(str(INDEX_PATH), embeddings)
        except Exception:
            return None
    except Exception:
        return None


def phrase_points(phrase, query_text, query_tokens):
    normalized_phrase = normalize_text(phrase)
    phrase_tokens = [token for token in tokenize(phrase) if token not in STOPWORDS]
    if not phrase_tokens:
        return 0

    if normalized_phrase and normalized_phrase in query_text:
        return 12

    overlap = sum(1 for token in phrase_tokens if token in query_tokens)
    if overlap == len(phrase_tokens) and len(phrase_tokens) > 1:
        return 8 + overlap

    if overlap >= 2:
        return 4 + overlap

    if overlap == 1 and len(phrase_tokens) == 1:
        return 3

    return 0


def score_ticket(ticket, query, query_text, query_tokens, query_tcodes):
    score = 0
    reasons = []

    for signal in ticket["error_signals"]:
        points = phrase_points(signal, query_text, query_tokens)
        if points:
            score += points + 6
            reasons.append(f"matched error signal '{signal}'")

    for symptom in ticket["symptoms"]:
        points = phrase_points(symptom, query_text, query_tokens)
        if points:
            score += points + 3
            reasons.append(f"matched symptom '{symptom}'")

    title_points = phrase_points(ticket["title"], query_text, query_tokens)
    if title_points:
        score += title_points + 2
        reasons.append("matched the incident title")

    keyword_hits = [keyword for keyword in ticket["keywords"] if normalize_text(keyword) in query_text or normalize_token(keyword) in query_tokens]
    if keyword_hits:
        score += len(keyword_hits) * 2
        reasons.append(f"matched keywords {', '.join(keyword_hits[:4])}")

    matched_tcodes = sorted(query_tcodes.intersection(ticket["tcodes"]))
    if matched_tcodes:
        score += len(matched_tcodes) * 8
        reasons.append(f"matched T-code {', '.join(matched_tcodes)}")

    if ticket["area"].lower() in query_text:
        score += 3
        reasons.append(f"matched area '{ticket['area']}'")

    if query.strip().lower() == ticket["title"].strip().lower():
        score += 8
        reasons.append("exact title match")

    return {
        "ticket": ticket,
        "score": score,
        "reasons": reasons,
    }


def search_local_notes(query, top_k=3):
    filtered_query_tokens = [token for token in tokenize(query) if token not in STOPWORDS]
    query_terms = Counter(filtered_query_tokens)
    if not query_terms:
        return []

    scored_notes = []
    query_text = normalize_text(query)

    for note in load_local_notes():
        score = calculate_note_score(query_terms, query_text, note)
        if normalize_text(note) in query_text:
            score += 4
        if score >= 3:
            scored_notes.append((score, note))

    scored_notes.sort(key=lambda item: item[0], reverse=True)
    return [note for _, note in scored_notes[:top_k]]


def search_vector_context(query):
    if not is_vector_context_enabled():
        return []

    vector_db = load_vector_db()
    if vector_db is None:
        return []

    try:
        docs = vector_db.similarity_search(query, k=3)
    except Exception:
        return []

    snippets = []
    for doc in docs:
        content = getattr(doc, "page_content", "").strip()
        if content:
            snippets.append(content)
    return snippets


def find_ticket_matches(query, top_k=3):
    query_text = normalize_text(query)
    query_tokens = {token for token in tokenize(query) if token not in STOPWORDS}
    query_tcodes = extract_tcodes(query)

    scored = [
        score_ticket(ticket, query, query_text, query_tokens, query_tcodes)
        for ticket in TICKET_CATALOG
    ]
    scored = [item for item in scored if item["score"] > 0]
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def summarize_confidence(matches):
    if not matches:
        return "Low"

    top_score = matches[0]["score"]
    runner_up = matches[1]["score"] if len(matches) > 1 else 0

    if top_score >= 32 and top_score - runner_up >= 8:
        return "High"
    if top_score >= 18:
        return "Medium"
    return "Low"


def build_context(matches, query, include_vector=False):
    note_matches = search_local_notes(query)

    snippets = []
    sources = []

    if note_matches:
        snippets.extend(note_matches)
        sources.append("local SAP ticket notes")

    if include_vector:
        vector_matches = search_vector_context(query)
        if vector_matches:
            snippets.extend(vector_matches)
            sources.append("FAISS knowledge base")

    if matches:
        snippets.append(f"Matched playbook: {matches[0]['ticket']['title']}")
        sources.append("structured SAP ticket catalog")
    else:
        sources.append("structured SAP ticket catalog")

    return snippets[:4], ", ".join(dict.fromkeys(sources))


def format_list(items, default_message):
    if not items:
        return f"- {default_message}"
    return "\n".join(f"- {item}" for item in items)


def resolve_environment(environment, query):
    if environment:
        normalized = environment.strip().upper()
        if normalized in ENVIRONMENT_PROFILES or normalized == "ALL":
            return normalized

    query_tokens = set(tokenize(query))
    for key, profile in ENVIRONMENT_PROFILES.items():
        if profile["aliases"].intersection(query_tokens):
            return key

    return "ALL"


def build_environment_section(environment):
    if environment == "ALL":
        lines = []
        for key in ["DEV", "QA", "TEST", "PROD"]:
            profile = ENVIRONMENT_PROFILES[key]
            lines.append(f"- {key}: {profile['guidance'][0]}")
        return "Landscape Plan\n" + "\n".join(lines)

    profile = ENVIRONMENT_PROFILES[environment]
    return (
        f"Environment\n"
        f"- {environment} ({profile['label']})\n\n"
        f"Environment Guidance\n"
        f"{format_list(profile['guidance'], 'Follow approved support process for this system.')}\n\n"
        f"Environment Guardrails\n"
        f"{format_list(profile['guardrails'], 'Use the standard operational controls for this landscape.')}"
    )


def build_openai_prompt(query, environment, base_answer, context_snippets, context_source):
    context_block = "\n\n".join(context_snippets) if context_snippets else "No additional SAP context snippets were available."
    environment_name = environment or "ALL"

    return f"""
You are an SAP support lead helping resolve enterprise SAP tickets.

Use the local runbook answer as the primary source of truth. You may improve clarity,
reorder steps, and make the answer more actionable, but do not invent T-codes or
unsafe production actions. If the local answer is uncertain, say so clearly.

Ticket:
{query}

SAP landscape:
{environment_name}

Local runbook answer:
{base_answer}

Retrieved SAP context:
{context_block}

Context sources:
{context_source}

    Return a concise ticket-resolution playbook with these sections:
- Incident
- Root Cause
- T-codes
- Checks
- Fix Plan
- Risks / Escalation
""".strip()


def shorten_text(text, limit=500):
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    shortened = compact[:limit].rsplit(" ", 1)[0].strip()
    return f"{shortened}..."


def condense_playbook_for_local_llm(base_answer):
    important_sections = {
        "Incident",
        "Likely Root Cause",
        "Environment",
        "Environment Guidance",
        "Best T-codes",
        "Checks",
        "Resolution",
        "Escalate If",
    }
    sections = []
    current_header = None
    current_items = []

    for raw_line in base_answer.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if not line.startswith("- "):
            if current_header is not None:
                sections.append((current_header, current_items))
            current_header = line
            current_items = []
            continue

        if current_header is None:
            continue

        if len(current_items) < 2:
            current_items.append(line)

    if current_header is not None:
        sections.append((current_header, current_items))

    if not sections:
        return shorten_text(base_answer, limit=1200)

    condensed_lines = []
    for header, items in sections:
        if header not in important_sections:
            continue
        condensed_lines.append(header)
        condensed_lines.extend(items[:2] or ["- No details available."])

    condensed = "\n".join(condensed_lines).strip()
    return shorten_text(condensed or base_answer, limit=1500)


def build_ollama_prompt(query, environment, base_answer, context_snippets, context_source):
    condensed_answer = condense_playbook_for_local_llm(base_answer)
    context_block = (
        shorten_text(context_snippets[0], limit=400)
        if context_snippets
        else "No additional SAP context snippets were available."
    )
    environment_name = environment or "ALL"

    return f"""
You are improving an SAP ticket resolution draft for a local Ollama model.

Rules:
- Keep the answer concise and practical.
- Preserve the same structure and headings from the draft answer.
- Use short bullets and at most 2 bullets per section.
- Do not invent unsafe production actions or fake T-codes.
- If the draft is already solid, lightly improve wording only.

Ticket:
{query}

SAP landscape:
{environment_name}

Draft answer:
{condensed_answer}

Optional SAP context:
{context_block}

Context source:
{context_source}

Return only the improved answer.
""".strip()


def try_openai_enhancement(query, environment, base_answer, context_snippets, context_source):
    client = load_openai_client()
    if client is None:
        return None

    failure_notice = get_openai_failure_notice()
    if failure_notice:
        return None

    prompt = build_openai_prompt(query, environment, base_answer, context_snippets, context_source)

    try:
        response = client.responses.create(
            model=get_openai_model(),
            input=[
                {
                    "role": "system",
                    "content": "You generate safe, practical SAP support playbooks from supplied context.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_output_tokens=1200,
        )
    except Exception as exc:
        notice = summarize_openai_failure(exc)
        set_openai_failure_notice(notice)
        return None

    output_text = getattr(response, "output_text", "") or ""
    clear_openai_failure_notice()
    text = output_text.strip()
    return text or None


def enhance_answer_with_openai(query, environment, base_answer, context_snippets, context_source):
    answer = try_openai_enhancement(
        query,
        environment,
        base_answer,
        context_snippets,
        context_source,
    )
    if answer:
        return answer

    notice = get_openai_failure_notice()
    if notice:
        return f"{notice}\n\n{base_answer}"

    return base_answer


def summarize_openai_failure(exc):
    message = str(exc).strip()
    lowered = message.lower()

    if "insufficient_quota" in lowered or "quota" in lowered:
        return "OpenAI is configured, but the current API project is out of quota. Showing the local SAP runbook answer instead."
    if "429" in lowered or "rate limit" in lowered:
        return "OpenAI is temporarily rate limited. Showing the local SAP runbook answer instead."
    if "401" in lowered or "invalid api key" in lowered:
        return "OpenAI authentication failed. Showing the local SAP runbook answer instead."
    if "timeout" in lowered:
        return "OpenAI timed out. Showing the local SAP runbook answer instead."

    return "OpenAI enhancement is currently unavailable. Showing the local SAP runbook answer instead."


def try_ollama_enhancement(query, environment, base_answer, context_snippets, context_source):
    if not ollama_is_available():
        return None

    try:
        import requests
    except ImportError:
        set_ollama_failure_notice("The local Ollama integration requires the 'requests' package, which is missing.")
        return None

    prompt = build_ollama_prompt(query, environment, base_answer, context_snippets, context_source)

    try:
        response = requests.post(
            f"{get_ollama_base_url()}/api/generate",
            json={
                "model": get_ollama_model(),
                "stream": False,
                "prompt": prompt,
                "options": {
                    "temperature": 0.1,
                    "num_ctx": 2048,
                    "num_predict": 220,
                },
            },
            timeout=float(get_config("OLLAMA_TIMEOUT_SECONDS", "70")),
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        notice = summarize_ollama_failure(exc)
        set_ollama_failure_notice(notice)
        return None

    clear_ollama_failure_notice()
    text = clean_ollama_response(str(payload.get("response", "")))
    return text or None


def enhance_answer_with_ollama(query, environment, base_answer, context_snippets, context_source):
    answer = try_ollama_enhancement(
        query,
        environment,
        base_answer,
        context_snippets,
        context_source,
    )
    if answer:
        return answer

    notice = get_ollama_failure_notice()
    if notice:
        return f"{notice}\n\n{base_answer}"

    return base_answer


def summarize_ollama_failure(exc):
    message = str(exc).strip()
    lowered = message.lower()

    if "connection refused" in lowered or "failed to establish a new connection" in lowered:
        return "Ollama is not reachable at the configured local endpoint. Showing the local SAP runbook answer instead."
    if "404" in lowered:
        return f"Ollama model '{get_ollama_model()}' is not available. Showing the local SAP runbook answer instead."
    if "timeout" in lowered:
        return "Ollama timed out while generating the answer. Showing the local SAP runbook answer instead."

    return "Ollama enhancement is currently unavailable. Showing the local SAP runbook answer instead."


def clean_ollama_response(text):
    header_names = {
        "Incident",
        "Likely Root Cause",
        "Environment",
        "Guidance",
        "Best T-codes",
        "Checks",
        "Resolution",
        "Escalate If",
    }
    cleaned_lines = []
    last_line_was_header = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            cleaned_lines.append("")
            last_line_was_header = False
            continue

        normalized = line.replace("•", "- ").replace("–", "- ").replace("—", "- ")
        if normalized.lower() in {
            "here is the improved answer:",
            "improved answer:",
            "final answer:",
        }:
            continue
        if re.match(r"^-?\s*(here is|below is).*(answer|draft)", normalized, re.IGNORECASE):
            continue
        inline_header = re.match(
            r"^(Incident|Likely Root Cause|Environment|Guidance|Best T-codes|Checks|Resolution|Escalate If)\s*[:-]\s*(.+)$",
            normalized,
        )
        if inline_header:
            cleaned_lines.append(inline_header.group(1))
            remainder = inline_header.group(2).strip()
            remainder_parts = [part.strip() for part in re.split(r"\s+-\s+", remainder) if part.strip()]
            for part in remainder_parts or [remainder]:
                cleaned_lines.append(f"- {part}")
            last_line_was_header = False
            continue
        if normalized.endswith(":") and normalized[:-1] in header_names:
            cleaned_lines.append(normalized[:-1])
            last_line_was_header = True
            continue
        if last_line_was_header and not normalized.startswith("-"):
            normalized = f"- {normalized}"
        if normalized.startswith("-"):
            normalized = f"- {normalized[1:].strip()}"
        elif ":" in normalized and normalized.split(":", 1)[0] not in header_names:
            normalized = f"- {normalized}"
        cleaned_lines.append(normalized)
        last_line_was_header = normalized in header_names

    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned or text.strip()


def get_open_source_backend_order(backend=None):
    selected_backend = normalize_open_source_backend(
        backend if backend not in (None, "") else get_open_source_backend()
    )
    if selected_backend == "auto":
        return ["ollama", "openai_compatible", "hf_local"]
    return [selected_backend]


def try_openai_compatible_enhancement(query, environment, base_answer, context_snippets, context_source):
    if not openai_compatible_is_configured():
        return None

    client = load_openai_compatible_client()
    if client is None:
        set_openai_compatible_failure_notice(
            "The OpenAI-compatible open-source backend could not be initialized."
        )
        return None

    prompt = build_ollama_prompt(query, environment, base_answer, context_snippets, context_source)

    try:
        response = client.chat.completions.create(
            model=get_openai_compatible_model(),
            messages=[
                {
                    "role": "system",
                    "content": "You generate safe, concise SAP ticket resolution playbooks.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.1,
            max_tokens=int(get_config("OPEN_SOURCE_API_MAX_TOKENS", "260")),
        )
        text = (
            response.choices[0].message.content
            if getattr(response, "choices", None)
            else ""
        )
    except Exception as exc:
        set_openai_compatible_failure_notice(summarize_openai_compatible_failure(exc))
        return None

    clear_openai_compatible_failure_notice()
    return clean_ollama_response(str(text or ""))


def enhance_answer_with_openai_compatible(query, environment, base_answer, context_snippets, context_source):
    answer = try_openai_compatible_enhancement(
        query,
        environment,
        base_answer,
        context_snippets,
        context_source,
    )
    if answer:
        return answer

    notice = get_openai_compatible_failure_notice()
    if notice:
        return f"{notice}\n\n{base_answer}"

    return base_answer


def summarize_openai_compatible_failure(exc):
    message = str(exc).strip()
    lowered = message.lower()

    if "connection refused" in lowered or "failed to establish a new connection" in lowered:
        return "The OpenAI-compatible local model endpoint is not reachable. Showing the local SAP runbook answer instead."
    if "404" in lowered:
        return "The configured OpenAI-compatible model or endpoint was not found. Showing the local SAP runbook answer instead."
    if "401" in lowered or "unauthorized" in lowered or "invalid api key" in lowered:
        return "Authentication failed for the OpenAI-compatible local model endpoint. Showing the local SAP runbook answer instead."
    if "timeout" in lowered:
        return "The OpenAI-compatible local model endpoint timed out. Showing the local SAP runbook answer instead."

    return "The OpenAI-compatible open-source backend is currently unavailable. Showing the local SAP runbook answer instead."


def try_hf_local_enhancement(query, environment, base_answer, context_snippets, context_source):
    if not hf_local_is_configured():
        return None

    try:
        generator = load_hf_local_pipeline()
        if generator is None:
            raise RuntimeError("No Hugging Face local generation pipeline could be created.")
    except Exception as exc:
        set_hf_local_failure_notice(summarize_hf_local_failure(exc))
        return None

    prompt = build_ollama_prompt(query, environment, base_answer, context_snippets, context_source)

    try:
        outputs = generator(
            prompt,
            max_new_tokens=int(get_config("HF_LOCAL_MAX_NEW_TOKENS", "220")),
            do_sample=False,
            return_full_text=False,
        )
    except Exception as exc:
        set_hf_local_failure_notice(summarize_hf_local_failure(exc))
        return None

    clear_hf_local_failure_notice()
    text = ""
    if outputs and isinstance(outputs, list):
        first = outputs[0]
        if isinstance(first, dict):
            text = str(first.get("generated_text", "") or first.get("text", ""))
    return clean_ollama_response(text)


def enhance_answer_with_hf_local(query, environment, base_answer, context_snippets, context_source):
    answer = try_hf_local_enhancement(
        query,
        environment,
        base_answer,
        context_snippets,
        context_source,
    )
    if answer:
        return answer

    notice = get_hf_local_failure_notice()
    if notice:
        return f"{notice}\n\n{base_answer}"

    return base_answer


def summarize_hf_local_failure(exc):
    message = str(exc).strip()
    lowered = message.lower()

    if "pytorch" in lowered or "tensorflow" in lowered or "flax" in lowered:
        return "The Hugging Face local backend needs a supported model runtime such as PyTorch. Showing the local SAP runbook answer instead."
    if "out of memory" in lowered or "cuda" in lowered:
        return "The Hugging Face local backend ran out of memory or device capacity. Showing the local SAP runbook answer instead."
    if "not found" in lowered or "404" in lowered:
        return "The configured Hugging Face local model could not be found. Showing the local SAP runbook answer instead."
    if "timeout" in lowered:
        return "The Hugging Face local backend timed out. Showing the local SAP runbook answer instead."

    return "The Hugging Face local backend is currently unavailable. Showing the local SAP runbook answer instead."


def try_open_source_enhancement(
    query,
    environment,
    base_answer,
    context_snippets,
    context_source,
    backend=None,
):
    for candidate in get_open_source_backend_order(backend):
        if candidate == "ollama":
            answer = try_ollama_enhancement(
                query,
                environment,
                base_answer,
                context_snippets,
                context_source,
            )
            if answer:
                return answer
        elif candidate == "openai_compatible":
            answer = try_openai_compatible_enhancement(
                query,
                environment,
                base_answer,
                context_snippets,
                context_source,
            )
            if answer:
                return answer
        elif candidate == "hf_local":
            answer = try_hf_local_enhancement(
                query,
                environment,
                base_answer,
                context_snippets,
                context_source,
            )
            if answer:
                return answer

    return None


def get_open_source_failure_notice(backend=None):
    backend_order = get_open_source_backend_order(backend)
    notice_map = {
        "ollama": get_ollama_failure_notice(),
        "openai_compatible": get_openai_compatible_failure_notice(),
        "hf_local": get_hf_local_failure_notice(),
    }
    for candidate in backend_order:
        if notice_map.get(candidate):
            return notice_map[candidate]
    return None


def enhance_answer_with_open_source(
    query,
    environment,
    base_answer,
    context_snippets,
    context_source,
    backend=None,
):
    answer = try_open_source_enhancement(
        query,
        environment,
        base_answer,
        context_snippets,
        context_source,
        backend=backend,
    )
    if answer:
        return answer

    notice = get_open_source_failure_notice(backend)
    if notice:
        return f"{notice}\n\n{base_answer}"

    return base_answer


def build_ticket_answer(query, environment):
    matches = find_ticket_matches(query)
    context_snippets, context_source = build_context(matches, query, include_vector=False)
    resolved_environment = resolve_environment(environment, query)

    if not matches or matches[0]["score"] < 10:
        return build_generic_triage_answer(query, context_snippets, context_source, resolved_environment)

    best_match = matches[0]
    ticket = best_match["ticket"]
    confidence = summarize_confidence(matches)
    related = [match["ticket"]["title"] for match in matches[1:] if match["score"] >= 10]

    reason_lines = best_match["reasons"][:3] or ["matched the closest SAP incident pattern available"]
    context_preview = context_snippets[0] if context_snippets else "No additional note snippet was available."

    return f"""Incident
- {ticket['title']}
- Area: {ticket['area']}
- Confidence: {confidence}

Likely Root Cause
- {ticket['root_cause']}

{build_environment_section(resolved_environment)}

Best T-codes
{format_list(ticket['tcodes'], "No T-code available")}

Checks
{format_list(ticket['checks'], "Capture the exact error and validate the impacted transaction.")}

Resolution
{format_list(ticket['resolution'], "Apply the safest validated fix and retest.")}

Escalate If
{format_list(ticket['escalate_if'], "Escalate when the issue affects production or cannot be reproduced safely.")}

Why This Matched
{format_list(reason_lines, "matched the closest catalog runbook")}

Supporting Context
- Source: {context_source}
- Snippet: {context_preview}

Related Playbooks
{format_list(related, "No close secondary match found.")}"""


def build_generic_triage_answer(query, context_snippets, context_source, environment):
    query_tcodes = sorted(extract_tcodes(query))
    default_checks = [
        "Capture the exact SAP error text, screenshot, and business step that failed.",
        "Identify the affected user, client, company code, plant, and document number if applicable.",
        "Check the most relevant technical traces such as SU53, ST22, SM21, SM13, SM37, or SM59 based on the symptom.",
    ]
    if query_tcodes:
        default_checks.insert(1, f"Validate the mentioned transaction code(s): {', '.join(query_tcodes)}.")

    return f"""Incident
- Unclassified SAP ticket
- Area: Needs more detail
- Confidence: Low

Likely Root Cause
- The current ticket text does not strongly match a single runbook in the local SAP catalog.

{build_environment_section(environment)}

Best T-codes
- SU53
- ST22
- SM21
- SM37

Checks
{format_list(default_checks, "Collect enough detail to classify the incident.")}

Resolution
- Start with the error log or failed transaction that reproduces the issue.
- Gather the exact technical message and route the ticket to the right module after basic checks.
- Add ticket details such as module, transaction, error code, and impact to improve the diagnosis.

Escalate If
- The issue blocks production users or financial/logistics postings.
- The ticket requires functional customizing or ABAP changes.

Supporting Context
- Source: {context_source}
- Snippet: {context_snippets[0] if context_snippets else 'No supporting note snippet was available.'}"""


def ask_sap(query, environment=None, provider="auto"):
    clean_query = query.strip()
    if not clean_query:
        return "Please enter an SAP ticket, issue, or question."

    provider = str(provider or "auto").strip().lower()
    provider_aliases = {
        "oss": "open_source",
        "open_source_api": "openai_compatible",
        "local_api": "openai_compatible",
        "hf": "hf_local",
        "transformers": "hf_local",
    }
    provider = provider_aliases.get(provider, provider)

    resolved_environment = resolve_environment(environment, clean_query)
    base_answer = build_ticket_answer(clean_query, resolved_environment)

    if provider == "rules":
        return base_answer

    if provider == "openai" and not openai_is_configured():
        return (
            "OpenAI mode was selected, but OPENAI_API_KEY is not configured.\n\n"
            f"{base_answer}"
        )

    if provider == "ollama" and not ollama_is_available():
        return (
            "Ollama mode was selected, but the configured Ollama model is not available at the local endpoint.\n\n"
            f"{base_answer}"
        )

    if provider == "open_source" and not open_source_is_available():
        return (
            "Open Source AI mode was selected, but no supported open-source backend is available. Configure Ollama, an OpenAI-compatible local API, or a Hugging Face local model.\n\n"
            f"{base_answer}"
        )

    if provider == "openai_compatible" and not openai_compatible_is_configured():
        return (
            "OpenAI-compatible mode was selected, but OPEN_SOURCE_API_BASE_URL and OPEN_SOURCE_API_MODEL are not configured.\n\n"
            f"{base_answer}"
        )

    if provider == "hf_local" and not hf_local_is_configured():
        return (
            "HF local mode was selected, but HF_LOCAL_MODEL is not configured.\n\n"
            f"{base_answer}"
        )

    matches = find_ticket_matches(clean_query)
    context_snippets, context_source = build_context(
        matches,
        clean_query,
        include_vector=is_vector_context_enabled(),
    )

    if provider == "openai" and openai_is_configured():
        return enhance_answer_with_openai(
            clean_query,
            resolved_environment,
            base_answer,
            context_snippets,
            context_source,
        )

    if provider == "ollama" and ollama_is_available():
        return enhance_answer_with_ollama(
            clean_query,
            resolved_environment,
            base_answer,
            context_snippets,
            context_source,
        )

    if provider == "open_source":
        return enhance_answer_with_open_source(
            clean_query,
            resolved_environment,
            base_answer,
            context_snippets,
            context_source,
            backend=get_open_source_backend(),
        )

    if provider == "openai_compatible":
        return enhance_answer_with_open_source(
            clean_query,
            resolved_environment,
            base_answer,
            context_snippets,
            context_source,
            backend="openai_compatible",
        )

    if provider == "hf_local":
        return enhance_answer_with_open_source(
            clean_query,
            resolved_environment,
            base_answer,
            context_snippets,
            context_source,
            backend="hf_local",
        )

    if provider == "auto" and openai_is_configured():
        openai_answer = try_openai_enhancement(
            clean_query,
            resolved_environment,
            base_answer,
            context_snippets,
            context_source,
        )
        if openai_answer:
            return openai_answer
        if get_openai_failure_notice():
            return f"{get_openai_failure_notice()}\n\n{base_answer}"

    if provider == "auto" and not openai_is_configured() and open_source_is_available():
        open_source_answer = try_open_source_enhancement(
            clean_query,
            resolved_environment,
            base_answer,
            context_snippets,
            context_source,
        )
        if open_source_answer:
            return open_source_answer

    if provider == "auto":
        open_source_notice = get_open_source_failure_notice()
        if open_source_notice:
            return f"{open_source_notice}\n\n{base_answer}"
        return base_answer

    return base_answer


if __name__ == "__main__":
    print(ask_sap("Transport failed with RC 8 because object missing"))
