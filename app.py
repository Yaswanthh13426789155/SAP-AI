from collections import Counter
from functools import lru_cache
import os
from pathlib import Path
import re

from dotenv import load_dotenv

from sap_ticket_catalog import TICKET_CATALOG


load_dotenv(Path(__file__).resolve().parent / ".env")


DATA_FILES = [
    "sap_tickets.txt",
    "sap_dataset.txt",
    "sap_web_data.txt",
]
INDEX_PATH = Path("sap_index")
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_/.-]+")
TCODE_PATTERN = re.compile(r"\b[A-Z]{2,5}\d{1,4}[A-Z]?\b")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
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


def normalize_token(token):
    cleaned = token.lower().strip(".,:;()[]{}")
    return TOKEN_ALIASES.get(cleaned, cleaned)


def tokenize(text):
    return [normalize_token(token) for token in TOKEN_PATTERN.findall(text.lower())]


def normalize_text(text):
    return " ".join(tokenize(text))


def extract_tcodes(text):
    return {match.upper() for match in TCODE_PATTERN.findall(text.upper())}


def openai_is_configured():
    return bool(os.getenv("OPENAI_API_KEY"))


@lru_cache(maxsize=1)
def load_openai_client():
    if not openai_is_configured():
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    kwargs = {}
    if os.getenv("OPENAI_ORGANIZATION"):
        kwargs["organization"] = os.getenv("OPENAI_ORGANIZATION")
    if os.getenv("OPENAI_PROJECT"):
        kwargs["project"] = os.getenv("OPENAI_PROJECT")

    return OpenAI(**kwargs)


def runtime_status():
    return {
        "openai_configured": openai_is_configured(),
        "openai_model": DEFAULT_OPENAI_MODEL,
        "vector_index_present": INDEX_PATH.exists(),
        "sap_web_data_present": Path("sap_web_data.txt").exists(),
        "sap_dataset_present": Path("sap_dataset.txt").exists(),
    }


def calculate_note_score(query_terms, query_text, note):
    note_terms = Counter(token for token in tokenize(note) if token not in STOPWORDS)
    overlap = sum(min(query_terms[token], note_terms[token]) for token in query_terms)
    return overlap


@lru_cache(maxsize=1)
def load_local_notes():
    notes = []

    for filename in DATA_FILES:
        path = Path(filename)
        if not path.exists():
            continue

        chunks = [chunk.strip() for chunk in path.read_text(encoding="utf-8").split("\n\n")]
        notes.extend(chunk for chunk in chunks if chunk)

    return notes


@lru_cache(maxsize=1)
def load_vector_db():
    if not INDEX_PATH.exists():
        return None

    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
    except ImportError:
        return None

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
    except Exception:
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


def build_context(matches, query):
    note_matches = search_local_notes(query)
    vector_matches = search_vector_context(query)

    snippets = []
    sources = []

    if note_matches:
        snippets.extend(note_matches)
        sources.append("local SAP ticket notes")

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


def enhance_answer_with_openai(query, environment, base_answer, context_snippets, context_source):
    client = load_openai_client()
    if client is None:
        return base_answer

    prompt = build_openai_prompt(query, environment, base_answer, context_snippets, context_source)

    try:
        response = client.responses.create(
            model=DEFAULT_OPENAI_MODEL,
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
    except Exception:
        return base_answer

    output_text = getattr(response, "output_text", "") or ""
    return output_text.strip() or base_answer


def build_ticket_answer(query, environment):
    matches = find_ticket_matches(query)
    context_snippets, context_source = build_context(matches, query)
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

    base_answer = build_ticket_answer(clean_query, environment)

    if provider == "rules":
        return base_answer

    if provider == "openai" and not openai_is_configured():
        return (
            "OpenAI mode was selected, but OPENAI_API_KEY is not configured.\n\n"
            f"{base_answer}"
        )

    if provider in {"auto", "openai"} and openai_is_configured():
        matches = find_ticket_matches(clean_query)
        context_snippets, context_source = build_context(matches, clean_query)
        return enhance_answer_with_openai(
            clean_query,
            resolve_environment(environment, clean_query),
            base_answer,
            context_snippets,
            context_source,
        )

    return base_answer


if __name__ == "__main__":
    print(ask_sap("Transport failed with RC 8 because object missing"))
