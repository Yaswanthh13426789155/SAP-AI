from collections import Counter
from functools import lru_cache
from importlib.util import find_spec
import os
from pathlib import Path
import re

from dotenv import load_dotenv

from sap_intelligence import neural_nlp_is_available, ocr_is_available
from sap_landscape import get_landscape_counts, has_landscape_override, resolve_system_context
from sap_reasoner import build_advanced_reasoning
from sap_ticket_catalog import TICKET_CATALOG
from sap_training import load_training_status, predict_ticket_candidates, router_model_available


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
RUNBOOK_HEADINGS = {
    "Agent Mode",
    "Objective",
    "Investigation Plan",
    "Expert Assessment",
    "Failure Boundary",
    "Dependency Map",
    "Tool Findings",
    "Layer Coordination",
    "Evidence Correlation",
    "Hypothesis Ranking",
    "Autonomous Next Step",
    "Validation Gate",
    "Safe Change Plan",
    "Specialist Handoff",
    "Open Questions",
    "Incident",
    "Likely Root Cause",
    "Priority",
    "Likely Owner",
    "Required Inputs",
    "Environment",
    "Landscape Plan",
    "Environment Guidance",
    "Environment Guardrails",
    "System",
    "Subsystem",
    "System Context",
    "Integration Points",
    "Integration Guidance",
    "Integration Specifications",
    "NLP Signals",
    "Neural Matches",
    "Image Findings",
    "Issue Mix",
    "Parallel Workstreams",
    "Cross-Issue Risks",
    "Advanced Diagnosis",
    "Failure Chain",
    "Decision Path",
    "Guidance",
    "Best T-codes",
    "Checks",
    "Resolution",
    "Fix Plan",
    "Escalate If",
    "Risks / Escalation",
    "Why This Matched",
    "Supporting Context",
    "Related Playbooks",
}
RUNBOOK_HEADING_ALIASES = {
    "Root Cause": "Likely Root Cause",
    "T-codes": "Best T-codes",
    "T Codes": "Best T-codes",
    "Fix Plan": "Resolution",
    "System Scope": "System",
    "Subsystem Scope": "Subsystem",
    "Integration Targets": "Integration Points",
    "Interface Specifications": "Integration Specifications",
    "Integration Specs": "Integration Specifications",
    "OCR Findings": "Image Findings",
    "Workstreams": "Parallel Workstreams",
    "Cross-System Risks": "Cross-Issue Risks",
    "Reasoning": "Advanced Diagnosis",
    "Diagnostic Path": "Decision Path",
}
AREA_OWNERS = {
    "Basis": [
        "SAP Basis support",
        "SAP operations or infrastructure team for host, database, or instance issues",
    ],
    "Security": [
        "SAP security or authorization team",
        "GRC or access approver if role changes need approval",
    ],
    "Integration": [
        "SAP integration or middleware team",
        "Basis support if RFC, qRFC, or gateway connectivity is involved",
    ],
    "FI": [
        "SAP FI functional support",
        "Basis or ABAP team if posting fails because of technical dumps or updates",
    ],
    "MM": [
        "SAP MM functional support",
        "Basis team if the failure is technical rather than process or master-data related",
    ],
    "SD": [
        "SAP SD functional support",
        "ABAP or integration team if pricing logic, outputs, or interfaces are custom",
    ],
    "MM/SD": [
        "SAP MM or SD functional support depending on the document flow affected",
        "ABAP or Basis support if the failure is technical rather than transactional",
    ],
    "Fiori/Gateway": [
        "SAP Fiori or Gateway support team",
        "Security or backend application support if the app failure is role or service dependent",
    ],
    "Analytics": [
        "SAP BW/4HANA or analytics support team",
        "Source-system or Basis support if extraction, queues, or process chains are failing",
    ],
    "HANA / DB": [
        "SAP HANA database support",
        "Basis or infrastructure team for host, memory, replication, or backup issues",
    ],
    "Workflow / MDG": [
        "SAP workflow or MDG functional support",
        "ABAP or Basis support if the workflow runtime or replication layer is failing technically",
    ],
    "ALM / Governance": [
        "SAP Solution Manager, ChaRM, or GRC support team",
        "Basis or security support if the issue is caused by connectors, roles, or transport governance",
    ],
    "Cross-System": [
        "SAP application support",
        "Basis, integration, or functional owner depending on the first failing component identified",
    ],
}
UNIVERSAL_SUPPORT_PATTERNS = [
    {
        "id": "authorization",
        "title": "Authorization or role issue",
        "area": "Security",
        "signals": ["authorization", "not authorized", "access denied", "su53", "role", "pfcg"],
        "tcodes": ["SU53", "PFCG", "SUIM", "ST01"],
        "checks": [
            "Capture the failed business step and run SU53 immediately for the affected user.",
            "Compare the missing authorization object with the assigned role design in PFCG.",
            "If the issue is unclear, trace the failed check with ST01.",
        ],
        "resolution": [
            "Add or adjust the required authorization in the correct role.",
            "Regenerate and transport the role only after security approval if needed.",
            "Retest the exact failed action with the same user and context.",
        ],
        "inputs": ["affected user", "failed transaction or app", "SU53 output", "role name"],
        "causes": [
            "A required authorization object is missing from the assigned role.",
            "The user buffer or derived role assignment is outdated.",
        ],
    },
    {
        "id": "transport_change",
        "title": "Transport, release, or environment mismatch issue",
        "area": "Basis",
        "signals": ["transport", "release", "import", "stms", "rc ", "return code", "object missing"],
        "tcodes": ["STMS", "SE09", "SE10", "SE03"],
        "checks": [
            "Review the import or release log to find the failing object and phase.",
            "Check transport sequence, prerequisites, and object lock or repair status.",
            "Compare behavior across DEV, QA, TEST, and PROD for the same change.",
        ],
        "resolution": [
            "Import prerequisites first or repair the transport sequence.",
            "Re-release or rebuild the request if the object list is inconsistent.",
            "Retest after confirming the transported objects are active in the target environment.",
        ],
        "inputs": ["transport number", "target system", "import log text", "failing object"],
        "causes": [
            "A prerequisite change is missing or out of sequence.",
            "The transport contains inconsistent or inactive objects.",
        ],
    },
    {
        "id": "interface_connectivity",
        "title": "Interface, RFC, IDoc, or queue processing issue",
        "area": "Integration",
        "signals": ["idoc", "rfc", "sm59", "queue", "qrfc", "smq1", "smq2", "bd87", "we02", "status 51"],
        "tcodes": ["WE02", "WE05", "BD87", "SM59", "SM58", "SMQ1", "SMQ2"],
        "checks": [
            "Identify the exact message, partner, destination, queue, or IDoc status record.",
            "Check whether the failure is master data, credentials, connectivity, or posting logic.",
            "Review retries, stuck queues, and downstream acknowledgements before reprocessing.",
        ],
        "resolution": [
            "Fix the root cause first, then reprocess only the failed IDoc, tRFC, or queue entries.",
            "Validate middleware, credentials, and partner profile settings if connectivity is involved.",
            "Monitor the backlog until the interface drains successfully.",
        ],
        "inputs": ["IDoc number or queue name", "destination", "message text", "partner or interface name"],
        "causes": [
            "Interface processing is blocked by application, credential, or connectivity failures.",
            "The message cannot post because of missing master data or downstream errors.",
        ],
    },
    {
        "id": "background_job",
        "title": "Background job, batch, or scheduling issue",
        "area": "Basis",
        "signals": ["job", "batch", "sm37", "sm36", "variant", "cancelled", "scheduler", "spool"],
        "tcodes": ["SM37", "SM36", "SE38", "SA38", "SU53"],
        "checks": [
            "Review the job log, spool, runtime user, and variant in SM37.",
            "Identify whether the failure is authorization, variant, spool, or downstream application logic.",
            "Check recent transports or schedule changes that may have broken the job definition.",
        ],
        "resolution": [
            "Correct the missing authorization, variant, or spool setup.",
            "Reschedule or rerun the job in a controlled way after fixing the root cause.",
            "Confirm that the output and dependent follow-on steps complete successfully.",
        ],
        "inputs": ["job name", "run timestamp", "technical user", "job log text", "variant name"],
        "causes": [
            "The scheduled job references invalid parameters or missing authorizations.",
            "A downstream process or spool/output dependency is failing during runtime.",
        ],
    },
    {
        "id": "performance_capacity",
        "title": "Performance, workload, or system capacity issue",
        "area": "Basis",
        "signals": ["slow", "performance", "cpu", "memory", "dump", "st22", "response time", "work process"],
        "tcodes": ["ST03N", "ST06", "ST22", "SM50", "SM66", "DBACOCKPIT"],
        "checks": [
            "Identify whether the issue is isolated to one transaction, one user group, or the whole system.",
            "Check workload, resource usage, dumps, and long-running work processes.",
            "Correlate the slowdown with jobs, locks, expensive SQL, or infrastructure alarms.",
        ],
        "resolution": [
            "Contain runaway jobs or blocking processes before changing system parameters.",
            "Tune the responsible workload, SQL, or memory settings based on the evidence collected.",
            "Retest business response times after each corrective action.",
        ],
        "inputs": ["affected transaction", "time window", "number of users impacted", "dump or alert text"],
        "causes": [
            "Resource saturation, expensive SQL, or blocked processes are degrading response time.",
            "An abnormal workload spike or code path is exhausting work processes or memory.",
        ],
    },
    {
        "id": "financial_posting",
        "title": "FI posting, payment, or document control issue",
        "area": "FI",
        "signals": ["invoice", "payment", "fb60", "f110", "posting", "ob52", "fb08", "number range", "fbn1"],
        "tcodes": ["FB03", "FB08", "FB60", "F110", "OB52", "FBN1"],
        "checks": [
            "Capture the document type, company code, fiscal period, and exact posting error.",
            "Check whether posting period, payment setup, or number range maintenance is blocking the transaction.",
            "Confirm whether the document is already cleared, reversed, or partially processed.",
        ],
        "resolution": [
            "Fix the financial control point such as posting period, number range, or payment master data.",
            "Reverse or repost only after confirming finance controls and downstream dependencies.",
            "Validate the final accounting document and related subledger impact after correction.",
        ],
        "inputs": ["company code", "document number", "fiscal period", "vendor or customer", "error text"],
        "causes": [
            "Financial control settings or master data are blocking the posting flow.",
            "The document needs reversal, reset, or period maintenance before reposting.",
        ],
    },
    {
        "id": "logistics_master_data",
        "title": "MM or SD process blocked by master data, pricing, or stock conditions",
        "area": "MM",
        "signals": ["migo", "material", "stock", "pricing", "condition", "va01", "vl02n", "purchase", "release"],
        "tcodes": ["MIGO", "MMBE", "ME29N", "ME28", "VA01", "VA02", "VK11", "VL02N"],
        "checks": [
            "Capture the exact transaction, material or document, and business step that failed.",
            "Check whether the issue is pricing, release strategy, stock, period, or master-data validity.",
            "Validate organizational data such as plant, sales area, purchasing group, or delivery status.",
        ],
        "resolution": [
            "Correct the missing condition record, stock data, release authorization, or master-data setting.",
            "Retest the full business flow after correction, not only the single error screen.",
            "Confirm that downstream delivery, billing, or inventory steps still behave correctly.",
        ],
        "inputs": ["document number", "material", "plant", "customer or vendor", "pricing or stock message"],
        "causes": [
            "The business process is blocked by incomplete master data or document controls.",
            "Pricing, release strategy, stock, or status settings do not match the transaction context.",
        ],
    },
    {
        "id": "workflow_fiori_application",
        "title": "Workflow, Fiori, or application-layer process issue",
        "area": "Basis",
        "signals": ["workflow", "approval", "fiori", "launchpad", "odata", "app error", "http 500", "ui"],
        "tcodes": ["SWI1", "SWIA", "/IWFND/ERROR_LOG", "/IWBEP/ERROR_LOG", "SU53"],
        "checks": [
            "Identify whether the failure happens in workflow routing, Fiori launch, OData service, or backend authorization.",
            "Review workflow logs, gateway error logs, and the exact user action that failed.",
            "Check whether the issue is limited to one app, one role, or one document scenario.",
        ],
        "resolution": [
            "Correct the failing workflow agent, gateway service, role assignment, or backend application error.",
            "Retest from the same app and business context after the fix is applied.",
            "Validate both frontend behavior and backend document posting after recovery.",
        ],
        "inputs": ["app name or tile", "workflow item or document number", "gateway error text", "affected business role"],
        "causes": [
            "The workflow or Fiori flow is blocked by authorization, gateway, or backend application failures.",
            "The frontend app is calling a service or approval step that is misconfigured or failing in the backend.",
        ],
    },
]


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


def get_config_aliases(names, default=None):
    for name in names:
        value = get_config(name)
        if value not in (None, ""):
            return value
    return default


def get_openai_model():
    return get_config("OPENAI_MODEL", "gpt-4.1-mini")


def get_ollama_base_url():
    return get_config("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")


def get_ollama_model():
    return get_config("OLLAMA_MODEL", "llama3")


def get_open_source_backend():
    return normalize_open_source_backend(
        get_config_aliases(["OPEN_SOURCE_BACKEND", "OPEN_LLM_BACKEND"], "auto")
    )


def get_openai_compatible_base_url():
    return str(
        get_config_aliases(
            [
                "OPEN_SOURCE_API_BASE_URL",
                "OPEN_LLM_API_BASE_URL",
                "OPEN_LLM_BASE_URL",
            ],
            "",
        )
    ).strip().rstrip("/")


def get_openai_compatible_model():
    return str(
        get_config_aliases(
            [
                "OPEN_SOURCE_API_MODEL",
                "OPEN_LLM_API_MODEL",
                "OPEN_LLM_MODEL",
            ],
            "",
        )
    ).strip()


def get_openai_compatible_api_key():
    return str(
        get_config_aliases(
            [
                "OPEN_SOURCE_API_KEY",
                "OPEN_LLM_API_KEY",
            ],
            "open-source-local",
        )
    ).strip()


def get_openai_compatible_timeout_seconds():
    return float(
        get_config_aliases(
            [
                "OPEN_SOURCE_API_TIMEOUT_SECONDS",
                "OPEN_LLM_TIMEOUT_SECONDS",
            ],
            "4",
        )
    )


def get_openai_compatible_max_tokens():
    return int(
        get_config_aliases(
            [
                "OPEN_SOURCE_API_MAX_TOKENS",
                "OPEN_LLM_MAX_TOKENS",
            ],
            "180",
        )
    )


def get_huggingface_token():
    return str(
        get_config_aliases(
            [
                "HF_TOKEN",
                "HUGGINGFACEHUB_API_TOKEN",
                "HUGGING_FACE_HUB_TOKEN",
            ],
            "",
        )
    ).strip()


def configure_huggingface_auth():
    token = get_huggingface_token()
    if not token:
        return False

    os.environ["HF_TOKEN"] = token
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    return True


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
        "open_llm": "openai_compatible",
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
        "timeout": float(get_config("OPENAI_TIMEOUT_SECONDS", "4")),
    }
    organization = get_config("OPENAI_ORGANIZATION")
    project = get_config("OPENAI_PROJECT")
    if organization:
        kwargs["organization"] = organization
    if project:
        kwargs["project"] = project

    return OpenAI(**kwargs)


def runtime_status():
    landscape_counts = get_landscape_counts()
    open_source_backends = get_available_open_source_backends()
    training_status = load_training_status()
    return {
        "agentic_ready": True,
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
        "supported_systems": landscape_counts["systems"],
        "supported_subsystems": landscape_counts["subsystems"],
        "custom_landscape_present": has_landscape_override(),
        "ocr_available": ocr_is_available(),
        "neural_nlp_available": neural_nlp_is_available(),
        "trained_router_ready": router_model_available(),
        "training_state": training_status.get("state", "idle"),
        "training_message": training_status.get("message", ""),
        "training_best_val_accuracy": training_status.get("best_val_accuracy", 0.0),
        "training_best_val_macro_f1": training_status.get("best_val_macro_f1", 0.0),
        "training_last_updated": training_status.get("updated_at", ""),
        "training_last_trained_at": training_status.get("last_trained_at", ""),
        "training_elapsed_minutes": training_status.get("elapsed_minutes", 0.0),
        "training_examples_total": training_status.get("examples_total", 0),
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


def normalize_runbook_heading(heading):
    normalized = str(heading or "").strip().rstrip(":")
    return RUNBOOK_HEADING_ALIASES.get(normalized, normalized)


def parse_runbook_sections(text):
    sections = {}
    notices = []
    current_heading = None

    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue

        normalized_heading = normalize_runbook_heading(line)
        if normalized_heading in RUNBOOK_HEADINGS:
            current_heading = normalized_heading
            sections.setdefault(current_heading, [])
            continue

        if current_heading is None:
            notices.append(line)
            continue

        if line.startswith("- "):
            sections[current_heading].append(line[2:].strip())
        else:
            sections[current_heading].append(line)

    return notices, sections


def join_section_items(items, limit=2):
    trimmed = [item for item in (items or []) if item][:limit]
    return "; ".join(trimmed)


def first_section_item(sections, section_name):
    items = sections.get(section_name) or []
    return items[0] if items else ""


def build_next_best_actions(sections):
    actions = []
    for section_name, limit in [
        ("Checks", 2),
        ("Resolution", 2),
        ("Escalate If", 1),
    ]:
        for item in (sections.get(section_name) or [])[:limit]:
            if item and item not in actions:
                actions.append(item)
    return actions[:5]


def build_solve_now_plan(sections, environment):
    steps = []
    incident = join_section_items(sections.get("Incident"), limit=1) or "the SAP incident"
    reasoning_step = first_section_item(sections, "Decision Path")
    primary_check = first_section_item(sections, "Checks")
    primary_fix = first_section_item(sections, "Resolution")
    confirmation_step = (sections.get("Resolution") or [""])[1] or (sections.get("Checks") or ["", ""])[1]
    escalation = first_section_item(sections, "Escalate If")

    steps.append(f"Stabilize scope in {environment}: confirm the affected users, business step, and exact symptom for {incident}.")
    if reasoning_step:
        steps.append(reasoning_step)
    if primary_check:
        steps.append(primary_check)
    if primary_fix:
        steps.append(primary_fix)
    if confirmation_step:
        steps.append(f"Confirm recovery: {confirmation_step}")
    if escalation:
        steps.append(f"Escalate only if needed: {escalation}")

    unique_steps = []
    for step in steps:
        if step and step not in unique_steps:
            unique_steps.append(step)
    return unique_steps[:5]


def build_expected_outcome(sections):
    incident = join_section_items(sections.get("Incident"), limit=1) or "the affected SAP process"
    resolution = first_section_item(sections, "Resolution")
    tcodes = ", ".join((sections.get("Best T-codes") or [])[:2])
    outcome_parts = [f"Expected outcome: {incident} completes successfully without the current error."]
    if resolution:
        outcome_parts.append(f"Primary fix path: {resolution}.")
    if tcodes:
        outcome_parts.append(f"Validation evidence should be captured with {tcodes}.")
    return " ".join(outcome_parts)


def build_business_update(query, sections, environment):
    incident = join_section_items(sections.get("Incident"), limit=2) or "SAP issue under investigation"
    root_cause = join_section_items(sections.get("Likely Root Cause"), limit=1) or "Root cause is still being validated"
    plan = join_section_items(sections.get("Resolution"), limit=2) or join_section_items(sections.get("Checks"), limit=2)
    risk = join_section_items(sections.get("Escalate If"), limit=1) or "No immediate escalation trigger captured yet"
    system_scope = join_section_items(sections.get("System"), limit=1)
    subsystem_scope = join_section_items(sections.get("Subsystem"), limit=1)
    scope = environment
    if system_scope:
        scope = f"{scope} / {system_scope}"
    if subsystem_scope and subsystem_scope != system_scope:
        scope = f"{scope} / {subsystem_scope}"

    return (
        f"Business update for {scope}: {incident}. "
        f"Likely cause: {root_cause}. "
        f"Current action plan: {plan}. "
        f"Escalate if: {risk}."
    )


def build_end_user_update(query, sections, environment):
    incident = join_section_items(sections.get("Incident"), limit=1) or "the reported SAP issue"
    plan = first_section_item(sections, "Resolution") or first_section_item(sections, "Checks") or "the current recovery plan"
    outcome = build_expected_outcome(sections).replace("Expected outcome: ", "")
    return (
        f"We identified the issue in {environment} as {incident}. "
        f"The support team is now working on {plan}. "
        f"Once complete, {outcome}"
    )


def build_technical_handoff(query, sections, environment):
    lines = [
        f"Landscape: {environment}",
        f"Ticket: {query}",
    ]

    system_scope = join_section_items(sections.get("System"), limit=1)
    if system_scope:
        lines.append(f"System: {system_scope}")

    subsystem_scope = join_section_items(sections.get("Subsystem"), limit=1)
    if subsystem_scope:
        lines.append(f"Subsystem: {subsystem_scope}")

    incident = join_section_items(sections.get("Incident"), limit=2)
    if incident:
        lines.append(f"Incident summary: {incident}")

    root_cause = join_section_items(sections.get("Likely Root Cause"), limit=1)
    if root_cause:
        lines.append(f"Likely root cause: {root_cause}")

    tcodes = ", ".join(sections.get("Best T-codes", [])[:4])
    if tcodes:
        lines.append(f"Primary T-codes: {tcodes}")

    checks = join_section_items(sections.get("Checks"), limit=3)
    if checks:
        lines.append(f"Checks completed or recommended: {checks}")

    resolution = join_section_items(sections.get("Resolution"), limit=3)
    if resolution:
        lines.append(f"Resolution path: {resolution}")

    escalation = join_section_items(sections.get("Escalate If"), limit=2)
    if escalation:
        lines.append(f"Escalation conditions: {escalation}")

    return "\n".join(lines)


def build_follow_up_prompts(query, sections, environment):
    prompts = []

    incident = sections.get("Incident") or []
    tcodes = sections.get("Best T-codes") or []
    system_scope = join_section_items(sections.get("System"), limit=1) or "SAP system"
    subsystem_scope = join_section_items(sections.get("Subsystem"), limit=1)
    scope = system_scope
    if subsystem_scope and subsystem_scope != system_scope:
        scope = f"{scope} / {subsystem_scope}"

    if incident:
        prompts.append(f"Summarize this {environment} incident in {scope} for a business manager.")
    if tcodes:
        prompts.append(f"Give me a production-safe validation checklist using {tcodes[0]}.")
    if sections.get("Resolution"):
        prompts.append(f"Turn this into an operations handoff note for the {environment} {scope} support team.")
        prompts.append(f"Draft an end-user update for this {environment} SAP issue.")
    if sections.get("Escalate If"):
        prompts.append("What should I monitor after the fix to make sure the issue does not come back?")

    prompts.append(f"Explain the likely root cause of this SAP ticket in simpler terms: {query}")
    return prompts[:4]


def build_joule_workspace(query, answer, environment, provider):
    notices, sections = parse_runbook_sections(answer)
    normalized_environment = resolve_environment(environment, query)
    next_actions = build_next_best_actions(sections)
    primary_tcode = (sections.get("Best T-codes") or [""])[0]

    return {
        "environment": normalized_environment,
        "provider": provider,
        "notices": notices,
        "sections": sections,
        "next_actions": next_actions,
        "solve_now_plan": build_solve_now_plan(sections, normalized_environment),
        "expected_outcome": build_expected_outcome(sections),
        "business_update": build_business_update(query, sections, normalized_environment),
        "end_user_update": build_end_user_update(query, sections, normalized_environment),
        "technical_handoff": build_technical_handoff(query, sections, normalized_environment),
        "follow_up_prompts": build_follow_up_prompts(query, sections, normalized_environment),
        "primary_incident": join_section_items(sections.get("Incident"), limit=1),
        "primary_system": join_section_items(sections.get("System"), limit=1),
        "primary_subsystem": join_section_items(sections.get("Subsystem"), limit=1),
        "primary_tcode": primary_tcode,
    }


@lru_cache(maxsize=1)
def fetch_ollama_tags():
    try:
        import requests
    except ImportError:
        return None

    try:
        response = requests.get(
            f"{get_ollama_base_url()}/api/tags",
            timeout=float(get_config("OLLAMA_DISCOVERY_TIMEOUT_SECONDS", "3")),
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
            timeout=min(get_openai_compatible_timeout_seconds(), 3.0),
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
    bare_names = {name.split(":")[0] for name in names if name}
    return configured_model in names or configured_model in bare_names


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
        timeout=get_openai_compatible_timeout_seconds(),
    )


def hf_local_is_configured():
    return bool(get_hf_local_model())


@lru_cache(maxsize=1)
def module_is_installed(module_name):
    return find_spec(module_name) is not None


def hf_local_is_available():
    if not hf_local_is_configured():
        return False

    return module_is_installed("transformers") and module_is_installed("torch")


@lru_cache(maxsize=1)
def load_hf_local_pipeline():
    if not hf_local_is_configured():
        return None

    configure_huggingface_auth()

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
    configure_huggingface_auth()

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


def ticket_reference_terms(ticket):
    terms = set()
    for value in (
        [ticket["title"], ticket["root_cause"], ticket["area"]]
        + ticket.get("keywords", [])
        + ticket.get("error_signals", [])
        + ticket.get("symptoms", [])
        + ticket.get("tcodes", [])
    ):
        for token in tokenize(value):
            if token not in STOPWORDS:
                terms.add(token)
    return terms


def ticket_query_overlap(ticket, query_tokens):
    return len(ticket_reference_terms(ticket).intersection(set(query_tokens or [])))


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


def blend_router_matches(query, scored, top_k=3):
    router_candidates = predict_ticket_candidates(query, top_k=max(6, top_k))
    if not router_candidates:
        return scored

    query_tokens = {token for token in tokenize(query) if token not in STOPWORDS}
    router_by_index = {candidate["ticket_index"]: candidate for candidate in router_candidates}
    enriched = []
    for item in scored:
        candidate = router_by_index.get(item["ticket_index"])
        overlap = ticket_query_overlap(item["ticket"], query_tokens)
        if candidate and candidate["probability"] >= 0.12:
            if overlap == 0 and candidate["probability"] < 0.55:
                enriched.append(item)
                continue
            boost = max(1, int(round(candidate["probability"] * (12 + min(overlap, 4)))))
            item["score"] += boost
            item["reasons"] = extend_unique(
                item["reasons"],
                [f"trained SAP router confidence {candidate['probability']:.2f}"],
                limit=6,
            )
        enriched.append(item)

    existing_indices = {item["ticket_index"] for item in enriched}
    for candidate in router_candidates[: max(4, top_k)]:
        if candidate["ticket_index"] in existing_indices:
            continue
        ticket = TICKET_CATALOG[candidate["ticket_index"]]
        overlap = ticket_query_overlap(ticket, query_tokens)
        if candidate["probability"] < 0.72 or overlap < 2:
            continue
        enriched.append(
            {
                "ticket": ticket,
                "ticket_index": candidate["ticket_index"],
                "score": 4 + int(round(candidate["probability"] * 14)) + overlap,
                "reasons": [f"trained SAP router confidence {candidate['probability']:.2f}"],
            }
        )

    return enriched


def find_ticket_matches(query, top_k=3):
    query_text = normalize_text(query)
    query_tokens = {token for token in tokenize(query) if token not in STOPWORDS}
    query_tcodes = extract_tcodes(query)

    scored = []
    for ticket_index, ticket in enumerate(TICKET_CATALOG):
        item = score_ticket(ticket, query, query_text, query_tokens, query_tcodes)
        item["ticket_index"] = ticket_index
        scored.append(item)

    scored = blend_router_matches(query, scored, top_k=top_k)
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


def derive_preferred_areas(system_context, analysis_context=None):
    preferred = set()
    system_id = str((system_context or {}).get("system_id", "")).upper()
    subsystem_id = str((system_context or {}).get("subsystem_id", "")).upper()

    system_map = {
        "FIORI_GATEWAY": {"Fiori/Gateway"},
        "HANA_DB": {"HANA / DB"},
        "BW4HANA": {"Analytics", "Integration"},
        "MDG": {"Workflow / MDG", "Integration"},
        "SOLMAN": {"ALM / Governance", "Basis"},
        "GRC": {"ALM / Governance", "Security"},
        "PI_PO": {"Integration"},
        "INTEGRATION_SUITE": {"Integration"},
        "ECC": {"Basis", "Integration", "FI", "MM/SD", "Security"},
        "S4HANA": {"Basis", "FI", "MM/SD", "Fiori/Gateway", "Workflow / MDG", "Cross-System"},
    }
    subsystem_map = {
        "LAUNCHPAD": {"Fiori/Gateway"},
        "ODATA": {"Fiori/Gateway"},
        "PERFORMANCE": {"HANA / DB", "Basis"},
        "REPLICATION": {"HANA / DB", "Integration", "Workflow / MDG"},
        "DATA": {"Analytics", "Integration"},
        "REPORTING": {"Analytics"},
        "WORKFLOW": {"Workflow / MDG", "Fiori/Gateway"},
        "CHARM": {"ALM / Governance", "Basis"},
        "MONITORING": {"ALM / Governance", "Basis"},
        "ACCESS": {"ALM / Governance", "Security"},
        "FIREFIGHTER": {"ALM / Governance", "Security"},
        "IFLOWS": {"Integration"},
        "API": {"Integration", "Fiori/Gateway"},
        "CHANNELS": {"Integration"},
        "MESSAGING": {"Integration"},
        "FINANCE": {"FI"},
        "LOGISTICS": {"MM/SD"},
    }
    preferred.update(system_map.get(system_id, set()))
    preferred.update(subsystem_map.get(subsystem_id, set()))

    for signal in (analysis_context or {}).get("domain_signals", [])[:2]:
        mapped = map_domain_signal_to_area(signal.get("domain"))
        if mapped:
            preferred.add(mapped)
    return preferred


def match_has_direct_evidence(match):
    reasons = match.get("reasons", [])
    direct_prefixes = (
        "matched error signal",
        "matched symptom",
        "matched T-code",
        "matched the incident title",
        "exact title match",
    )
    return any(str(reason).startswith(direct_prefixes) for reason in reasons)


def match_has_strong_reason(match):
    for reason in match.get("reasons", []):
        normalized = str(reason)
        if normalized.startswith(
            (
                "matched error signal",
                "matched T-code",
                "matched the incident title",
                "exact title match",
            )
        ):
            return True
        if normalized.startswith("trained SAP router confidence"):
            try:
                probability = float(normalized.rsplit(" ", 1)[-1])
            except Exception:
                probability = 0.0
            if probability >= 0.75:
                return True
    return False


def pattern_has_direct_evidence(item):
    reasons = item.get("reasons", [])
    direct_prefixes = ("matched signal", "matched T-code")
    return any(str(reason).startswith(direct_prefixes) for reason in reasons)


def rerank_matches_for_precision(matches, system_context, analysis_context=None):
    preferred_areas = derive_preferred_areas(system_context, analysis_context=analysis_context)
    specialized_scope = has_specific_scope((system_context or {}).get("system_label")) or has_specific_scope(
        (system_context or {}).get("subsystem_label")
    )
    reranked = []

    for item in matches or []:
        adjusted = dict(item)
        adjusted["reasons"] = list(item.get("reasons", []))
        area = item["ticket"]["area"]
        direct_evidence = match_has_direct_evidence(item)

        if preferred_areas and area in preferred_areas:
            adjusted["score"] += 6 if direct_evidence else 4
            adjusted["reasons"] = extend_unique(adjusted["reasons"], [f"aligned with {area} scope"], limit=6)
        elif preferred_areas and specialized_scope and area not in preferred_areas:
            penalty = 6 if not direct_evidence else 3
            adjusted["score"] -= penalty
            adjusted["reasons"] = extend_unique(adjusted["reasons"], [f"less aligned with scoped SAP landscape"], limit=6)

        if not direct_evidence and any(
            str(reason).startswith("trained SAP router confidence") for reason in adjusted["reasons"]
        ):
            adjusted["score"] -= 3

        reranked.append(adjusted)

    reranked = [item for item in reranked if item["score"] > 0]
    reranked.sort(key=lambda entry: entry["score"], reverse=True)
    return reranked


def rerank_patterns_for_precision(patterns, system_context, analysis_context=None):
    preferred_areas = derive_preferred_areas(system_context, analysis_context=analysis_context)
    specialized_scope = has_specific_scope((system_context or {}).get("system_label")) or has_specific_scope(
        (system_context or {}).get("subsystem_label")
    )
    reranked = []

    for item in patterns or []:
        adjusted = dict(item)
        adjusted["reasons"] = list(item.get("reasons", []))
        area = item["pattern"]["area"]
        direct_evidence = pattern_has_direct_evidence(item)

        if preferred_areas and area in preferred_areas:
            adjusted["score"] += 5
            adjusted["reasons"] = extend_unique(adjusted["reasons"], [f"aligned with {area} scope"], limit=6)
        elif preferred_areas and specialized_scope and area not in preferred_areas:
            adjusted["score"] -= 4 if direct_evidence else 6
            adjusted["reasons"] = extend_unique(adjusted["reasons"], [f"less aligned with scoped SAP landscape"], limit=6)

        reranked.append(adjusted)

    reranked = [item for item in reranked if item["score"] > 0]
    reranked.sort(key=lambda entry: entry["score"], reverse=True)
    return reranked


def score_universal_pattern(pattern, query_text, query_tokens, query_tcodes):
    score = 0
    reasons = []

    for signal in pattern["signals"]:
        points = phrase_points(signal, query_text, query_tokens)
        if points:
            score += points + 2
            reasons.append(f"matched signal '{signal}'")

    if pattern["area"].lower() in query_text:
        score += 2
        reasons.append(f"matched area '{pattern['area']}'")

    pattern_tcodes = set(pattern.get("tcodes", []))
    matched_tcodes = sorted(query_tcodes.intersection(pattern_tcodes))
    if matched_tcodes:
        score += len(matched_tcodes) * 6
        reasons.append(f"matched T-code {', '.join(matched_tcodes)}")

    return {
        "pattern": pattern,
        "score": score,
        "reasons": reasons,
    }


def find_universal_pattern(query, top_k=3):
    query_text = normalize_text(query)
    query_tokens = {token for token in tokenize(query) if token not in STOPWORDS}
    query_tcodes = extract_tcodes(query)

    scored = [
        score_universal_pattern(pattern, query_text, query_tokens, query_tcodes)
        for pattern in UNIVERSAL_SUPPORT_PATTERNS
    ]
    scored = [item for item in scored if item["score"] > 0]
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def infer_priority(query, environment):
    lowered = query.lower()
    resolved_environment = resolve_environment(environment, query)

    if any(term in lowered for term in ["sev1", "p1", "critical outage", "system down", "production down"]):
        return "Critical"
    if resolved_environment == "PROD" and any(
        term in lowered
        for term in ["blocked", "cannot post", "cannot login", "month-end", "payroll", "all users", "integration stopped"]
    ):
        return "High"
    if resolved_environment == "PROD":
        return "High"
    if any(term in lowered for term in ["urgent", "asap", "today", "go live", "cutover"]):
        return "High"
    return "Medium"


def get_area_owners(area):
    return AREA_OWNERS.get(
        area,
        [
            "SAP application support",
            "Basis or functional team depending on whether the issue is technical or process related",
        ],
    )


def derive_required_inputs(query, area, tcodes, pattern=None, analysis_context=None, system_context=None):
    inputs = [
        "Exact SAP error text or message class and number",
        "Affected user, batch user, or interface technical account",
        "Client and system where the issue occurred",
        "Business impact and whether production users are blocked",
    ]

    if tcodes:
        inputs.append(f"Transaction code or app involved: {', '.join(tcodes[:4])}")

    lowered = query.lower()
    conditional_inputs = [
        ("document", "Document number or business object ID"),
        ("invoice", "Invoice or accounting document number"),
        ("idoc", "IDoc number and status record text"),
        ("queue", "Queue name and exact SYSFAIL or stop reason"),
        ("transport", "Transport number and import phase"),
        ("job", "Job name, job count, and job log"),
        ("vendor", "Vendor, customer, material, or master-data identifier"),
        ("material", "Vendor, customer, material, or master-data identifier"),
        ("printer", "Output device, spool request, or printer name"),
    ]
    for signal, input_text in conditional_inputs:
        if signal in lowered and input_text not in inputs:
            inputs.append(input_text)

    if pattern:
        for item in pattern.get("inputs", []):
            if item and item not in inputs:
                inputs.append(item[0].upper() + item[1:] if item[0].islower() else item)

    entities = (analysis_context or {}).get("entities", {})
    if entities.get("users"):
        inputs.append(f"Affected user(s) already detected: {', '.join(entities['users'][:3])}")
    if entities.get("transports"):
        inputs.append(f"Transport reference already detected: {', '.join(entities['transports'][:2])}")
    if entities.get("idocs"):
        inputs.append(f"IDoc number(s) already detected: {', '.join(entities['idocs'][:3])}")
    if entities.get("queues"):
        inputs.append(f"Queue name(s) already detected: {', '.join(entities['queues'][:2])}")
    if entities.get("objects"):
        inputs.append(f"Business object(s) already detected: {', '.join(entities['objects'][:3])}")
    if entities.get("http_codes"):
        inputs.append(f"HTTP status already detected: {', '.join(entities['http_codes'][:2])}")

    if system_context and system_context.get("system_label"):
        scope_text = system_context["system_label"]
        subsystem_label = system_context.get("subsystem_label")
        if subsystem_label and subsystem_label != system_context["system_label"]:
            scope_text = f"{scope_text} / {subsystem_label}"
        inputs.append(f"Owning SAP scope: {scope_text}")

    if area in {"FI", "MM", "SD", "MM/SD", "Workflow / MDG"}:
        inputs.append("Organizational data such as company code, plant, sales area, or purchasing org")

    return extend_unique([], inputs, limit=8)


def derive_best_tcodes(query_tcodes, matched_tcodes, pattern=None, system_context=None):
    ordered = []
    for bucket in [sorted(query_tcodes), matched_tcodes or [], pattern.get("tcodes", []) if pattern else []]:
        for tcode in bucket:
            if tcode and tcode not in ordered:
                ordered.append(tcode)

    if system_context:
        for tool in system_context.get("system_tools", [])[:6]:
            if tool and tool not in ordered:
                ordered.append(tool)

    defaults = ["SU53", "ST22", "SM21", "SM37", "SM59"]
    for tcode in defaults:
        if tcode not in ordered:
            ordered.append(tcode)

    return ordered[:6]


def extend_unique(items, additions, limit=None):
    combined = list(items or [])
    for item in additions or []:
        if item and item not in combined:
            combined.append(item)
        if limit and len(combined) >= limit:
            break
    return combined[:limit] if limit else combined


def build_mixed_issue_workstreams(matches, universal_patterns):
    workstreams = []
    seen_keys = set()

    for match in matches[:4]:
        if match["score"] < 12:
            continue
        ticket = match["ticket"]
        key = ("ticket", ticket["title"].lower())
        if key in seen_keys:
            continue
        if any(
            stream["area"] == ticket["area"]
            and set(stream["tcodes"]).intersection(ticket["tcodes"])
            for stream in workstreams
        ):
            continue
        workstreams.append(
            {
                "title": ticket["title"],
                "area": ticket["area"],
                "score": match["score"],
                "reasons": match["reasons"][:3],
                "tcodes": ticket["tcodes"],
                "checks": ticket["checks"],
                "resolution": ticket["resolution"],
                "inputs": [],
                "owners": get_area_owners(ticket["area"]),
            }
        )
        seen_keys.add(key)

    if len(workstreams) >= 2:
        workstreams.sort(key=lambda stream: stream["score"], reverse=True)
        return workstreams[:3]

    for item in universal_patterns[:4]:
        if item["score"] < 8:
            continue
        pattern = item["pattern"]
        key = ("pattern", pattern["id"])
        if key in seen_keys:
            continue
        if any(stream["area"] == pattern["area"] for stream in workstreams):
            continue
        workstreams.append(
            {
                "title": pattern["title"],
                "area": pattern["area"],
                "score": item["score"],
                "reasons": item["reasons"][:3],
                "tcodes": pattern.get("tcodes", []),
                "checks": pattern.get("checks", []),
                "resolution": pattern.get("resolution", []),
                "inputs": pattern.get("inputs", []),
                "owners": get_area_owners(pattern["area"]),
            }
        )
        seen_keys.add(key)

    workstreams.sort(key=lambda stream: stream["score"], reverse=True)
    return workstreams[:3]


def detect_mixed_issue_workstreams(query, matches, universal_patterns):
    workstreams = build_mixed_issue_workstreams(matches, universal_patterns)
    if len(workstreams) < 2:
        return []

    lowered = query.lower()
    cue_terms = [
        " and ",
        " also ",
        " plus ",
        " both ",
        " while ",
        " meanwhile ",
        " multiple issues ",
        " multiple errors ",
        " two issues ",
        " mixed ",
        " issue 1",
        " issue 2",
        ";",
        "\n",
    ]
    has_multi_cue = any(term in lowered for term in cue_terms)
    distinct_areas = {stream["area"] for stream in workstreams}
    distinct_titles = {stream["title"] for stream in workstreams}
    strong_secondary = len(workstreams) >= 2 and workstreams[1]["score"] >= max(10, workstreams[0]["score"] - 10)
    minimum_secondary = len(workstreams) >= 2 and workstreams[1]["score"] >= 12

    if len(distinct_areas) >= 2 and strong_secondary:
        return workstreams
    if len(distinct_areas) >= 2 and has_multi_cue and minimum_secondary:
        return workstreams
    if has_multi_cue and len(distinct_titles) >= 2 and strong_secondary:
        return workstreams
    return []


def build_mixed_issue_answer(query, context_snippets, context_source, environment, system_context, workstreams, analysis_context=None):
    primary_areas = []
    owners = []
    best_tcodes = []
    required_inputs = []
    issue_mix = []
    parallel_workstreams = []
    shared_checks = [
        "Split the ticket into separate failing steps and confirm whether one symptom is upstream of the others.",
        "Align timestamps, users, and business objects so you can see whether the second issue is a consequence of the first one.",
        "Validate the owning system, subsystem, and support team for each symptom before applying transports or message reprocessing.",
    ]
    shared_resolution = [
        "Fix the upstream dependency first, then retest the downstream symptom before applying a second change.",
        "Keep transports, role changes, and interface reprocessing as separate controlled actions when the ticket spans multiple teams.",
    ]
    shared_risks = [
        "Do not assume one workaround fixes every symptom until the dependency order is proven.",
        "Avoid replaying queues, IDocs, or jobs before the upstream authorization, transport, or service issue is corrected.",
        "Coordinate multiple owners so one team does not overwrite or mask the evidence needed by another team.",
    ]
    escalation_lines = [
        "The incident requires coordinated action from more than one SAP team and the ownership sequence is unclear.",
        "Production remains blocked after the first workstream is cleared.",
        "The issue spans security, transport, integration, or application layers and needs a controlled change plan.",
    ]

    for index, workstream in enumerate(workstreams, start=1):
        primary_areas = extend_unique(primary_areas, [workstream["area"]], limit=4)
        owners = extend_unique(owners, workstream["owners"], limit=6)
        best_tcodes = extend_unique(best_tcodes, workstream["tcodes"], limit=6)
        required_inputs = extend_unique(
            required_inputs,
            derive_required_inputs(
                query,
                workstream["area"],
                workstream["tcodes"],
                analysis_context=analysis_context,
                system_context=system_context,
            ),
            limit=8,
        )
        required_inputs = extend_unique(
            required_inputs,
            [
                item[0].upper() + item[1:] if item and item[0].islower() else item
                for item in workstream.get("inputs", [])
                if item
            ],
            limit=8,
        )
        reason_text = "; ".join(workstream["reasons"][:2]) or "multiple strong SAP signals"
        issue_mix.append(f"Workstream {index}: {workstream['title']} ({workstream['area']}) because {reason_text}")
        primary_check = workstream["checks"][0] if workstream["checks"] else "Collect the first diagnostic trace for this workstream."
        primary_fix = workstream["resolution"][0] if workstream["resolution"] else "Apply the safest corrective action for this workstream and retest."
        parallel_workstreams.append(
            f"{workstream['area']}: {workstream['title']} - Check: {primary_check} Fix path: {primary_fix}"
        )
        shared_checks = extend_unique(shared_checks, workstream["checks"][:1], limit=6)
        shared_resolution = extend_unique(shared_resolution, workstream["resolution"][:1], limit=5)

    best_tcodes = derive_best_tcodes(extract_tcodes(query), best_tcodes, system_context=system_context)
    priority = infer_priority(query, environment)
    context_preview = context_snippets[0] if context_snippets else "No additional note snippet was available."
    reasoning = build_advanced_reasoning(
        query,
        system_context,
        analysis_context=analysis_context,
        matches=[],
        universal_patterns=[],
    )

    return f"""Incident
- Mixed SAP incident with multiple likely failure domains
- Areas: {', '.join(primary_areas) if primary_areas else 'Basis'}
- Confidence: Medium

Likely Root Cause
- The ticket text combines multiple strong SAP symptoms, so the incident should be split into parallel workstreams instead of forcing a single diagnosis.

Priority
- {priority}

Likely Owner
{format_list(owners, "SAP application support")}

Required Inputs
{format_list(required_inputs, "Collect enough detail to separate the individual issues.")}

{build_environment_section(environment)}

{build_system_section(system_context)}

{build_analysis_section(analysis_context)}

{build_reasoning_sections(reasoning)}

Issue Mix
{format_list(issue_mix, "No mixed-issue breakdown was derived.")}

Parallel Workstreams
{format_list(parallel_workstreams, "No workstreams were derived.")}

Best T-codes
{format_list(best_tcodes, "No T-code available")}

Checks
{format_list(shared_checks, "Collect enough detail to classify the incident.")}

Resolution
{format_list(shared_resolution, "Apply the safest validated fix and retest.")}

Cross-Issue Risks
{format_list(shared_risks, "Use a controlled change plan when multiple symptoms are involved.")}

Escalate If
{format_list(escalation_lines, "Escalate when the issue affects production or cannot be reproduced safely.")}

Why This Matched
- Multiple strong runbook candidates were detected in the same ticket text.
- The incident appears to span more than one SAP failure domain or support team.

Supporting Context
- Source: {context_source}
- Snippet: {context_preview}"""


def build_context(matches, query, include_vector=False):
    note_matches = search_local_notes(query)

    snippets = []
    sources = []

    if note_matches:
        snippets.extend(shorten_text(note, limit=420) for note in note_matches)
        sources.append("local SAP ticket notes")

    if include_vector:
        vector_matches = search_vector_context(query)
        if vector_matches:
            snippets.extend(shorten_text(note, limit=420) for note in vector_matches)
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


def build_system_section(system_context):
    system_lines = [system_context["system_label"]]
    if system_context.get("system_type"):
        system_lines.append(f"Platform: {system_context['system_type']}")
    if system_context.get("system_tools"):
        system_lines.append(f"Core tools: {', '.join(system_context['system_tools'][:6])}")

    subsystem_lines = [system_context.get("subsystem_label", "Shared service or general system scope")]
    if system_context.get("subsystem_focus"):
        subsystem_lines.append(system_context["subsystem_focus"])

    context_lines = []
    if system_context.get("matched_terms"):
        context_lines.append(f"Detected from: {', '.join(system_context['matched_terms'][:4])}")
    context_lines.append("Use the owning system boundary before assigning fixes, transports, or message reprocessing.")

    return (
        f"System\n"
        f"{format_list(system_lines, 'Cross-system SAP landscape')}\n\n"
        f"Subsystem\n"
        f"{format_list(subsystem_lines, 'Shared SAP service area')}\n\n"
        f"System Context\n"
        f"{format_list(context_lines, 'Narrow the incident to the owning SAP stack and connected services.')}\n\n"
        f"Integration Points\n"
        f"{format_list(system_context.get('integration_points'), 'No specific integration point was identified yet.')}\n\n"
        f"Integration Guidance\n"
        f"{format_list(system_context.get('integration_guidance'), 'Validate the owning system and subsystem before changing the fix path.')}\n\n"
        f"Integration Specifications\n"
        f"{format_list(system_context.get('integration_specs'), 'No detailed integration specification was derived for this SAP scope yet.')}"
    )


def build_reasoning_sections(reasoning):
    if not reasoning:
        return ""

    return (
        f"Advanced Diagnosis\n"
        f"{format_list(reasoning.get('diagnosis_lines'), 'No advanced diagnosis was derived.')}\n\n"
        f"Failure Chain\n"
        f"{format_list(reasoning.get('failure_chain'), 'No failure chain was derived.')}\n\n"
        f"Decision Path\n"
        f"{format_list(reasoning.get('decision_path'), 'No decision path was derived.')}"
    )


def has_specific_scope(label):
    text = str(label or "").strip().lower()
    if not text:
        return False
    generic_markers = [
        "cross-system",
        "shared service",
        "shared sap service",
        "general system scope",
        "not yet classified",
    ]
    return not any(marker in text for marker in generic_markers)


def map_domain_signal_to_area(domain_name):
    mapping = {
        "Security": "Security",
        "Integration": "Integration",
        "Basis": "Basis",
        "FI": "FI",
        "MM/SD": "MM/SD",
        "Fiori/Gateway": "Fiori/Gateway",
    }
    return mapping.get(domain_name, "")


def derive_flexible_area(query, system_context, analysis_context=None, best_pattern=None):
    if best_pattern:
        return best_pattern["area"]

    domain_signals = (analysis_context or {}).get("domain_signals", [])
    for signal in domain_signals:
        mapped = map_domain_signal_to_area(signal.get("domain"))
        if mapped:
            return mapped

    subsystem_id = str(system_context.get("subsystem_id", "")).upper()
    system_id = str(system_context.get("system_id", "")).upper()
    subsystem_label = str(system_context.get("subsystem_label", "")).lower()
    system_label = str(system_context.get("system_label", "")).lower()
    lowered = query.lower()

    if system_id == "FIORI_GATEWAY" or subsystem_id in {"LAUNCHPAD", "ODATA"}:
        return "Fiori/Gateway"
    if system_id == "HANA_DB" or subsystem_id in {"PERFORMANCE", "REPLICATION"}:
        return "HANA / DB"
    if system_id == "BW4HANA" or subsystem_id in {"DATA", "REPORTING"}:
        return "Analytics"
    if system_id == "MDG" or subsystem_id in {"WORKFLOW", "REPLICATION"} or "workflow" in subsystem_label:
        return "Workflow / MDG"
    if system_id in {"SOLMAN", "GRC"} or subsystem_id in {"CHARM", "MONITORING", "ACCESS", "FIREFIGHTER"}:
        return "ALM / Governance"
    if subsystem_id == "FINANCE" or any(term in lowered for term in ["invoice", "fb60", "fb08", "f110", "ob52", "company code"]):
        return "FI"
    if subsystem_id == "LOGISTICS" or any(term in lowered for term in ["migo", "material", "pricing", "va01", "vl02n", "delivery", "stock"]):
        return "MM/SD"
    if system_id in {"PI_PO", "INTEGRATION_SUITE"} or any(term in lowered for term in ["idoc", "rfc", "queue", "iflow", "api"]):
        return "Integration"
    if any(term in lowered for term in ["authorization", "role", "su53", "locked user", "access denied"]):
        return "Security"
    if any(term in lowered for term in ["sql", "expensive statement", "dbacockpit", "replication", "backup", "memory", "cpu"]):
        return "HANA / DB"
    if any(term in lowered for term in ["process chain", "odp", "odqmon", "bex", "analysis office", "reporting"]):
        return "Analytics"
    if any(term in lowered for term in ["workflow", "approval", "drflog", "change request", "firefighter", "grc", "mdg"]):
        return "Workflow / MDG"
    if has_specific_scope(system_label):
        return "Cross-System"
    return "Basis"


def refine_area_with_scope(base_area, system_context, analysis_context=None):
    area = base_area or "Cross-System"
    system_id = str(system_context.get("system_id", "")).upper()
    subsystem_id = str(system_context.get("subsystem_id", "")).upper()

    if system_id == "FIORI_GATEWAY" or subsystem_id in {"LAUNCHPAD", "ODATA"}:
        if area in {"Basis", "Integration", "Cross-System"}:
            return "Fiori/Gateway"
    if system_id == "HANA_DB" or subsystem_id in {"PERFORMANCE", "REPLICATION"}:
        if area in {"Basis", "Cross-System"}:
            return "HANA / DB"
    if system_id == "BW4HANA" or subsystem_id in {"DATA", "REPORTING"}:
        if area in {"Basis", "Integration", "Cross-System"}:
            return "Analytics"
    if system_id == "MDG" or subsystem_id in {"WORKFLOW", "REPLICATION"}:
        if area in {"Basis", "Integration", "Cross-System"}:
            return "Workflow / MDG"
    if system_id in {"SOLMAN", "GRC"} or subsystem_id in {"CHARM", "MONITORING", "ACCESS", "FIREFIGHTER"}:
        if area in {"Basis", "Integration", "Security", "Cross-System"}:
            return "ALM / Governance"
    if system_id in {"PI_PO", "INTEGRATION_SUITE"} and area in {"Basis", "Cross-System"}:
        return "Integration"
    return area


def derive_flexible_title(system_context, area, best_pattern=None):
    if best_pattern:
        return best_pattern["title"]

    subsystem_label = system_context.get("subsystem_label", "")
    system_label = system_context.get("system_label", "")
    if has_specific_scope(subsystem_label):
        return f"{subsystem_label} issue requiring adaptive SAP triage"
    if has_specific_scope(system_label):
        return f"{system_label} issue requiring adaptive SAP triage"
    if area and area != "Cross-System":
        return f"{area} issue requiring adaptive SAP triage"
    return "Cross-system SAP issue requiring adaptive triage"


def derive_flexible_owners(area, system_context):
    owners = get_area_owners(area)
    if has_specific_scope(system_context.get("system_label")):
        owners = extend_unique(owners, [f"Owning system support team: {system_context['system_label']}"], limit=6)
    if has_specific_scope(system_context.get("subsystem_label")):
        owners = extend_unique(owners, [f"Primary subsystem owner: {system_context['subsystem_label']}"], limit=6)
    return owners[:6]


def summarize_detected_evidence(analysis_context):
    entities = (analysis_context or {}).get("entities", {})
    evidence = []
    if entities.get("http_codes"):
        evidence.append(f"HTTP {', '.join(entities['http_codes'][:2])}")
    if entities.get("status_codes"):
        evidence.append(f"status {', '.join(entities['status_codes'][:2])}")
    if entities.get("transports"):
        evidence.append(f"transport {', '.join(entities['transports'][:2])}")
    if entities.get("idocs"):
        evidence.append(f"IDoc {', '.join(entities['idocs'][:2])}")
    if entities.get("queues"):
        evidence.append(f"queue {', '.join(entities['queues'][:2])}")
    if entities.get("users"):
        evidence.append(f"user {', '.join(entities['users'][:2])}")
    if entities.get("objects"):
        evidence.append(f"object {', '.join(entities['objects'][:2])}")
    return evidence[:4]


def build_adaptive_guidance(query, environment, area, system_context, analysis_context=None):
    guidance = []
    system_label = system_context.get("system_label", "SAP landscape")
    subsystem_label = system_context.get("subsystem_label", "")
    scope = system_label
    if has_specific_scope(subsystem_label) and subsystem_label != system_label:
        scope = f"{scope} / {subsystem_label}"

    guidance.append(f"Treat this as a {scope} investigation first so fixes stay inside the owning SAP boundary.")
    if system_context.get("subsystem_focus"):
        guidance.append(system_context["subsystem_focus"])
    if system_context.get("integration_guidance"):
        guidance.append(system_context["integration_guidance"][0])

    domain_signals = (analysis_context or {}).get("domain_signals", [])
    if domain_signals:
        top_domain = domain_signals[0]
        guidance.append(
            f"Current evidence points most strongly to {top_domain['domain']} because of {', '.join(top_domain['signals'][:3])}."
        )

    evidence = summarize_detected_evidence(analysis_context)
    if evidence:
        guidance.append(f"Keep the detected evidence together during triage: {', '.join(evidence)}.")

    if environment == "PROD":
        guidance.append("Use production-safe containment first and avoid mass reprocessing, broad role changes, or risky transports without approval.")
    elif environment in {"QA", "TEST"}:
        guidance.append("Retest the exact failing business flow plus one nearby happy-path scenario before sign-off.")
    else:
        guidance.append("Reproduce the issue with enough technical detail to isolate the first failing object, service, or document.")

    return extend_unique([], guidance, limit=6)


def build_adaptive_checks(query, query_tcodes, system_context, analysis_context=None, best_pattern=None):
    checks = []
    entities = (analysis_context or {}).get("entities", {})

    if best_pattern:
        checks = extend_unique(checks, best_pattern.get("checks", [])[:2], limit=7)

    if query_tcodes:
        checks.append(f"Validate the exact SAP transaction or app mentioned in the ticket: {', '.join(query_tcodes[:4])}.")

    if entities.get("http_codes"):
        checks.append("Review Gateway, ICF, and backend service errors with /IWFND/ERROR_LOG, /IWBEP/ERROR_LOG, SICF, and ST22.")
    if entities.get("transports") or entities.get("return_codes"):
        checks.append("Review STMS import history, sequence, activation state, and object ownership before reimporting anything.")
    if entities.get("idocs") or entities.get("queues") or entities.get("status_codes"):
        checks.append("Trace the failed payload in WE02, BD87, SM58, SMQ1, or SMQ2 and confirm whether the blocker is posting, master data, or connectivity.")
    if entities.get("users"):
        checks.append("Run SU53 or ST01 for the affected user and compare the failing step with the assigned role or technical user context.")
    if entities.get("objects"):
        checks.append(f"Validate the business object lifecycle and current status for {', '.join(entities['objects'][:3])}.")

    if system_context.get("system_tools"):
        checks.append(f"Start with the main tools for this stack: {', '.join(system_context['system_tools'][:5])}.")

    checks.extend(
        [
            "Capture the exact SAP error text, timestamp, user, and business step before making corrective changes.",
            "Identify the first technical artifact that fails: document posting, service call, queue entry, transport, workflow step, job, or SQL statement.",
            "Confirm whether the issue is isolated to one user or object, or affects a wider batch, interface, or business population.",
        ]
    )
    return extend_unique([], checks, limit=7)


def build_adaptive_resolution(query, environment, system_context, analysis_context=None, best_pattern=None):
    resolution = []
    entities = (analysis_context or {}).get("entities", {})

    if best_pattern:
        resolution = extend_unique(resolution, best_pattern.get("resolution", [])[:2], limit=6)

    if entities.get("transports") or entities.get("return_codes"):
        resolution.append("Correct the missing prerequisite, sequence, or inactive object first, then reimport only the failing transport in the approved order.")
    if entities.get("idocs") or entities.get("queues") or entities.get("status_codes"):
        resolution.append("Fix the application, master-data, or connectivity blocker first, then reprocess only the failed IDoc, queue, or tRFC entries.")
    if entities.get("http_codes"):
        resolution.append("Correct the OData service, ICF activation, role, RFC trust, or backend exception before retesting the same user action in the app.")
    if entities.get("users") and not entities.get("http_codes"):
        resolution.append("Correct the affected user, role, or technical account issue and retest with the same user context to confirm the blocker is gone.")
    if not resolution:
        resolution.extend(
            [
                "Reproduce the exact failure and correct the smallest validated configuration, authorization, transport, or data issue that explains the first failing step.",
                "Retest the same business step immediately after the change so you confirm the fix before touching related components.",
            ]
        )

    resolution.append("Validate one downstream step after the primary fix so the wider SAP process still completes correctly.")
    if environment == "PROD":
        resolution.append("Use change controls, communications, and rollback planning for any correction that affects multiple users, postings, or integrations.")

    return extend_unique([], resolution, limit=6)


def build_adaptive_risks(query, environment, area, system_context, analysis_context=None):
    risks = []
    entities = (analysis_context or {}).get("entities", {})

    if environment == "PROD":
        risks.append("Production users, postings, or interfaces are affected and the fix has not been validated in a safer environment.")
    if entities.get("transports"):
        risks.append("The issue may require transport sequencing, object repair, or activation checks rather than a direct change in the target system.")
    if entities.get("idocs") or entities.get("queues"):
        risks.append("Mass reprocessing could duplicate payloads or hide the original application error if the root cause is not fixed first.")
    if entities.get("objects") and area in {"FI", "MM", "SD", "MM/SD", "Workflow / MDG"}:
        risks.append("Document correction, reversal, or reprocessing may have downstream financial, logistics, or workflow impact.")
    if has_specific_scope(system_context.get("subsystem_label")):
        risks.append(f"The owning subsystem {system_context['subsystem_label']} may need specialist support beyond first-line triage.")
    risks.append("The incident still lacks enough evidence for a one-step fix and may require cross-team coordination.")
    return extend_unique([], risks, limit=5)


def build_related_playbooks(query, universal_patterns=None, best_pattern=None, preferred_area=None):
    related = []
    for item in universal_patterns or []:
        pattern = item["pattern"]
        if best_pattern and pattern["title"] == best_pattern["title"]:
            continue
        if item.get("score", 0) < 4:
            continue
        related.append(f"{pattern['title']} ({pattern['area']})")

    for candidate in predict_ticket_candidates(query, top_k=4):
        if best_pattern and candidate["title"] == best_pattern["title"]:
            continue
        if candidate["probability"] < 0.2:
            continue
        if preferred_area and preferred_area not in {"Basis", "Security", "Integration", "FI", "MM", "SD", "MM/SD"}:
            continue
        related.append(f"{candidate['title']} [{candidate['area']}, router {candidate['probability']:.2f}]")

    if not related:
        return []
    return extend_unique([], related, limit=4)


def build_analysis_section(analysis_context):
    if not analysis_context:
        return ""

    nlp_lines = []
    entities = analysis_context.get("entities", {})
    if entities.get("tcodes"):
        nlp_lines.append(f"T-codes from ticket or image: {', '.join(entities['tcodes'])}")
    if entities.get("transports"):
        nlp_lines.append(f"Transport references: {', '.join(entities['transports'])}")
    if entities.get("status_codes"):
        nlp_lines.append(f"Status codes: {', '.join(entities['status_codes'])}")
    if entities.get("http_codes"):
        nlp_lines.append(f"HTTP errors: {', '.join(entities['http_codes'])}")
    if entities.get("users"):
        nlp_lines.append(f"Users detected: {', '.join(entities['users'])}")
    if analysis_context.get("domain_signals"):
        for signal in analysis_context["domain_signals"][:3]:
            nlp_lines.append(
                f"{signal['domain']} signal from keywords: {', '.join(signal['signals'])}"
            )

    neural_lines = [
        f"{item['label']} [{item['type']}] score {item['score']}"
        for item in analysis_context.get("semantic_matches", [])[:4]
    ]

    image_lines = list(analysis_context.get("image_findings", [])[:5])
    if analysis_context.get("ocr_text"):
        image_lines.append(
            f"OCR preview: {analysis_context['ocr_text'][:220]}{'...' if len(analysis_context['ocr_text']) > 220 else ''}"
        )

    sections = []
    if nlp_lines:
        sections.append(f"NLP Signals\n{format_list(nlp_lines, 'No NLP signals detected.')}")
    if neural_lines:
        sections.append(f"Neural Matches\n{format_list(neural_lines, 'No neural semantic matches were available.')}")
    if image_lines:
        sections.append(f"Image Findings\n{format_list(image_lines, 'No image findings were captured.')}")
    return "\n\n".join(sections)


def build_matching_query(query, analysis_context):
    if not analysis_context:
        return query

    evidence_lines = [query.strip()]
    ocr_text = shorten_text(analysis_context.get("ocr_text", ""), limit=500)
    if ocr_text:
        evidence_lines.append(f"OCR text: {ocr_text}")

    entities = analysis_context.get("entities", {})
    exact_evidence = []
    for label, key in [
        ("T-codes", "tcodes"),
        ("Transports", "transports"),
        ("HTTP errors", "http_codes"),
        ("Statuses", "status_codes"),
        ("Return codes", "return_codes"),
        ("Queues", "queues"),
        ("IDocs", "idocs"),
        ("Users", "users"),
    ]:
        values = entities.get(key)
        if values:
            exact_evidence.append(f"{label}: {', '.join(values[:4])}")

    evidence_lines.extend(exact_evidence[:6])
    return "\n".join(line for line in evidence_lines if line)


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

Return a concise ticket-resolution playbook that preserves the section headings already present
in the local runbook answer whenever possible. Keep the response safe for enterprise SAP
operations, especially for production systems.
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
        "System",
        "Subsystem",
        "System Context",
        "Integration Points",
        "Integration Guidance",
        "NLP Signals",
        "Neural Matches",
        "Image Findings",
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
                    "num_ctx": 1280,
                    "num_predict": 128,
                },
            },
            timeout=float(get_config("OLLAMA_TIMEOUT_SECONDS", "8")),
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
    header_names = RUNBOOK_HEADINGS.union(RUNBOOK_HEADING_ALIASES)
    header_pattern = "|".join(re.escape(name) for name in sorted(header_names, key=len, reverse=True))
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
            rf"^({header_pattern})\s*[:-]\s*(.+)$",
            normalized,
        )
        if inline_header:
            cleaned_lines.append(normalize_runbook_heading(inline_header.group(1)))
            remainder = inline_header.group(2).strip()
            remainder_parts = [part.strip() for part in re.split(r"\s+-\s+", remainder) if part.strip()]
            for part in remainder_parts or [remainder]:
                cleaned_lines.append(f"- {part}")
            last_line_was_header = False
            continue
        normalized_header = normalize_runbook_heading(normalized)
        if normalized_header in RUNBOOK_HEADINGS:
            cleaned_lines.append(normalized_header)
            last_line_was_header = True
            continue
        if normalized.endswith(":") and normalize_runbook_heading(normalized[:-1]) in RUNBOOK_HEADINGS:
            cleaned_lines.append(normalize_runbook_heading(normalized[:-1]))
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


def sanitize_output_text(text):
    replacements = {
        "•": "-",
        "–": "-",
        "—": "-",
        "−": "-",
        "∕": "/",
        "／": "/",
        "’": "'",
        "“": '"',
        "”": '"',
        "\xa0": " ",
    }
    cleaned_chars = []
    for char in str(text):
        char = replacements.get(char, char)
        codepoint = ord(char)
        if codepoint in {10, 13, 9}:
            cleaned_chars.append(char)
            continue
        if 0xE000 <= codepoint <= 0xF8FF:
            continue
        if char.isprintable():
            cleaned_chars.append(char)
    return "".join(cleaned_chars).encode("ascii", "ignore").decode("ascii")


def get_open_source_backend_order(backend=None):
    selected_backend = normalize_open_source_backend(
        backend if backend not in (None, "") else get_open_source_backend()
    )
    if selected_backend == "auto":
        return ["ollama", "openai_compatible"]
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
            max_tokens=get_openai_compatible_max_tokens(),
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


def build_solver_bundle(query, environment, system=None, subsystem=None, analysis_context=None, matching_query=None, scope_query=None):
    matching_query = matching_query or query
    scope_query = scope_query or query
    resolved_environment = resolve_environment(environment, scope_query)
    system_context = resolve_system_context(scope_query, system=system, subsystem=subsystem)
    matches = rerank_matches_for_precision(
        find_ticket_matches(matching_query, top_k=6),
        system_context,
        analysis_context=analysis_context,
    )[:4]
    universal_patterns = rerank_patterns_for_precision(
        find_universal_pattern(matching_query, top_k=6),
        system_context,
        analysis_context=analysis_context,
    )[:4]
    context_snippets, context_source = build_context(matches, matching_query, include_vector=False)
    reasoning = build_advanced_reasoning(
        matching_query,
        system_context,
        analysis_context=analysis_context,
        matches=matches,
        universal_patterns=universal_patterns,
    )
    mixed_issue_workstreams = detect_mixed_issue_workstreams(matching_query, matches, universal_patterns)

    return {
        "query": query,
        "matching_query": matching_query,
        "scope_query": scope_query,
        "resolved_environment": resolved_environment,
        "system_context": system_context,
        "analysis_context": analysis_context,
        "matches": matches,
        "universal_patterns": universal_patterns,
        "context_snippets": context_snippets,
        "context_source": context_source,
        "reasoning": reasoning,
        "mixed_issue_workstreams": mixed_issue_workstreams,
    }


def build_ticket_answer_from_bundle(bundle):
    matching_query = bundle["matching_query"]
    resolved_environment = bundle["resolved_environment"]
    system_context = bundle["system_context"]
    analysis_context = bundle.get("analysis_context")
    matches = bundle["matches"]
    universal_patterns = bundle["universal_patterns"]
    context_snippets = bundle["context_snippets"]
    context_source = bundle["context_source"]
    reasoning = bundle["reasoning"]
    mixed_issue_workstreams = bundle["mixed_issue_workstreams"]

    if mixed_issue_workstreams:
        return build_mixed_issue_answer(
            matching_query,
            context_snippets,
            context_source,
            resolved_environment,
            system_context,
            mixed_issue_workstreams,
            analysis_context=analysis_context,
        )

    if not matches or matches[0]["score"] < 10:
        return build_generic_triage_answer(
            matching_query,
            context_snippets,
            context_source,
            resolved_environment,
            system_context=system_context,
            analysis_context=analysis_context,
            universal_patterns=universal_patterns,
        )

    best_match = matches[0]
    ticket = best_match["ticket"]
    confidence = summarize_confidence(matches)
    has_strong_reason = match_has_strong_reason(best_match)
    preferred_areas = derive_preferred_areas(system_context, analysis_context=analysis_context)
    top_area_aligned = not preferred_areas or ticket["area"] in preferred_areas

    if (confidence == "Low" and not has_strong_reason) or (preferred_areas and not top_area_aligned and best_match["score"] < 24):
        return build_generic_triage_answer(
            matching_query,
            context_snippets,
            context_source,
            resolved_environment,
            system_context=system_context,
            analysis_context=analysis_context,
            universal_patterns=universal_patterns,
        )

    related = [match["ticket"]["title"] for match in matches[1:] if match["score"] >= 10]
    priority = infer_priority(matching_query, resolved_environment)
    display_area = refine_area_with_scope(ticket["area"], system_context, analysis_context=analysis_context)
    owners = derive_flexible_owners(display_area, system_context)
    query_tcodes = sorted(extract_tcodes(matching_query))
    best_tcodes = derive_best_tcodes(query_tcodes, ticket["tcodes"], system_context=system_context)
    required_inputs = derive_required_inputs(
        matching_query,
        display_area,
        best_tcodes,
        analysis_context=analysis_context,
        system_context=system_context,
    )

    reason_lines = best_match["reasons"][:3] or ["matched the closest SAP incident pattern available"]
    context_preview = context_snippets[0] if context_snippets else "No additional note snippet was available."

    return f"""Incident
- {ticket['title']}
- Area: {display_area}
- Confidence: {confidence}

Likely Root Cause
- {ticket['root_cause']}

Priority
- {priority}

Likely Owner
{format_list(owners, "SAP application support")}

Required Inputs
{format_list(required_inputs, "Collect the exact SAP error text and affected business object.")}

{build_environment_section(resolved_environment)}

{build_system_section(system_context)}

{build_analysis_section(analysis_context)}

{build_reasoning_sections(reasoning)}

Best T-codes
{format_list(best_tcodes, "No T-code available")}

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


def build_ticket_answer(query, environment, system=None, subsystem=None, analysis_context=None, matching_query=None, scope_query=None):
    bundle = build_solver_bundle(
        query,
        environment,
        system=system,
        subsystem=subsystem,
        analysis_context=analysis_context,
        matching_query=matching_query,
        scope_query=scope_query,
    )
    return build_ticket_answer_from_bundle(bundle)


def build_generic_triage_answer(query, context_snippets, context_source, environment, system_context=None, analysis_context=None, universal_patterns=None):
    system_context = system_context or resolve_system_context(query)
    query_tcodes = sorted(extract_tcodes(query))
    universal_patterns = universal_patterns if universal_patterns is not None else rerank_patterns_for_precision(
        find_universal_pattern(query, top_k=6),
        system_context,
        analysis_context=analysis_context,
    )
    best_pattern_item = universal_patterns[0] if universal_patterns else None
    scoped_area = derive_flexible_area(query, system_context, analysis_context=analysis_context, best_pattern=None)
    pattern_is_strong = bool(best_pattern_item and best_pattern_item["score"] >= 8)
    if (
        pattern_is_strong
        and scoped_area
        and best_pattern_item["pattern"]["area"] != scoped_area
        and has_specific_scope(system_context.get("system_label"))
        and scoped_area in {"Analytics", "HANA / DB", "Workflow / MDG", "ALM / Governance", "Fiori/Gateway"}
        and best_pattern_item["score"] < 14
    ):
        pattern_is_strong = False
    best_pattern = best_pattern_item["pattern"] if pattern_is_strong else None
    pattern_reasons = best_pattern_item["reasons"][:3] if pattern_is_strong else []
    area = refine_area_with_scope(
        derive_flexible_area(query, system_context, analysis_context=analysis_context, best_pattern=best_pattern) or scoped_area,
        system_context,
        analysis_context=analysis_context,
    )
    title = derive_flexible_title(
        system_context,
        area,
        best_pattern=best_pattern if not best_pattern or area == best_pattern["area"] else None,
    )
    priority = infer_priority(query, environment)
    owners = derive_flexible_owners(area, system_context)
    best_tcodes = derive_best_tcodes(query_tcodes, [], best_pattern, system_context=system_context)
    required_inputs = derive_required_inputs(
        query,
        area,
        best_tcodes,
        pattern=best_pattern,
        analysis_context=analysis_context,
        system_context=system_context,
    )
    guidance_lines = build_adaptive_guidance(
        query,
        environment,
        area,
        system_context,
        analysis_context=analysis_context,
    )
    default_checks = build_adaptive_checks(
        query,
        query_tcodes,
        system_context,
        analysis_context=analysis_context,
        best_pattern=best_pattern,
    )

    root_cause_lines = []
    if best_pattern:
        root_cause_lines = list(best_pattern["causes"])
    else:
        root_cause_lines.append("The current ticket does not strongly match one existing SAP runbook, so adaptive triage is being used instead of forcing a narrow diagnosis.")
        if has_specific_scope(system_context.get("system_label")):
            root_cause_lines.append(
                f"The issue most likely sits within {system_context['system_label']}"
                + (
                    f" / {system_context['subsystem_label']}"
                    if has_specific_scope(system_context.get("subsystem_label")) and system_context.get("subsystem_label") != system_context.get("system_label")
                    else ""
                )
                + "."
            )
        domain_signals = (analysis_context or {}).get("domain_signals", [])
        if domain_signals:
            top_domain = domain_signals[0]
            root_cause_lines.append(
                f"Top issue signal points to {top_domain['domain']} because of {', '.join(top_domain['signals'][:3])}."
            )
        evidence = summarize_detected_evidence(analysis_context)
        if evidence:
            root_cause_lines.append(f"Current evidence already includes {', '.join(evidence)}.")

    resolution_lines = build_adaptive_resolution(
        query,
        environment,
        system_context,
        analysis_context=analysis_context,
        best_pattern=best_pattern,
    )
    risk_lines = build_adaptive_risks(
        query,
        environment,
        area,
        system_context,
        analysis_context=analysis_context,
    )
    escalation_lines = [
        "The issue blocks production users, critical postings, interfaces, or business approvals.",
        "The first failing technical component cannot be isolated with the available logs and evidence.",
        f"The identified area '{area}' needs specialist support or a controlled change path.",
    ]
    why_matched = pattern_reasons or [
        "adaptive fallback was used because no single SAP runbook matched strongly enough",
        f"the owning scope was inferred as {system_context.get('system_label', 'SAP landscape')}",
    ]
    related_playbooks = build_related_playbooks(
        query,
        universal_patterns=universal_patterns,
        best_pattern=best_pattern,
        preferred_area=area,
    )
    reasoning = build_advanced_reasoning(
        query,
        system_context,
        analysis_context=analysis_context,
        matches=[],
        universal_patterns=universal_patterns,
    )

    return f"""Incident
- {title}
- Area: {area}
- Confidence: {"Medium" if best_pattern else "Low"}

Likely Root Cause
{format_list(root_cause_lines, "The root cause still needs more evidence.")}

Priority
- {priority}

Likely Owner
{format_list(owners, "SAP application support")}

Required Inputs
{format_list(required_inputs, "Collect enough detail to classify the incident.")}

{build_environment_section(environment)}

{build_system_section(system_context)}

{build_analysis_section(analysis_context)}

{build_reasoning_sections(reasoning)}

Guidance
{format_list(guidance_lines, "Use adaptive SAP triage when the ticket does not map cleanly to one known runbook.")}

Best T-codes
{format_list(best_tcodes, "No T-code available")}

Checks
{format_list(default_checks, "Collect enough detail to classify the incident.")}

Resolution
{format_list(resolution_lines, "Apply the safest validated fix and retest.")}

Risks / Escalation
{format_list(risk_lines, "Use a controlled support path when the issue is still under investigation.")}

Escalate If
{format_list(escalation_lines, "Escalate when the issue affects production or cannot be reproduced safely.")}

Why This Matched
{format_list(why_matched, "matched the closest universal SAP support pattern")}

Supporting Context
- Source: {context_source}
- Snippet: {context_snippets[0] if context_snippets else 'No supporting note snippet was available.'}

Related Playbooks
{format_list(related_playbooks, "No close supporting playbook was identified yet.")}"""


def ask_sap(query, environment=None, provider="auto", system=None, subsystem=None, analysis_context=None):
    clean_query = query.strip()
    if not clean_query:
        return "Please enter an SAP ticket, issue, or question."

    provider = str(provider or "auto").strip().lower()
    provider_aliases = {
        "agent": "agentic",
        "autonomous": "agentic",
        "copilot": "agentic",
        "advanced": "agentic",
        "advanced agent": "agentic",
        "advanced_agent": "agentic",
        "oss": "open_source",
        "open_llm": "openai_compatible",
        "open_source_api": "openai_compatible",
        "local_api": "openai_compatible",
        "hf": "hf_local",
        "transformers": "hf_local",
    }
    provider = provider_aliases.get(provider, provider)

    if provider == "agentic":
        from sap_agent import run_sap_agent

        return sanitize_output_text(
            run_sap_agent(
                clean_query,
                environment=environment,
                system=system,
                subsystem=subsystem,
                analysis_context=analysis_context,
            )
        )

    bundle = build_solver_bundle(
        clean_query,
        environment,
        system=system,
        subsystem=subsystem,
        analysis_context=analysis_context,
    )
    matching_query = bundle["matching_query"]
    resolved_environment = bundle["resolved_environment"]
    context_snippets = bundle["context_snippets"]
    context_source = bundle["context_source"]
    base_answer = build_ticket_answer_from_bundle(bundle)

    if provider == "rules":
        return sanitize_output_text(base_answer)

    if provider == "openai" and not openai_is_configured():
        return sanitize_output_text(
            (
            "OpenAI mode was selected, but OPENAI_API_KEY is not configured.\n\n"
            f"{base_answer}"
            )
        )

    if provider == "ollama" and not ollama_is_available():
        return sanitize_output_text(
            (
            "Ollama mode was selected, but the configured Ollama model is not available at the local endpoint.\n\n"
            f"{base_answer}"
            )
        )

    if provider == "open_source" and not open_source_is_available():
        return sanitize_output_text(
            (
            "Open Source AI mode was selected, but no supported open-source backend is available. Configure Ollama, an OpenAI-compatible local API, or a Hugging Face local model.\n\n"
            f"{base_answer}"
            )
        )

    if provider == "openai_compatible" and not openai_compatible_is_configured():
        return sanitize_output_text(
            (
            "Open LLM mode was selected, but OPEN_LLM_API_BASE_URL and OPEN_LLM_MODEL are not configured. The legacy OPEN_SOURCE_API_* names also still work.\n\n"
            f"{base_answer}"
            )
        )

    if provider == "hf_local" and not hf_local_is_configured():
        return sanitize_output_text(
            (
            "HF local mode was selected, but HF_LOCAL_MODEL is not configured.\n\n"
            f"{base_answer}"
            )
        )

    if is_vector_context_enabled():
        vector_context_snippets, vector_context_source = build_context(
            bundle["matches"],
            matching_query,
            include_vector=True,
        )
        if vector_context_snippets:
            context_snippets = vector_context_snippets
            context_source = vector_context_source

    if provider == "openai" and openai_is_configured():
        return sanitize_output_text(
            enhance_answer_with_openai(
                matching_query,
                resolved_environment,
                base_answer,
                context_snippets,
                context_source,
            )
        )

    if provider == "ollama" and ollama_is_available():
        return sanitize_output_text(
            enhance_answer_with_ollama(
                matching_query,
                resolved_environment,
                base_answer,
                context_snippets,
                context_source,
            )
        )

    if provider == "open_source":
        return sanitize_output_text(
            enhance_answer_with_open_source(
                matching_query,
                resolved_environment,
                base_answer,
                context_snippets,
                context_source,
                backend=get_open_source_backend(),
            )
        )

    if provider == "openai_compatible":
        return sanitize_output_text(
            enhance_answer_with_open_source(
                matching_query,
                resolved_environment,
                base_answer,
                context_snippets,
                context_source,
                backend="openai_compatible",
            )
        )

    if provider == "hf_local":
        return sanitize_output_text(
            enhance_answer_with_open_source(
                matching_query,
                resolved_environment,
                base_answer,
                context_snippets,
                context_source,
                backend="hf_local",
            )
        )

    if provider == "auto" and openai_is_configured():
        openai_answer = try_openai_enhancement(
            matching_query,
            resolved_environment,
            base_answer,
            context_snippets,
            context_source,
        )
        if openai_answer:
            return sanitize_output_text(openai_answer)
        if get_openai_failure_notice():
            return sanitize_output_text(f"{get_openai_failure_notice()}\n\n{base_answer}")

    if provider == "auto" and not openai_is_configured() and open_source_is_available():
        open_source_answer = try_open_source_enhancement(
            matching_query,
            resolved_environment,
            base_answer,
            context_snippets,
            context_source,
        )
        if open_source_answer:
            return sanitize_output_text(open_source_answer)

    if provider == "auto":
        open_source_notice = get_open_source_failure_notice()
        if open_source_notice:
            return sanitize_output_text(f"{open_source_notice}\n\n{base_answer}")
        return sanitize_output_text(base_answer)

    return sanitize_output_text(base_answer)


if __name__ == "__main__":
    print(ask_sap("Transport failed with RC 8 because object missing"))
