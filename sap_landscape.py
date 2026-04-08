import json
import re
from copy import deepcopy
from functools import lru_cache
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
LANDSCAPE_OVERRIDE_PATH = BASE_DIR / "sap_landscape.json"
TOKEN_PATTERN = re.compile(r"[a-z0-9_./-]+")

DEFAULT_SYSTEM_CONTEXT = {
    "system_id": "AUTO",
    "system_label": "Cross-system SAP landscape",
    "system_type": "Shared SAP enterprise estate",
    "system_tools": ["SM59", "ST22", "SM21", "STMS"],
    "subsystem_id": "AUTO",
    "subsystem_label": "Shared service or subsystem not yet classified",
    "subsystem_focus": "Use the system and subsystem selectors to narrow the incident to the owning SAP stack.",
    "integration_points": [
        "RFC / BAPI",
        "IDoc / ALE",
        "OData, SOAP, or REST",
        "Jobs, queues, and middleware",
    ],
    "integration_guidance": [
        "Map the incident to the owning SAP product before retesting or reprocessing.",
        "Document the upstream source, downstream target, and technical connector involved.",
    ],
    "integration_specs": [
        "Connector baseline: confirm the source system, target system, client, business object, and technical owner before changing retries or reprocessing.",
        "Security baseline: validate destination credentials, trusted-system setup, certificates, OAuth tokens, or API keys before changing payload logic.",
    ],
    "matched_terms": [],
}

INTEGRATION_SPEC_LIBRARY = {
    "LANDSCAPE": [
        "Connector baseline: confirm the source system, target system, client, business object, and technical owner before changing retries or reprocessing.",
        "Security baseline: validate destination credentials, trusted-system setup, certificates, OAuth tokens, or API keys before changing payload logic.",
    ],
    "RFC": [
        "RFC/BAPI: verify SM59 destination, target client and system number, gateway host/service, Unicode or SNC settings, and trusted-RFC or logon credentials.",
        "RFC monitoring: use SM59, SMGW, SM58, ST22, and traces on both caller and target systems before retrying.",
    ],
    "QUEUES": [
        "qRFC/tRFC/bgRFC: capture queue name, LUW key, SYSFAIL text, and retry state before replaying work.",
        "Queue control: review SMQ1, SMQ2, SM58, and upstream application errors so reprocessing does not duplicate payloads.",
    ],
    "IDOC": [
        "IDoc/ALE: validate WE20 partner profile, WE21 port, BD64 distribution model, message type/basic type, and logical-system mapping.",
        "IDoc recovery: review WE02/WE05/BD87 status flow and reprocess only the failed payload after the posting error is fixed.",
    ],
    "ODATA": [
        "OData/Gateway: verify /IWFND/MAINT_SERVICE registration, system alias, SICF activation, metadata, backend RFC destination, and service user authorization.",
        "Gateway monitoring: use /IWFND/ERROR_LOG, /IWBEP/ERROR_LOG, ST22, SU53, and browser traces for the same failing user action.",
    ],
    "API": [
        "API/REST: confirm endpoint URL, method, payload schema, OAuth or API-key policy, timeout, rate limits, and correlation/message IDs.",
        "API troubleshooting: compare proxy logs, backend HTTP response, token validity, and caller headers before changing the consuming app.",
    ],
    "SOAP": [
        "SOAP/XI: validate WSDL or service contract version, endpoint URL, certificate or trust chain, SOAP action, and runtime credentials.",
    ],
    "PI_PO": [
        "PI/PO: check sender and receiver channels, adapter credentials, certificates, ESR mapping objects, and communication channel status.",
        "PI/PO monitoring: use SXMB_MONI, NWA, adapter engine logs, SMQ1/SMQ2, and payload trace to isolate adapter, mapping, or application failure.",
    ],
    "CPI": [
        "Integration Suite/CPI: confirm iFlow artifact version, runtime configuration, security material, endpoint URL, and adapter connectivity.",
        "CPI monitoring: use message monitoring, trace mode, MPL IDs, API analytics, and backend response logs before redeploying.",
    ],
    "JOBS": [
        "Background interfaces: capture job name/count, variant, event trigger, batch user, and downstream connector or file/API handoff before rerunning.",
    ],
    "DB": [
        "Database connectivity: confirm replication or secondary target, consumer connection owner, and backup or recovery dependencies before restart actions.",
    ],
}

DEFAULT_SAP_LANDSCAPE = {
    "ECC": {
        "label": "SAP ECC",
        "type": "Core ERP / ABAP stack",
        "aliases": ["ecc", "erp", "r/3", "sap gui"],
        "integration_points": ["RFC / BAPI", "IDoc / ALE", "Batch jobs"],
        "monitoring_tools": ["SM59", "WE02", "WE20", "ST22", "STMS"],
        "guidance": ["Confirm the client, transport path, and custom object ownership before moving fixes."],
        "subsystems": {
            "ABAP": {
                "label": "ABAP stack and custom code",
                "aliases": ["abap", "z program", "zreport", "se38", "dump"],
                "focus": "Custom code, dumps, enhancements, and runtime failures.",
                "integration_points": ["Transports", "RFC modules", "Jobs"],
                "guidance": ["Check dump, activation state, and transport history before rerunning."],
            },
            "IDOC": {
                "label": "ALE and IDoc processing",
                "aliases": ["idoc", "ale", "we02", "bd87", "partner profile"],
                "focus": "IDoc generation, partner profiles, and reprocessing.",
                "integration_points": ["Partner profiles", "Ports", "qRFC"],
                "guidance": ["Fix the posting or partner issue first, then replay only failed payloads."],
            },
        },
    },
    "S4HANA": {
        "label": "SAP S/4HANA",
        "type": "Digital core ERP",
        "aliases": ["s/4", "s4", "s4hana", "s/4hana"],
        "integration_points": ["Released APIs", "OData and Fiori", "IDoc and SOAP"],
        "monitoring_tools": ["ST22", "SM21", "SM37", "STMS", "SAT"],
        "guidance": ["Decide whether the issue belongs to the digital core, Fiori channel, or integration edge."],
        "subsystems": {
            "FINANCE": {
                "label": "Finance and controlling",
                "aliases": ["finance", "fi", "co", "fb60", "fb08", "f110"],
                "focus": "Posting controls, reversals, close activities, and payments.",
                "integration_points": ["Universal journal", "Subledger postings", "Approvals"],
                "guidance": ["Validate periods, document status, and downstream accounting impact before reposting."],
            },
            "LOGISTICS": {
                "label": "Logistics and execution",
                "aliases": ["sd", "mm", "pp", "ewm", "tm", "va01", "migo", "delivery"],
                "focus": "Sales, procurement, inventory, warehousing, and execution.",
                "integration_points": ["Pricing", "ATP", "Warehouse and transport handoffs"],
                "guidance": ["Retest the full document flow so dependent logistics steps still complete correctly."],
            },
        },
    },
    "BW4HANA": {
        "label": "SAP BW/4HANA",
        "type": "Analytics and data warehousing",
        "aliases": ["bw", "bw4", "bw/4", "bw4hana", "process chain"],
        "integration_points": ["ODP loads", "Process chains", "Source system RFC"],
        "monitoring_tools": ["RSPC", "RSA1", "ODQMON", "SM37"],
        "guidance": ["Separate source extraction failures from BW transformation or process-chain failures."],
        "subsystems": {
            "DATA": {
                "label": "Data acquisition and extraction",
                "aliases": ["extractor", "odp", "datasource", "delta", "odqmon"],
                "focus": "Source extraction, delta handling, and source system connectivity.",
                "integration_points": ["Source systems", "ODP queues", "RFC"],
                "guidance": ["Validate queue state and source-system health before changing BW logic."],
            },
            "REPORTING": {
                "label": "Queries and reporting",
                "aliases": ["query", "reporting", "bex", "analysis office", "sac"],
                "focus": "Query output, reporting semantics, and consumer issues.",
                "integration_points": ["Queries", "Providers", "Consumer tools"],
                "guidance": ["Confirm whether the issue is data correctness, authorization, or layout."],
            },
        },
    },
    "FIORI_GATEWAY": {
        "label": "SAP Fiori and Gateway",
        "type": "UX, launchpad, and OData layer",
        "aliases": ["fiori", "launchpad", "gateway", "odata", "tile", "ui5"],
        "integration_points": ["OData services", "ICF services", "Backend RFC trusts"],
        "monitoring_tools": ["/IWFND/ERROR_LOG", "/IWBEP/ERROR_LOG", "SICF", "SU53"],
        "guidance": ["Check whether the failure is in the UI layer, OData service, role design, or backend posting step."],
        "subsystems": {
            "LAUNCHPAD": {
                "label": "Launchpad, catalogs, and roles",
                "aliases": ["launchpad", "tile", "catalog", "target mapping", "role"],
                "focus": "App launch, tile visibility, and role assignment issues.",
                "integration_points": ["Catalogs", "Roles", "ICF services"],
                "guidance": ["Validate the app catalog, role assignment, and target mapping first."],
            },
            "ODATA": {
                "label": "Gateway OData services",
                "aliases": ["odata", "http 500", "/iwfnd/error_log", "service"],
                "focus": "OData runtime, metadata, service registration, and backend service failures.",
                "integration_points": ["Registered services", "RFC destinations", "Metadata"],
                "guidance": ["Use Gateway error logs and the failing user action to isolate the service."],
            },
        },
    },
    "PI_PO": {
        "label": "SAP PI/PO",
        "type": "Middleware and enterprise integration",
        "aliases": ["pi", "po", "adapter engine", "aex", "sxmb"],
        "integration_points": ["Channels", "Mappings", "SOAP, IDoc, RFC, file adapters"],
        "monitoring_tools": ["SXMB_MONI", "SMQ1", "SMQ2", "SM58", "NWA"],
        "guidance": ["Confirm whether the issue is in adapter connectivity, mapping, or the sender or receiver application."],
        "subsystems": {
            "CHANNELS": {
                "label": "Adapter channels and connectivity",
                "aliases": ["channel", "adapter", "receiver", "sender"],
                "focus": "Channel connectivity, credentials, certificates, and protocol failures.",
                "integration_points": ["SOAP", "SFTP", "IDoc", "RFC", "REST"],
                "guidance": ["Validate channel credentials and network reachability before changing mappings."],
            },
            "MESSAGING": {
                "label": "Messaging and queue processing",
                "aliases": ["sxmb_moni", "queue", "stuck message", "retry", "mapping"],
                "focus": "Pipeline processing, payload errors, retries, and queue backlogs.",
                "integration_points": ["Pipeline steps", "Queues", "Mapping runtime"],
                "guidance": ["Fix the mapping or application error first, then replay only failed messages."],
            },
        },
    },
    "INTEGRATION_SUITE": {
        "label": "SAP Integration Suite",
        "type": "Cloud integration and APIs",
        "aliases": ["cpi", "integration suite", "cloud integration", "iflow"],
        "integration_points": ["iFlows", "APIs", "SAP and non-SAP cloud connectivity"],
        "monitoring_tools": ["Message monitoring", "iFlow trace", "API analytics"],
        "guidance": ["Separate connector or credential issues from mapping or payload issues before redeploying."],
        "subsystems": {
            "IFLOWS": {
                "label": "Cloud Integration iFlows",
                "aliases": ["iflow", "cpi", "trace", "mapping"],
                "focus": "iFlow routing, mappings, traces, and endpoint behavior.",
                "integration_points": ["HTTP", "SOAP", "SFTP", "RFC", "OData"],
                "guidance": ["Confirm whether the failure is in the artifact, adapter, or payload mapping."],
            },
            "API": {
                "label": "API management and proxy layer",
                "aliases": ["api management", "api proxy", "rate limit", "token", "api key"],
                "focus": "API policies, authentication, proxy configuration, and traffic controls.",
                "integration_points": ["OAuth", "API keys", "Backend HTTP services"],
                "guidance": ["Validate tokens, proxy policies, and backend reachability before blaming the caller."],
            },
        },
    },
    "GRC": {
        "label": "SAP GRC",
        "type": "Access governance and risk controls",
        "aliases": ["grc", "access request", "firefighter", "sod", "risk analysis"],
        "integration_points": ["Role sync", "Access requests", "Emergency access logs"],
        "monitoring_tools": ["NWBC", "GRACREQ", "GRACROLE", "SU53"],
        "guidance": ["Differentiate a GRC workflow issue from a backend role or connector issue before routing."],
        "subsystems": {
            "ACCESS": {
                "label": "Access requests and provisioning",
                "aliases": ["access request", "provisioning", "grac", "request number"],
                "focus": "Access request lifecycle, approval routing, and provisioning status.",
                "integration_points": ["Provisioning jobs", "Approvals", "Connectors"],
                "guidance": ["Validate request status, approval stage, and connector health before retriggering."],
            },
            "FIREFIGHTER": {
                "label": "Emergency access management",
                "aliases": ["firefighter", "ffid", "emergency access"],
                "focus": "Emergency IDs, owners, controllers, and log review flows.",
                "integration_points": ["Emergency IDs", "Log sync", "Approvals"],
                "guidance": ["Check owner and controller assignments before changing firefighter credentials."],
            },
        },
    },
    "SOLMAN": {
        "label": "SAP Solution Manager",
        "type": "ALM, ChaRM, and monitoring",
        "aliases": ["solman", "solution manager", "charm", "focused run"],
        "integration_points": ["Change control", "Monitoring", "Diagnostics"],
        "monitoring_tools": ["SM_WORKCENTER", "SOLMAN_SETUP", "ChaRM"],
        "guidance": ["Decide whether the ticket is a monitoring symptom, a change-control problem, or a managed-system issue."],
        "subsystems": {
            "CHARM": {
                "label": "Change request management",
                "aliases": ["charm", "change request", "normal change", "urgent change"],
                "focus": "Change documents, approvals, transport assignment, and deployment governance.",
                "integration_points": ["Transport management", "Approvals", "Managed systems"],
                "guidance": ["Validate document status and transport linkage before forcing movement outside ChaRM."],
            },
            "MONITORING": {
                "label": "System and interface monitoring",
                "aliases": ["alert", "monitoring", "health check", "mai"],
                "focus": "Alerting, metrics, dashboards, and monitored object health.",
                "integration_points": ["Managed systems", "Interfaces", "Alert notifications"],
                "guidance": ["Confirm whether the alert reflects a real outage or stale monitoring setup."],
            },
        },
    },
    "HANA_DB": {
        "label": "SAP HANA Database",
        "type": "Database and platform runtime",
        "aliases": ["hana", "database", "dbacockpit", "sql", "memory", "cpu"],
        "integration_points": ["Database connectivity", "Replication", "Backup and recovery"],
        "monitoring_tools": ["DBACOCKPIT", "HANA Studio", "ST04"],
        "guidance": ["Confirm whether the issue is application logic, expensive SQL, or platform capacity."],
        "subsystems": {
            "PERFORMANCE": {
                "label": "Database runtime and SQL performance",
                "aliases": ["sql", "expensive statement", "performance", "plan cache", "st04"],
                "focus": "Expensive SQL, locks, and statement performance.",
                "integration_points": ["SQL statements", "Sessions", "Indexes"],
                "guidance": ["Tie the expensive statement back to the SAP transaction or job before tuning."],
            },
            "REPLICATION": {
                "label": "Replication, backup, and recovery",
                "aliases": ["replication", "backup", "restore", "log shipping"],
                "focus": "System replication, backup chains, and recovery readiness.",
                "integration_points": ["Primary and secondary systems", "Backup storage", "Recovery procedures"],
                "guidance": ["Validate replication status and backup consistency before restarting services."],
            },
        },
    },
    "MDG": {
        "label": "SAP Master Data Governance",
        "type": "Master data governance and workflows",
        "aliases": ["mdg", "governance", "change request", "master data workflow"],
        "integration_points": ["Workflow", "Replication", "Validation rules"],
        "monitoring_tools": ["NWBC", "DRFLOG", "SLG1", "SWI1"],
        "guidance": ["Separate workflow routing, validation rule, and replication failures before routing the ticket."],
        "subsystems": {
            "WORKFLOW": {
                "label": "Change requests and workflow",
                "aliases": ["workflow", "change request", "approval", "cr"],
                "focus": "Change request lifecycle, approvals, and agent determination.",
                "integration_points": ["Workflow", "Business rules", "UI apps"],
                "guidance": ["Check workflow status and approver determination before changing the data model."],
            },
            "REPLICATION": {
                "label": "Replication and distribution",
                "aliases": ["replication", "drf", "drflog", "distribution model"],
                "focus": "Replication to target systems, DRF configuration, and outbound messages.",
                "integration_points": ["DRF", "Target systems", "Queues and messages"],
                "guidance": ["Fix the target-system or queue issue before resending master data."],
            },
        },
    },
}


def _tokenize(text):
    return TOKEN_PATTERN.findall(str(text or "").lower())


def _normalize_choice(value):
    return str(value or "").strip().upper()


def _deep_merge(base, override):
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _score_aliases(query_tokens, aliases):
    if not aliases:
        return 0, []
    query_joined = " ".join(query_tokens)
    query_set = set(query_tokens)
    score = 0
    reasons = []
    for alias in aliases:
        normalized = " ".join(_tokenize(alias))
        if not normalized:
            continue
        if " " in normalized or "/" in normalized or "-" in normalized:
            if normalized in query_joined:
                score += 4
                reasons.append(alias)
        elif normalized in query_set:
            score += 3
            reasons.append(alias)
    return score, reasons[:5]


def _unique_list(items):
    unique = []
    seen = set()
    for item in items:
        normalized = str(item).strip().lower()
        if not normalized or normalized in seen:
            continue
        unique.append(str(item).strip())
        seen.add(normalized)
    return unique


def _derive_integration_specifications(system_id, subsystem_id, system_profile, subsystem_profile):
    use_system_wide_connectors = subsystem_id in {"", "AUTO"}
    catalog = []
    for item in [
        system_id,
        subsystem_id,
        system_profile.get("label", ""),
        system_profile.get("type", ""),
        subsystem_profile.get("label", ""),
        subsystem_profile.get("focus", ""),
    ]:
        catalog.append(str(item or ""))
    catalog.extend(system_profile.get("aliases", []))
    catalog.extend(subsystem_profile.get("aliases", []))
    if use_system_wide_connectors:
        catalog.extend(system_profile.get("integration_points", []))
        catalog.extend(system_profile.get("monitoring_tools", []))
    catalog.extend(subsystem_profile.get("integration_points", []))
    joined = " ".join(catalog).lower()
    tokens = set(_tokenize(joined))

    spec_keys = ["LANDSCAPE"]
    if tokens.intersection({"rfc", "bapi", "sm59"}) or system_id in {"ECC", "BW4HANA"}:
        spec_keys.append("RFC")
    if tokens.intersection({"qrfc", "trfc", "bgfrc", "queue", "queues", "smq1", "smq2", "sm58"}):
        spec_keys.append("QUEUES")
    if tokens.intersection({"idoc", "idocs", "ale", "we02", "we05", "we20", "bd87", "bd64"}):
        spec_keys.append("IDOC")
    if tokens.intersection({"odata", "gateway", "sicf"}) or any(term in joined for term in ["/iwfnd", "/iwbep", "system alias"]) or system_id == "FIORI_GATEWAY":
        spec_keys.append("ODATA")
    if tokens.intersection({"api", "apis", "rest", "oauth", "proxy", "token"}) or any(term in joined for term in ["api key", "rate limit"]) or subsystem_id == "API":
        spec_keys.append("API")
    if tokens.intersection({"soap", "wsdl"}):
        spec_keys.append("SOAP")
    if system_id == "PI_PO" or tokens.intersection({"sxmb", "channel", "channels", "mapping", "aex", "nwa"}) or "adapter engine" in joined:
        spec_keys.append("PI_PO")
    if system_id == "INTEGRATION_SUITE" or tokens.intersection({"iflow", "iflows", "cpi", "mpl"}) or any(term in joined for term in ["integration suite", "message monitoring"]):
        spec_keys.append("CPI")
    if tokens.intersection({"job", "jobs", "batch", "sm37"}):
        spec_keys.append("JOBS")
    if system_id == "HANA_DB" or tokens.intersection({"database", "replication", "jdbc", "odbc", "dbacockpit"}):
        spec_keys.append("DB")

    specs = []
    for key in _unique_list(spec_keys):
        specs.extend(INTEGRATION_SPEC_LIBRARY.get(key, []))
    return _unique_list(specs)[:8]


@lru_cache(maxsize=1)
def get_sap_landscape():
    landscape = deepcopy(DEFAULT_SAP_LANDSCAPE)
    if LANDSCAPE_OVERRIDE_PATH.exists():
        try:
            override = json.loads(LANDSCAPE_OVERRIDE_PATH.read_text(encoding="utf-8"))
        except Exception:
            override = {}
        _deep_merge(landscape, override)
    return landscape


def has_landscape_override():
    return LANDSCAPE_OVERRIDE_PATH.exists()


def get_landscape_counts():
    landscape = get_sap_landscape()
    return {
        "systems": len(landscape),
        "subsystems": sum(len(profile.get("subsystems", {})) for profile in landscape.values()),
    }


def get_system_choices(include_auto=True):
    choices = []
    if include_auto:
        choices.append(("AUTO", "Auto detect / cross-system"))
    for system_id, profile in get_sap_landscape().items():
        choices.append((system_id, profile.get("label", system_id)))
    return choices


def get_subsystem_choices(system_id, include_auto=True):
    choices = []
    if include_auto:
        choices.append(("AUTO", "Auto detect / shared service"))
    profile = get_sap_landscape().get(_normalize_choice(system_id))
    if not profile:
        return choices
    for subsystem_id, subsystem in profile.get("subsystems", {}).items():
        choices.append((subsystem_id, subsystem.get("label", subsystem_id)))
    return choices


def resolve_system_context(query, system=None, subsystem=None):
    query_tokens = _tokenize(query)
    landscape = get_sap_landscape()
    selected_system = _normalize_choice(system)
    selected_subsystem = _normalize_choice(subsystem)
    best_system_id = None
    best_system_score = 0
    best_system_reasons = []

    if selected_system not in {"", "AUTO"} and selected_system in landscape:
        best_system_id = selected_system
    else:
        for system_id, profile in landscape.items():
            aliases = profile.get("aliases", []) + [system_id, profile.get("label", "")]
            score, reasons = _score_aliases(query_tokens, aliases)
            for subsystem_profile in profile.get("subsystems", {}).values():
                subsystem_score, subsystem_reasons = _score_aliases(
                    query_tokens,
                    subsystem_profile.get("aliases", []) + [subsystem_profile.get("label", "")],
                )
                if subsystem_score:
                    score += min(subsystem_score, 5)
                    reasons.extend(subsystem_reasons)
            if score > best_system_score:
                best_system_id = system_id
                best_system_score = score
                best_system_reasons = reasons[:5]

    if not best_system_id:
        return deepcopy(DEFAULT_SYSTEM_CONTEXT)

    system_profile = landscape[best_system_id]
    subsystem_profiles = system_profile.get("subsystems", {})
    best_subsystem_id = None
    best_subsystem_score = 0
    best_subsystem_reasons = []

    if selected_subsystem not in {"", "AUTO"} and selected_subsystem in subsystem_profiles:
        best_subsystem_id = selected_subsystem
    else:
        for subsystem_id, subsystem_profile in subsystem_profiles.items():
            aliases = subsystem_profile.get("aliases", []) + [subsystem_id, subsystem_profile.get("label", "")]
            score, reasons = _score_aliases(query_tokens, aliases)
            if score > best_subsystem_score:
                best_subsystem_id = subsystem_id
                best_subsystem_score = score
                best_subsystem_reasons = reasons[:5]

    subsystem_profile = subsystem_profiles.get(best_subsystem_id, {})
    return {
        "system_id": best_system_id,
        "system_label": system_profile.get("label", best_system_id),
        "system_type": system_profile.get("type", "SAP system"),
        "system_tools": system_profile.get("monitoring_tools", []),
        "subsystem_id": best_subsystem_id or "AUTO",
        "subsystem_label": subsystem_profile.get("label", "Shared service or general system scope"),
        "subsystem_focus": subsystem_profile.get(
            "focus",
            "Use subsystem-specific logs and ownership boundaries to narrow the incident quickly.",
        ),
        "integration_points": _unique_list(
            system_profile.get("integration_points", []) + subsystem_profile.get("integration_points", [])
        )
        or deepcopy(DEFAULT_SYSTEM_CONTEXT["integration_points"]),
        "integration_guidance": _unique_list(
            system_profile.get("guidance", []) + subsystem_profile.get("guidance", [])
        )
        or deepcopy(DEFAULT_SYSTEM_CONTEXT["integration_guidance"]),
        "integration_specs": _derive_integration_specifications(
            best_system_id,
            best_subsystem_id or "AUTO",
            system_profile,
            subsystem_profile,
        )
        or deepcopy(DEFAULT_SYSTEM_CONTEXT["integration_specs"]),
        "matched_terms": _unique_list(best_system_reasons + best_subsystem_reasons),
    }
