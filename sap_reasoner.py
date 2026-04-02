import re


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_/.-]+")


LAYER_PROFILES = {
    "Access / Roles": {
        "signals": [
            "authorization",
            "not authorized",
            "access denied",
            "role",
            "su53",
            "pfcg",
            "user locked",
            "logon",
            "login",
        ],
        "areas": {"Security", "Fiori/Gateway", "ALM / Governance"},
        "systems": {"FIORI_GATEWAY", "GRC", "MDG"},
        "subsystems": {"LAUNCHPAD", "ACCESS", "FIREFIGHTER"},
        "entity_keys": {"users"},
        "domain_signals": {"Security"},
    },
    "Transport / Change": {
        "signals": [
            "transport",
            "import",
            "stms",
            "return code",
            "rc 8",
            "change request",
            "release",
            "charm",
        ],
        "areas": {"Basis", "ALM / Governance"},
        "systems": {"SOLMAN", "ECC", "S4HANA"},
        "subsystems": {"CHARM"},
        "entity_keys": {"transports", "return_codes"},
        "domain_signals": {"Basis"},
    },
    "Interface / Middleware": {
        "signals": [
            "idoc",
            "queue",
            "qrfc",
            "trfc",
            "sm58",
            "sm59",
            "destination",
            "partner",
            "iflow",
            "api",
            "mapping",
            "middleware",
        ],
        "areas": {"Integration", "Analytics"},
        "systems": {"PI_PO", "INTEGRATION_SUITE", "BW4HANA", "ECC"},
        "subsystems": {"CHANNELS", "MESSAGING", "DATA", "IDOC", "IFLOWS", "API"},
        "entity_keys": {"idocs", "queues", "status_codes"},
        "domain_signals": {"Integration"},
    },
    "UI / Workflow": {
        "signals": [
            "fiori",
            "launchpad",
            "odata",
            "http 500",
            "workflow",
            "approval",
            "tile",
            "gateway",
            "service",
        ],
        "areas": {"Fiori/Gateway", "Workflow / MDG"},
        "systems": {"FIORI_GATEWAY", "MDG", "S4HANA"},
        "subsystems": {"LAUNCHPAD", "ODATA", "WORKFLOW"},
        "entity_keys": {"http_codes", "objects"},
        "domain_signals": {"Fiori/Gateway"},
    },
    "Application / Process": {
        "signals": [
            "invoice",
            "posting",
            "payment",
            "pricing",
            "condition",
            "delivery",
            "material",
            "stock",
            "vendor",
            "customer",
            "document",
        ],
        "areas": {"FI", "MM", "SD", "MM/SD", "Workflow / MDG"},
        "systems": {"S4HANA", "ECC", "MDG"},
        "subsystems": {"FINANCE", "LOGISTICS", "WORKFLOW"},
        "entity_keys": {"objects"},
        "domain_signals": {"FI", "MM/SD"},
    },
    "Batch / Scheduling": {
        "signals": [
            "job",
            "batch",
            "sm37",
            "sm36",
            "variant",
            "spool",
            "scheduler",
            "cancelled",
        ],
        "areas": {"Basis", "Analytics"},
        "systems": {"ECC", "S4HANA", "BW4HANA"},
        "subsystems": {"ABAP", "REPORTING"},
        "entity_keys": set(),
        "domain_signals": {"Basis"},
    },
    "Performance / Capacity": {
        "signals": [
            "slow",
            "performance",
            "cpu",
            "memory",
            "response time",
            "work process",
            "blocked process",
            "expensive statement",
        ],
        "areas": {"Basis", "HANA / DB", "Analytics"},
        "systems": {"HANA_DB", "BW4HANA", "S4HANA", "ECC"},
        "subsystems": {"PERFORMANCE", "REPORTING"},
        "entity_keys": set(),
        "domain_signals": {"Basis"},
    },
    "Database / Infrastructure": {
        "signals": [
            "hana",
            "database",
            "sql",
            "dbacockpit",
            "replication",
            "backup",
            "restore",
            "host",
            "disk",
            "tablespace",
            "oom",
        ],
        "areas": {"HANA / DB", "Basis"},
        "systems": {"HANA_DB"},
        "subsystems": {"PERFORMANCE", "REPLICATION"},
        "entity_keys": set(),
        "domain_signals": set(),
    },
}


SYSTEM_ROUTES = {
    "FIORI_GATEWAY": [
        "User action or app tile",
        "Gateway or OData service",
        "Authorization or RFC trust",
        "Backend application logic",
        "Database or runtime layer",
    ],
    "BW4HANA": [
        "Process chain or reporting trigger",
        "ODP extraction or source queue",
        "RFC or source connectivity",
        "Transformation or provider load",
        "Target reporting object",
    ],
    "HANA_DB": [
        "Application workload or batch trigger",
        "Expensive SQL or lock contention",
        "Memory, CPU, or I/O pressure",
        "Host, replication, or backup services",
    ],
    "PI_PO": [
        "Sender application",
        "Adapter or channel connectivity",
        "Queue or mapping runtime",
        "Receiver application",
        "Business acknowledgement",
    ],
    "INTEGRATION_SUITE": [
        "Caller or source application",
        "iFlow or API policy",
        "Connector or credential layer",
        "Receiver service",
        "Business acknowledgement",
    ],
    "MDG": [
        "Change request or workflow step",
        "Validation or rule framework",
        "Replication or DRF layer",
        "Target system posting",
    ],
    "SOLMAN": [
        "Change or monitoring document",
        "Governance or approval step",
        "Transport linkage",
        "Managed system import or runtime",
    ],
}


LAYER_ACTIONS = {
    "Access / Roles": [
        "Validate the affected user, role, technical account, and the exact failed authorization before changing application logic.",
        "Use SU53, ST01, role design, and the same user context to prove the first failing control.",
    ],
    "Transport / Change": [
        "Confirm the transport or change document status, sequence, prerequisites, and activation state before reimporting or rebuilding requests.",
        "Keep transport correction and application retest as separate controlled steps.",
    ],
    "Interface / Middleware": [
        "Prove whether the first blocker is payload content, master data, credentials, mapping, or connectivity before replaying messages.",
        "Fix the upstream issue first, then reprocess only the failed queue, IDoc, or message set.",
    ],
    "UI / Workflow": [
        "Start with the same failing user action and trace the app, workflow step, and service call end to end.",
        "Use Gateway or workflow logs to locate the first failing layer before changing backend configuration.",
    ],
    "Application / Process": [
        "Validate the business object state, master data, organizational data, and process controls before changing custom code or infrastructure.",
        "Retest the full process flow after the fix, not just the error screen.",
    ],
    "Batch / Scheduling": [
        "Read the job log, variant, spool, and runtime user first so the failure is not mistaken for an application issue.",
        "Only reschedule after the missing dependency or authorization is corrected.",
    ],
    "Performance / Capacity": [
        "Tie the slowdown to one workload, statement, batch, or user population before changing system parameters.",
        "Contain blocking work or abnormal load before tuning memory or process counts.",
    ],
    "Database / Infrastructure": [
        "Validate database alerts, host health, replication, backup, and expensive SQL evidence before changing application logic.",
        "Use the safest containment action first, especially in production.",
    ],
}


def normalize_text(text):
    return " ".join(str(text or "").split())


def tokenize(text):
    return TOKEN_PATTERN.findall(normalize_text(text).lower())


def summarize_confidence(top_score, runner_up):
    if top_score >= 16 and top_score >= runner_up + 4:
        return "High"
    if top_score >= 8:
        return "Medium"
    return "Low"


def build_query_evidence(query, analysis_context):
    parts = [normalize_text(query)]
    if analysis_context and analysis_context.get("ocr_text"):
        parts.append(normalize_text(analysis_context["ocr_text"]))
    return " ".join(part for part in parts if part).strip().lower()


def score_layer(layer_name, profile, query_text, query_tokens, system_context, analysis_context, matches, universal_patterns):
    score = 0
    reasons = []
    matched_signals = []

    for signal in profile["signals"]:
        if signal in query_text:
            score += 3
            matched_signals.append(signal)
    if matched_signals:
        reasons.append(f"query signals: {', '.join(matched_signals[:3])}")

    entities = (analysis_context or {}).get("entities", {})
    present_entity_keys = [key for key in profile["entity_keys"] if entities.get(key)]
    if present_entity_keys:
        score += len(present_entity_keys) * 2
        entity_summary = []
        for key in present_entity_keys[:2]:
            entity_summary.append(f"{key}={', '.join(entities[key][:2])}")
        reasons.append(f"entities: {', '.join(entity_summary)}")

    system_id = str(system_context.get("system_id", "")).upper()
    subsystem_id = str(system_context.get("subsystem_id", "")).upper()
    if system_id in profile["systems"]:
        score += 3
        reasons.append(f"system scope {system_id}")
    if subsystem_id in profile["subsystems"]:
        score += 4
        reasons.append(f"subsystem scope {subsystem_id}")

    domain_signals = (analysis_context or {}).get("domain_signals", [])
    for signal in domain_signals[:2]:
        if signal.get("domain") in profile["domain_signals"]:
            score += 3
            reasons.append(f"nlp domain {signal['domain']}")
            break

    for match in matches or []:
        if match["ticket"]["area"] in profile["areas"]:
            score += max(2, min(6, int(match.get("score", 0) / 15) + 1))
            reasons.append(f"runbook area {match['ticket']['area']}")
            if match["reasons"]:
                reasons.append(f"runbook evidence {match['reasons'][0]}")
            break

    for item in universal_patterns or []:
        if item["pattern"]["area"] in profile["areas"]:
            score += 1
            reasons.append(f"pattern area {item['pattern']['area']}")
            break

    return {
        "layer": layer_name,
        "score": score,
        "reasons": reasons[:3],
    }


def build_failure_chain(system_context, primary_layer):
    system_id = str(system_context.get("system_id", "")).upper()
    chain = list(SYSTEM_ROUTES.get(system_id, []))
    if not chain:
        chain = [
            "User action or business trigger",
            "UI, batch, or interface entry point",
            "Application or workflow logic",
            "Master data or configuration",
            "Database or infrastructure layer",
        ]

    if primary_layer == "Access / Roles":
        chain.insert(1, "Authorization and role check")
    elif primary_layer == "Transport / Change":
        chain.insert(1, "Transport, activation, or change-control layer")
    elif primary_layer == "Interface / Middleware":
        chain.insert(1, "Queue, RFC, or middleware transport")
    elif primary_layer == "Performance / Capacity":
        chain.insert(1, "Runtime pressure and blocking workload")
    elif primary_layer == "Database / Infrastructure":
        chain.insert(1, "Database or host health")

    unique_chain = []
    for item in chain:
        if item and item not in unique_chain:
            unique_chain.append(item)
    return unique_chain[:6]


def build_decision_path(system_context, primary_layer, analysis_context, failure_chain):
    steps = []
    steps.extend(LAYER_ACTIONS.get(primary_layer, []))

    if system_context.get("system_tools"):
        steps.append(f"Use the main tools for this stack first: {', '.join(system_context['system_tools'][:5])}.")

    entities = (analysis_context or {}).get("entities", {})
    if entities.get("http_codes"):
        steps.append("Use the HTTP error evidence to trace the exact failing service, log entry, and backend user context.")
    if entities.get("transports"):
        steps.append("Do not retry the change blindly until the failing object, prerequisite, or import sequence is confirmed.")
    if entities.get("idocs") or entities.get("queues"):
        steps.append("Do not replay queues or IDocs until the upstream application, mapping, or master-data blocker is fixed.")
    if entities.get("objects"):
        steps.append(f"Keep the same business object in scope during troubleshooting: {', '.join(entities['objects'][:3])}.")

    if failure_chain:
        steps.append(f"Trace the issue in this order: {' -> '.join(failure_chain[:4])}.")

    unique_steps = []
    for step in steps:
        if step and step not in unique_steps:
            unique_steps.append(step)
    return unique_steps[:6]


def build_hypotheses(matches, universal_patterns, top_layers):
    hypotheses = []
    if matches:
        top_match = matches[0]
        reason_text = "; ".join(top_match["reasons"][:2]) or "strong SAP runbook similarity"
        hypotheses.append(
            f"Primary runbook hypothesis: {top_match['ticket']['title']} ({top_match['ticket']['area']}) because {reason_text}."
        )

    if universal_patterns:
        top_pattern = universal_patterns[0]
        reason_text = "; ".join(top_pattern["reasons"][:2]) or "generic SAP signal overlap"
        hypotheses.append(
            f"Fallback pattern hypothesis: {top_pattern['pattern']['title']} ({top_pattern['pattern']['area']}) because {reason_text}."
        )

    if top_layers:
        top_layer = top_layers[0]
        reason_text = "; ".join(top_layer["reasons"][:2]) or "weighted issue signals"
        hypotheses.append(
            f"Failure-layer hypothesis: {top_layer['layer']} is the most likely first failing control point because of {reason_text}."
        )

    unique = []
    for hypothesis in hypotheses:
        if hypothesis not in unique:
            unique.append(hypothesis)
    return unique[:4]


def build_advanced_reasoning(query, system_context, analysis_context=None, matches=None, universal_patterns=None):
    query_text = build_query_evidence(query, analysis_context)
    query_tokens = set(tokenize(query_text))
    scored_layers = []

    for layer_name, profile in LAYER_PROFILES.items():
        scored_layers.append(
            score_layer(
                layer_name,
                profile,
                query_text,
                query_tokens,
                system_context or {},
                analysis_context or {},
                matches or [],
                universal_patterns or [],
            )
        )

    scored_layers = [item for item in scored_layers if item["score"] > 0]
    scored_layers.sort(key=lambda item: item["score"], reverse=True)
    top_layers = scored_layers[:3]

    primary_layer = top_layers[0]["layer"] if top_layers else "Application / Process"
    runner_up = top_layers[1]["score"] if len(top_layers) > 1 else 0
    confidence = summarize_confidence(top_layers[0]["score"], runner_up) if top_layers else "Low"
    hypotheses = build_hypotheses(matches or [], universal_patterns or [], top_layers)
    failure_chain = build_failure_chain(system_context or {}, primary_layer)
    decision_path = build_decision_path(system_context or {}, primary_layer, analysis_context or {}, failure_chain)

    diagnosis_lines = [
        f"Primary failure layer: {primary_layer}",
        f"Algorithm confidence: {confidence}",
    ]
    if top_layers:
        for layer in top_layers[:2]:
            if layer["reasons"]:
                diagnosis_lines.append(
                    f"{layer['layer']} score {layer['score']}: {'; '.join(layer['reasons'])}"
                )
    diagnosis_lines.extend(hypotheses)

    return {
        "primary_layer": primary_layer,
        "confidence": confidence,
        "top_layers": top_layers,
        "diagnosis_lines": diagnosis_lines[:6],
        "failure_chain": failure_chain[:5],
        "decision_path": decision_path[:6],
    }
