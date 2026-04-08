from app import (
    build_solver_bundle,
    build_ticket_answer_from_bundle,
    build_joule_workspace,
    extract_tcodes,
    join_section_items,
    shorten_text,
)
from sap_intelligence import analyze_issue_evidence
from tools import get_tcode_info


def _format_bullets(items, fallback):
    cleaned = [str(item).strip() for item in (items or []) if str(item).strip()]
    if not cleaned:
        cleaned = [fallback]
    return "\n".join(f"- {item}" for item in cleaned)


def _append_unique(items, value, limit=None):
    value = str(value or "").strip()
    if not value:
        return items
    normalized_existing = {item.lower() for item in items}
    if value.lower() in normalized_existing:
        return items
    items.append(value)
    if limit:
        return items[:limit]
    return items


def _summarize_entities(analysis_context):
    entities = (analysis_context or {}).get("entities", {})
    findings = []

    entity_labels = [
        ("tcodes", "T-codes"),
        ("transports", "Transports"),
        ("status_codes", "Statuses"),
        ("http_codes", "HTTP errors"),
        ("queues", "Queues"),
        ("users", "Users"),
        ("idocs", "IDocs"),
        ("objects", "Objects"),
    ]
    for entity_key, label in entity_labels:
        values = entities.get(entity_key) or []
        if values:
            findings.append(f"{label}: {', '.join(values[:4])}")

    return findings


def _summarize_tcode_findings(query, analysis_context):
    tcodes = sorted(set(extract_tcodes(query)).union((analysis_context or {}).get("entities", {}).get("tcodes", [])))
    findings = []
    for tcode in tcodes[:3]:
        findings.append(f"{tcode}: {get_tcode_info(tcode)}")
    return findings


def _summarize_hypotheses(matches):
    hypotheses = []
    for index, match in enumerate(matches[:3], start=1):
        ticket = match.get("ticket", {})
        title = ticket.get("title", "Closest SAP playbook")
        score = match.get("score", 0)
        reasons = "; ".join(match.get("reasons", [])[:2]) or "matched the closest SAP support pattern"
        hypotheses.append(f"{index}. {title} (score {score}) because {reasons}")
    return hypotheses


def _build_issue_mix_lines(workstreams):
    lines = []
    for index, workstream in enumerate(workstreams[:4], start=1):
        reasons = "; ".join(workstream.get("reasons", [])[:2]) or "multiple strong SAP signals"
        lines.append(
            f"Workstream {index}: {workstream.get('title', 'Mixed SAP issue')} ({workstream.get('area', 'SAP')}) because {reasons}"
        )
    return lines


def _build_parallel_workstream_lines(workstreams):
    lines = []
    for workstream in workstreams[:4]:
        primary_check = (workstream.get("checks") or ["Collect the first diagnostic trace for this workstream."])[0]
        primary_fix = (workstream.get("resolution") or ["Apply the safest corrective action for this workstream and retest."])[0]
        lines.append(
            f"{workstream.get('area', 'SAP')}: {workstream.get('title', 'Mixed issue')} - Check: {primary_check} Fix path: {primary_fix}"
        )
    return lines


def _build_cross_issue_risks(workstreams):
    if not workstreams:
        return []

    risks = [
        "Do not assume one workaround fixes every symptom until the dependency order is proven.",
        "Keep transports, unlocks, interface reprocessing, and role changes as separate controlled actions when multiple SAP teams are involved.",
        "Avoid replaying queues, IDocs, or jobs before the upstream authorization, transport, or service issue is corrected.",
    ]
    for workstream in workstreams[:3]:
        area = workstream.get("area", "SAP")
        risks = _append_unique(
            risks,
            f"{area} changes may hide evidence needed by the other workstreams if they are applied out of order.",
            limit=5,
        )
    return risks[:5]


def _build_investigation_plan(clean_query, resolved_environment, system_context, analysis_context, mixed_workstreams=None):
    plan = [
        f"Classify the SAP incident in {resolved_environment} and confirm the owning support boundary.",
        "Extract concrete SAP evidence such as T-codes, status codes, queues, users, and HTTP failures.",
        "Compare the ticket against known runbooks and the trained SAP router to rank the most likely playbooks.",
        "Start with the safest validation step before proposing transports, unlocks, reprocessing, or master-data changes.",
        "Package the fix path, validation steps, and escalation threshold for operations and business communication.",
    ]

    if mixed_workstreams:
        plan.insert(1, "Split the ticket into separate SAP workstreams and determine whether one symptom is upstream of the others.")
        plan.insert(2, "Sequence fixes so the most upstream blocker is validated before changing downstream symptoms.")

    matched_terms = system_context.get("matched_terms") or []
    if matched_terms:
        plan.insert(1, f"Use detected SAP scope terms ({', '.join(matched_terms[:4])}) to narrow the system and subsystem.")

    if (analysis_context or {}).get("ocr_text"):
        plan.insert(2, "Fold OCR evidence into the troubleshooting path so screenshot-only clues are not lost.")

    if extract_tcodes(clean_query):
        plan.insert(3, "Use the detected transaction codes to anchor validation in the exact SAP execution path.")

    return plan[:6]


def _build_tool_findings(resolved_environment, system_context, analysis_context, context_snippets, context_source, mixed_workstreams=None):
    findings = [
        f"Landscape resolved to {resolved_environment}.",
        f"System boundary: {system_context.get('system_label', 'Cross-system SAP landscape')}.",
        f"Subsystem boundary: {system_context.get('subsystem_label', 'Shared service or subsystem not yet classified')}.",
    ]

    if mixed_workstreams:
        findings.append(f"Mixed issue detection identified {len(mixed_workstreams)} parallel SAP workstreams.")

    matched_terms = system_context.get("matched_terms") or []
    if matched_terms:
        findings.append(f"System scope was inferred from: {', '.join(matched_terms[:5])}.")

    for entity_line in _summarize_entities(analysis_context)[:5]:
        findings.append(entity_line)

    domain_signals = (analysis_context or {}).get("domain_signals", [])
    if domain_signals:
        top_signal = domain_signals[0]
        findings.append(
            f"Top NLP signal: {top_signal['domain']} ({', '.join(top_signal.get('signals', [])[:4])})."
        )

    for tcode_line in _summarize_tcode_findings((analysis_context or {}).get("combined_text", ""), analysis_context):
        findings.append(f"T-code lookup: {tcode_line}")

    if context_snippets:
        findings.append(f"Knowledge source: {context_source}.")
        findings.append(f"Best supporting snippet: {shorten_text(context_snippets[0], limit=220)}")

    return findings[:8]


def _build_evidence_correlation(matches, context_snippets, analysis_context, mixed_workstreams=None):
    lines = []
    if mixed_workstreams:
        for workstream in mixed_workstreams[:3]:
            reasons = "; ".join(workstream.get("reasons", [])[:3]) or "strong SAP evidence"
            lines.append(f"{workstream.get('title', 'Mixed issue')}: {reasons}.")
    for match in matches[:3]:
        ticket = match.get("ticket", {})
        title = ticket.get("title", "Closest SAP playbook")
        reasons = "; ".join(match.get("reasons", [])[:3]) or "matched the strongest available evidence"
        candidate_line = f"{title}: {reasons}."
        if candidate_line not in lines:
            lines.append(candidate_line)

    semantic_matches = (analysis_context or {}).get("semantic_matches", [])
    if semantic_matches:
        top_semantic = ", ".join(
            f"{item['label']} [{item['type']}, {item['score']}]"
            for item in semantic_matches[:3]
        )
        lines.append(f"Semantic evidence: {top_semantic}.")

    if context_snippets:
        lines.append(f"Context correlation: {shorten_text(context_snippets[0], limit=180)}")

    return lines[:6]


def _build_autonomous_next_step(workspace, system_context, mixed_workstreams=None):
    sections = workspace.get("sections", {})
    steps = []

    if mixed_workstreams:
        first_stream = mixed_workstreams[0]
        second_stream = mixed_workstreams[1] if len(mixed_workstreams) > 1 else None
        first_check = (first_stream.get("checks") or ["Validate the first failing step in this workstream."])[0]
        steps.append(f"Start with the upstream workstream {first_stream.get('area', 'SAP')}: {first_check}")
        if second_stream:
            steps.append(
                f"Do not change the downstream workstream {second_stream.get('area', 'SAP')} until the first workstream is retested cleanly."
            )

    checks = sections.get("Checks") or []
    resolution = sections.get("Resolution") or []
    tcodes = sections.get("Best T-codes") or []

    if checks:
        steps.append(f"Run the first validation check: {checks[0]}")
    if tcodes:
        steps.append(f"Anchor the investigation in {tcodes[0]} before making broader changes.")
    if resolution:
        steps.append(f"Preferred corrective path after validation: {resolution[0]}")
    if system_context.get("integration_guidance"):
        steps.append(system_context["integration_guidance"][0])

    return steps[:4]


def _build_open_questions(workspace, analysis_context, query, mixed_workstreams=None):
    sections = workspace.get("sections", {})
    required_inputs = sections.get("Required Inputs") or []
    lower_query = query.lower()
    questions = []

    for required_input in required_inputs[:5]:
        normalized = required_input.lower()
        if normalized in lower_query:
            continue
        if any(token in lower_query for token in normalized.replace(",", " ").split()[:2]):
            continue
        questions = _append_unique(questions, f"Confirm: {required_input}", limit=4)

    if not questions and (analysis_context or {}).get("image_findings"):
        questions.append("Confirm whether the screenshot evidence matches the exact user action that failed.")

    if mixed_workstreams:
        questions = _append_unique(
            questions,
            "Confirm whether one issue started earlier and caused the later SAP symptoms.",
            limit=4,
        )
        questions = _append_unique(
            questions,
            "Confirm which team owns each workstream before applying parallel changes.",
            limit=4,
        )

    return questions[:4]


def _derive_agent_mode_description():
    return "Grounded SAP runbooks, ticket evidence extraction, landscape scoping, and hypothesis ranking"


def _section_summary(workspace, section_name, limit=2, fallback=""):
    return join_section_items((workspace.get("sections") or {}).get(section_name), limit=limit) or fallback


def _derive_operational_confidence(matches, mixed_workstreams):
    if mixed_workstreams and len(mixed_workstreams) >= 2:
        if len(mixed_workstreams) >= 2 and mixed_workstreams[1].get("score", 0) >= max(10, mixed_workstreams[0].get("score", 0) - 8):
            return "Medium"
        return "Medium"
    if not matches:
        return "Low"
    top_score = matches[0].get("score", 0)
    second_score = matches[1].get("score", 0) if len(matches) > 1 else 0
    if top_score >= 32 and top_score - second_score >= 8:
        return "High"
    if top_score >= 18:
        return "Medium"
    return "Low"


def _derive_failure_layers(clean_query, analysis_context, system_context, mixed_workstreams=None):
    layers = []
    lowered = clean_query.lower()
    entities = (analysis_context or {}).get("entities", {})
    domain_signals = (analysis_context or {}).get("domain_signals", [])
    subsystem_label = str(system_context.get("subsystem_label", "")).lower()
    system_label = str(system_context.get("system_label", "")).lower()

    area_to_layer = {
        "Security": "Access / Roles",
        "Integration": "Interface / Middleware",
        "Basis": "Transport / Change",
        "Fiori/Gateway": "UI / Workflow",
        "Workflow / MDG": "UI / Workflow",
        "HANA / DB": "Database / Infrastructure",
        "ALM / Governance": "Transport / Change",
        "FI": "Application / Process",
        "MM": "Application / Process",
        "SD": "Application / Process",
        "MM/SD": "Application / Process",
        "Analytics": "Application / Process",
        "Cross-System": "Cross-System Dependency",
    }

    for workstream in mixed_workstreams or []:
        layer = area_to_layer.get(workstream.get("area"))
        if layer:
            layers = _append_unique(layers, layer, limit=5)

    if domain_signals:
        top_domain = domain_signals[0].get("domain")
        domain_layer = {
            "Fiori/Gateway": "UI / Workflow",
            "Integration": "Interface / Middleware",
            "Security": "Access / Roles",
            "Basis": "Transport / Change",
        }.get(top_domain)
        if domain_layer:
            layers = _append_unique(layers, domain_layer, limit=5)

    if entities.get("http_codes") or "fiori" in lowered or "gateway" in lowered or "workflow" in lowered:
        layers = _append_unique(layers, "UI / Workflow", limit=5)
    if subsystem_label and "odata" in subsystem_label:
        layers = _append_unique(layers, "UI / Workflow", limit=5)
    if subsystem_label and "workflow" in subsystem_label:
        layers = _append_unique(layers, "UI / Workflow", limit=5)
    if entities.get("transports") or entities.get("return_codes") or "transport" in lowered or "rc " in lowered:
        layers = _append_unique(layers, "Transport / Change", limit=5)
    if entities.get("idocs") or entities.get("queues") or entities.get("status_codes") or "idoc" in lowered or "queue" in lowered:
        layers = _append_unique(layers, "Interface / Middleware", limit=5)
    if entities.get("users") or "authorization" in lowered or "locked" in lowered or "su53" in lowered:
        layers = _append_unique(layers, "Access / Roles", limit=5)
    if "dbif_" in lowered or "sql" in lowered or "hana" in lowered or "database" in lowered or "dump" in lowered:
        layers = _append_unique(layers, "Database / Infrastructure", limit=5)
    if "batch" in lowered or "job" in lowered or "sm37" in lowered:
        layers = _append_unique(layers, "Batch / Scheduling", limit=5)
    if subsystem_label and "idoc" in subsystem_label:
        layers = _append_unique(layers, "Interface / Middleware", limit=5)
    if system_label and "hana" in system_label:
        layers = _append_unique(layers, "Database / Infrastructure", limit=5)

    if not layers:
        layers.append("Application / Process")

    return layers[:5]


def _build_expert_assessment(clean_query, workspace, system_context, analysis_context, matches, mixed_workstreams, resolved_environment):
    owners = (workspace.get("sections") or {}).get("Likely Owner") or []
    layers = _derive_failure_layers(clean_query, analysis_context, system_context, mixed_workstreams=mixed_workstreams)
    topology = "Mixed SAP incident" if mixed_workstreams else "Single dominant SAP issue"
    areas = ", ".join(stream.get("area", "SAP") for stream in mixed_workstreams[:3]) if mixed_workstreams else _section_summary(workspace, "Incident", limit=1, fallback="SAP issue")
    confidence = _derive_operational_confidence(matches, mixed_workstreams)

    lines = [
        f"Case topology: {topology}.",
        f"Likely first-failing layer: {layers[0]}.",
        f"Operational confidence: {confidence}.",
        f"Primary owner bias: {owners[0] if owners else 'SAP application support'}.",
        f"Current issue footprint: {areas}.",
    ]
    if resolved_environment == "PROD":
        lines.append("Production-safe execution is required before any transport, unlock, replay, or master-data correction.")
    return lines[:6]


def _build_failure_boundary(system_context, analysis_context):
    entities = (analysis_context or {}).get("entities", {})
    lines = [
        f"Owning system: {system_context.get('system_label', 'Cross-system SAP landscape')}.",
        f"Owning subsystem: {system_context.get('subsystem_label', 'Shared service or subsystem not yet classified')}.",
    ]

    integration_points = system_context.get("integration_points") or []
    if integration_points:
        lines.append(f"Primary integration boundary: {', '.join(integration_points[:4])}.")

    affected = []
    for key in ["users", "idocs", "transports", "queues", "objects", "http_codes"]:
        values = entities.get(key) or []
        affected.extend(values[:2])
    if affected:
        lines.append(f"Affected identities or objects already visible: {', '.join(affected[:6])}.")

    matched_terms = system_context.get("matched_terms") or []
    if matched_terms:
        lines.append(f"Boundary was inferred from: {', '.join(matched_terms[:5])}.")
    return lines[:5]


def _build_dependency_map(system_context, mixed_workstreams, workspace):
    lines = []
    if mixed_workstreams:
        order = " -> ".join(stream.get("area", "SAP") for stream in mixed_workstreams[:3])
        lines.append(f"Suggested execution order: {order}.")
        lines.append("Retest each workstream before touching the next one so downstream symptoms are not mistaken for root cause.")
        lines.append("Use shared timestamps, users, and business objects to prove whether the second symptom depends on the first.")
        return lines[:4]

    system_label = system_context.get("system_label", "SAP system")
    subsystem_label = system_context.get("subsystem_label", "SAP subsystem")
    lines.append(f"Primary path: user action or batch step -> {subsystem_label} -> {system_label}.")
    integration_points = system_context.get("integration_points") or []
    if integration_points:
        lines.append(f"Watch the technical handoff between: {', '.join(integration_points[:4])}.")
    checks = (workspace.get("sections") or {}).get("Checks") or []
    if checks:
        lines.append(f"Retest path should begin with: {checks[0]}")
    return lines[:4]


def _build_validation_gate(workspace, analysis_context, resolved_environment):
    sections = workspace.get("sections") or {}
    entities = (analysis_context or {}).get("entities", {})
    lines = []

    for item in (sections.get("Checks") or [])[:2]:
        lines = _append_unique(lines, item, limit=5)
    for item in (sections.get("Required Inputs") or [])[:2]:
        lines = _append_unique(lines, f"Confirm before change: {item}", limit=5)

    if entities.get("transports"):
        lines = _append_unique(lines, "Confirm transport sequence, prerequisite requests, and target-system activation state before importing again.", limit=5)
    if entities.get("idocs") or entities.get("queues"):
        lines = _append_unique(lines, "Confirm replay scope before reprocessing so duplicate payloads are not created.", limit=5)
    if entities.get("users"):
        lines = _append_unique(lines, "Limit authorization, role, or unlock changes to the affected user or technical account first.", limit=5)
    if resolved_environment == "PROD":
        lines = _append_unique(lines, "Capture rollback owner, approval path, and business validation contact before making a production change.", limit=5)

    return lines[:5]


def _build_safe_change_plan(workspace, mixed_workstreams, resolved_environment):
    sections = workspace.get("sections") or {}
    lines = [
        "Use the smallest reversible correction that explains the first failing step.",
        "Retest with the same user, object, and environment immediately after the change.",
    ]
    if mixed_workstreams:
        lines.append("Do not execute risky workstreams in parallel unless each team has isolated ownership and rollback coverage.")
    if sections.get("Resolution"):
        lines.append(f"Preferred corrective path: {sections['Resolution'][0]}")
    if resolved_environment == "PROD":
        lines.append("Use change controls, business communication, and rollback criteria before a production transport, unlock, replay, or configuration update.")
    return lines[:5]


def _build_specialist_handoff(workspace, system_context, analysis_context):
    sections = workspace.get("sections") or {}
    entities = (analysis_context or {}).get("entities", {})
    owners = sections.get("Likely Owner") or ["SAP application support"]
    lines = [
        f"Escalate to: {owners[0]}.",
        f"System scope to include: {system_context.get('system_label', 'SAP system')} / {system_context.get('subsystem_label', 'SAP subsystem')}.",
    ]

    evidence_bits = []
    for key in ["users", "transports", "idocs", "queues", "http_codes", "status_codes"]:
        values = entities.get(key) or []
        evidence_bits.extend(values[:2])
    if evidence_bits:
        lines.append(f"Attach concrete evidence: {', '.join(evidence_bits[:6])}.")

    required_inputs = sections.get("Required Inputs") or []
    if required_inputs:
        lines.append(f"Missing inputs to request: {'; '.join(required_inputs[:3])}.")

    return lines[:5]


def _build_layer_coordination(bundle, workspace):
    sections = workspace.get("sections") or {}
    system_context = bundle.get("system_context") or {}
    lines = [
        "One shared solver bundle now carries evidence, SAP scope, ranked playbooks, and the final runbook into one execution path.",
        f"Routing query reused across layers: {shorten_text(bundle.get('matching_query', ''), limit=120)}",
        f"Coordinated scope: {system_context.get('system_label', 'Cross-system SAP landscape')} / {system_context.get('subsystem_label', 'Shared SAP subsystem')}.",
        f"Grounding source: {bundle.get('context_source', 'catalog context')}.",
    ]
    if bundle.get("mixed_issue_workstreams"):
        lines = _append_unique(lines, "Mixed-issue detection is shared with execution so all workstreams stay aligned to the same SAP evidence set.", limit=6)
    if sections.get("Advanced Diagnosis"):
        lines = _append_unique(lines, f"Reasoning layer stays attached to execution: {sections['Advanced Diagnosis'][0]}", limit=6)
    if sections.get("Validation Gate"):
        lines = _append_unique(lines, f"Validation stays synchronized with the chosen fix path: {sections['Validation Gate'][0]}", limit=6)
    return lines[:6]


def run_sap_agent(query, environment=None, system=None, subsystem=None, analysis_context=None):
    clean_query = str(query or "").strip()
    if not clean_query:
        return "Please enter an SAP ticket, issue, or question."

    if not analysis_context:
        analysis_context = analyze_issue_evidence(clean_query)

    scope_query = clean_query
    if analysis_context.get("ocr_text"):
        scope_query = f"{clean_query}\n{shorten_text(analysis_context.get('ocr_text', ''), limit=500)}"

    bundle = build_solver_bundle(
        clean_query,
        environment,
        system=system,
        subsystem=subsystem,
        analysis_context=analysis_context,
        scope_query=scope_query,
    )
    matching_query = bundle["matching_query"]
    resolved_environment = bundle["resolved_environment"]
    system_context = bundle["system_context"]
    matches = bundle["matches"]
    mixed_workstreams = bundle["mixed_issue_workstreams"]
    context_snippets = bundle["context_snippets"]
    context_source = bundle["context_source"]

    base_answer = build_ticket_answer_from_bundle(bundle)
    workspace = build_joule_workspace(clean_query, base_answer, resolved_environment, "rules")
    agent_report = f"""Agent Mode
- ADVANCED AGENT: multi-tool autonomous SAP troubleshooting agent
- Workflow: analyze evidence -> share solver bundle -> rank hypotheses -> choose the safest next action
- Operating model: {_derive_agent_mode_description()}

Objective
- {clean_query}

Investigation Plan
{_format_bullets(_build_investigation_plan(clean_query, resolved_environment, system_context, analysis_context, mixed_workstreams=mixed_workstreams), "Start with evidence collection and system scoping.")}

Expert Assessment
{_format_bullets(_build_expert_assessment(clean_query, workspace, system_context, analysis_context, matches, mixed_workstreams, resolved_environment), "Use the grounded SAP runbook as the primary expert guidance path.")}

Failure Boundary
{_format_bullets(_build_failure_boundary(system_context, analysis_context), "Use the system and subsystem selectors to tighten the SAP boundary.")}

Dependency Map
{_format_bullets(_build_dependency_map(system_context, mixed_workstreams, workspace), "Retest the same user action after each safe change so dependencies stay visible.")}

Tool Findings
{_format_bullets(_build_tool_findings(resolved_environment, system_context, analysis_context, context_snippets, context_source, mixed_workstreams=mixed_workstreams), "No tool findings were collected.")}

Layer Coordination
{_format_bullets(_build_layer_coordination(bundle, workspace), "Keep evidence, scope, routing, and execution aligned in one shared SAP path.")}

Evidence Correlation
{_format_bullets(_build_evidence_correlation(matches, context_snippets, analysis_context, mixed_workstreams=mixed_workstreams), "No strong evidence correlation was available.")}

Hypothesis Ranking
{_format_bullets(_summarize_hypotheses(matches), "No strong playbook hypothesis was identified, so the agent will rely on generic SAP triage." )}

Issue Mix
{_format_bullets(_build_issue_mix_lines(mixed_workstreams), "No mixed issue split was needed; the agent is treating this as one primary SAP problem.")}

Parallel Workstreams
{_format_bullets(_build_parallel_workstream_lines(mixed_workstreams), "No parallel workstreams were required for this SAP ticket.")}

Validation Gate
{_format_bullets(_build_validation_gate(workspace, analysis_context, resolved_environment), "Capture the exact error and validate the first safe check before changing SAP configuration.")}

Autonomous Next Step
{_format_bullets(_build_autonomous_next_step(workspace, system_context, mixed_workstreams=mixed_workstreams), "Start with the first safe validation check from the SAP runbook below.")}

Safe Change Plan
{_format_bullets(_build_safe_change_plan(workspace, mixed_workstreams, resolved_environment), "Apply the smallest safe change and retest before touching adjacent SAP components.")}

Cross-Issue Risks
{_format_bullets(_build_cross_issue_risks(mixed_workstreams), "Use a controlled change plan when multiple SAP symptoms are present.")}

Specialist Handoff
{_format_bullets(_build_specialist_handoff(workspace, system_context, analysis_context), "Escalate with the exact SAP evidence, owning scope, and missing inputs.")}

Open Questions
{_format_bullets(_build_open_questions(workspace, analysis_context, clean_query, mixed_workstreams=mixed_workstreams), "No blocking questions. Proceed with the checks and resolution plan below.")}

{base_answer}
"""
    return agent_report.strip()
