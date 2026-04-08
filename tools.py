from dataclasses import dataclass
from typing import Callable, Any

from app import ask_sap, build_context, build_matching_query, find_ticket_matches, rerank_matches_for_precision
from sap_intelligence import analyze_issue_evidence
from sap_landscape import resolve_system_context
from sap_ticket_catalog import TICKET_CATALOG


@dataclass
class Tool:
    name: str
    func: Callable[..., Any]
    description: str


def build_tcode_map():
    tcode_map = {
        "FB08": "Reverse a posted FI document",
        "FB60": "Enter a vendor invoice",
        "SE16N": "Display SAP table data",
    }

    for ticket in TICKET_CATALOG:
        description = ticket["title"]
        for tcode in ticket["tcodes"]:
            tcode_map.setdefault(tcode, description)

    return tcode_map


TCODE_MAP = build_tcode_map()

# Tool 1: SAP T-code info
def get_tcode_info(tcode):
    return TCODE_MAP.get(tcode.upper(), "T-code not found in the local SAP runbook catalog.")


def get_sap_knowledge(query):
    return ask_sap(query)


def collect_issue_evidence(query, image_bytes=None, filename=None):
    return analyze_issue_evidence(query, image_bytes=image_bytes, filename=filename)


def describe_system_scope(query, system=None, subsystem=None):
    return resolve_system_context(query, system=system, subsystem=subsystem)


def lookup_related_playbooks(query, system=None, subsystem=None, analysis_context=None, top_k=3):
    analysis_context = analysis_context or analyze_issue_evidence(query)
    matching_query = build_matching_query(query, analysis_context)
    system_context = resolve_system_context(query, system=system, subsystem=subsystem)
    matches = rerank_matches_for_precision(
        find_ticket_matches(matching_query, top_k=max(top_k, 4)),
        system_context,
        analysis_context=analysis_context,
    )[:top_k]
    context_snippets, context_source = build_context(matches, matching_query, include_vector=False)

    lines = []
    for index, match in enumerate(matches, start=1):
        ticket = match.get("ticket", {})
        reasons = "; ".join(match.get("reasons", [])[:2]) or "matched the closest SAP runbook"
        lines.append(
            f"{index}. {ticket.get('title', 'Closest SAP playbook')} (score {match.get('score', 0)}): {reasons}"
        )
    if context_snippets:
        lines.append(f"Context source: {context_source}")
        lines.append(f"Best snippet: {context_snippets[0]}")
    return "\n".join(lines) if lines else "No related SAP playbooks found."


def build_agent_toolkit():
    return {
        "tcode_lookup": get_tcode_info,
        "sap_knowledge": get_sap_knowledge,
        "issue_evidence": collect_issue_evidence,
        "system_scope": describe_system_scope,
        "playbook_lookup": lookup_related_playbooks,
    }


tcode_tool = Tool(
    name="TCodeTool",
    func=get_tcode_info,
    description="Useful for SAP T-code related queries"
)


rag_tool = Tool(
    name="SAPKnowledgeTool",
    func=get_sap_knowledge,
    description="Useful for SAP process, troubleshooting, BASIS, and ticket resolution questions."
)
