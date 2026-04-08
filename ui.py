import html
import os
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent

# Streamlit on this Windows machine cannot write to the default home directory.
# Keep the workaround local to Windows so Linux cloud hosts can use their normal home directory.
if os.name == "nt":
    os.environ["USERPROFILE"] = str(WORKSPACE_DIR)
    os.environ["HOME"] = str(WORKSPACE_DIR)
    (WORKSPACE_DIR / ".streamlit").mkdir(exist_ok=True)

import streamlit as st

from app import ask_sap, build_joule_workspace, runtime_status
from sap_intelligence import analyze_issue_evidence
from sap_landscape import get_subsystem_choices, get_system_choices
from sap_ticket_catalog import TICKET_CATALOG

ARCHITECTURE_FLOW = """
User
  |
Frontend (Chat UI)
  |
LLM (Ollama / OpenAI)
  |
Agent Layer (LangChain / CrewAI)
  |
Tools (SAP APIs, Logs, DB, Scripts)
  |
SAP System (ECC / S4HANA)
""".strip()

SUGGESTED_PROMPTS = [
    "PROD user cannot log in. Error: User locked after multiple attempts. Need safe fix steps.",
    "QA transport failed with RC 8. Import log says object missing. What should I check first?",
    "TEST IDoc stuck in status 51 after master data change. Need T-codes and resolution plan.",
    "DEV sales order pricing condition missing in VA01. Help me troubleshoot step by step.",
]

KNOWN_HEADINGS = {
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

SECTION_ORDER = [
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
]

PAGE_CSS = """
<style>
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top right, rgba(61, 163, 219, 0.18), transparent 26%),
        radial-gradient(circle at top left, rgba(8, 61, 119, 0.18), transparent 24%),
        linear-gradient(180deg, #f2f7fb 0%, #edf4f9 42%, #f7fafc 100%);
}

.block-container {
    max-width: 1320px;
    padding-top: 1.4rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2438 0%, #102c43 100%);
}

[data-testid="stSidebar"] * {
    color: #e8f1f7;
}

.hero-shell {
    background: linear-gradient(135deg, #08243d 0%, #0f4774 60%, #1f86c7 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 26px;
    box-shadow: 0 24px 60px rgba(12, 30, 53, 0.18);
    color: #ffffff;
    padding: 1.6rem 1.7rem;
    margin-bottom: 1rem;
}

.hero-kicker {
    color: rgba(255, 255, 255, 0.82);
    font-size: 0.8rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.hero-title {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.15;
    margin-top: 0.35rem;
}

.hero-copy {
    color: rgba(255, 255, 255, 0.9);
    font-size: 0.98rem;
    margin-top: 0.7rem;
    max-width: 62rem;
}

.pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-top: 1rem;
}

.info-pill {
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 999px;
    color: #ffffff;
    font-size: 0.86rem;
    padding: 0.35rem 0.75rem;
}

.workspace-card {
    background: rgba(255, 255, 255, 0.82);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(12, 42, 74, 0.08);
    border-radius: 24px;
    box-shadow: 0 16px 32px rgba(14, 34, 55, 0.08);
    padding: 1rem 1rem 1.1rem 1rem;
}

.section-card {
    background: linear-gradient(180deg, #ffffff 0%, #f8fbfd 100%);
    border: 1px solid #d9e4ee;
    border-radius: 18px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.8rem;
}

.section-title {
    color: #113554;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
}

.summary-card {
    background: linear-gradient(180deg, #ffffff 0%, #f7fbfe 100%);
    border: 1px solid #d6e4ef;
    border-radius: 18px;
    padding: 0.95rem 1rem;
    min-height: 160px;
}

.summary-label {
    color: #1d5074;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.4rem;
}

.kpi-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin: 0.4rem 0 0.85rem 0;
}

.mini-pill {
    background: #ebf4fa;
    border: 1px solid #cfe0ec;
    border-radius: 999px;
    color: #174263;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 0.3rem 0.7rem;
}

.action-card {
    background: linear-gradient(180deg, #ffffff 0%, #f9fbfd 100%);
    border: 1px solid #d9e5ef;
    border-radius: 16px;
    padding: 0.85rem 0.95rem;
    margin-bottom: 0.7rem;
}

.action-title {
    color: #163f60;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.4rem;
}

.message-meta {
    color: #53708a;
    font-size: 0.78rem;
    margin-bottom: 0.45rem;
}

.empty-state {
    border: 1px dashed #b8ccdc;
    border-radius: 18px;
    padding: 1rem 1.1rem;
    background: rgba(255, 255, 255, 0.7);
}

.sidebar-card {
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(255, 255, 255, 0.06);
    border-radius: 18px;
    padding: 0.95rem 1rem;
    margin-bottom: 0.85rem;
}

.sidebar-title {
    font-size: 0.86rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
}

div.stButton > button {
    border-radius: 14px;
    border: 1px solid #b8cfdf;
    background: linear-gradient(180deg, #ffffff 0%, #f2f7fb 100%);
    color: #143652;
    font-weight: 600;
}

div.stButton > button:hover {
    border-color: #1f86c7;
    color: #0f4d7a;
}

[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.55);
    border: 1px solid rgba(17, 53, 84, 0.08);
    border-radius: 18px;
    padding: 0.2rem 0.6rem;
}
</style>
"""


def init_state():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("composer_text", "")
    st.session_state.setdefault("clear_composer_on_rerun", False)


def apply_pending_widget_updates():
    if st.session_state.pop("clear_composer_on_rerun", False):
        st.session_state["composer_text"] = ""


def queue_prompt(prompt):
    st.session_state["composer_text"] = prompt


def clear_conversation():
    st.session_state["messages"] = []
    st.session_state["clear_composer_on_rerun"] = True


def get_recommended_provider(status):
    if status.get("agentic_ready") and status.get("trained_router_ready"):
        return "agentic"
    if status.get("trained_router_ready"):
        return "rules"
    if status.get("openai_configured") and not status.get("openai_last_error"):
        return "openai"
    if status.get("ollama_available") and not status.get("ollama_last_error"):
        return "ollama"
    if status.get("open_source_ready"):
        return "open_source"
    return "rules"


def summarize_message_for_context(message):
    if message["role"] == "user":
        return f"Previous user request: {message['content']}"

    workspace = build_joule_workspace(
        message.get("query", ""),
        message["content"],
        message.get("environment", "ALL"),
        message.get("provider", "auto"),
    )
    summary_parts = []
    if workspace["primary_incident"]:
        summary_parts.append(f"Incident: {workspace['primary_incident']}")
    if workspace["sections"].get("Resolution"):
        summary_parts.append(
            f"Resolution: {'; '.join(workspace['sections']['Resolution'][:2])}"
        )
    if workspace["sections"].get("Escalate If"):
        summary_parts.append(
            f"Escalation: {'; '.join(workspace['sections']['Escalate If'][:1])}"
        )
    return "Previous assistant guidance: " + " | ".join(summary_parts)


def build_contextual_prompt(prompt, messages):
    if not messages:
        return prompt

    context_lines = []
    for message in messages[-4:]:
        summary = summarize_message_for_context(message)
        if summary:
            context_lines.append(summary)

    if not context_lines:
        return prompt

    return (
        f"{prompt}\n\nConversation memory:\n"
        + "\n".join(f"- {line}" for line in context_lines)
    )


def parse_sections(text):
    sections = {}
    notices = []
    current_heading = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        normalized_heading = line.rstrip(":")
        if normalized_heading in KNOWN_HEADINGS:
            current_heading = normalized_heading
            sections.setdefault(current_heading, [])
            continue

        if current_heading is None:
            notices.append(line)
            continue

        if line.startswith("- "):
            sections[current_heading].append(line)
        else:
            sections[current_heading].append(f"- {line}")

    return notices, sections


def format_section_markdown(items):
    return "\n".join(items) if items else "- No details available."


def items_to_html(items):
    if not items:
        return "<ul><li>No details available.</li></ul>"

    bullets = []
    for item in items:
        content = item[2:] if item.startswith("- ") else item
        bullets.append(f"<li>{html.escape(content)}</li>")
    return f"<ul>{''.join(bullets)}</ul>"


def render_summary_card(title, items):
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-label">{html.escape(title)}</div>
            {items_to_html(items)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_action_card(title, body):
    st.markdown(
        f"""
        <div class="action-card">
            <div class="action-title">{html.escape(title)}</div>
            {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_assistant_response(message, message_key):
    content = message["content"]
    environment = message.get("environment", "ALL")
    provider = message.get("provider", "auto")
    workspace = build_joule_workspace(
        message.get("query", ""),
        content,
        environment,
        provider,
    )
    notices = workspace["notices"]
    sections = workspace["sections"]
    system_scope = workspace.get("primary_system") or message.get("system", "")
    subsystem_scope = workspace.get("primary_subsystem") or message.get("subsystem", "")
    meta_parts = [f"Landscape: {html.escape(environment)}"]
    if system_scope and system_scope not in {"AUTO", "Auto detect / cross-system"}:
        meta_parts.append(f"System: {html.escape(system_scope)}")
    if subsystem_scope and subsystem_scope not in {"AUTO", "Auto detect / shared service"} and subsystem_scope != system_scope:
        meta_parts.append(f"Subsystem: {html.escape(subsystem_scope)}")
    meta_parts.append(f"Engine: {html.escape(provider)}")

    st.markdown(
        f"""
        <div class="message-meta">
            {' | '.join(meta_parts)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    kpis = []
    if workspace["primary_incident"]:
        kpis.append(f"<div class='mini-pill'>Incident: {html.escape(workspace['primary_incident'])}</div>")
    if workspace["primary_system"]:
        kpis.append(f"<div class='mini-pill'>System: {html.escape(workspace['primary_system'])}</div>")
    if workspace["primary_subsystem"] and workspace["primary_subsystem"] != workspace["primary_system"]:
        kpis.append(f"<div class='mini-pill'>Subsystem: {html.escape(workspace['primary_subsystem'])}</div>")
    if workspace["primary_tcode"]:
        kpis.append(f"<div class='mini-pill'>Primary T-code: {html.escape(workspace['primary_tcode'])}</div>")
    if kpis:
        st.markdown(f"<div class='kpi-strip'>{''.join(kpis)}</div>", unsafe_allow_html=True)

    for notice in notices:
        st.info(notice)

    summary_pairs = [
        ("Incident", sections.get("Incident")),
        ("Likely Root Cause", sections.get("Likely Root Cause")),
    ]
    available_summaries = [(title, items) for title, items in summary_pairs if items]
    if available_summaries:
        cols = st.columns(len(available_summaries))
        for col, (title, items) in zip(cols, available_summaries):
            with col:
                render_summary_card(title, items)

    solve_tab, resolution_tab, business_tab, handoff_tab, enduser_tab, followup_tab = st.tabs(
        ["Solve Now", "Resolution", "Business Update", "Technical Handoff", "End User", "Follow-Ups"]
    )

    with solve_tab:
        if sections.get("Expert Assessment"):
            render_action_card(
                "Expert Assessment",
                items_to_html([f"- {item}" for item in sections["Expert Assessment"]]),
            )
        if sections.get("Validation Gate"):
            render_action_card(
                "Validation Gate",
                items_to_html([f"- {item}" for item in sections["Validation Gate"]]),
            )
        if sections.get("Layer Coordination"):
            render_action_card(
                "Layer Coordination",
                items_to_html([f"- {item}" for item in sections["Layer Coordination"]]),
            )
        if sections.get("Investigation Plan"):
            render_action_card(
                "Investigation Plan",
                items_to_html([f"- {item}" for item in sections["Investigation Plan"]]),
            )
        if sections.get("Autonomous Next Step"):
            render_action_card(
                "Autonomous Next Step",
                items_to_html([f"- {item}" for item in sections["Autonomous Next Step"]]),
            )
        if workspace["solve_now_plan"]:
            for step_index, action in enumerate(workspace["solve_now_plan"], start=1):
                render_action_card(f"Step {step_index}", f"<div>{html.escape(action)}</div>")
        else:
            st.info("No solve-now plan was derived from this response yet.")

        render_action_card(
            "Expected Outcome",
            f"<div>{html.escape(workspace['expected_outcome'])}</div>",
        )

        if sections.get("Advanced Diagnosis"):
            render_action_card(
                "Advanced Diagnosis",
                items_to_html([f"- {item}" for item in sections["Advanced Diagnosis"]]),
            )
        if sections.get("Decision Path"):
            render_action_card(
                "Decision Path",
                items_to_html([f"- {item}" for item in sections["Decision Path"]]),
            )
        if sections.get("Safe Change Plan"):
            render_action_card(
                "Safe Change Plan",
                items_to_html([f"- {item}" for item in sections["Safe Change Plan"]]),
            )
        if sections.get("Why This Matched"):
            render_action_card(
                "Why This Match Was Chosen",
                items_to_html([f"- {item}" for item in sections["Why This Matched"]]),
            )
        if sections.get("Supporting Context"):
            render_action_card(
                "Supporting Context",
                items_to_html([f"- {item}" for item in sections["Supporting Context"]]),
            )

    with resolution_tab:
        for heading in SECTION_ORDER:
            items = sections.get(heading)
            if not items or heading in {"Incident", "Likely Root Cause"}:
                continue
            st.markdown(
                f"""
                <div class="section-card">
                    <div class="section-title">{html.escape(heading)}</div>
                    {items_to_html(items)}
                </div>
                """,
                unsafe_allow_html=True,
            )

    with business_tab:
        st.caption("Share this with a manager or business stakeholder.")
        st.code(workspace["business_update"], language="text")

    with handoff_tab:
        st.caption("Use this as an operator handoff or support note.")
        st.code(workspace["technical_handoff"], language="text")

    with enduser_tab:
        st.caption("Use this to update the requester in plain language.")
        st.code(workspace["end_user_update"], language="text")

    with followup_tab:
        st.caption("Ask the assistant to continue like an enterprise copilot.")
        for follow_up in workspace["follow_up_prompts"]:
            if st.button(
                follow_up,
                key=f"followup_{message_key}_{follow_up}",
                use_container_width=True,
            ):
                queue_prompt(follow_up)


def render_message(message, index):
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
        return

    with st.chat_message("assistant"):
        render_assistant_response(message, f"{index}")


def render_sidebar(status):
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-card">
                <div class="sidebar-title">SAP AI Workspace</div>
                <div>Enterprise ticket resolution with a Joule-style copilot experience for SAP support teams.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="sidebar-card">
                <div class="sidebar-title">Coverage</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.metric("Runbooks", len(TICKET_CATALOG))
        st.metric("Landscapes", 4)
        st.metric("Systems", status["supported_systems"])
        st.metric("Subsystems", status["supported_subsystems"])
        st.caption("BASIS | Security | FI | MM | SD | Integration")

        st.markdown(
            """
            <div class="sidebar-card">
                <div class="sidebar-title">Runtime</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.metric("OpenAI", "Configured" if status["openai_configured"] else "Missing")
        st.metric("Agentic", "Ready" if status["agentic_ready"] else "Missing")
        st.metric("Open Source", "Ready" if status["open_source_ready"] else "Missing")
        st.metric("Ollama", "Ready" if status["ollama_available"] else "Missing")
        st.metric("Open LLM API", "Ready" if status["openai_compatible_available"] else "Missing")
        st.metric("HF Local", "Ready" if status["hf_local_available"] else "Missing")
        st.metric("Vector Index", "Ready" if status["vector_index_present"] else "Missing")
        st.metric("OCR", "Ready" if status["ocr_available"] else "Missing")
        st.metric("Neural NLP", "Ready" if status["neural_nlp_available"] else "Missing")
        router_state = "Ready" if status["trained_router_ready"] else (
            "Training" if status["training_state"] == "running" else "Missing"
        )
        st.metric("SAP Router", router_state)
        st.metric("Custom Landscape", "Ready" if status["custom_landscape_present"] else "Default")

        if status["openai_last_error"]:
            st.warning(status["openai_last_error"])
        if status["ollama_last_error"]:
            st.warning(status["ollama_last_error"])
        if status["openai_compatible_last_error"]:
            st.warning(status["openai_compatible_last_error"])
        if status["hf_local_last_error"]:
            st.warning(status["hf_local_last_error"])
        if status["training_message"]:
            st.info(status["training_message"])
        if status["training_best_val_accuracy"]:
            st.caption(
                "Router validation accuracy: "
                f"{status['training_best_val_accuracy']:.2%} | macro F1: {status['training_best_val_macro_f1']:.2%}"
            )

        with st.expander("Architecture", expanded=False):
            st.code(ARCHITECTURE_FLOW, language="text")

        with st.expander("Response Tips", expanded=False):
            st.markdown("- Include exact error text, dump names, or return codes.")
            st.markdown("- Mention T-codes, jobs, programs, or document numbers.")
            st.markdown("- Add business impact and the SAP landscape when known.")
            st.markdown("- Pick the SAP system and subsystem to route the answer to the right stack.")
            st.markdown("- Upload screenshots so OCR and NLP can extract issue signals automatically.")
            st.markdown("- Use `open_llm` for API-key-based OpenAI-compatible endpoints.")

        with st.expander("Model Tuning", expanded=False):
            st.markdown("- Run `python sap_training.py --time-budget-hours 8` to tune the SAP router.")
            st.markdown("- The tuned router improves runbook matching for ambiguous tickets on this machine.")
            st.markdown("- Progress is written to `.cache/sap_training/status.json`.")

        with st.expander("Supported Systems", expanded=False):
            for _, label in get_system_choices(include_auto=False):
                st.markdown(f"- {label}")

        st.button("Clear Conversation", use_container_width=True, on_click=clear_conversation)


st.set_page_config(page_title="SAP AI", layout="wide")
st.markdown(PAGE_CSS, unsafe_allow_html=True)
init_state()
apply_pending_widget_updates()
status = runtime_status()

render_sidebar(status)

st.markdown(
    """
    <div class="hero-shell">
        <div class="hero-kicker">SAP Enterprise Copilot</div>
        <div class="hero-title">Joule-style SAP support assistant for tickets, incidents, and operations.</div>
        <div class="hero-copy">
            Ask SAP AI the way an analyst would ask Joule: describe the issue, business impact, and landscape,
            then get a guided playbook with checks, T-codes, fixes, escalation advice, system-aware integration guidance,
            plus OCR and NLP extraction from issue screenshots.
        </div>
        <div class="pill-row">
            <div class="info-pill">DEV / QA / TEST / PROD aware</div>
            <div class="info-pill">System / subsystem aware</div>
            <div class="info-pill">OCR + NLP + neural similarity</div>
            <div class="info-pill">Chat-first workflow</div>
            <div class="info-pill">OpenAI + Open Source AI + Ollama</div>
            <div class="info-pill">Grounded SAP runbooks</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

top_left, top_right = st.columns([1.5, 1])
with top_left:
    st.markdown("### Start From A Real Ticket")
    st.caption("Choose your SAP landscape, system, subsystem, and preferred engine, then chat with the assistant the way you would in an enterprise support workspace.")
with top_right:
    st.markdown(
        f"""
        <div class="workspace-card">
            <div class="section-title">Workspace Status</div>
            <ul>
                <li>Open Source backends: <code>{html.escape(status["open_source_backends"])}</code></li>
                <li>Agentic mode: <code>{"Ready" if status["agentic_ready"] else "Missing"}</code></li>
                <li>Preferred backend: <code>{html.escape(status["open_source_backend"])}</code></li>
                <li>Web SAP corpus: <code>{"Ready" if status["sap_web_data_present"] else "Missing"}</code></li>
                <li>Vector context: <code>{"On" if status["vector_context_enabled"] else "Off"}</code></li>
                <li>OCR pipeline: <code>{"Ready" if status["ocr_available"] else "Missing"}</code></li>
                <li>Neural NLP: <code>{"Ready" if status["neural_nlp_available"] else "Missing"}</code></li>
                <li>SAP router: <code>{"Ready" if status["trained_router_ready"] else status["training_state"].title()}</code></li>
                <li>Router accuracy: <code>{status["training_best_val_accuracy"]:.2%}</code></li>
                <li>Supported systems: <code>{status["supported_systems"]}</code></li>
                <li>Supported subsystems: <code>{status["supported_subsystems"]}</code></li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

controls_col, prompts_col = st.columns([1.1, 1])
with controls_col:
    provider_options = ["agentic", "rules", "auto", "open_source", "ollama", "open_llm", "hf_local", "openai"]
    recommended_provider = get_recommended_provider(status)
    system_options = get_system_choices()
    system_ids = [option[0] for option in system_options]
    system_labels = {option[0]: option[1] for option in system_options}
    environment = st.selectbox(
        "SAP landscape",
        ["ALL", "DEV", "QA", "TEST", "PROD"],
        index=0,
        help="Choose one environment or ALL to get rollout-aware guidance across the SAP landscape.",
    )
    system_id = st.selectbox(
        "SAP system",
        system_ids,
        index=0,
        format_func=lambda key: system_labels.get(key, key),
        help="Pick the owning SAP product stack or leave it on auto-detect for cross-system tickets.",
    )
    subsystem_options = get_subsystem_choices(system_id)
    subsystem_ids = [option[0] for option in subsystem_options]
    subsystem_labels = {option[0]: option[1] for option in subsystem_options}
    subsystem_id = st.selectbox(
        "Subsystem or service",
        subsystem_ids,
        index=0,
        format_func=lambda key: subsystem_labels.get(key, key),
        help="Narrow the answer to the subsystem, service boundary, or shared service involved in the incident.",
    )
    provider = st.selectbox(
        "Assistant engine",
        provider_options,
        index=provider_options.index(recommended_provider),
        help="`agentic` runs an autonomous SAP investigation workflow on top of the grounded runbook engine. `rules` is still the fastest direct playbook mode.",
    )
    st.caption(f"Recommended for this workspace: `{recommended_provider}`")

with prompts_col:
    st.markdown("### Suggested Prompts")
    prompt_cols = st.columns(2)
    for index, prompt in enumerate(SUGGESTED_PROMPTS):
        with prompt_cols[index % 2]:
            if st.button(prompt, key=f"prompt_{index}", use_container_width=True):
                queue_prompt(prompt)

chat_shell = st.container(border=True)
with chat_shell:
    st.markdown("### Conversation")

    if not st.session_state["messages"]:
        st.markdown(
            """
            <div class="empty-state">
                <strong>No conversation yet.</strong><br/>
                Start with a production issue, failed transport, IDoc error, authorization issue, or finance posting problem.
                The assistant will answer in a guided SAP runbook format with system and subsystem context, then generate
                next-best actions, business updates, and technical handoff notes like an enterprise copilot.
            </div>
            """,
            unsafe_allow_html=True,
        )

    for index, message in enumerate(st.session_state["messages"]):
        render_message(message, index)

    with st.form("sap_chat_form", clear_on_submit=False):
        prompt = st.text_area(
            "Describe the SAP incident",
            key="composer_text",
            height=140,
            placeholder=(
                "Example:\n"
                "Background job cancelled in PROD.\n"
                "Error: Authorization missing.\n"
                "Program runs in SM37 with batch user SAPBATCH.\n"
                "Need safe fix steps and T-codes."
            ),
        )
        issue_image = st.file_uploader(
            "Optional screenshot or error image",
            type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
            help="Upload an SAP screenshot so OCR, image preprocessing, and NLP can extract issue signals automatically.",
        )
        submitted = st.form_submit_button("Ask SAP AI", use_container_width=True)

if submitted:
    clean_prompt = prompt.strip()
    if not clean_prompt:
        st.warning("Paste a ticket or incident description before submitting.")
    else:
        image_bytes = issue_image.getvalue() if issue_image is not None else None
        analysis_notices = []
        try:
            analysis_context = analyze_issue_evidence(
                clean_prompt,
                image_bytes=image_bytes,
                filename=getattr(issue_image, "name", None),
            )
        except Exception:
            analysis_context = {
                "original_text": clean_prompt,
                "ocr_text": "",
                "combined_text": clean_prompt,
                "entities": {},
                "domain_signals": [],
                "semantic_matches": [],
                "image_findings": [],
                "warnings": [],
                "summary_lines": [],
                "resolver_evidence": "",
            }
            analysis_notices.append(
                "Advanced NLP analysis was unavailable, so SAP AI used the core resolver path."
            )
        contextual_prompt = build_contextual_prompt(
            clean_prompt,
            st.session_state["messages"],
        )
        st.session_state["messages"].append(
            {
                "role": "user",
                "content": clean_prompt,
            }
        )
        with st.spinner("Building SAP ticket playbook..."):
            try:
                response = ask_sap(
                    contextual_prompt,
                    environment=environment,
                    provider=provider,
                    system=system_id,
                    subsystem=subsystem_id,
                    analysis_context=analysis_context,
                )
            except Exception:
                analysis_notices.append(
                    f"The `{provider}` engine failed, so SAP AI returned the core SAP runbook answer instead."
                )
                try:
                    response = ask_sap(
                        contextual_prompt,
                        environment=environment,
                        provider="rules",
                        system=system_id,
                        subsystem=subsystem_id,
                        analysis_context=analysis_context,
                    )
                except Exception:
                    response = (
                        "SAP AI could not build the ticket answer right now.\n\n"
                        "Try the `rules` engine again or restart the app."
                    )
            if not str(response or "").strip():
                analysis_notices.append(
                    "The selected engine returned an empty answer, so SAP AI rebuilt the response with the core runbook engine."
                )
                response = ask_sap(
                    contextual_prompt,
                    environment=environment,
                    provider="rules",
                    system=system_id,
                    subsystem=subsystem_id,
                    analysis_context=analysis_context,
                )
            if analysis_notices:
                response = "\n".join(analysis_notices) + "\n\n" + response
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": response,
                "environment": environment,
                "system": system_labels.get(system_id, system_id),
                "subsystem": subsystem_labels.get(subsystem_id, subsystem_id),
                "provider": provider,
                "query": clean_prompt,
                "analysis_summary": analysis_context.get("summary_lines", []),
            }
        )
        st.session_state["clear_composer_on_rerun"] = True
        st.rerun()
