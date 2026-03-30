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
from app import ask_sap, runtime_status
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

st.set_page_config(page_title="SAP AI", layout="wide")
st.title("SAP AI")
status = runtime_status()
st.caption("Paste an SAP incident and get a structured runbook with checks, T-codes, fixes, and escalation guidance.")

left_col, right_col = st.columns([1.3, 1])

with left_col:
    st.subheader("Resolve Ticket")
    environment = st.selectbox(
        "SAP landscape:",
        ["ALL", "DEV", "QA", "TEST", "PROD"],
        index=0,
        help="Choose one environment or ALL to get a rollout-style answer across the SAP landscape.",
    )
    provider = st.selectbox(
        "Answer engine:",
        ["auto", "open_source", "ollama", "openai_compatible", "hf_local", "openai", "rules"],
        index=0,
        help="Auto uses OpenAI when configured, otherwise any available open-source backend, and otherwise the built-in SAP runbooks. Use open_source to prefer open-source AI, or choose a specific backend directly.",
    )
    query = st.text_area(
        "Paste the SAP ticket, error, or issue:",
        height=180,
        placeholder="Example:\nBackground job cancelled in PROD.\nError: Authorization missing.\nProgram runs in SM37 with batch user SAPBATCH.\nNeed safe fix steps and T-code for DEV, QA, TEST, and PROD.\nUse OpenAI and SAP internet data if available.",
    )

    if st.button("Resolve Ticket", use_container_width=True):
        if query.strip():
            try:
                with st.spinner("Building SAP ticket playbook..."):
                    response = ask_sap(query, environment=environment, provider=provider)
                st.subheader("Resolution Playbook")
                st.text(response)
            except Exception as exc:
                st.error(f"Unable to resolve the SAP ticket: {exc}")
        else:
            st.warning("Paste a ticket or incident description before submitting.")

    with st.expander("Example Tickets", expanded=True):
        st.markdown("- PROD: User cannot login to SAP. Error: User locked after multiple attempts.")
        st.markdown("- QA: Transport failed with RC 8. Import log says object missing.")
        st.markdown("- TEST: IDoc stuck in status 51 after master data change.")
        st.markdown("- DEV: Sales order pricing condition missing in VA01.")
        st.markdown("- PROD: Posting period not open for FI invoice in FB60.")

with right_col:
    st.subheader("Runtime")
    openai_state = "Configured" if status["openai_configured"] else "Missing"
    if status["openai_last_error"]:
        openai_state = "Fallback"
    st.metric("OpenAI", openai_state)
    st.metric("OpenAI Model", status["openai_model"])
    st.metric("Open Source AI", "Ready" if status["open_source_ready"] else "Missing")
    st.metric("OSS Default", status["open_source_backend"])
    st.metric("OSS Backends", status["open_source_backends"])
    ollama_state = "Ready" if status["ollama_available"] else "Missing"
    if status["ollama_last_error"]:
        ollama_state = "Fallback"
    st.metric("Ollama", ollama_state)
    st.metric("Ollama Model", status["ollama_model"])
    compatible_state = "Ready" if status["openai_compatible_available"] else "Missing"
    if status["openai_compatible_last_error"]:
        compatible_state = "Fallback"
    st.metric("OpenAI-Compatible", compatible_state)
    st.metric("Compatible Model", status["openai_compatible_model"])
    hf_state = "Ready" if status["hf_local_available"] else "Missing"
    if status["hf_local_last_error"]:
        hf_state = "Fallback"
    st.metric("HF Local", hf_state)
    st.metric("HF Model", status["hf_local_model"])
    st.metric("Vector Index", "Ready" if status["vector_index_present"] else "Missing")
    st.metric("Web SAP Corpus", "Ready" if status["sap_web_data_present"] else "Missing")
    st.metric("Vector Context", "On" if status["vector_context_enabled"] else "Off")

    if status["openai_last_error"]:
        st.warning(status["openai_last_error"])
    if status["ollama_last_error"]:
        st.warning(status["ollama_last_error"])
    if status["openai_compatible_last_error"]:
        st.warning(status["openai_compatible_last_error"])
    if status["hf_local_last_error"]:
        st.warning(status["hf_local_last_error"])

    st.subheader("Coverage")
    st.metric("Runbooks", len(TICKET_CATALOG))
    st.metric("Landscapes", 4)
    st.markdown("- BASIS")
    st.markdown("- Security")
    st.markdown("- FI")
    st.markdown("- MM")
    st.markdown("- SD")
    st.markdown("- Integration")

    st.subheader("Architecture")
    st.code(ARCHITECTURE_FLOW, language="text")

st.subheader("How To Get Better Answers")
st.markdown(
    """
    - Include the exact error text, dump name, status code, or return code.
    - Mention the transaction code, program, job name, interface, or document number.
    - Specify whether the ticket is in DEV, QA, TEST, or PROD when you know it.
    - Add the business impact, such as production blocked, month-end issue, or user-specific problem.
    - Paste the full ticket description instead of a short generic question when possible.
    - Set `OPENAI_API_KEY` to enable OpenAI-backed answer generation.
    - Set `OPEN_SOURCE_BACKEND=openai_compatible` plus `OPEN_SOURCE_API_BASE_URL` and `OPEN_SOURCE_API_MODEL` to use LM Studio, vLLM, LocalAI, or llama.cpp servers that expose an OpenAI-style API.
    - Set `HF_LOCAL_MODEL` to run a local Hugging Face Transformers model when the machine has a supported runtime.
    - Set `OLLAMA_BASE_URL` and `OLLAMA_MODEL`, then choose `ollama` or `open_source`, to use your local Ollama server.
    - Refresh `sap_web_data.txt` with `python sap_web_ingest.py` to use SAP internet content.
    """
)
