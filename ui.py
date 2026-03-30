import os
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent

# Streamlit writes machine/session metadata under Path.home()/.streamlit.
# Redirect that into the writable workspace to avoid home-directory permission errors.
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

st.set_page_config(page_title="SAP AI Assistant", layout="wide")
st.title("SAP Ticket Resolver")
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
        ["auto", "openai", "rules"],
        index=0,
        help="Auto uses OpenAI when OPENAI_API_KEY is configured, otherwise it falls back to local SAP runbooks.",
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
    st.metric("OpenAI Key", "Configured" if status["openai_configured"] else "Missing")
    st.metric("OpenAI Model", status["openai_model"])
    st.metric("Web SAP Corpus", "Ready" if status["sap_web_data_present"] else "Missing")

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
    - Refresh `sap_web_data.txt` with `python sap_web_ingest.py` to use SAP internet content.
    """
)
