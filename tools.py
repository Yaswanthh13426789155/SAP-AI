from langchain.tools import Tool

from app import ask_sap
from sap_ticket_catalog import TICKET_CATALOG


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

tcode_tool = Tool(
    name="TCodeTool",
    func=lambda x: get_tcode_info(x),
    description="Useful for SAP T-code related queries"
)


rag_tool = Tool(
    name="SAPKnowledgeTool",
    func=ask_sap,
    description="Useful for SAP process, troubleshooting, BASIS, and ticket resolution questions."
)
