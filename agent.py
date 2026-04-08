from app import ask_sap, extract_tcodes
from sap_agent import run_sap_agent
from tools import get_tcode_info


def should_answer_as_tcode_lookup(question):
    lowered = question.lower()
    if "t-code" in lowered or "tcode" in lowered or "transaction code" in lowered:
        return True
    return len(extract_tcodes(question)) == 1 and len(question.split()) <= 8


def ask_agent(question, environment=None, provider="agentic", system=None, subsystem=None, analysis_context=None):
    if should_answer_as_tcode_lookup(question):
        tcodes = sorted(extract_tcodes(question))
        if tcodes:
            return get_tcode_info(tcodes[0])
    if str(provider or "agentic").strip().lower() in {"agentic", "agent", "autonomous"}:
        return run_sap_agent(
            question,
            environment=environment,
            system=system,
            subsystem=subsystem,
            analysis_context=analysis_context,
        )
    return ask_sap(
        question,
        environment=environment,
        provider=provider,
        system=system,
        subsystem=subsystem,
        analysis_context=analysis_context,
    )


if __name__ == "__main__":
    print(ask_agent("How to reverse invoice in SAP?"))
