from app import ask_sap, extract_tcodes
from tools import get_tcode_info


def should_answer_as_tcode_lookup(question):
    lowered = question.lower()
    if "t-code" in lowered or "tcode" in lowered or "transaction code" in lowered:
        return True
    return len(extract_tcodes(question)) == 1 and len(question.split()) <= 8


def ask_agent(question, environment=None, provider="auto"):
    if should_answer_as_tcode_lookup(question):
        tcodes = sorted(extract_tcodes(question))
        if tcodes:
            return get_tcode_info(tcodes[0])
    return ask_sap(question, environment=environment, provider=provider)


if __name__ == "__main__":
    print(ask_agent("How to reverse invoice in SAP?"))
