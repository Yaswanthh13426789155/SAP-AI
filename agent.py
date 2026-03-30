from langchain_community.llms import Ollama
from langchain.agents import initialize_agent
from tools import tcode_tool, rag_tool

llm = Ollama(model="llama3")

tools = [tcode_tool, rag_tool]

agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)


def ask_agent(question):
    return agent.run(question)


if __name__ == "__main__":
    print(ask_agent("How to reverse invoice in SAP?"))
