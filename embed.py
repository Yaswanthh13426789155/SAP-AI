from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


DATA_FILES = [
    "sap_tickets.txt",
    "sap_dataset.txt",
    "sap_web_data.txt",
]


def load_documents():
    docs = []

    for name in DATA_FILES:
        path = Path(name)
        if not path.exists():
            continue

        loader = TextLoader(str(path), encoding="utf-8")
        docs.extend(loader.load())

    if not docs:
        raise FileNotFoundError(
            "No source documents found. Add sap_tickets.txt or sap_dataset.txt before building sap_index."
        )

    return docs


def build_index():
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )
    docs = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("sap_index")
    return len(docs)


if __name__ == "__main__":
    count = build_index()
    print(f"Vector DB created from {count} document(s).")
