from pathlib import Path

import faiss
import ollama
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
DATA_FILES = [
    "sap_tickets.txt",
    "sap_dataset.txt",
]


def load_data():
    data = []

    for name in DATA_FILES:
        path = Path(name)
        if not path.exists():
            continue

        chunks = [chunk.strip() for chunk in path.read_text(encoding="utf-8").split("\n\n")]
        data.extend(chunk for chunk in chunks if chunk)

    if not data:
        raise FileNotFoundError(
            "No local SAP text data found. Add sap_tickets.txt or sap_dataset.txt first."
        )

    return data


def build_index(model, data):
    embeddings = model.encode(data)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def ask_sap(question, model, data, index):
    q_embed = model.encode([question])
    _, neighbors = index.search(q_embed, k=min(5, len(data)))
    context = "\n".join(data[i] for i in neighbors[0])

    response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "system",
                "content": "You are an SAP BASIS support engineer. Give practical, safe troubleshooting guidance.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )

    return response["message"]["content"]


if __name__ == "__main__":
    sentence_model = SentenceTransformer(MODEL_NAME)
    dataset = load_data()
    faiss_index = build_index(sentence_model, dataset)

    while True:
        query = input("Ask SAP: ").strip()
        if not query or query.lower() in {"exit", "quit"}:
            break
        print(ask_sap(query, sentence_model, dataset, faiss_index))
