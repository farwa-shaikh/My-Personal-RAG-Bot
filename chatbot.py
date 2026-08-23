import os
import json

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import CHAT_MODEL, TOP_K, INDEX_PATH, METADATA_PATH

load_dotenv()
client = OpenAI()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

SYSTEM_PROMPT = """You are a helpful assistant answering questions using ONLY the provided context.
If the answer isn't in the context, say you don't know based on the available documents.
Always mention which source(s) you used."""


def load_ingest():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            "No ingest found. Run `python ingest.py` first to build one from your documents."
        )
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return index, data["chunks"], data["metadata"]


def embed_query(query: str) -> np.ndarray:
    vector = embeddings.embed_query(query)
    vec = np.array(vector, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec


def retrieve(index, chunks, metadata, query: str, k: int):
    vec = embed_query(query)
    scores, indices = index.search(vec, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append(
            {
                "text": chunks[idx],
                "source": metadata[idx]["source"],
                "score": float(score),
            }
        )
    return results


def build_prompt(query: str, retrieved):
    context_blocks = []
    for r in retrieved:
        context_blocks.append(f"[Source: {r['source']} | relevance: {r['score']:.3f}]\n{r['text']}")
    context = "\n\n---\n\n".join(context_blocks)
    return f"""Context:
{context}

Question: {query}

Answer using only the context above."""


def ask(query: str, index, chunks, metadata):
    retrieved = retrieve(index, chunks, metadata, query, TOP_K)
    if not retrieved:
        return "No relevant content found in your documents.", []

    user_prompt = build_prompt(query, retrieved)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    answer = response.choices[0].message.content
    return answer, retrieved


def main():
    index, chunks, metadata = load_ingest()
    print("RAG bot ready. Type your question (or 'exit' to quit).\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue

        answer, retrieved = ask(query, index, chunks, metadata)

        print(f"\nBot: {answer}\n")
        print("--- Retrieved chunks used ---")
        for r in retrieved:
            print(f"  [{r['score']:.3f}] {r['source']}")
            print()
if __name__ == "__main__":
    main()