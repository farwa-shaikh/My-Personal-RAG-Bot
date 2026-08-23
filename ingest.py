import os
import json
import glob

import numpy as np
import faiss
import tiktoken
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DOCUMENTS_DIR,
    INDEX_PATH,
    METADATA_PATH,
)

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
encoding = tiktoken.get_encoding("cl100k_base")


def read_file(path: str) -> str:
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def chunk_text(text: str, chunk_size: int, overlap: int):
    tokens = encoding.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(encoding.decode(chunk_tokens))
        start += chunk_size - overlap
    return chunks


def embed_texts(texts: list[str]) -> np.ndarray:
    vectors = embeddings.embed_documents(texts)
    vec = np.array(vectors, dtype="float32")
    faiss.normalize_L2(vec)
    return vec


def main():
    os.makedirs("storage", exist_ok=True)
    filepaths = glob.glob(os.path.join(DOCUMENTS_DIR, "**", "*"), recursive=True)
    filepaths = [f for f in filepaths if f.lower().endswith((".txt", ".md", ".pdf"))]

    if not filepaths:
        print(f"No .txt/.md/.pdf files found in '{DOCUMENTS_DIR}/'. Add some and re-run.")
        return

    all_chunks = []
    all_metadata = []

    for path in filepaths:
        print(f"Reading {path} ...")
        text = read_file(path)
        if not text.strip():
            print(f"  (skipped — no extractable text)")
            continue
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"  -> {len(chunks)} chunks")
        for i, c in enumerate(chunks):
            all_chunks.append(c)
            all_metadata.append({"source": path, "chunk_index": i})

    print(f"\nEmbedding {len(all_chunks)} chunks with HuggingFace (384 dims) ...")
    batch_size = 100
    all_vectors = []
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i: i + batch_size]
        vectors = embed_texts(batch)
        all_vectors.append(vectors)
        print(f"  embedded {min(i + batch_size, len(all_chunks))}/{len(all_chunks)}")

    matrix = np.vstack(all_vectors)
    dimension = matrix.shape[1]
    index = faiss.IndexFlatIP(dimension)

    # ADD VECTORS TO FAISS INDEX
    index.add(matrix)

    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump({"chunks": all_chunks, "metadata": all_metadata}, f)

    print(f"\nDone. Index saved to {INDEX_PATH}, metadata saved to {METADATA_PATH}.")


if __name__ == "__main__":
    main()