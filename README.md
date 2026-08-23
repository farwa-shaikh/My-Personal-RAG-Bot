# Personal RAG Bot 

A minimal, practical RAG pipeline you can run end-to-end with nothing but an
OpenAI API key. Vector storage is local (FAISS), so there are no other paid
services involved.

## How it works

```
documents/*.txt,.md,.pdf
        |
        v
   [ingest.py]
   - chunk text (token-based, overlapping)
   - embed chunks -> OpenAI Embeddings API
   - store vectors -> local FAISS index (storage/index.faiss)
   - store chunk text/metadata -> storage/metadata.json
        |
        v
   [chatbot.py]
   - embed your question -> OpenAI Embeddings API
   - similarity search against FAISS index
   - stuff top-k chunks into a prompt
   - ask OpenAI Chat API to answer using only that context
```

## Setup

```bash
cd rag-bot
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and paste your real key:
# OPENAI_API_KEY=sk-...
```

## Usage

1. Drop a few `.txt`, `.md`, or `.pdf` files into `documents/`.
2. Build the index:
   ```bash
   python ingest.py
   ```
3. Chat with your documents:
   ```bash
   python chatbot.py
   ```

Example session:
```
You: What does the document say about refund policy?
Bot: According to policy.txt, refunds are issued within 14 days of purchase...

--- Retrieved chunks used ---
  [0.842] documents/policy.txt
  [0.791] documents/faq.txt
```

## What's actually happening (the practical bits worth studying)

- **Chunking** (`ingest.py::chunk_text`) — splits by token count with overlap,
  so ideas that span a chunk boundary aren't lost entirely. Try changing
  `CHUNK_SIZE`/`CHUNK_OVERLAP` in `config.py` and re-running `ingest.py` to
  see how retrieval quality changes.
- **Embeddings** — `text-embedding-3-small` turns each chunk (and later, each
  query) into a 1536-dim vector. Similar meaning -> nearby vectors.
- **Vector search** — FAISS `IndexFlatIP` does exact cosine similarity search
  (vectors are L2-normalized, so inner product = cosine similarity). It's a
  brute-force index — fine for personal-scale document sets (thousands of
  chunks). For much larger corpora you'd swap in an approximate index
  (`IndexIVFFlat`, `IndexHNSWFlat`), which is a good next experiment.
- **Retrieval + generation** — top-k chunks get stuffed into the prompt with
  their source filenames, and the system prompt instructs the model to
  answer only from that context and cite sources — this is the core RAG
  pattern (retrieve, then generate grounded on what was retrieved).

  Results:
<img width="1365" height="768" alt="RAGBOT" src="https://github.com/user-attachments/assets/e042483c-b390-4df2-ba4b-e33d2de3bb19" />

This Project is done by Farwa Shaikh with the help of AI Tools ;)
