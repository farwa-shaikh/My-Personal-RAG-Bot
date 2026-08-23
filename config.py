"""
Shared config for the RAG bot.
Tweak these knobs to see how they change retrieval quality.
"""

# --- Models (both billed only through your OpenAI key) ---
# --- Models ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHAT_MODEL = "gpt-4o-mini"

# --- Chunking ---
CHUNK_SIZE = 500      # tokens per chunk
CHUNK_OVERLAP = 50    # tokens of overlap between consecutive chunks

# --- Retrieval ---
TOP_K = 4             # how many chunks to retrieve per query

# --- Storage paths (all local, no external DB) ---
DOCUMENTS_DIR = "documents"
INDEX_PATH = "storage/index.faiss"
METADATA_PATH = "storage/metadata.json"
