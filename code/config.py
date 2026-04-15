"""
config.py — Central configuration for Sanskrit RAG System.
CPU-only · No external APIs
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
CODE_DIR  = Path(__file__).resolve().parent
BASE_DIR  = CODE_DIR.parent
DATA_DIR  = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"

INDEX_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Document Settings ──────────────────────────────────────────────────────────
CORPUS_FILE   = DATA_DIR / "sanskrit_corpus.txt"
CHUNK_SIZE    = 300
CHUNK_OVERLAP = 50

# ── Embedding Model ────────────────────────────────────────────────────────────
# paraphrase-multilingual-MiniLM-L12-v2
#   ~420 MB, supports Devanagari + transliterated Sanskrit, fast on CPU
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ── FAISS Index ────────────────────────────────────────────────────────────────
FAISS_INDEX_FILE = str(INDEX_DIR / "sanskrit.index")
CHUNKS_FILE      = str(INDEX_DIR / "chunks.json")
TOP_K            = 4

# ── Generator Model ────────────────────────────────────────────────────────────
# google/flan-t5-small — ~300 MB, instruction-tuned, fast on CPU
GENERATOR_MODEL    = "google/flan-t5-small"
MAX_NEW_TOKENS     = 256
NUM_BEAMS          = 4

# ── Prompt Template ────────────────────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = (
    "You are a Sanskrit literature expert. "
    "Answer the question using ONLY the context provided. "
    "If the answer is not in the context, say 'Not found in the documents.'\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\nAnswer:"
)
