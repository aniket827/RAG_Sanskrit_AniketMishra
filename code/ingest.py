"""
ingest.py — Load → Chunk → Embed → FAISS Index

Speed improvements over naive approach:
  • batch_size=64 for embedding (faster throughput on CPU)
  • normalize_embeddings=True enables cosine via inner-product (no extra step)
  • IndexFlatIP — exact search, no approximation needed for small corpora

Run once to build the index:
    python code/ingest.py
"""

import os
import sys
import json
import glob
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# ── 1. Document Loader ─────────────────────────────────────────────────────────

def load_documents() -> list:
    """Load all .txt files from DATA_DIR."""
    documents = []
    pattern = str(config.DATA_DIR / "*.txt")
    for filepath in glob.glob(pattern):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
        fname = os.path.basename(filepath)
        documents.append({"source": fname, "text": text})
        print(f"  ✓ Loaded: {fname}  ({len(text):,} chars)")
    return documents


# ── 2. Chunker ─────────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list:
    """
    Character-level sliding-window chunking.
    chunk_size=300, overlap=50 as required by assignment.
    Works well for mixed Devanagari + transliterated Sanskrit.
    """
    chunks, start = [], 0
    text = text.strip()
    while start < len(text):
        chunk = text[start : start + config.CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += config.CHUNK_SIZE - config.CHUNK_OVERLAP
    return chunks


def build_chunks(documents: list) -> list:
    all_chunks, cid = [], 0
    for doc in documents:
        for chunk in chunk_text(doc["text"]):
            all_chunks.append({"chunk_id": cid, "source": doc["source"], "text": chunk})
            cid += 1
    print(f"  ✓ Total chunks: {len(all_chunks)}")
    return all_chunks


# ── 3. Embedder (batched, fast) ────────────────────────────────────────────────

def embed_chunks(chunks: list, model: SentenceTransformer) -> np.ndarray:
    """
    Encode chunks in batches of 64 — significantly faster than batch_size=32.
    normalize_embeddings=True → cosine similarity becomes dot product (FAISS IP).
    """
    texts = [c["text"] for c in chunks]
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=64,            # ← doubled from naive 32, faster on CPU
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    print(f"  ✓ Embedded {len(texts)} chunks in {time.time()-t0:.1f}s")
    return embeddings.astype("float32")


# ── 4. FAISS Index ─────────────────────────────────────────────────────────────

def build_and_save_index(embeddings: np.ndarray, chunks: list):
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"  ✓ FAISS index: {index.ntotal} vectors, dim={dim}")

    faiss.write_index(index, config.FAISS_INDEX_FILE)
    with open(config.CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"  ✓ Saved → {config.FAISS_INDEX_FILE}")
    print(f"  ✓ Saved → {config.CHUNKS_FILE}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run_ingestion():
    print("\n" + "=" * 52)
    print("  Sanskrit RAG — Ingestion Pipeline")
    print("=" * 52)

    print("\n[1/4] Loading documents …")
    docs = load_documents()
    if not docs:
        print(f"  ERROR: No .txt files in {config.DATA_DIR}"); return

    print(f"\n[2/4] Chunking (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}) …")
    chunks = build_chunks(docs)

    print("\n[3/4] Loading embedding model …")
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    print("\n[4/4] Embedding + building FAISS index …")
    embeddings = embed_chunks(chunks, model)
    build_and_save_index(embeddings, chunks)

    print("\n✅ Ingestion complete! Run:  streamlit run code/app.py\n")
    return chunks


if __name__ == "__main__":
    run_ingestion()
