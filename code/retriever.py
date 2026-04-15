"""
retriever.py — FAISS Vector Retriever

Loads the pre-built index and returns top-K chunks for any query.
The same embedding model used during ingestion is reused here.
"""

import os
import sys
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


class Retriever:
    """
    Loads FAISS index once; reuses for every query.

    Flow:
        query → embed (multilingual-MiniLM) → FAISS IP search → top-K chunks
    """

    def __init__(self):
        print("  Loading embedding model …")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)

        if not os.path.exists(config.FAISS_INDEX_FILE):
            raise FileNotFoundError(
                f"Index not found at {config.FAISS_INDEX_FILE}\n"
                "Run:  python code/ingest.py  first."
            )

        print("  Loading FAISS index …")
        self.index = faiss.read_index(config.FAISS_INDEX_FILE)

        with open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        print(f"  ✓ Retriever ready — {self.index.ntotal} vectors indexed\n")

    def is_index_built(self) -> bool:
        return os.path.exists(config.FAISS_INDEX_FILE) and os.path.exists(config.CHUNKS_FILE)

    def retrieve(self, query: str, top_k: int = config.TOP_K) -> list:
        """Return top_k most similar chunks for the query."""
        vec = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(vec, top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({
                "rank":     rank + 1,
                "score":    float(score),
                "source":   chunk["source"],
                "text":     chunk["text"],
                "metadata": {"section": chunk["source"]},
            })
        return results
