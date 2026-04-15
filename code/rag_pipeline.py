"""
rag_pipeline.py — End-to-End Sanskrit RAG Pipeline

Orchestrates: Ingest → Retrieve → Generate
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from retriever import Retriever
from generator import Generator


class SanskritRAGPipeline:
    """
    Full RAG pipeline.

    Usage:
        rag = SanskritRAGPipeline()
        rag.ingest()                # build index (skips if already built)
        result = rag.query("Who is Kalidasa?")
    """

    def __init__(self, top_k: int = config.TOP_K):
        self.top_k     = top_k
        self.retriever = None
        self.generator = None
        self._ready    = False

    def ingest(self, force_rebuild: bool = False):
        """Build index from corpus (skips if index already exists)."""
        from ingest import run_ingestion

        index_exists = (
            os.path.exists(config.FAISS_INDEX_FILE)
            and os.path.exists(config.CHUNKS_FILE)
        )

        if index_exists and not force_rebuild:
            print("[RAG] Existing index found — skipping ingestion.")
        else:
            print("[RAG] Building index from corpus …")
            run_ingestion()

        print("[RAG] Loading retriever + generator …")
        self.retriever = Retriever()
        self.generator = Generator()
        self._ready    = True
        print("[RAG] ✓ System ready\n")

    def query(self, question: str) -> dict:
        if not self._ready:
            raise RuntimeError("Call ingest() before query().")

        t0      = time.time()
        chunks  = self.retriever.retrieve(question, top_k=self.top_k)
        answer  = self.generator.generate(question, chunks)

        return {
            "query":            question,
            "answer":           answer,
            "retrieved_chunks": chunks,
            "retrieval_scores": [c["score"] for c in chunks],
            "latency_seconds":  round(time.time() - t0, 3),
        }
