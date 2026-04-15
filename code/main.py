"""
main.py — CLI interface for Sanskrit RAG System

Usage:
    python code/main.py                    # interactive mode
    python code/main.py --query "Who is Kalidasa?"
    python code/main.py --demo
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_pipeline import SanskritRAGPipeline
import config

DEMO_QUERIES = [
    "Who is Shankhanaad?",
    "What is the story of King Bhoj and Kalidasa?",
    "How did the old woman defeat Ghantakarna?",
    "What is the moral of the devotee and God story?",
    "शंखनादः कः अस्ति?",
]


def print_result(result: dict):
    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  Question : {result['question'] if 'question' in result else result['query']}")
    print(f"  Answer   : {result['answer']}")
    print(f"  Latency  : {result.get('latency_seconds', '—')}s")
    print(f"\n  Retrieved Chunks:")
    for c in result.get("retrieved_chunks", result.get("sources", [])):
        score = c.get("score", 0)
        print(f"\n  [{c.get('rank','-')}] score={score:.4f}  src={c.get('source','')}")
        print(f"      {c['text'][:160]}…")
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Sanskrit RAG CLI")
    parser.add_argument("--query", type=str, default=None, help="Single query")
    parser.add_argument("--demo",  action="store_true",    help="Run demo queries")
    parser.add_argument("--rebuild", action="store_true",  help="Force re-ingest")
    args = parser.parse_args()

    rag = SanskritRAGPipeline(top_k=config.TOP_K)
    rag.ingest(force_rebuild=args.rebuild)

    if args.demo:
        for q in DEMO_QUERIES:
            print_result(rag.query(q))
        return

    if args.query:
        print_result(rag.query(args.query))
        return

    # Interactive
    print("\n" + "=" * 52)
    print("  Sanskrit RAG — Interactive CLI")
    print("  Type 'quit' to exit · 'demo' to run samples")
    print("=" * 52 + "\n")

    while True:
        try:
            q = input("  >> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!"); break
        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            print("Goodbye!"); break
        if q.lower() == "demo":
            for dq in DEMO_QUERIES:
                print_result(rag.query(dq))
            continue
        print_result(rag.query(q))


if __name__ == "__main__":
    main()
