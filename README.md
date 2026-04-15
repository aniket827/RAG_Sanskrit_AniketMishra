# 📜 Sanskrit RAG System
**Aniket Mishra · AI/ML Intern Assignment**

A fully local, CPU-only Retrieval-Augmented Generation (RAG) system for Sanskrit documents.
No OpenAI, no Claude, no external APIs — everything runs on your machine.

---

## 🏗️ Architecture

```
Sanskrit .txt files
       │
       ▼
┌─────────────┐  chunk_size=300, overlap=50
│   Chunker   │  (character-level sliding window)
└──────┬──────┘
       │
       ▼
┌──────────────────────────────┐
│  Embedder                    │  → 384-dim vectors
│  paraphrase-multilingual-    │
│  MiniLM-L12-v2               │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────┐
│  FAISS IndexFlatIP   │  Exact cosine search (CPU)
│  (persisted to disk) │
└──────────────────────┘

─── Query Time ──────────────────────────────────────
User Query  →  Embed  →  FAISS top-K  →  Context chunks
                                               │
                                               ▼
                              ┌────────────────────────┐
                              │  google/flan-t5-small  │
                              │  (CPU, ~300 MB)        │
                              └──────────┬─────────────┘
                                         ▼
                                    Final Answer
```

---

## 📁 Project Structure

```
RAG_Sanskrit_AniketMishra/
├── code/
│   ├── app.py           ← Streamlit UI (main entry point)
│   ├── config.py        ← All settings (chunk size, models, paths)
│   ├── ingest.py        ← Load → Chunk → Embed → FAISS index
│   ├── retriever.py     ← FAISS vector search
│   ├── generator.py     ← flan-t5-small answer generation
│   ├── rag_pipeline.py  ← Ties everything together
│   └── main.py          ← CLI fallback
├── data/
│   └── sanskrit_corpus.txt   ← Sanskrit source documents
├── index/                    ← Auto-created after ingestion
│   ├── sanskrit.index        ← FAISS binary index
│   └── chunks.json           ← Chunk text + metadata
├── report/                   ← Place your PDF report here
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Build the index (run once)
```bash
python code/ingest.py
```

### 3. Launch the Streamlit UI
```bash
streamlit run code/app.py
```

### 4. CLI (optional)
```bash
python code/main.py                          # interactive
python code/main.py --query "Who is Kalidasa?"
python code/main.py --demo                   # sample queries
```

---

## ⚙️ Configuration (`code/config.py`)

| Parameter | Value | Notes |
|---|---|---|
| `CHUNK_SIZE` | 300 | Characters per chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `EMBEDDING_MODEL` | paraphrase-multilingual-MiniLM-L12-v2 | ~420MB, supports Devanagari |
| `GENERATOR_MODEL` | google/flan-t5-small | ~300MB, CPU-friendly |
| `TOP_K` | 4 | Chunks retrieved per query |

---

## ⚡ Speed Improvements (over baseline)

| Change | Impact |
|---|---|
| `batch_size=64` in embedding | ~2× faster ingestion |
| `normalize_embeddings=True` | Enables cosine via inner-product (no extra step) |
| `IndexFlatIP` | Exact, fast search for small corpora |
| `@st.cache_resource` | Model loads once per session |

---

## 🖥️ UI Features

- **Dark theme** with Sanskrit-inspired purple/indigo palette
- **Sample query buttons** — one-click test queries in English + Sanskrit
- **Sidebar settings** — Top-K slider, force rebuild toggle
- **Session stats** — Query counter, history panel
- **Query history** — Click to re-run past queries
- **Chunk viewer** — Expandable retrieved context with similarity scores
- **Latency display** — Per-query response time

---

## 📊 Tech Stack

| Component | Technology |
|---|---|
| Embeddings | sentence-transformers (MiniLM-L12-v2) |
| Vector Store | FAISS (faiss-cpu) |
| Generator | HuggingFace Transformers (flan-t5-small) |
| UI | Streamlit |
| Runtime | Python 3.10+ · CPU-only |
