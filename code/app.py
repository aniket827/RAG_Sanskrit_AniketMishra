"""
app.py — Streamlit UI for Sanskrit RAG System

"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from rag_pipeline import SanskritRAGPipeline
from config import TOP_K, GENERATOR_MODEL

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sanskrit RAG",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Header banner */
    .header-banner {
        background: linear-gradient(135deg, #1a1f2e 0%, #2d1b4e 50%, #1a1f2e 100%);
        border: 1px solid #6c3483;
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 24px;
        text-align: center;
    }
    .header-banner h1 {
        color: #d4a0ff;
        font-size: 2.2rem;
        margin: 0;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .header-banner p {
        color: #a07cc5;
        margin: 8px 0 0 0;
        font-size: 1rem;
    }

    /* Answer box */
    .answer-box {
        background: linear-gradient(135deg, #1a2744 0%, #0d1b2a 100%);
        border: 1px solid #2e5bff;
        border-left: 4px solid #7b2ff7;
        border-radius: 10px;
        padding: 20px 24px;
        margin: 12px 0;
        color: #e8eaf6;
        font-size: 1.05rem;
        line-height: 1.7;
    }

    /* Chunk card */
    .chunk-card {
        background: #161b27;
        border: 1px solid #2a2f45;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
        font-size: 0.88rem;
        color: #c5cae9;
        line-height: 1.6;
    }
    .chunk-score {
        display: inline-block;
        background: #2d1b4e;
        color: #d4a0ff;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    /* Sample query buttons */
    .stButton > button {
        background: #1a1f2e;
        color: #c5cae9;
        border: 1px solid #3a3f5a;
        border-radius: 20px;
        font-size: 0.78rem;
        padding: 4px 12px;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: #2d1b4e;
        border-color: #7b2ff7;
        color: #d4a0ff;
    }

    /* Primary ask button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7b2ff7, #2e5bff);
        border: none;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        font-size: 1rem;
    }

    /* Stat cards */
    .stat-card {
        background: #161b27;
        border: 1px solid #2a2f45;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .stat-value { color: #d4a0ff; font-size: 1.4rem; font-weight: 700; }
    .stat-label { color: #7986cb; font-size: 0.78rem; margin-top: 2px; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: #0d1117; }

    /* Input box */
    .stTextInput > div > div > input {
        background: #161b27 !important;
        border: 1px solid #3a3f5a !important;
        color: #e8eaf6 !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
    }

    /* Divider */
    hr { border-color: #2a2f45; }

    /* History item */
    .history-item {
        background: #161b27;
        border-left: 3px solid #7b2ff7;
        padding: 8px 12px;
        border-radius: 0 6px 6px 0;
        margin: 6px 0;
        font-size: 0.85rem;
        color: #a0a8cc;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State ──────────────────────────────────────────────────────────────
for key, default in [
    ("query_text", ""),
    ("result", None),
    ("error", None),
    ("history", []),
    ("query_count", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Load RAG (cached) ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading Sanskrit RAG System…")
def load_rag(top_k: int, force_rebuild: bool) -> SanskritRAGPipeline:
    rag = SanskritRAGPipeline(top_k=top_k)
    rag.ingest(force_rebuild=force_rebuild)
    return rag


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    top_k         = st.slider("Top-K retrieved chunks", 1, 8, TOP_K)
    force_rebuild = st.checkbox("Force re-ingest on startup", value=False)

    st.markdown("---")
    st.markdown("### 📊 Session Stats")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-value">{st.session_state.query_count}</div>'
            f'<div class="stat-label">Queries</div></div>',
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-value">{len(st.session_state.history)}</div>'
            f'<div class="stat-label">History</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 🕘 Query History")
    if st.session_state.history:
        for i, h in enumerate(reversed(st.session_state.history[-8:])):
            if st.button(f"↩ {h[:40]}…" if len(h) > 40 else f"↩ {h}", key=f"hist_{i}"):
                st.session_state.query_text = h
                st.session_state.result     = None
    else:
        st.caption("No queries yet.")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption(f"**Embedding:** multilingual-MiniLM-L12-v2")
    st.caption(f"**Generator:** flan-t5-small")
    st.caption(f"**Vector Store:** FAISS (CPU)")
    st.caption(f"**Chunk size:** 300 · Overlap: 50")
    st.caption("No GPU · No external API")


# ── Load the RAG ───────────────────────────────────────────────────────────────
rag = load_rag(top_k, force_rebuild)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1>📜 Sanskrit RAG System</h1>
    <p>Ask questions about Sanskrit texts — in <b>Sanskrit (Devanagari)</b>,
    <b>transliteration</b>, or <b>English</b> · CPU-only · No external APIs</p>
</div>
""", unsafe_allow_html=True)


# ── Sample Queries ─────────────────────────────────────────────────────────────
SAMPLES = [
    "Who is Shankhanaad?",
    "What did Kalidasa do with the poem?",
    "How did the old woman defeat Ghantakarna?",
    "What is the moral of the devotee story?",
    "Who was King Bhoj?",
    "शंखनादः कः अस्ति?",
    "घण्टाकर्णः किम् अभवत्?",
]

st.markdown("**💡 Try a sample query:**")
cols = st.columns(len(SAMPLES))
for col, sample in zip(cols, SAMPLES):
    if col.button(sample, key=f"s_{sample}"):
        st.session_state.query_text = sample
        st.session_state.result     = None
        st.session_state.error      = None

st.markdown("---")

# ── Query Input ────────────────────────────────────────────────────────────────
st.markdown("### 🔍 Ask a Question")
col1, col2 = st.columns([5, 1])
with col1:
    typed = st.text_input(
        label="query",
        label_visibility="collapsed",
        placeholder="e.g.  Who is Kalidasa?  /  कालिदासः कः अस्ति?",
        value=st.session_state.query_text,
        key="input_box",
    )
with col2:
    ask_clicked = st.button("Ask →", use_container_width=True, type="primary")


final_query = (typed or st.session_state.query_text).strip()

should_run = ask_clicked or (
    st.session_state.query_text
    and st.session_state.result is None
    and st.session_state.error  is None
    and final_query
)

# ── Run Query ──────────────────────────────────────────────────────────────────
if should_run and final_query:
    with st.spinner("🔎 Retrieving context and generating answer…"):
        try:
            result = rag.query(final_query)
            st.session_state.result     = result
            st.session_state.error      = None
            st.session_state.query_text = final_query
            st.session_state.query_count += 1
            if final_query not in st.session_state.history:
                st.session_state.history.append(final_query)
        except Exception as e:
            st.session_state.error  = str(e)
            st.session_state.result = None

elif ask_clicked and not final_query:
    st.warning("⚠️ Please enter a question first.")


# ── Display Results ────────────────────────────────────────────────────────────
if st.session_state.error:
    st.error(f"❌ {st.session_state.error}")

elif st.session_state.result:
    r = st.session_state.result

    st.markdown("### 💬 Answer")
    st.markdown(
        f'<div class="answer-box">{r["answer"]}</div>',
        unsafe_allow_html=True,
    )

    # Latency + query info
    lat_col, q_col = st.columns([1, 3])
    with lat_col:
        st.metric("⏱ Latency", f"{r['latency_seconds']}s")
    with q_col:
        st.caption(f"**Query:** _{r['query']}_")

    st.markdown("---")

    # Retrieved chunks
    st.markdown("### 📚 Retrieved Context Chunks")
    for i, (chunk, score) in enumerate(
        zip(r["retrieved_chunks"], r["retrieval_scores"]), 1
    ):
        section = chunk.get("metadata", {}).get("section", chunk.get("source", "—"))
        with st.expander(f"Chunk [{i}]  ·  {section}  ·  similarity: {score:.4f}"):
            st.markdown(
                f'<div class="chunk-card">{chunk["text"]}</div>',
                unsafe_allow_html=True,
            )
