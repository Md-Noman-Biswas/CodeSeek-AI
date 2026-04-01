"""Streamlit semantic search app for CodeSeek AI."""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from typing import List

import streamlit as st
from sentence_transformers import SentenceTransformer

from search_engine import SemanticSearchEngine


# ================= CONFIG =================
DATASET_PATH = Path("data/stackoverflow_sample_3000.json")


# ================= DATASET SETUP =================
def ensure_dataset():
    if not DATASET_PATH.exists():
        with st.spinner("Preparing dataset (first run only)..."):
            script = Path(__file__).parent / "prepare_stackoverflow_sample.py"
            result = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                st.error(f"Dataset preparation failed:\n\n{result.stderr}")
                st.stop()


# ================= ENGINE =================
@st.cache_resource(show_spinner=False)
def load_engine() -> SemanticSearchEngine:
    return SemanticSearchEngine(DATASET_PATH)


# ================= EMBEDDING =================
@st.cache_resource(show_spinner=False)
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_query_embedding(query: str) -> List[float]:
    model = load_embedder()
    return model.encode(query).tolist()


# ================= MAIN APP =================
def main():
    st.set_page_config(page_title="CodeSeek AI", page_icon="🔎", layout="wide")

    st.title("🔎 CodeSeek AI")
    st.subheader("Semantic Programming Search")

    ensure_dataset()

    query = st.text_area(
        "Ask a programming question:",
        placeholder="e.g. How to declare array in Python?",
        height=120,
    )

    if not query.strip():
        st.info("Enter a query to begin search.")
        return

    try:
        with st.spinner("Searching..."):
            engine = load_engine()
            query_embedding = get_query_embedding(query.strip())
            results = engine.search(query_embedding, top_k=5)

    except Exception as e:
        st.error(f"Search failed: {e}")
        return

    st.markdown("### Top Results")

    for i, item in enumerate(results, start=1):
        st.markdown(f"**{i}. {item['question']}**")
        st.write(item["answer"])
        st.caption(f"Similarity score: {item['score']:.4f}")
        st.divider()


if __name__ == "__main__":
    main()