"""Streamlit semantic search app for CodeSeek AI."""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from typing import List

import requests
import streamlit as st

from search_engine import SemanticSearchEngine


# ================= CONFIG =================
DEFAULT_MODEL = os.getenv("GITHUB_EMBEDDING_MODEL", "openai/text-embedding-3-small")
API_VERSION = os.getenv("GITHUB_API_VERSION", "2026-03-10")
BASE_URL = "https://models.github.ai"
DATASET_PATH = Path("data/stackoverflow_sample_3000.json")


# ================= ERRORS =================
class GitHubModelsError(RuntimeError):
    pass


# ================= DATASET SETUP =================
def ensure_dataset():
    if not DATASET_PATH.exists():
        with st.spinner("Preparing dataset (first run only)..."):
            subprocess.run(
                [sys.executable, "prepare_stackoverflow_sample.py"],
                check=True
            )


# ================= ENGINE =================
@st.cache_resource(show_spinner=False)
def load_engine() -> SemanticSearchEngine:
    return SemanticSearchEngine(DATASET_PATH)


# ================= EMBEDDING =================
def get_query_embedding(query: str) -> List[float]:
    token = os.getenv("GITHUB_TOKEN")
    org = os.getenv("GITHUB_ORG")

    if not token:
        raise GitHubModelsError("Missing GITHUB_TOKEN environment variable.")

    endpoint = f"{BASE_URL}/inference/embeddings"
    if org:
        endpoint = f"{BASE_URL}/orgs/{org}/inference/embeddings"

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": API_VERSION,
        "Content-Type": "application/json",
    }

    payload = {
        "model": DEFAULT_MODEL,
        "input": query
    }

    response = requests.post(endpoint, headers=headers, json=payload, timeout=30)

    if response.status_code >= 400:
        raise GitHubModelsError(f"API Error {response.status_code}: {response.text[:300]}")

    data = response.json()
    return data["data"][0]["embedding"]


# ================= MAIN APP =================
def main():
    st.set_page_config(page_title="CodeSeek AI", page_icon="🔎", layout="wide")

    st.title("🔎 CodeSeek AI")
    st.subheader("Semantic Programming Search")

    # Ensure dataset exists
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