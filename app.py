"""Streamlit semantic search app for CodeSeek AI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import requests
import streamlit as st

from search_engine import SemanticSearchEngine
import subprocess, pathlib


if not pathlib.Path("data/stackoverflow_sample_3000.json").exists():
    subprocess.run(["python", "prepare_stackoverflow_sample.py"], check=True)


DEFAULT_MODEL = os.getenv("GITHUB_EMBEDDING_MODEL", "openai/text-embedding-3-small")
API_VERSION = os.getenv("GITHUB_API_VERSION", "2026-03-10")
BASE_URL = "https://models.github.ai"
DATASET_PATH = Path("data/stackoverflow_sample_3000.json")


class GitHubModelsError(RuntimeError):
    """Raised when embedding generation fails."""


@st.cache_resource(show_spinner=False)
def load_engine() -> SemanticSearchEngine:
    return SemanticSearchEngine(DATASET_PATH)


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
    payload = {"model": DEFAULT_MODEL, "input": query}

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
    except requests.RequestException as exc:
        raise GitHubModelsError(f"Embedding API request failed: {exc}") from exc

    if response.status_code >= 400:
        detail = response.text.strip()
        raise GitHubModelsError(
            f"Embedding API error {response.status_code}: {detail[:500]}"
        )

    data = response.json()
    try:
        return data["data"][0]["embedding"]
    except (KeyError, IndexError, TypeError) as exc:
        raise GitHubModelsError("Unexpected embedding API response format.") from exc


def main() -> None:
    st.set_page_config(page_title="CodeSeek AI", page_icon="🔎", layout="wide")

    st.title("CodeSeek AI")
    st.subheader("Semantic Programming Search")

    query = st.text_area(
        "Ask a programming question",
        placeholder="e.g. How can I optimize Python list comprehensions?",
        height=140,
    )

    if not DATASET_PATH.exists():
        st.error(
            f"Dataset file not found at {DATASET_PATH}. "
            "Generate it first using prepare_stackoverflow_sample.py"
        )
        st.stop()

    if not query.strip():
        st.info("Enter a query to search semantically similar Q&A pairs.")
        st.stop()

    try:
        with st.spinner("Generating embedding and searching..."):
            engine = load_engine()
            query_embedding = get_query_embedding(query.strip())
            results = engine.search(query_embedding, top_k=5)
    except Exception as exc:  # Surface user-facing issues cleanly
        st.error(f"Search failed: {exc}")
        st.stop()

    st.markdown("### Top Results")
    for i, item in enumerate(results, start=1):
        st.markdown(f"**{i}. {item['question']}**")
        st.write(item["answer"])
        st.caption(f"Similarity score: {item['score']:.4f}")
        st.divider()


if __name__ == "__main__":
    main()
