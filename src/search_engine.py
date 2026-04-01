"""Semantic vector search engine backed by FAISS.

Expected dataset format (JSON array):
[
  {
    "question": "...",
    "answer": "...",
    "embeddings": [0.1, 0.2, ...]
  },
  ...
]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np


DEFAULT_DATASET_PATH = Path("data/stackoverflow_sample_3000.json")


class SemanticSearchEngine:
    """FAISS-based semantic search using cosine similarity via inner product."""

    def __init__(self, dataset_path: str | Path = DEFAULT_DATASET_PATH) -> None:
        self.dataset_path = Path(dataset_path)
        self.metadata: List[Dict[str, str]] = []
        self.embeddings: np.ndarray
        self.index: faiss.IndexFlatIP
        self._load_and_build()

    def _load_and_build(self) -> None:
        with self.dataset_path.open("r", encoding="utf-8") as f:
            rows: List[Dict[str, Any]] = json.load(f)

        if not isinstance(rows, list):
            raise ValueError("Dataset must be a JSON array of objects.")
        if not rows:
            raise ValueError("Dataset is empty; expected at least one row.")

        self.metadata = [
            {
                "question": row["question"],
                "answer": row["answer"],
            }
            for row in rows
        ]

        embeddings = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D matrix [num_rows, dim].")

        self.embeddings = self._normalize(embeddings)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalize vectors for cosine similarity search via inner product."""
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return vectors / norms

    def search(self, query_embedding: List[float] | np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search nearest neighbors and return question/answer plus similarity score."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.embeddings.shape[1]:
            raise ValueError(
                f"Query dimension {query.shape[1]} does not match index dimension {self.embeddings.shape[1]}."
            )

        query = self._normalize(query)
        scores, indices = self.index.search(query, min(top_k, len(self.metadata)))

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            item = self.metadata[int(idx)]
            results.append(
                {
                    "question": item["question"],
                    "answer": item["answer"],
                    "score": float(score),
                }
            )
        return results


def search(query_embedding: List[float] | np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
    """Module-level convenience function using the default dataset path."""
    engine = SemanticSearchEngine(DEFAULT_DATASET_PATH)
    return engine.search(query_embedding=query_embedding, top_k=top_k)
