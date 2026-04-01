"""Prepare a lightweight semantic search dataset from Hugging Face.

Usage:
    python prepare_stackoverflow_sample.py
"""

from pathlib import Path
import json

from datasets import load_dataset


DATASET_ID = "MartinElMolon/stackoverflow_preguntas_con_embeddings"
OUTPUT_PATH = Path("data/stackoverflow_sample_3000.json")
SAMPLE_SIZE = 3000


def main() -> None:
    ds = load_dataset(DATASET_ID, split="train[:3000]")

    if len(ds) < SAMPLE_SIZE:
        raise ValueError(f"Dataset has only {len(ds)} rows; expected at least {SAMPLE_SIZE}.")

    sampled = ds.shuffle(seed=42).select(range(SAMPLE_SIZE))
    sampled = sampled.select_columns(["question", "answer", "embeddings"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(sampled.to_list(), f, ensure_ascii=False)

    print(f"Saved {len(sampled)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
