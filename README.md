---
title: CodeSeek AI
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
tags:
- streamlit
- semantic-search
- faiss
- stackoverflow
- nlp
pinned: false
short_description: Semantic search engine for programming questions
---

# 🔎 CodeSeek AI

A semantic search engine for programming questions, powered by FAISS and Sentence Transformers.

## How it works

1. Type any programming question in the search box
2. Your query is converted into a vector embedding using `all-MiniLM-L6-v2`
3. FAISS performs a cosine similarity search against 3,000 StackOverflow Q&A pairs
4. The most relevant results are returned ranked by similarity score

## Tech Stack

- **Frontend:** Streamlit
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Search:** FAISS
- **Dataset:** StackOverflow Q&A with pre-computed embeddings from Hugging Face
- **Hosting:** Hugging Face Spaces (Docker)

## Local Development
```bash
pip install -r src/requirements.txt
python src/prepare_stackoverflow_sample.py
streamlit run src/streamlit_app.py
```
