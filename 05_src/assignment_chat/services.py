from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
import requests
from chromadb.utils.embedding_functions import (
    DefaultEmbeddingFunction,
    SentenceTransformerEmbeddingFunction,
)
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv(".env")
load_dotenv(".secrets")

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "knowledge_base.json"
CHROMA_PATH = HERE / "chroma_store"
COLLECTION_NAME = "assignment_2_knowledge_local_v1"


def _load_documents() -> list[dict[str, str]]:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _make_embedding_function():
    # Use local embedding backends to keep this assignment runnable without external keys.
    try:
        return SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    except Exception:
        return DefaultEmbeddingFunction()


def _setup_collection() -> Any:
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    embedding_fn = _make_embedding_function()
    try:
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
        )
    except ValueError as exc:
        # Recover from stale persisted metadata that was created with a different embedding backend.
        if "Embedding function conflict" not in str(exc):
            raise
        client.delete_collection(name=COLLECTION_NAME)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
        )

    if collection.count() == 0:
        docs = _load_documents()
        collection.add(
            ids=[d["id"] for d in docs],
            documents=[d["text"] for d in docs],
            metadatas=[{"title": d["title"]} for d in docs],
        )

    return collection


COLLECTION = _setup_collection()


@tool
def fetch_space_news(limit: int = 3) -> str:
    """Service 1 (API): fetch recent space-news headlines and return a rewritten summary."""
    safe_limit = max(1, min(limit, 5))
    url = f"https://api.spaceflightnewsapi.net/v4/articles/?limit={safe_limit}"

    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    payload = resp.json()

    articles = payload.get("results", [])
    if not articles:
        return "I checked the news endpoint but could not find fresh articles right now."

    summaries = []
    for item in articles:
        title = item.get("title", "Untitled story")
        site = item.get("news_site", "Unknown source")
        published = item.get("published_at", "recently")
        summaries.append(f"{title} ({site}, published {published}).")

    return "Here is a quick rewritten space-news digest: " + " ".join(summaries)


@tool
def semantic_course_lookup(query: str, n_results: int = 3) -> str:
    """Service 2 (Semantic Query): search local course notes with Chroma persistent storage."""
    safe_n = max(1, min(n_results, 5))
    result = COLLECTION.query(query_texts=[query], n_results=safe_n)

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]

    if not docs:
        return "I could not find a close semantic match in the local knowledge base."

    lines = []
    for idx, text in enumerate(docs):
        title = (metas[idx] or {}).get("title", f"Source {idx + 1}")
        # Keep a concise transformed snippet rather than dumping raw text.
        snippet = text.strip().replace("\n", " ")
        if len(snippet) > 180:
            snippet = snippet[:177] + "..."
        lines.append(f"{title}: {snippet}")

    return "I found relevant material in the local semantic index. " + " | ".join(lines)


@tool
def build_study_plan(topic: str, hours_per_day: int = 1, days: int = 5) -> str:
    """Service 3 (Function Calling): create a compact study plan for a topic."""
    h = max(1, min(hours_per_day, 6))
    d = max(1, min(days, 14))

    schedule = []
    for i in range(1, d + 1):
        schedule.append(
            f"Day {i}: {h}h on {topic} (30m review, {max(20, h * 30)}m practice, 10m reflection)."
        )

    return "Custom study plan generated via function call: " + " ".join(schedule)


TOOLS = [fetch_space_news, semantic_course_lookup, build_study_plan]
