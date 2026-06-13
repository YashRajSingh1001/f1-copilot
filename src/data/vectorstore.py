"""Qdrant Cloud vector store for race reports, context, and session summaries."""

import uuid
from typing import Optional

from openai import OpenAI
from qdrant_client import QdrantClient
from ..config import get
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

COLLECTION_NAME = "f1_race_context"
VECTOR_DIM = 1536  # text-embedding-3-small


def _get_client() -> QdrantClient:
    url = get("QDRANT_URL")
    api_key = get("QDRANT_API_KEY")
    if not url or not api_key:
        raise ValueError(
            "QDRANT_URL and QDRANT_API_KEY must be set. "
            "Get a free cluster at cloud.qdrant.io"
        )
    return QdrantClient(url=url, api_key=api_key)


def _ensure_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )


def _embed(texts: list[str]) -> list[list[float]]:
    oai = OpenAI(api_key=get("OPENAI_API_KEY"))
    response = oai.embeddings.create(
        model=get("EMBEDDING_MODEL", "text-embedding-3-small"),
        input=texts,
    )
    return [item.embedding for item in response.data]


def ingest_document(
    text: str,
    doc_id: str,
    metadata: Optional[dict] = None,
) -> None:
    """Chunk, embed, and upsert a document into Qdrant."""
    client = _get_client()
    _ensure_collection(client)

    chunks = _chunk_text(text, chunk_size=800, overlap=100)
    embeddings = _embed(chunks)

    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}_chunk_{i}")),
            vector=embeddings[i],
            payload={
                **(metadata or {}),
                "text": chunks[i],
                "doc_id": doc_id,
                "chunk_index": i,
            },
        )
        for i in range(len(chunks))
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)


def search(query: str, n_results: int = 5, filter_meta: Optional[dict] = None) -> list[dict]:
    """Semantic search over ingested race context."""
    client = _get_client()
    _ensure_collection(client)

    query_vector = _embed([query])[0]

    qdrant_filter = None
    if filter_meta:
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filter_meta.items()
        ]
        qdrant_filter = Filter(must=conditions)

    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=n_results,
        query_filter=qdrant_filter,
        with_payload=True,
    )

    return [
        {
            "text": hit.payload.get("text", ""),
            "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
            "score": hit.score,
        }
        for hit in hits
    ]


def ingest_session_summary(
    year: int,
    grand_prix: str,
    session_type: str,
    summary_text: str,
) -> None:
    doc_id = f"{year}_{grand_prix.replace(' ', '_')}_{session_type}"
    ingest_document(
        text=summary_text,
        doc_id=doc_id,
        metadata={
            "year": year,
            "grand_prix": grand_prix,
            "session_type": session_type,
            "type": "session_summary",
        },
    )


def get_collection_stats() -> dict:
    client = _get_client()
    _ensure_collection(client)
    info = client.get_collection(COLLECTION_NAME)
    return {
        "total_vectors": info.vectors_count,
        "collection": COLLECTION_NAME,
        "status": str(info.status),
    }


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks
