"""Pinecone vector store for race reports, context, and session summaries."""

import hashlib
from typing import Optional

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from ..config import get

INDEX_NAME = "f1-race-context"
VECTOR_DIM = 1536  # text-embedding-3-small


def _get_index():
    pc = Pinecone(api_key=get("PINECONE_API_KEY"))
    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(INDEX_NAME)


def _embed(texts: list[str]) -> list[list[float]]:
    oai = OpenAI(api_key=get("OPENAI_API_KEY"))
    response = oai.embeddings.create(
        model=get("EMBEDDING_MODEL", "text-embedding-3-small"),
        input=texts,
    )
    return [item.embedding for item in response.data]


def _make_id(doc_id: str, chunk_index: int) -> str:
    raw = f"{doc_id}_chunk_{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def ingest_document(
    text: str,
    doc_id: str,
    metadata: Optional[dict] = None,
) -> None:
    """Chunk, embed, and upsert a document into Pinecone."""
    index = _get_index()
    chunks = _chunk_text(text, chunk_size=800, overlap=100)
    embeddings = _embed(chunks)

    vectors = [
        {
            "id": _make_id(doc_id, i),
            "values": embeddings[i],
            "metadata": {
                **(metadata or {}),
                "text": chunks[i],
                "doc_id": doc_id,
                "chunk_index": i,
            },
        }
        for i in range(len(chunks))
    ]

    # Pinecone recommends batches of 100
    for batch_start in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[batch_start:batch_start + 100])


def search(query: str, n_results: int = 5, filter_meta: Optional[dict] = None) -> list[dict]:
    """Semantic search over ingested race context."""
    index = _get_index()
    query_vector = _embed([query])[0]

    results = index.query(
        vector=query_vector,
        top_k=n_results,
        include_metadata=True,
        filter=filter_meta or None,
    )

    return [
        {
            "text": match.metadata.get("text", ""),
            "metadata": {k: v for k, v in match.metadata.items() if k != "text"},
            "score": match.score,
        }
        for match in results.matches
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
    index = _get_index()
    stats = index.describe_index_stats()
    return {
        "total_vectors": stats.total_vector_count,
        "collection": INDEX_NAME,
        "dimension": VECTOR_DIM,
    }


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks
