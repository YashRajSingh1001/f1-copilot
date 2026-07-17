"""Ingestion pipeline — pull F1 session data, generate summaries, store in vector DB."""

import json
from openai import OpenAI
from . import fastf1_client as ff1
from .vectorstore import ingest_session_summary, ingest_document
from ..config import get


def _llm_summarize(prompt: str) -> str:
    client = OpenAI(api_key=get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_completion_tokens=1000,
    )
    return response.choices[0].message.content


def ingest_race_session(year: int, grand_prix: str, session_type: str = "R") -> str:
    """
    Full ingestion pipeline for one session:
    1. Pull all data via FastF1
    2. Generate an LLM narrative summary
    3. Store in ChromaDB
    Returns the generated summary.
    """
    print(f"Loading session: {year} {grand_prix} {session_type}...")
    session = ff1.load_session_basic(year, grand_prix, session_type)

    results = ff1.get_race_results(session)
    weather = ff1.get_session_weather(session)

    raw_data = {
        "event": f"{year} {grand_prix} {session_type}",
        "results": results,
        "weather": weather,
    }

    prompt = f"""You are an expert F1 analyst. Generate a detailed, factual race summary based on the data below.
Include: race winner, key battles, tire strategies used, weather impact, notable incidents, and performance insights.
Write in a factual, technical tone suitable for a racing engineer. 400-600 words.

DATA:
{json.dumps(raw_data, indent=2)}
"""
    print("Generating LLM summary...")
    summary = _llm_summarize(prompt)

    ingest_session_summary(
        year=year,
        grand_prix=grand_prix,
        session_type=session_type,
        summary_text=summary,
    )

    raw_text = f"RAW DATA for {year} {grand_prix} {session_type}:\n{json.dumps(raw_data, indent=2)}"
    ingest_document(
        text=raw_text,
        doc_id=f"{year}_{grand_prix.replace(' ', '_')}_{session_type}_raw",
        metadata={
            "year": year,
            "grand_prix": grand_prix,
            "session_type": session_type,
            "type": "raw_data",
        },
    )

    print(f"Ingested {year} {grand_prix} {session_type} into vector store.")
    return summary


def ingest_custom_article(text: str, title: str, source: str = "manual") -> None:
    """Ingest any F1 article or commentary into the RAG store."""
    ingest_document(
        text=text,
        doc_id=f"article_{title.replace(' ', '_')[:40]}",
        metadata={"type": "article", "title": title, "source": source},
    )
    print(f"Ingested article: {title}")
