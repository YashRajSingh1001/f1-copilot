# 🏎️ F1 Copilot — Agentic RAG for Formula 1

> "Why was Norris slower than Verstappen in Bahrain?"
> → Agent pulls telemetry, tire data, weather, race context → answers with exact numbers.

**Stack: GPT-5.4 nano · LangGraph · FastF1 · Qdrant Cloud · Streamlit**

---

## Architecture

```
User Question
     │
     ▼
LangGraph Agent (GPT-5.4 nano)
     │
     ├──► get_telemetry()        ──► FastF1 API
     ├──► compare_telemetry()    ──► FastF1 API
     ├──► get_sector_times()     ──► FastF1 API
     ├──► get_tire_data()        ──► FastF1 API
     ├──► get_weather()          ──► FastF1 API
     ├──► get_race_results()     ──► FastF1 API
     └──► search_race_context()  ──► Qdrant Cloud (RAG)
                                       ▲
                              Ingestion Pipeline
                          FastF1 → GPT-5.4 nano summary → embed → store
     │
     ▼
Synthesised answer with exact numbers
```

---

## Setup (local)

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY
```

Get a **free Qdrant Cloud cluster** at [cloud.qdrant.io](https://cloud.qdrant.io) — takes 2 minutes, no credit card.

```bash
# Ingest race data into the knowledge base
python scripts/ingest_race.py --year 2024 --gp Bahrain --session R

# Or bulk-ingest a curated set of memorable races
python scripts/ingest_race.py --bulk

# Launch the app
streamlit run ui/app.py
```

---

## Deploy to Streamlit Community Cloud (free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Set **Main file path** to `ui/app.py`
4. Under **Advanced → Secrets**, paste:

```toml
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-5.4-nano"
OPENAI_REASONING_EFFORT = "low"
OPENAI_TEXT_VERBOSITY = "low"
OPENAI_MAX_OUTPUT_TOKENS = "2000"
EMBEDDING_MODEL = "text-embedding-3-small"
QDRANT_URL = "https://your-cluster.qdrant.io"
QDRANT_API_KEY = "your-qdrant-key"
FASTF1_CACHE_PATH = "/tmp/f1_cache"
```

5. Deploy. Done.

> **Why not Vercel?** FastF1 data loading takes 30–90s (cold) and needs a persistent cache.
> Vercel serverless functions timeout at 10–60s and have no persistent filesystem.
> Streamlit Community Cloud is the right fit — purpose-built for Python data apps.

---

## Project Structure

```
f1-copilot/
├── src/
│   ├── config.py             # st.secrets → env var fallback
│   ├── agents/
│   │   ├── f1_agent.py       # LangGraph ReAct loop
│   │   └── tools.py          # 7 OpenAI-compatible tool definitions
│   └── data/
│       ├── fastf1_client.py  # FastF1 wrapper (telemetry, tires, weather, results)
│       ├── vectorstore.py    # Qdrant Cloud RAG
│       └── ingestion.py      # FastF1 → GPT-5.4 nano summary → embed → Qdrant
├── ui/app.py                 # Streamlit chat UI
├── scripts/ingest_race.py    # CLI ingestion
├── .streamlit/
│   └── config.toml           # Dark F1 theme
├── packages.txt              # System deps for Streamlit Cloud
└── requirements.txt
```

---

## Example Questions

- *"Why was Norris slower than Verstappen in Bahrain 2024?"*
- *"Compare Leclerc and Hamilton tire strategies in Monaco 2024"*
- *"What was the weather impact in Singapore 2023?"*
- *"Who had the fastest Sector 2 in Silverstone 2024 qualifying?"*
- *"Analyse Verstappen's braking telemetry in Abu Dhabi 2023"*
