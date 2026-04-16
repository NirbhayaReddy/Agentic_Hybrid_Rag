# Production Agentic Hybrid RAG System

An AI-powered research assistant that automatically fetches the latest AI papers from arXiv daily, indexes them, and lets you ask natural language questions. Instead of reading 10 new papers every day, you ask a question and get a sourced answer generated from the actual paper content.

---

## What It Does

**Every day (automated):**
- Fetches new AI/ML papers from arXiv
- Downloads and parses the PDFs using Docling
- Chunks the text, generates vector embeddings with Jina AI
- Indexes everything into OpenSearch (BM25 + vector search)

**When you ask a question:**
- A LangGraph agent validates your question (guardrail)
- Searches indexed papers using hybrid search — BM25 keyword + Jina AI semantic + RRF re-ranking
- Grades retrieved chunks for relevance
- Rewrites your query and retries if needed
- Passes the best chunks to GPT-4o-mini to generate a cited answer

---

## Architecture

```
arXiv API → Airflow DAG → Docling (PDF) → PostgreSQL
                                               ↓
                              Jina AI embeddings → OpenSearch

User → FastAPI → Redis Cache → LangGraph Agent → OpenSearch → GPT-4o-mini → Answer
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| API | FastAPI | REST endpoints, health checks |
| Database | PostgreSQL 16 | Paper metadata and full text |
| Search Engine | OpenSearch 2.19 | BM25 + vector hybrid search |
| Embeddings | Jina AI | Text → 1024-dim vectors |
| LLM | OpenAI GPT-4o-mini | Guardrail, grading, generation |
| Agent | LangGraph | 5-node agentic RAG workflow |
| Scheduler | Apache Airflow | Daily paper ingestion DAG |
| PDF Parser | Docling | Academic PDF → structured text |
| Cache | Redis | Exact-match query caching |
| Infrastructure | Docker Compose | 4-service containerized stack |

---

## Prerequisites

- **Docker Desktop** (with Docker Compose)
- **Python 3.12**
- **uv** package manager — [install guide](https://docs.astral.sh/uv/getting-started/installation/)
- **OpenAI API key** — [platform.openai.com](https://platform.openai.com)
- **Jina AI API key** — [jina.ai](https://jina.ai) (free tier available)

---

## How to Run

### Step 1 — Clone and set up environment

```bash
git clone https://github.com/NirbhayaReddy/Agentic_Hybrid_Rag.git
cd Agentic_Hybrid_Rag
```

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

Open `.env` and set:
```
OPENAI_API_KEY=sk-your-key-here
JINA_API_KEY=your-jina-key-here
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini
```

### Step 2 — Install Python dependencies

```bash
uv sync
```

### Step 3 — Start infrastructure (Docker)

```bash
docker compose up -d --build
```

This starts: PostgreSQL, OpenSearch, Redis, Airflow. Wait ~60 seconds for all services to be healthy.

Verify:
```bash
docker ps
```

All 4 containers should show `(healthy)`.

### Step 4 — Start the API (local)

Open a new terminal:

```bash
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

Check it's running:
```bash
curl http://localhost:8000/api/v1/health
```

### Step 5 — Trigger paper ingestion

Open Airflow at [http://localhost:8080](http://localhost:8080)

- Username: `admin`
- Password: check `airflow/simple_auth_manager_passwords.json.generated`

Find the `arxiv_paper_ingestion` DAG → click the play button → trigger it manually.

This fetches papers, parses PDFs, and indexes everything into OpenSearch. Takes ~2-5 minutes depending on `ARXIV__MAX_RESULTS` in your `.env`.

### Step 6 — Ask a question

```bash
curl -X POST http://localhost:8000/api/v1/ask-agentic \
  -H "Content-Type: application/json" \
  -d '{"query": "How does mixture of depths attention work in transformers?"}'
```

You'll get back an answer with source paper URLs.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Check all services status |
| `/api/v1/ask` | POST | Simple RAG (single-shot) |
| `/api/v1/ask-agentic` | POST | Full LangGraph agent pipeline |
| `/api/v1/hybrid-search/` | POST | Raw search, returns chunks |

API docs (interactive): [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Services

| Service | URL |
|---------|-----|
| FastAPI | http://localhost:8000 |
| Airflow | http://localhost:8080 |
| OpenSearch | http://localhost:9200 |

---

## Configuration

Key `.env` variables:

```bash
# LLM
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# Embeddings
JINA_API_KEY=...

# Paper ingestion
ARXIV__MAX_RESULTS=10        # how many papers to fetch per day
ARXIV__CATEGORIES=cs.AI      # arXiv category

# Chunking
CHUNKING__CHUNK_SIZE=600
CHUNKING__OVERLAP_SIZE=100

# Cache TTL
REDIS__TTL_HOURS=24
```

---

## RAG Strategies Used

| Component | Strategy |
|-----------|---------|
| Chunking | Fixed-size, 600 words, 100-word overlap |
| Embedding | Jina AI `jina-embeddings-v3`, 1024 dimensions |
| Retrieval | Hybrid BM25 + vector search with RRF fusion |
| Re-ranking | LLM grading — GPT judges each chunk relevance |
| Query rewriting | Adaptive retry — rewrites query if 0 relevant chunks found |
| Guardrail | LLM scoring — rejects off-topic questions (threshold: 60/100) |

---

## Project Structure

```
├── src/
│   ├── main.py                    # FastAPI app, service initialization
│   ├── config.py                  # All settings via env vars
│   ├── dependencies.py            # Dependency injection
│   ├── routers/                   # API endpoints
│   │   ├── agentic_ask.py         # Agentic RAG endpoint
│   │   ├── ask.py                 # Simple RAG endpoint
│   │   ├── hybrid_search.py       # Raw search endpoint
│   │   └── ping.py                # Health check
│   ├── services/
│   │   ├── agents/                # LangGraph agent
│   │   │   ├── nodes/             # 5 agent nodes
│   │   │   ├── agentic_rag.py     # Workflow orchestration
│   │   │   ├── state.py           # Agent state definition
│   │   │   ├── context.py         # Runtime context/services
│   │   │   ├── tools.py           # retrieve_papers tool
│   │   │   └── prompts.py         # All LLM prompts
│   │   ├── opensearch/            # Search client + index config
│   │   ├── embeddings/            # Jina AI client
│   │   ├── indexing/              # Chunker + hybrid indexer
│   │   ├── openai/                # OpenAI LLM client
│   │   ├── cache/                 # Redis cache client
│   │   ├── arxiv/                 # arXiv API client
│   │   ├── pdf_parser/            # Docling PDF parser
│   │   └── metadata_fetcher.py    # Ingestion orchestrator
│   ├── models/                    # SQLAlchemy DB models
│   ├── repositories/              # DB query layer
│   └── schemas/                   # Pydantic request/response models
├── airflow/
│   ├── Dockerfile                 # Airflow image (includes libgl1 for Docling)
│   ├── dags/
│   │   ├── arxiv_paper_ingestion.py   # Main DAG definition
│   │   └── arxiv_ingestion/
│   │       ├── fetching.py            # Fetch + parse papers
│   │       ├── indexing.py            # Chunk + embed + index
│   │       ├── reporting.py           # Daily stats report
│   │       └── setup.py              # DB initialization
│   └── requirements-airflow.txt
├── compose.yml                    # Docker services
├── pyproject.toml                 # Python dependencies
└── .env.example                   # Environment variable template
```

---

## Stopping

```bash
# Stop the API
Ctrl+C in the terminal running uvicorn

# Stop Docker services
docker compose down
```

Data persists in Docker volumes — papers and indexes are saved between restarts.


## License

MIT
