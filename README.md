<h1 align="center">Clinical RAG Agent</h1>

<p align="center">
  <strong>Production-ready agentic RAG system for medical literature Q&A — powered by LangGraph, FastAPI, and PostgreSQL</strong>
</p>

<p align="center">
  <a href="#overview">Overview</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#api">API</a> &bull;
  <a href="#evaluation">Evaluation</a> &bull;
  <a href="#quickstart">Quickstart</a> &bull;
  <a href="#deployment">Deployment</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-async-009688?logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangGraph-1.0%2B-orange?logo=langchain&logoColor=white"/>
  <img src="https://img.shields.io/badge/Qdrant-vector%20db-DC244C"/>
  <img src="https://img.shields.io/badge/PostgreSQL-16-336791?logo=postgresql&logoColor=white"/>
  <img src="https://img.shields.io/badge/Redis-cache-DC382D?logo=redis&logoColor=white"/>
  <img src="https://img.shields.io/badge/Docker-compose-2496ED?logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/Gemini%202.5-Flash-4285F4?logo=google&logoColor=white"/>
  <img src="https://img.shields.io/badge/license-MIT-green"/>
</p>

<p align="center">
  <strong>🚀 Live demo coming soon — AWS deployment in progress</strong>
</p>

---

## Overview

**Clinical RAG Agent** is a production-ready retrieval-augmented generation backend for medical literature question answering. It ingests clinical textbooks, indexes them into a hybrid vector store, and answers queries with full source attribution — grounded strictly in the evidence.

Built on [LangGraph](https://github.com/langchain-ai/langgraph) with an async [FastAPI](https://fastapi.tiangolo.com/) backend, PostgreSQL persistence, and Redis caching, the system is containerized with Docker and designed for cloud deployment.

### Highlights

- **87% accuracy on USMLE-style MedQA** benchmark (in-KB questions)
- **11-node LangGraph state machine** with conditional routing, context compression, and retry logic
- **Hybrid retrieval** (dense + BM25) with cross-encoder reranking over 350K+ indexed chunks
- **Redis-backed cache** for embeddings and LLM responses — cuts repeat-query latency significantly
- **PostgreSQL** for parent chunk storage and evaluation metrics tracking
- **REST API + SSE streaming** — `/api/chat`, `/api/chat/stream`, `/api/documents`
- **Automated evaluation pipeline** with PostgreSQL metrics tracking (67% → 87% across 3 optimization cycles)

> **Disclaimer**: For research and educational purposes only. Not a substitute for professional medical judgment.

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Backend                       │
│                                                          │
│   POST /api/chat          POST /api/chat/stream          │
│   GET  /api/documents     POST /api/documents/upload     │
│   GET  /api/health                                       │
│                        │                                 │
│            ┌───────────▼────────────┐                   │
│            │   Redis LLM Cache      │ ← cache hit?      │
│            └───────────┬────────────┘                   │
│                        │ miss                            │
│            ┌───────────▼────────────┐                   │
│            │   LangGraph Agent      │                    │
│            └───────────┬────────────┘                   │
└────────────────────────┼────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    ┌─────────┐    ┌──────────┐    ┌──────────┐
    │ Qdrant  │    │PostgreSQL│    │  Redis   │
    │ Vectors │    │  Chunks  │    │Emb Cache │
    └─────────┘    └──────────┘    └──────────┘
```

### LangGraph State Machine (11 nodes)

```
User Query
    │
    ▼
[summarize_history] ──► [rewrite_query] ──► [request_clarification]?
                               │
                    ┌──────────▼──────────┐
                    │   Agent Subgraph     │
                    │                      │
                    │  [orchestrator]      │
                    │      │  ▲            │
                    │      ▼  │            │
                    │  [tools] (hybrid     │
                    │   search + rerank)   │
                    │      │               │
                    │      ▼               │
                    │  [should_compress]   │
                    │      │               │
                    │  [compress_context]──┘
                    │      │
                    │  [fallback_response]
                    │      │
                    │  [collect_answer]
                    └──────┬──────────────┘
                           │
                    [aggregate_answers]
                           │
                    Final Answer + Sources
```

### Retrieval Pipeline

```
Query
  │
  ▼
Redis Embedding Cache ──hit──► [cached vector]
  │ miss
  ▼
HuggingFace all-mpnet-base-v2 (768-dim dense)
  +
Qdrant/BM25 (sparse)
  │
  ▼
Hybrid Search → over-fetch 3× candidates
  │
  ▼
cross-encoder/ms-marco-MiniLM-L-6-v2 reranking → top-5
  │
  ▼
Retrieve Parent Chunks (PostgreSQL)
  │
  ▼
LLM Synthesis (Gemini 2.5 Flash)
  │
  ▼
Answer + Source Citations
```

---

## API

The FastAPI backend exposes a REST API alongside the Gradio UI.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Service health check (Qdrant, PostgreSQL, Redis) |
| `POST` | `/api/chat` | Synchronous Q&A |
| `POST` | `/api/chat/stream` | Server-Sent Events streaming response |
| `POST` | `/api/chat/reset` | Reset conversation session |
| `GET` | `/api/documents` | List indexed documents |
| `POST` | `/api/documents/upload` | Upload PDF or Markdown |
| `DELETE` | `/api/documents` | Clear knowledge base |

```bash
# Example
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the main causes of myocardial infarction?"}'
```

Interactive API docs: `http://localhost:8000/docs`

---

## Evaluation

### MedQA Benchmark (USMLE-style)

#### Iterative Optimization (20-question subset)

| Metric | v1 Baseline | v2 | v3 |
|--------|:-----------:|:--:|:--:|
| **In-KB Accuracy** | 67% | 80% | **87%** |
| **Overall Accuracy** | 65% | 75% | **80%** |
| **Avg Time / Question** | 32.1s | 35.3s | 26.4s |

#### Scaled Evaluation (100 questions, 18 textbooks)

| Metric | Result |
|--------|:------:|
| **Overall Accuracy** | **80%** |
| **In-KB Accuracy** | 78% (62/80) |
| **Source Attribution** | 86% (69/80) |
| **Avg Time / Question** | 26.0s |

Evaluation results are persisted to PostgreSQL (`eval_runs` + `eval_results` tables) for tracking across optimization cycles.

```bash
python scripts/evaluate.py                          # 100 questions (default)
python scripts/evaluate.py --num-in-kb 15 --num-out-kb 5
python scripts/evaluate.py --dry-run                # preview only
```

---

## Quickstart

### Option A — Docker Compose (recommended)

```bash
git clone https://github.com/rzhang-tech/clinical-rag-agent.git
cd clinical-rag-agent

# Set your API key
echo "GOOGLE_API_KEY=your-key-here" > .env

# Start all services (app + Qdrant + PostgreSQL + Redis)
docker compose up -d --build

# Import knowledge base (one-time, ~30-60 min)
docker compose exec app python scripts/import_medrag.py

# UI:  http://localhost:8000/ui
# API: http://localhost:8000/api/health
```

### Option B — Local development

```bash
# Start backing services only
docker run -d -p 5432:5432 -e POSTGRES_DB=clinical_rag -e POSTGRES_PASSWORD=password postgres:16-alpine
docker run -d -p 6379:6379 redis:7-alpine

# Install dependencies
conda create -n clinical-rag python=3.11 -y
conda activate clinical-rag
pip install -r requirements.txt

# Configure
cp project/.env.example project/.env  # add GOOGLE_API_KEY

cd project
python app.py
# UI: http://localhost:8000/ui
```

---

## Knowledge Base

18 medical textbooks from [MedRAG](https://github.com/Teddy-XiongGZ/MedRAG), totalling 350K+ indexed chunks.

| Textbook | Domain | Chunks |
|----------|--------|-------:|
| Harrison's Internal Medicine | Internal Medicine | 32,628 |
| Surgery (Schwartz) | Surgery | 14,349 |
| Neurology (Adams) | Neurology | 12,370 |
| Obstetrics (Williams) | Obstetrics | 9,166 |
| Gynecology (Novak) | Gynecology | 7,947 |
| Pharmacology (Katzung) | Pharmacology | 7,356 |
| Cell Biology (Alberts) | Cell Biology | 7,070 |
| Pathology (Robbins) | Pathology | 5,297 |
| Immunology (Janeway) | Immunology | 4,852 |
| Histology (Ross) | Histology | 4,411 |
| Physiology (Levy) | Physiology | 4,370 |
| Pediatrics (Nelson) | Pediatrics | 4,260 |
| Psychiatry (DSM-5) | Psychiatry | 4,057 |
| Anatomy (Gray's) | Anatomy | 3,017 |
| Biochemistry (Lippincott) | Biochemistry | 1,973 |
| First Aid Step 2 | Clinical Review | 1,369 |
| First Aid Step 1 | Basic Sciences | 850 |
| Pathoma (Husain) | Pathology | 505 |
| **Total** | **18 textbooks** | **~350K** |

---

## Deployment

### AWS EC2 (t3.large)

```bash
# On EC2 (Ubuntu 24.04)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker ubuntu && newgrp docker

git clone https://github.com/rzhang-tech/clinical-rag-agent.git
cd clinical-rag-agent
echo "GOOGLE_API_KEY=your-key" > .env

docker compose up -d --build
docker compose exec app python scripts/import_medrag.py
```

---

## Project Structure

```
clinical-rag-agent/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── project/
    ├── app.py                      # Entry point (FastAPI + Gradio mounted at /ui)
    ├── config.py                   # Central configuration
    ├── api/
    │   ├── main.py                 # FastAPI app factory + lifespan
    │   ├── schemas.py              # Pydantic request/response models
    │   └── routes/
    │       ├── chat.py             # /api/chat, /api/chat/stream
    │       ├── documents.py        # /api/documents
    │       └── health.py           # /api/health
    ├── core/
    │   ├── rag_system.py           # System bootstrap + process-level singleton
    │   ├── chat_interface.py       # Agent graph wrapper (Gradio path)
    │   ├── document_manager.py     # Document ingestion
    │   └── observability.py        # Langfuse tracing
    ├── db/
    │   ├── vector_db_manager.py    # Qdrant hybrid search + Redis embedding cache
    │   ├── parent_store_manager.py # PostgreSQL parent chunks (JSON fallback)
    │   ├── postgres_manager.py     # PostgreSQL DDL + CRUD
    │   └── cache_manager.py        # Redis client + CachedEmbeddings
    ├── rag_agent/
    │   ├── graph.py                # LangGraph workflow (11 nodes)
    │   ├── graph_state.py          # State schemas
    │   ├── nodes.py                # Node implementations
    │   ├── edges.py                # Conditional routing logic
    │   ├── prompts.py              # Medical-aware system prompts
    │   ├── tools.py                # Retrieval tools + cross-encoder singleton
    │   └── schemas.py              # Pydantic schemas
    ├── scripts/
    │   ├── import_medrag.py        # MedRAG textbook importer
    │   ├── evaluate.py             # MedQA evaluation → PostgreSQL metrics
    │   ├── migrate_parent_store.py # One-time JSON → PostgreSQL migration
    │   └── upload_qdrant_to_remote.py  # Local → remote Qdrant migration
    └── ui/
        ├── gradio_app.py           # Gradio chat + document management UI
        └── css.py                  # UI styling
```

---

## Configuration

```bash
# project/.env
LLM_PROVIDER=gemini           # or "ollama"
GOOGLE_API_KEY=...

# Set by docker-compose automatically:
QDRANT_URL=http://qdrant:6333
POSTGRES_URL=postgresql://postgres:password@postgres:5432/clinical_rag
REDIS_URL=redis://redis:6379
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DENSE_MODEL` | `all-mpnet-base-v2` | 768-dim dense embedding model |
| `SPARSE_MODEL` | `Qdrant/bm25` | BM25 sparse embeddings |
| `LLM_MODEL_GEMINI` | `gemini-2.5-flash` | Cloud LLM |
| `MAX_TOOL_CALLS` | `8` | Max retrieval calls per agent run |
| `REDIS_LLM_TTL` | `3600` | LLM response cache TTL (seconds) |

---

## Acknowledgments

- Architecture inspired by [Agentic RAG for Dummies](https://github.com/GiovanniPasq/agentic-rag-for-dummies)
- Medical knowledge base from [MedRAG](https://github.com/Teddy-XiongGZ/MedRAG)
- Evaluation benchmark from [MedQA](https://github.com/jind11/MedQA)

---

## License

MIT
