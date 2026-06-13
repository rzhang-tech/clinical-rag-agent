<h1 align="center">Clinical RAG Agent</h1>

<p align="center">
  <strong>Agentic RAG for medical-literature Q&A — and a rigorous study of when retrieval helps vs. hurts</strong>
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
  <strong>📊 Evaluation-first project — see <a href="#evaluation">Evaluation</a> for the key findings</strong>
</p>

---

## Overview

**Clinical RAG Agent** is an agentic retrieval-augmented generation system for medical-literature question answering — and, more importantly, a **rigorous evaluation study of when retrieval helps, hurts, or should be bypassed.** It ingests clinical textbooks, indexes them into a hybrid vector store, and answers queries with source attribution; the project's core contribution is a careful RAG-vs-no-RAG ablation that overturns the assumption that retrieval always improves accuracy.

Built on [LangGraph](https://github.com/langchain-ai/langgraph) with an async [FastAPI](https://fastapi.tiangolo.com/) backend, PostgreSQL persistence, Redis caching, and Docker.

### Highlights

- **Rigorous RAG-vs-no-RAG ablation** on 1,273 MedQA questions with 95% Wilson confidence intervals — finding that retrieval *reduced* closed-book accuracy (see [Evaluation](#evaluation))
- **Found and fixed two silent defects** via the eval: an incomplete vector index (only 4 of 18 textbooks) and a query-routing bug suppressing 22% of answers — recovering accuracy from 68.9% → 87.0%
- **11-node LangGraph state machine** — conditional routing, context compression, fan-out, retry/fallback
- **Hybrid retrieval** (dense + BM25) with cross-encoder reranking over **287K** indexed chunks
- **Multi-layer evaluation**: ablation, faithfulness/groundedness, and an agent-behavior regression suite
- **FastAPI + SSE streaming**, PostgreSQL, Redis cache, Docker Compose

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

## Retrieval Optimization

Before full-scale evaluation, the retrieval pipeline was iterated on a development subset. The single largest quality gain came from **cross-encoder reranking** — jointly scoring query–document pairs surfaced relevant passages that bi-encoder (dense) retrieval alone had ranked too low. Other changes: lowering the retrieval score threshold (0.7 → 0.4), strengthening grounding instructions to cut off-corpus hallucination, allowing the agent to *reason over* retrieved evidence rather than only quote it (which fixed a class of "refused despite having relevant evidence" failures), and tuning the answer-extraction path and context-compression threshold to cut average latency **~26% (35.3s → 26.0s)**.

> **The arc:** the "refused despite having evidence" failure addressed here is the same class of behavior later quantified and re-fixed at full scale (the 22% no-answer bug in [Evaluation](#evaluation)), and the **confident-answer vs. explicit-refusal calibration** flagged as an open problem is exactly what the faithfulness and agent-behavior suites below measure. Per-version dev-subset accuracies are intentionally *not* used as headline numbers (small n, single subset); the statistically robust full-set results follow.

---

## Evaluation

The central finding: **on closed-book MCQ, retrieval over public textbook knowledge did not help — it hurt — for both a strong and a weak model.** All numbers below are measured (full MedQA test set unless noted), with 95% Wilson confidence intervals.

### 1. RAG vs. no-RAG ablation (MedQA, n=1,273, Gemini 2.5 Flash)

| Condition | Accuracy | 95% CI |
|-----------|:--------:|:------:|
| no-RAG (parametric only) | **92.4%** | 90.8–93.7 |
| RAG, initial (with bugs) | 68.9% | 66.3–71.4 |
| **RAG, after bug fixes** | **87.0%** | 85.0–88.7 |

Most of the apparent 23-point "RAG penalty" was a **silent pipeline bug** (a query-rewrite node rejecting exam-style questions + answers not committing to an option → 22% of questions returned no answer). Root-causing and fixing it recovered 68.9% → 87.0%. A **real ~5.4-point residual penalty remains and is statistically significant** — retrieval adds noise to knowledge the model already has (cf. [Mallen et al., 2023](https://aclanthology.org/2023.acl-long.546/)).

A separate bug was found the same way: the vector index contained only **4 of 18 textbooks** (a silently interrupted import); re-indexing all 18 produced the 287K-chunk index used above.

### 2. Does RAG help a weaker, deployable model? (Qwen2.5-7B, n=200)

| Strategy | Accuracy |
|----------|:--------:|
| no-RAG | **66.5%** |
| simple-RAG (1 retrieval + 1 answer) | 58.0% |
| agentic-RAG (full pipeline) | 56.5% |

RAG hurt the weak model **more** than the strong one (−8.5 to −10 vs −5.4), overturning the "RAG helps weaker models" expectation. A **simple-RAG ablation isolates retrieval itself — not agent complexity — as the cause** (agentic adds only ~1.5 pts of extra penalty); weaker models integrate noisy context worse (cf. RGB, [Chen et al., 2024](https://ojs.aaai.org/index.php/AAAI/article/view/29728)).

### 3. Faithfulness & agent-behavior

- **Faithfulness** (RAGAS-style, n=120): context relevance **0.97**, but only **22%** of answers fully grounded in retrieved evidence → the bottleneck is generation faithfulness, not retrieval.
- **Agent-behavior regression** (14 qualitative cases): **12/14** pass — clarification, red-flag escalation, abstention, citation, fallback. Two documented gaps (over-deflection; imprecise partial-evidence abstention).

### Takeaway

Retrieval's value is **task-dependent**: harmful when the model already knows the answer (closed-book MCQ), valuable only for genuinely unseen / private / recent knowledge with auditable provenance. See [`project/eval_results/FINDINGS.md`](project/eval_results/FINDINGS.md) for full details.

```bash
python scripts/evaluate.py --mode both --full        # RAG vs no-RAG ablation
python scripts/eval_faithfulness.py --sample 120      # faithfulness / groundedness
python scripts/eval_behavior.py                       # agent-behavior regression suite
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

18 medical textbooks from [MedRAG](https://github.com/Teddy-XiongGZ/MedRAG): **287,420 child chunks / 38,979 parent chunks** indexed.

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
| **Total** | **18 textbooks** | **287,420 child** |

> The per-textbook counts above are MedRAG's original source-snippet counts; after parent–child re-chunking the indexed total is **287,420 child / 38,979 parent** chunks.

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
