<h1 align="center">Clinical RAG Agent</h1>

<p align="center">
  <strong>An agentic RAG system for medical literature question-answering, powered by LangGraph</strong>
</p>

<p align="center">
  <a href="#overview">Overview</a> &bull;
  <a href="#features">Features</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#evaluation">Evaluation</a> &bull;
  <a href="#installation">Installation</a> &bull;
  <a href="#usage">Usage</a> &bull;
  <a href="#configuration">Configuration</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/LangGraph-1.0%2B-orange?logo=langchain&logoColor=white" alt="LangGraph"/>
  <img src="https://img.shields.io/badge/Qdrant-vector%20db-DC244C" alt="Qdrant"/>
  <img src="https://img.shields.io/badge/Gemini%202.5-Flash-4285F4?logo=google&logoColor=white" alt="Gemini"/>
  <img src="https://img.shields.io/badge/domain-clinical%20medicine-red" alt="Clinical Medicine"/>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License"/>
</p>

---

## Overview

**Clinical RAG Agent** is an agentic retrieval-augmented generation system designed for **medical literature question answering**. It ingests medical textbooks and clinical documents, then answers queries grounded strictly in the source material with full source attribution.

Built on [LangGraph](https://github.com/langchain-ai/langgraph), the system uses an autonomous agent loop with tool-calling to search, retrieve, rerank, and synthesize answers from a knowledge base of 340K+ medical text chunks from 18 textbooks.

### Key Highlights

- **80% accuracy on 100 USMLE-style questions** from MedQA benchmark
- **Hybrid retrieval** (dense + sparse) with cross-encoder reranking
- **340K+ chunks** from 18 authoritative medical textbooks via [MedRAG](https://github.com/Teddy-XiongGZ/MedRAG)
- **Strict grounding** &mdash; refuses to answer when evidence is insufficient
- **Automated evaluation pipeline** with iterative optimization (65% &rarr; 80% overall accuracy across 3 iterations)

> **Disclaimer**: This tool is for **research and educational purposes only**. It is NOT a substitute for professional medical judgment. Always consult qualified healthcare professionals for clinical decisions.

## Features

| Feature | Description |
|---|---|
| **Hierarchical Indexing** | Parent-child chunking optimized for structured medical documents (sections, subsections) |
| **Hybrid Retrieval** | Dense (semantic) + Sparse (BM25) search via Qdrant for high-precision medical term matching |
| **Cross-Encoder Reranking** | Over-fetches 3x candidates then reranks with `ms-marco-MiniLM-L-6-v2` for better precision |
| **Medical-Aware Prompts** | System prompts tuned for clinical terminology, evidence grading, and safety-conscious responses |
| **Conversation Memory** | Multi-turn dialogue with context preservation for iterative clinical queries |
| **Query Clarification** | Rewrites ambiguous queries or asks for clarification (e.g., drug name disambiguation) |
| **Multi-Query Decomposition** | Splits complex clinical questions into parallel sub-queries, then aggregates |
| **Self-Correction** | Re-queries with broader/alternative terms if initial retrieval is insufficient |
| **Context Compression** | Keeps working memory lean across long retrieval loops |
| **Source Attribution** | Every answer cites the source document (textbook, guideline, paper) |
| **Automated Evaluation** | MedQA-based benchmark pipeline with accuracy, source grounding, and boundary awareness metrics |
| **Observability** | Optional Langfuse tracing for debugging and monitoring |

## Architecture

```
User Query
    |
    v
+-------------------+
| Summarize History |  <-- Compress prior conversation
+---------+---------+
          |
          v
+-------------------+
|  Rewrite Query    |  <-- Normalize medical terminology, split multi-part questions
+---------+---------+
          |
          v
+-------------------+
|  Route            |  <-- Clear? --> Fan-out agents  |  Unclear? --> Ask user
+---------+---------+
          |
          v
+---------------------------------------+
|  Agent Subgraph (per sub-question)    |
|                                       |
|  Orchestrator --> Tools --> Rerank    |
|       ^                    |          |
|       |   Compress Context |          |
|       +--------------------+          |
|                                       |
|  Tools:                               |
|    - search_child_chunks (hybrid)     |
|    - retrieve_parent_chunks           |
|    - cross-encoder reranking          |
+------------------+--------------------+
                   |
                   v
+-------------------+
| Aggregate         |  <-- Combine sub-answers into final response
+---------+---------+
          |
          v
   Final Answer + Sources
```

### Retrieval Pipeline

```
Query --> Hybrid Search (dense + BM25) --> Over-fetch 3x candidates
      --> Cross-Encoder Rerank (ms-marco-MiniLM-L-6-v2) --> Top-K results
      --> Retrieve Parent Chunks --> LLM Synthesis --> Answer
```

## Evaluation

### Benchmark: MedQA (USMLE-style)

#### Iterative Optimization (20-question subset)

| Metric | v1 (Baseline) | v2 | v3 |
|--------|:---:|:---:|:---:|
| **In-KB Accuracy** | 67% | 80% | **87%** |
| **Overall Accuracy** | 65% | 75% | **80%** |
| **Avg Time/Question** | 32.1s | 35.3s | **26.4s** |

#### Scaled Evaluation (100 questions, 18 textbooks)

| Metric | Result |
|--------|:---:|
| **Overall Accuracy** | **80%** |
| **In-KB Accuracy** | 78% (62/80) |
| **Source Attribution** | 86% (69/80) |
| **Avg Time/Question** | 26.0s |

### Optimization Journey

| Iteration | Changes | Impact |
|-----------|---------|--------|
| **v1 &rarr; v2** | Added cross-encoder reranker, lowered score threshold (0.7 &rarr; 0.4), strengthened grounding prompts | +13% in-KB accuracy |
| **v2 &rarr; v3** | Balanced grounding rules (allow reasoning over retrieved content, not just verbatim matching), improved answer format extraction | +7% in-KB accuracy, reduced UNKNOWN responses |

### Evaluation Dimensions

| Dimension | What it measures | Why it matters |
|-----------|------------------|----------------|
| **In-KB Accuracy** | Correct answers when relevant documents exist | Core retrieval + reasoning quality |
| **Source Grounding** | Answers cite source documents | Evidence traceability |
| **Boundary Awareness** | Refuses to answer when no relevant documents exist | Safety &mdash; prevents hallucinated medical advice |

Run the evaluation yourself:

```bash
cd project
python scripts/evaluate.py                                   # Full evaluation (100 questions)
python scripts/evaluate.py --dry-run                         # Preview questions only
python scripts/evaluate.py --num-in-kb 80 --num-out-kb 20   # Custom size
```

## Knowledge Base

The knowledge base is built from [MedRAG/textbooks](https://huggingface.co/datasets/MedRAG/textbooks), a curated collection of medical textbook content.

### Included Textbooks (18)

| Textbook | Domain | Chunks |
|----------|--------|--------|
| Harrison's (Internal Medicine) | Internal Medicine | 32,628 |
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
| **Total** | **18 textbooks** | **125K raw &rarr; 340K+ indexed chunks** |

### Import More Textbooks

```bash
# List all available textbooks
python scripts/import_medrag.py --list-titles

# Preview import (no writes)
python scripts/import_medrag.py --titles Immunology_Janeway --dry-run

# Import specific textbooks
python scripts/import_medrag.py --titles Immunology_Janeway Neurology_Adams
```

18 textbooks are available covering anatomy, biochemistry, cell biology, gynecology, histology, immunology, internal medicine, neurology, obstetrics, pathology, pediatrics, pharmacology, physiology, psychiatry, and surgery.

## Installation

### Prerequisites

- Python 3.10+
- [Docker](https://www.docker.com/) (recommended for Qdrant)
- Google Gemini API key (free tier available) **or** [Ollama](https://ollama.com/) for local LLM

### Setup

```bash
# Clone the repository
git clone https://github.com/rzhang-tech/clinical-rag-agent.git
cd clinical-rag-agent

# Create conda environment
conda create -n clinical-rag python=3.11 -y
conda activate clinical-rag

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp project/.env.example project/.env
# Edit project/.env with your API key
```

### Start Qdrant (Docker)

```bash
docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

> **Note**: Qdrant also supports a local file-based mode (no Docker needed), but Docker mode is recommended for datasets over 20K points for better performance.

### Import Knowledge Base

```bash
cd project
python scripts/import_medrag.py --titles Anatomy_Gray InternalMed_Harrison Pathology_Robbins Pharmacology_Katzung First_Aid_Step2 Neurology_Adams Surgery_Schwartz Pediatrics_Nelson Immunology_Janeway Histology_Ross Gynecology_Novak Obstentrics_Williams Psichiatry_DSM-5 Biochemistry_Lippinco Cell_Biology_Alberts Physiology_Levy First_Aid_Step1 Pathoma_Husain
```

### Run

```bash
python app.py
```

Open `http://localhost:7860` in your browser.

## Usage

1. **Ask Questions** &mdash; Go to the "Chat" tab, ask clinical questions like:
   - *"What are the major structures of the heart?"*
   - *"What is the first-line treatment for type 2 diabetes?"*
   - *"Describe the mechanism of action of ACE inhibitors"*
   - *"What are the contraindications for metformin?"*
2. **Upload Additional Documents** &mdash; Go to the "Documents" tab to upload clinical PDFs
3. **Review Sources** &mdash; Every answer includes source document citations

## Configuration

### LLM Provider

Edit `project/.env`:

```bash
# Google Gemini (recommended)
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your-api-key-here

# Ollama (local, requires GPU)
LLM_PROVIDER=ollama
```

### Key Parameters

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MODEL_GEMINI` | `gemini-2.5-flash` | Gemini model |
| `LLM_MODEL_OLLAMA` | `qwen3:4b` | Local Ollama model |
| `DENSE_MODEL` | `all-mpnet-base-v2` | Dense embedding model |
| `SPARSE_MODEL` | `Qdrant/bm25` | Sparse embedding (BM25) |
| `CHILD_CHUNK_SIZE` | `500` | Child chunk token size |
| `MIN_PARENT_SIZE` / `MAX_PARENT_SIZE` | `2000` / `4000` | Parent chunk size range |
| `MAX_TOOL_CALLS` | `8` | Max retrieval tool calls per agent run |
| `MAX_ITERATIONS` | `10` | Max agent loop iterations |

### Observability (Optional)

Enable [Langfuse](https://langfuse.com/) tracing:

```bash
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

## Project Structure

```
clinical-rag-agent/
├── README.md
├── requirements.txt
├── project/
│   ├── app.py                      # Entry point (Gradio UI)
│   ├── config.py                   # Central configuration
│   ├── document_chunker.py         # Parent-child chunking
│   ├── utils.py                    # PDF conversion utilities
│   ├── core/
│   │   ├── rag_system.py           # System bootstrap + LLM provider selection
│   │   ├── document_manager.py     # Document ingestion
│   │   ├── chat_interface.py       # Agent graph wrapper
│   │   └── observability.py        # Langfuse tracing
│   ├── db/
│   │   ├── vector_db_manager.py    # Qdrant hybrid search (dense + sparse)
│   │   └── parent_store_manager.py # Parent chunk storage
│   ├── rag_agent/
│   │   ├── graph.py                # LangGraph workflow definition
│   │   ├── graph_state.py          # State schemas
│   │   ├── nodes.py                # Node implementations (orchestrator, compress, etc.)
│   │   ├── edges.py                # Routing logic
│   │   ├── prompts.py              # Medical-aware system prompts
│   │   ├── tools.py                # Retrieval tools + cross-encoder reranking
│   │   └── schemas.py              # Pydantic schemas
│   ├── scripts/
│   │   ├── import_medrag.py        # MedRAG textbook importer
│   │   └── evaluate.py             # MedQA evaluation pipeline
│   ├── eval_results/               # Evaluation JSON outputs
│   └── ui/
│       ├── gradio_app.py           # Gradio interface
│       └── css.py                  # UI styling
```

## License

MIT

## Acknowledgments

- Architecture based on [Agentic RAG for Dummies](https://github.com/GiovanniPasq/agentic-rag-for-dummies) by Giovanni Pasquariello
- Medical knowledge base from [MedRAG](https://github.com/Teddy-XiongGZ/MedRAG) textbooks dataset
- Evaluation benchmark from [MedQA](https://github.com/jind11/MedQA) (USMLE-style questions)
