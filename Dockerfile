FROM python:3.11-slim

# System dependencies for PyMuPDF, sentence-transformers, psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding models into image layer so first startup is fast
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')"
RUN python -c "from langchain_qdrant import FastEmbedSparse; FastEmbedSparse(model_name='Qdrant/bm25')"

COPY project/ ./project/

WORKDIR /app/project

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
