import os
from dotenv import load_dotenv

load_dotenv()

# --- Directory Configuration ---
_BASE_DIR = os.path.dirname(__file__)

MARKDOWN_DIR = os.path.join(_BASE_DIR, "markdown_docs")
PARENT_STORE_PATH = os.path.join(_BASE_DIR, "parent_store")
QDRANT_DB_PATH = os.path.join(_BASE_DIR, "qdrant_db")
QDRANT_URL = os.environ.get("QDRANT_URL", "")  # empty = use local path mode

# --- Qdrant Configuration ---
CHILD_COLLECTION = "document_child_chunks"
SPARSE_VECTOR_NAME = "sparse"

# --- LLM Provider Configuration ---
# "gemini" -> Google AI Studio (API key); "vertex" -> Vertex AI (GCP, ADC auth, billed to project credit)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# --- Vertex AI (GCP) Configuration ---
GCP_PROJECT = os.environ.get("GCP_PROJECT", "")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")

# --- Together AI (open-weight models) ---
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")
LLM_MODEL_TOGETHER = os.environ.get("LLM_MODEL_TOGETHER", "meta-llama/Llama-3.1-8B-Instruct-Turbo")

# --- Model Configuration ---
DENSE_MODEL = "sentence-transformers/all-mpnet-base-v2"
SPARSE_MODEL = "Qdrant/bm25"
LLM_MODEL_GEMINI = "gemini-2.5-flash"
LLM_MODEL_OLLAMA = "qwen3:4b-instruct-2507-q4_K_M"
LLM_TEMPERATURE = 0

# --- Agent Configuration ---
MAX_TOOL_CALLS = 8
MAX_ITERATIONS = 10
GRAPH_RECURSION_LIMIT = 50
BASE_TOKEN_THRESHOLD = 2000
TOKEN_GROWTH_FACTOR = 0.9

# --- Text Splitter Configuration ---
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 100
MIN_PARENT_SIZE = 2000
MAX_PARENT_SIZE = 4000
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3")
]

# --- PostgreSQL Configuration ---
POSTGRES_URL = os.environ.get("POSTGRES_URL", "postgresql://postgres:password@localhost:5432/clinical_rag")

# --- Redis Configuration ---
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
REDIS_EMBEDDING_TTL = None   # embeddings are deterministic — cache forever
REDIS_LLM_TTL = 3600         # LLM responses: 1-hour TTL

# --- Langfuse Observability ---
LANGFUSE_ENABLED = os.environ.get("LANGFUSE_ENABLED", "false").lower() == "true"
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BASE_URL = os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000")
