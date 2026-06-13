import uuid
import logging
from typing import Optional

import config
from db.vector_db_manager import VectorDbManager
from db.parent_store_manager import ParentStoreManager
from db.postgres_manager import PostgresManager
from db.cache_manager import CacheManager
from document_chunker import DocumentChuncker
from rag_agent.tools import ToolFactory
from rag_agent.graph import create_agent_graph
from core.observability import Observability

logger = logging.getLogger(__name__)


def _create_llm():
    if config.LLM_PROVIDER == "vertex":
        # Vertex AI: billed to the GCP project (uses ADC, no API key).
        from langchain_google_vertexai import ChatVertexAI
        return ChatVertexAI(
            model=config.LLM_MODEL_GEMINI,
            temperature=config.LLM_TEMPERATURE,
            project=config.GCP_PROJECT,
            location=config.GCP_LOCATION,
        )
    elif config.LLM_PROVIDER == "together":
        # Together AI: open-weight models (Llama, etc.) via OpenAI-compatible API.
        from langchain_together import ChatTogether
        return ChatTogether(
            model=config.LLM_MODEL_TOGETHER,
            temperature=config.LLM_TEMPERATURE,
            together_api_key=config.TOGETHER_API_KEY,
        )
    elif config.LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.LLM_MODEL_GEMINI,
            temperature=config.LLM_TEMPERATURE,
            google_api_key=config.GOOGLE_API_KEY,
        )
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=config.LLM_MODEL_OLLAMA,
            temperature=config.LLM_TEMPERATURE,
        )


class RAGSystem:

    def __init__(self, collection_name: str = config.CHILD_COLLECTION):
        self.collection_name = collection_name
        self.pg = PostgresManager()
        self.cache = CacheManager()
        self.vector_db = VectorDbManager(cache=self.cache)
        self.parent_store = ParentStoreManager(pg=self.pg)
        self.chunker = DocumentChuncker()
        self.observability = Observability()
        self.agent_graph = None
        self.thread_id = str(uuid.uuid4())
        self.recursion_limit = config.GRAPH_RECURSION_LIMIT

    def initialize(self) -> None:
        # Connect backing services
        try:
            self.pg.connect()
        except Exception as exc:
            logger.warning("PostgreSQL unavailable at startup: %s — parent store may fail", exc)

        try:
            self.cache.connect()
        except Exception as exc:
            logger.warning("Redis unavailable at startup: %s — caching disabled", exc)

        self.vector_db.create_collection(self.collection_name)
        collection = self.vector_db.get_collection(self.collection_name)

        llm = _create_llm()
        tools = ToolFactory(collection, cache=self.cache).create_tools()
        self.agent_graph = create_agent_graph(llm, tools)

    def get_config(self) -> dict:
        cfg = {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": self.recursion_limit,
        }
        handler = self.observability.get_handler()
        if handler:
            cfg["callbacks"] = [handler]
        return cfg

    def reset_thread(self) -> None:
        try:
            self.agent_graph.checkpointer.delete_thread(self.thread_id)
        except Exception as exc:
            logger.warning("Could not delete thread %s: %s", self.thread_id, exc)
        self.thread_id = str(uuid.uuid4())


# ------------------------------------------------------------------ #
# Process-level singleton                                              #
# ------------------------------------------------------------------ #

_instance: Optional[RAGSystem] = None


def get_rag_system() -> RAGSystem:
    global _instance
    if _instance is None:
        _instance = RAGSystem()
        _instance.initialize()
    return _instance
