import hashlib
import json
import logging
from typing import List, Optional

import redis as redis_lib
from langchain_core.embeddings import Embeddings

import config

logger = logging.getLogger(__name__)


class CacheManager:

    def __init__(self, url: str = config.REDIS_URL):
        self._url = url
        self._client: Optional[redis_lib.Redis] = None

    def connect(self) -> None:
        try:
            self._client = redis_lib.from_url(self._url, decode_responses=True)
            self._client.ping()
            logger.info("Redis connected at %s", self._url)
        except Exception as exc:
            logger.warning("Redis unavailable (%s) — caching disabled", exc)
            self._client = None

    def get(self, key: str) -> Optional[str]:
        if self._client is None:
            return None
        try:
            return self._client.get(key)
        except Exception:
            return None

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        if self._client is None:
            return
        try:
            if ttl:
                self._client.setex(key, ttl, value)
            else:
                self._client.set(key, value)
        except Exception:
            pass

    def delete(self, key: str) -> None:
        if self._client is None:
            return
        try:
            self._client.delete(key)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Typed helpers                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def get_embedding(self, text: str) -> Optional[List[float]]:
        raw = self.get(f"emb:{self._hash(text)}")
        return json.loads(raw) if raw else None

    def set_embedding(self, text: str, vector: List[float]) -> None:
        self.set(f"emb:{self._hash(text)}", json.dumps(vector), ttl=config.REDIS_EMBEDDING_TTL)

    def get_llm_response(self, query: str) -> Optional[str]:
        raw = self.get(f"llm:{self._hash(query)}")
        return raw

    def set_llm_response(self, query: str, response: str) -> None:
        self.set(f"llm:{self._hash(query)}", response, ttl=config.REDIS_LLM_TTL)


# ------------------------------------------------------------------ #
# LangChain-compatible cached embedding wrapper                        #
# ------------------------------------------------------------------ #

class CachedEmbeddings(Embeddings):
    """Wraps any LangChain Embeddings with a Redis cache layer."""

    def __init__(self, base: Embeddings, cache: CacheManager):
        self._base = base
        self._cache = cache

    def embed_query(self, text: str) -> List[float]:
        cached = self._cache.get_embedding(text)
        if cached is not None:
            return cached
        vec = self._base.embed_query(text)
        self._cache.set_embedding(text, vec)
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results: List[Optional[List[float]]] = []
        miss_indices: List[int] = []
        miss_texts: List[str] = []

        for i, text in enumerate(texts):
            cached = self._cache.get_embedding(text)
            if cached is not None:
                results.append(cached)
            else:
                results.append(None)
                miss_indices.append(i)
                miss_texts.append(text)

        if miss_texts:
            fresh = self._base.embed_documents(miss_texts)
            for idx, vec in zip(miss_indices, fresh):
                self._cache.set_embedding(texts[idx], vec)
                results[idx] = vec

        return results
