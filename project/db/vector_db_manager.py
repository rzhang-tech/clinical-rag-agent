import logging
from typing import Optional

import config
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from db.cache_manager import CacheManager, CachedEmbeddings

logger = logging.getLogger(__name__)


class VectorDbManager:

    def __init__(self, cache: Optional[CacheManager] = None):
        if config.QDRANT_URL:
            self.__client = QdrantClient(url=config.QDRANT_URL)
        else:
            self.__client = QdrantClient(path=config.QDRANT_DB_PATH)

        base_dense = HuggingFaceEmbeddings(model_name=config.DENSE_MODEL)
        if cache is not None:
            self.__dense_embeddings = CachedEmbeddings(base_dense, cache)
        else:
            self.__dense_embeddings = base_dense

        self.__sparse_embeddings = FastEmbedSparse(model_name=config.SPARSE_MODEL)

    def create_collection(self, collection_name: str) -> None:
        if not self.__client.collection_exists(collection_name):
            logger.info("Creating Qdrant collection: %s", collection_name)
            self.__client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=len(self.__dense_embeddings.embed_query("test")),
                    distance=qmodels.Distance.COSINE,
                ),
                sparse_vectors_config={
                    config.SPARSE_VECTOR_NAME: qmodels.SparseVectorParams()
                },
            )
            logger.info("Collection created: %s", collection_name)
        else:
            logger.info("Collection already exists: %s", collection_name)

    def delete_collection(self, collection_name: str) -> None:
        try:
            if self.__client.collection_exists(collection_name):
                self.__client.delete_collection(collection_name)
        except Exception as e:
            logger.warning("Could not delete collection %s: %s", collection_name, e)

    def get_collection(self, collection_name: str) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=self.__client,
            collection_name=collection_name,
            embedding=self.__dense_embeddings,
            sparse_embedding=self.__sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_vector_name=config.SPARSE_VECTOR_NAME,
        )
