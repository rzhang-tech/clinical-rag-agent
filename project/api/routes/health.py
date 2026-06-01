import logging
from fastapi import APIRouter
from api.schemas import HealthResponse
import config

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    services: dict = {}

    # Qdrant
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=config.QDRANT_URL, timeout=2)
        client.get_collections()
        services["qdrant"] = "ok"
    except Exception as exc:
        services["qdrant"] = f"error: {exc}"

    # PostgreSQL
    try:
        from db.postgres_manager import PostgresManager
        pg = PostgresManager()
        pg.connect()
        pg.close()
        services["postgres"] = "ok"
    except Exception as exc:
        services["postgres"] = f"error: {exc}"

    # Redis
    try:
        from db.cache_manager import CacheManager
        cache = CacheManager()
        cache.connect()
        services["redis"] = "ok" if cache._client else "unavailable"
    except Exception as exc:
        services["redis"] = f"error: {exc}"

    overall = "ok" if all(v == "ok" for v in services.values()) else "degraded"
    return HealthResponse(status=overall, services=services)
