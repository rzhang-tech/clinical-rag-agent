import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import chat, documents, health

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initialising RAG system...")
    from core.rag_system import get_rag_system
    get_rag_system()
    logger.info("RAG system ready")
    yield
    logger.info("Shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Clinical RAG Agent API",
        description="Medical Q&A backend backed by 18 clinical textbooks (350K+ indexed chunks)",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api", tags=["Health"])
    app.include_router(chat.router, prefix="/api", tags=["Chat"])
    app.include_router(documents.router, prefix="/api", tags=["Documents"])

    return app
