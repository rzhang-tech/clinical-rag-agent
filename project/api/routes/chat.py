import asyncio
import json
import logging
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from api.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_rag_system():
    from core.rag_system import get_rag_system
    return get_rag_system()


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    rag = _get_rag_system()
    if not rag.agent_graph:
        raise HTTPException(status_code=503, detail="RAG system not initialised")

    # Check Redis LLM cache
    cache = rag.cache
    if cache:
        cached = cache.get_llm_response(req.message)
        if cached:
            session_id = req.session_id or str(uuid.uuid4())
            logger.info("LLM cache hit for query: %.60s", req.message)
            return ChatResponse(response=cached, session_id=session_id)

    try:
        result = await asyncio.to_thread(
            rag.agent_graph.invoke,
            {"messages": [HumanMessage(content=req.message.strip())]},
            rag.get_config(),
        )
        response_text = result["messages"][-1].content
    except Exception as exc:
        logger.error("Chat error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        rag.observability.flush()

    # Persist to LLM cache
    if cache:
        cache.set_llm_response(req.message, response_text)

    session_id = req.session_id or str(uuid.uuid4())
    return ChatResponse(response=response_text, session_id=session_id)


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Server-Sent Events streaming endpoint."""
    rag = _get_rag_system()
    if not rag.agent_graph:
        raise HTTPException(status_code=503, detail="RAG system not initialised")

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            events = rag.agent_graph.astream_events(
                {"messages": [HumanMessage(content=req.message.strip())]},
                rag.get_config(),
                version="v2",
            )
            async for event in events:
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        payload = json.dumps({"delta": chunk.content})
                        yield f"data: {payload}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.error("Streaming error: %s", exc)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        finally:
            rag.observability.flush()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/chat/reset")
async def reset_session():
    rag = _get_rag_system()
    await asyncio.to_thread(rag.reset_thread)
    return {"status": "session reset"}
