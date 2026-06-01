from pydantic import BaseModel
from typing import List, Optional


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


class DocumentInfo(BaseModel):
    name: str


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int


class UploadResponse(BaseModel):
    added: int
    skipped: int
    message: str


class HealthResponse(BaseModel):
    status: str
    services: dict
