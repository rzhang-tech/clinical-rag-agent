import asyncio
import logging
import os
import tempfile
from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File
from api.schemas import DocumentListResponse, DocumentInfo, UploadResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_rag_system():
    from core.rag_system import get_rag_system
    return get_rag_system()


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    rag = _get_rag_system()
    from core.document_manager import DocumentManager
    doc_manager = DocumentManager(rag)
    files = await asyncio.to_thread(doc_manager.get_markdown_files)
    docs = [DocumentInfo(name=f) for f in files]
    return DocumentListResponse(documents=docs, total=len(docs))


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    rag = _get_rag_system()
    from core.document_manager import DocumentManager
    doc_manager = DocumentManager(rag)

    tmp_paths: List[str] = []
    try:
        for upload in files:
            suffix = os.path.splitext(upload.filename)[1].lower()
            if suffix not in (".pdf", ".md"):
                continue
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await upload.read())
                tmp_paths.append(tmp.name)

        if not tmp_paths:
            raise HTTPException(status_code=400, detail="No valid PDF or Markdown files provided")

        added, skipped = await asyncio.to_thread(doc_manager.add_documents, tmp_paths)

    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    return UploadResponse(
        added=added,
        skipped=skipped,
        message=f"Added {added} document(s), skipped {skipped}",
    )


@router.delete("/documents")
async def clear_documents():
    rag = _get_rag_system()
    from core.document_manager import DocumentManager
    doc_manager = DocumentManager(rag)
    await asyncio.to_thread(doc_manager.clear_all)
    return {"status": "all documents cleared"}
