
import json
import logging
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.core.config import BASE_DIR
from app.ingestion.file_validator import (
    validate_file,
    MAX_FILE_SIZE,
)

import hashlib



from app.storage.metadata_store import (
    add_document,
    get_document_by_hash,
    get_document_by_id,
    delete_document_metadata,
)

from app.ingestion.document_extractor import extract_text
from app.ingestion.chunker import chunk_pages


from app.retrieval.embedding_service import generate_embeddings



from app.retrieval.vector_store import (
    initialize_vector_store,
    store_embeddings,
    delete_document_vectors,
)


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)

# Storage directories
UPLOAD_DIR = BASE_DIR / "app" / "storage" / "uploads"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document, validate it, extract text,
    divide the text into chunks, and save metadata.
    """

    # Use only the filename, not any directory supplied by the client.
    original_filename = Path(file.filename or "upload").name
    extension = Path(original_filename).suffix.lower()

    document_id = str(uuid4())
    stored_filename = f"{document_id}{extension}"

    temporary_path = UPLOAD_DIR / f"{stored_filename}.part"
    final_path = UPLOAD_DIR / stored_filename

    file_size = 0
    document_saved = False

    try:
        # 1. Save upload in chunks and enforce the size limit.
        with temporary_path.open("wb") as destination:
            while True:
                data = await file.read(1024 * 1024)

                if not data:
                    break

                file_size += len(data)

                if file_size > MAX_FILE_SIZE:
                    raise ValueError(
                        "File exceeds the 20 MB limit."
                    )

                destination.write(data)

        # 2. Validate filename extension and file size.
        validate_file(original_filename, file_size)

        # 3. Move completed upload to its final location.
        temporary_path.replace(final_path)


        # 4. Calculate the SHA-256 fingerprint of the uploaded file.
        sha256_hash = hashlib.sha256()

        with final_path.open("rb") as uploaded_file:
            for block in iter(
                lambda: uploaded_file.read(1024 * 1024),
                b"",
            ):
                sha256_hash.update(block)

        content_hash = sha256_hash.hexdigest()

        # 5. Check whether identical content already exists.
        existing_document = get_document_by_hash(content_hash)

        if existing_document:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "This document has already been uploaded.",
                    "existing_document_id": existing_document["document_id"],
                    "existing_filename": existing_document["filename"],
                },
            )

        # 4. Initialize extraction information.
        extraction_status = "awaiting_ocr"
        page_count = None
        extracted_text_path = None
        chunks_path = None
        chunk_count = 0

        # 5. Extract text from supported document formats.
        if extension in {".pdf", ".txt"}:
            try:
                extraction_result = extract_text(str(final_path))

                extracted_text = extraction_result["text"]
                page_count = extraction_result["page_count"]

                if extracted_text.strip():

                    # Save extracted text.
                    text_path = (
                        PROCESSED_DIR / f"{document_id}.txt"
                    )

                    text_path.write_text(
                        extracted_text,
                        encoding="utf-8",
                    )

                    extracted_text_path = str(
                        text_path.relative_to(BASE_DIR)
                    )

                    # 6. Divide extracted pages into chunks.
                    chunks = chunk_pages(
                        extraction_result["pages"],
                        chunk_size=250,
                        overlap=40,
                    )



                    # 6. Generate embeddings for every document chunk.
                    chunk_texts = [
                        chunk["text"]
                        for chunk in chunks
                    ]

                    embeddings = generate_embeddings(chunk_texts)

                    # Ensure every chunk has a corresponding embedding.
                    if len(embeddings) != len(chunks):
                        raise RuntimeError(
                            "Chunk and embedding counts do not match."
                        )

                    # 7. Initialize the vector database and store the vectors.
                    initialize_vector_store()

                    store_embeddings(
                        document_id=document_id,
                        chunks=chunks,
                        embeddings=embeddings,
                    )

                    # Save chunks as JSON.
                    chunk_file = (
                        PROCESSED_DIR
                        / f"{document_id}_chunks.json"
                    )

                    chunk_file.write_text(
                        json.dumps(
                            chunks,
                            indent=4,
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )

                    chunks_path = str(
                        chunk_file.relative_to(BASE_DIR)
                    )

                    chunk_count = len(chunks)
                    extraction_status = "completed"

                else:
                    # Likely a scanned PDF or an empty document.
                    extraction_status = "awaiting_ocr"

            except Exception:
                logger.exception(
                    "Text extraction failed for document %s",
                    document_id,
                )
                extraction_status = "failed"

        # 7. Create document metadata.
        metadata = {
            "content_hash": content_hash,
            "document_id": document_id,
            "filename": original_filename,
            "stored_filename": stored_filename,
            "extension": extension,
            "file_size_bytes": file_size,
            "page_count": page_count,
            "extraction_status": extraction_status,
            "extracted_text_path": extracted_text_path,
            "chunk_count": chunk_count,
            "chunks_path": chunks_path,
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        # 8. Save metadata.
        add_document(metadata)
        document_saved = True

        # 9. Return upload result.
        return {
            **metadata,
            "status": "uploaded",
            "message": "Document uploaded successfully.",
        }

    except HTTPException:
        # Preserve intentional HTTP errors, such as duplicate upload (409).
        raise

    except ValueError as error:
        # Invalid extension, empty file, or size limit exceeded.
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception:
        logger.exception(
            "Document upload failed: %s",
            original_filename,
        )

        raise HTTPException(
            status_code=500,
            detail="Document upload failed.",
        )

    finally:
        # Remove unfinished temporary upload.
        if temporary_path.exists():
            temporary_path.unlink()

        # Remove the final upload if metadata was never saved.
        if not document_saved and final_path.exists():
            final_path.unlink()

        await file.close()

    


@router.get("/")
def list_documents():
    """
    Return metadata for all uploaded documents.
    """

    from app.storage.metadata_store import get_all_documents

    try:
        documents = get_all_documents()

        return {
            "total_documents": len(documents),
            "documents": documents,
        }

    except Exception:
        logger.exception("Could not retrieve document metadata.")

        raise HTTPException(
            status_code=500,
            detail="Could not retrieve documents.",
        )



@router.delete("/{document_id}")
def delete_document(document_id: str):
    """Delete a document and all its associated local data."""

    document = get_document_by_id(document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    try:
        # Delete the document's vectors.
        deleted_vectors = delete_document_vectors(document_id)

        # Delete original uploaded file.
        stored_filename = document.get("stored_filename")

        if stored_filename:
            upload_path = UPLOAD_DIR / stored_filename

            if upload_path.exists():
                upload_path.unlink()

        # Delete extracted text and chunks files.
        for key in ("extracted_text_path", "chunks_path"):
            relative_path = document.get(key)

            if relative_path:
                file_path = (BASE_DIR / relative_path).resolve()

                if file_path.is_relative_to(PROCESSED_DIR.resolve()):
                    if file_path.is_file():
                        file_path.unlink()

        # Delete metadata last.
        delete_document_metadata(document_id)

        return {
            "message": "Document deleted successfully.",
            "document_id": document_id,
            "deleted_vectors": deleted_vectors,
        }

    except Exception:
        logger.exception(
            "Document deletion failed: %s",
            document_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Document deletion failed.",
        )

