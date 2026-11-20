
"""Agent API endpoints for text-only and multimodal requests."""

import logging
import time
import urllib.error

from fastapi import (
    APIRouter,
    HTTPException,
    UploadFile,
    File,
    Form,
)
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.agents.orchestrator import run_agent


router = APIRouter(
    prefix="/agent",
    tags=["Agent"],
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20 MB

ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


# ---------------------------------------------------------
# Request models
# ---------------------------------------------------------

class AgentRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Task or question for the agent",
    )


# ---------------------------------------------------------
# Image validation
# ---------------------------------------------------------

def validate_uploaded_image(
    filename: str | None,
    image_bytes: bytes,
) -> str:
    """
    Validate uploaded image extension, size and file signature.

    Returns:
        Validated image format: jpeg or png.
    """

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Image filename is required.",
        )

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty.",
        )

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 20 MB size limit.",
        )

    filename_lower = filename.lower()

    extension = None

    for allowed_extension in ALLOWED_IMAGE_EXTENSIONS:
        if filename_lower.endswith(allowed_extension):
            extension = allowed_extension
            break

    if extension is None:
        raise HTTPException(
            status_code=415,
            detail="Only JPG, JPEG and PNG images are supported.",
        )

    # JPEG file signature:
    # FF D8 FF
    is_jpeg = image_bytes.startswith(
        b"\xff\xd8\xff"
    )

    # PNG file signature:
    # 89 50 4E 47 0D 0A 1A 0A
    is_png = image_bytes.startswith(
        b"\x89PNG\r\n\x1a\n"
    )

    if extension in {".jpg", ".jpeg"} and not is_jpeg:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file does not contain a valid JPEG signature.",
        )

    if extension == ".png" and not is_png:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file does not contain a valid PNG signature.",
        )

    return "jpeg" if is_jpeg else "png"


# ---------------------------------------------------------
# Error handling helper
# ---------------------------------------------------------

def _handle_agent_exception(error: Exception):
    """Convert agent execution errors into appropriate API responses."""

    if isinstance(error, urllib.error.URLError):
        logger.exception(
            "Local LLM service unavailable."
        )

        raise HTTPException(
            status_code=503,
            detail="Local LLM service is unavailable.",
        )

    if isinstance(error, TimeoutError):
        logger.exception(
            "Agent request timed out."
        )

        raise HTTPException(
            status_code=504,
            detail="Agent request timed out.",
        )

    if isinstance(error, ValueError):
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    logger.exception(
        "Agent execution failed."
    )

    raise HTTPException(
        status_code=500,
        detail="Agent execution failed.",
    )


# ---------------------------------------------------------
# 1. Existing text-only endpoint
# ---------------------------------------------------------

@router.post("/run")
def run_agent_endpoint(
    request: AgentRequest,
):
    """
    Execute the agent using a text-only request.

    Existing endpoint retained for backward compatibility.
    """

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    start_time = time.perf_counter()

    try:
        result = run_agent(
            question=question,
        )

        return {
            **result,
            "api_latency_seconds": round(
                time.perf_counter() - start_time,
                3,
            ),
        }

    except Exception as error:
        _handle_agent_exception(error)


# ---------------------------------------------------------
# 2. New multimodal endpoint
# ---------------------------------------------------------

@router.post("/run-with-image")
async def run_agent_with_image_endpoint(
    question: str = Form(
        ...,
        min_length=1,
        max_length=2000,
        description="Question or task involving the uploaded image",
    ),
    image: UploadFile = File(
        ...,
        description="JPEG or PNG image",
    ),
):
    """
    Execute the multimodal agent using:

    - A text question
    - A validated uploaded image

    The image bytes are passed to the orchestrator as
    trusted request context, not as planner arguments.
    """

    question = question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    start_time = time.perf_counter()

    # Read at most one byte beyond the allowed limit.
    # This avoids reading an arbitrarily large upload into memory.
    try:
        image_bytes = await image.read(
            MAX_IMAGE_SIZE + 1
        )

    finally:
        await image.close()

    # Validate the uploaded file before agent execution.
    image_format = validate_uploaded_image(
        filename=image.filename,
        image_bytes=image_bytes,
    )

    try:
        # run_agent uses synchronous local Ollama requests.
        # Execute it in a worker thread rather than blocking
        # the FastAPI event loop.
        result = await run_in_threadpool(
            run_agent,
            question=question,
            image_bytes=image_bytes,
        )

        return {
            **result,
            "image_attached": True,
            "image_format": image_format,
            "api_latency_seconds": round(
                time.perf_counter() - start_time,
                3,
            ),
        }

    except Exception as error:
        _handle_agent_exception(error)
