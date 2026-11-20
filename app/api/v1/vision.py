
"""API endpoint for local image understanding."""

import logging
import urllib.error

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from starlette.concurrency import run_in_threadpool

from app.ingestion.file_validator import (
    validate_file,
    MAX_FILE_SIZE,
)
from app.vision.vision_service import analyze_image


router = APIRouter(
    prefix="/vision",
    tags=["Vision"],
)

logger = logging.getLogger(__name__)


@router.post("/analyze")
async def analyze_image_endpoint(
    file: UploadFile = File(...),
    task: str = Form("scene"),
    question: str | None = Form(None),
):
    """Analyze an uploaded image using the local vision model."""

    filename = file.filename or "upload"
    image_bytes = bytearray()

    try:
        # 1. Read the upload while enforcing the size limit.
        while True:
            chunk = await file.read(1024 * 1024)

            if not chunk:
                break

            image_bytes.extend(chunk)

            if len(image_bytes) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail="Image exceeds the 20 MB limit.",
                )

        # 2. Validate extension and file size.
        validate_file(filename, len(image_bytes))

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            raise HTTPException(
                status_code=400,
                detail="Only JPG, JPEG and PNG images are supported.",
            )

        # 3. Verify basic image file signatures.
        image_data = bytes(image_bytes)

        is_jpeg = image_data.startswith(b"\xff\xd8\xff")
        is_png = image_data.startswith(
            b"\x89PNG\r\n\x1a\n"
        )

        if not (is_jpeg or is_png):
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is not a valid JPEG or PNG image.",
            )

        # 4. Run local vision inference outside the async event loop.
        result = await run_in_threadpool(
            analyze_image,
            image_data,
            task,
            question,
        )

        return {
            "filename": filename,
            "status": "success",
            **result,
        }

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except urllib.error.URLError:
        logger.exception("Local vision model is unavailable.")
        raise HTTPException(
            status_code=503,
            detail="Local vision model is unavailable.",
        )

    except TimeoutError:
        logger.exception("Vision inference timed out.")
        raise HTTPException(
            status_code=504,
            detail="Vision inference timed out.",
        )

    except RuntimeError:
        logger.exception("Vision inference failed.")
        raise HTTPException(
            status_code=502,
            detail="Vision model inference failed.",
        )

    except Exception:
        logger.exception("Image analysis failed.")
        raise HTTPException(
            status_code=500,
            detail="Image analysis failed.",
        )

    finally:
        await file.close()
