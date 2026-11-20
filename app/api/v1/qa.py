
import logging
import time
import urllib.error

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.retrieval.answer_service import answer_question


router = APIRouter(
    prefix="/qa",
    tags=["Question Answering"]
)

logger = logging.getLogger(__name__)


class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Question about the uploaded documents"
    )

    document_id: str | None = Field(
        default=None,
        description="Optional document ID to restrict the search"
    )


@router.post("/ask")
def ask_question(request: QuestionRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    start_time = time.perf_counter()

    try:
        # Pass the optional document ID to the answer pipeline.
        result = answer_question(
            question=question,
            document_id=request.document_id
        )

        elapsed = time.perf_counter() - start_time

        return {
            "question": question,
            **result,
            "latency_seconds": round(elapsed, 3)
        }

    except urllib.error.URLError:
        logger.exception("Local LLM service unavailable.")
        raise HTTPException(
            status_code=503,
            detail="Local LLM service is unavailable."
        )

    except TimeoutError:
        logger.exception("Local LLM request timed out.")
        raise HTTPException(
            status_code=504,
            detail="Local LLM request timed out."
        )

    except Exception:
        logger.exception("Question answering failed.")
        raise HTTPException(
            status_code=500,
            detail="Question answering failed."
        )
