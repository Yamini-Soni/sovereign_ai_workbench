"""Shared Pydantic schemas for Sovereign On-Premise Agentic AI Workbench."""

from typing import Literal
from pydantic import BaseModel, Field


class ModelSelection(BaseModel):
    """Request schema for switching local Ollama text and vision models."""

    text_model: str = Field(min_length=1, max_length=200)
    vision_model: str = Field(min_length=1, max_length=200)


class QuestionRequest(BaseModel):
    """Request schema for grounded RAG question answering."""

    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Question about the uploaded documents",
    )
    document_id: str | None = Field(
        default=None,
        description="Optional document ID to restrict the search",
    )
    retrieval_mode: Literal["semantic", "hybrid"] = Field(
        default="semantic",
        description="Retrieval strategy: semantic or hybrid (semantic + BM25)",
    )


class AgentRequest(BaseModel):
    """Request schema for bounded text-only agent execution."""

    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Task or question for the agent",
    )


class SecurityPostureResponse(BaseModel):
    """Summary of verified local security controls and recent audit counts."""

    local_only_inference: bool
    offline_mode: bool
    ollama_endpoint_local: bool
    qdrant_endpoint_local: bool
    prompt_injection_guard: bool
    log_redaction_enabled: bool
    tool_allowlist_count: int
    audit_events_total: int
    rejected_events_total: int
