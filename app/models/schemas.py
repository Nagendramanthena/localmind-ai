"""
Pydantic request / response schemas for the FastAPI endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Body for ``POST /ask``."""

    query: str = Field(..., min_length=1, max_length=2000, description="Natural language query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve")


class IngestRequest(BaseModel):
    """Body for ``POST /ingest``."""

    texts: list[str] = Field(..., min_length=1, description="Document texts to ingest")
    metadata: list[dict] | None = Field(
        default=None,
        description="Optional per-document metadata dicts",
    )


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class QueryResponse(BaseModel):
    """Response for ``POST /ask``."""

    answer: str
    intent: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[str] = Field(default_factory=list)
    retries_used: int = Field(ge=0)
    critique: CritiqueDetail | None = None


class CritiqueDetail(BaseModel):
    """Breakdown of the self-critique scores."""

    relevance: float
    groundedness: float
    completeness: float
    feedback: str


# Rebuild QueryResponse to pick up the forward reference
QueryResponse.model_rebuild()


class IngestResponse(BaseModel):
    """Response for ``POST /ingest``."""

    ingested: int
    total_vectors: int


class HealthResponse(BaseModel):
    """Response for ``GET /health``."""

    status: str  # "healthy" | "degraded" | "unhealthy"
    ollama: bool
    onnx: bool
    index_size: int


class StatsResponse(BaseModel):
    """Response for ``GET /stats``."""

    index_size: int
    embedding_dimension: int
    ollama_model: str
    confidence_threshold: float
    max_retries: int
