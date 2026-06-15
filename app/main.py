"""
FastAPI application — the single entry point for the local AI agent.

Endpoints:
    POST /ask      — Query the agent (main RAG + self-critique workflow)
    POST /ingest   — Add documents to the knowledge base
    GET  /health   — Health check (Ollama + ONNX + index status)
    GET  /stats    — Index and configuration statistics
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models.schemas import (
    CritiqueDetail,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    StatsResponse,
)
from app.utils.logging import get_logger, setup_logging

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — initialise services on startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown lifecycle for the FastAPI app."""
    setup_logging()
    log.info("starting_up", host=settings.host, port=settings.port)

    # Eagerly initialise heavyweight singletons so the first request
    # is not penalised.  Imports are deferred to avoid circular deps.
    from app.services.embedding_service import get_embedding_service
    from app.services.retrieval_service import get_retriever

    embedder = get_embedding_service()
    retriever = get_retriever()

    log.info(
        "services_ready",
        embedding_dim=embedder.dimension,
        index_vectors=retriever.size,
    )

    yield  # ← application runs here

    log.info("shutting_down")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Local Agentic AI Assistant",
    description=(
        "A lightweight, fully offline agentic AI assistant. "
        "Accepts natural language queries, classifies intent, retrieves "
        "context via ONNX-powered semantic search, and returns grounded "
        "answers with a self-critique loop."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# POST /ask
# ---------------------------------------------------------------------------


@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest) -> QueryResponse:
    """
    Submit a natural language query to the agent.

    The agent classifies intent, retrieves relevant context (if needed),
    generates an answer, and self-critiques.  If confidence is below the
    threshold it re-retrieves up to ``max_retries`` times.
    """
    from app.agent.graph import agent_graph

    initial_state = {
        "query": request.query,
        "refined_query": request.query,
        "contexts": [],
        "context_scores": [],
        "answer": "",
        "confidence": 0.0,
        "critique_feedback": "",
        "retry_count": 0,
        "max_retries": settings.max_retries,
        "final": False,
    }

    try:
        result = agent_graph.invoke(initial_state)
    except Exception as exc:
        log.error("agent_error", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    # Extract source file names from contexts metadata (if available)
    sources: list[str] = []
    from app.services.retrieval_service import get_retriever

    retriever = get_retriever()
    if result.get("contexts"):
        # Try to match contexts back to metadata
        for ctx_text in result["contexts"]:
            for i, chunk in enumerate(retriever._chunks):
                if chunk == ctx_text and i < len(retriever._metadata):
                    src = retriever._metadata[i].get("source", "")
                    if src and src not in sources:
                        sources.append(src)
                    break

    # Build critique detail
    critique = None
    if result.get("critique_relevance") is not None:
        critique = CritiqueDetail(
            relevance=result.get("critique_relevance", 0.0),
            groundedness=result.get("critique_groundedness", 0.0),
            completeness=result.get("critique_completeness", 0.0),
            feedback=result.get("critique_feedback", ""),
        )

    return QueryResponse(
        answer=result.get("answer", ""),
        intent=result.get("intent", "unknown"),
        confidence=result.get("confidence", 0.0),
        sources=sources,
        retries_used=max(0, result.get("retry_count", 0) - 1),
        critique=critique,
    )


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    """Add new documents to the knowledge base at runtime."""
    from app.services.retrieval_service import get_retriever

    retriever = get_retriever()
    total = retriever.add_documents(request.texts, request.metadata)
    retriever.save()

    return IngestResponse(ingested=len(request.texts), total_vectors=total)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Check the health of all subsystems."""
    from app.services.embedding_service import get_embedding_service
    from app.services.retrieval_service import get_retriever

    # Check ONNX
    onnx_ok = False
    try:
        embedder = get_embedding_service()
        _ = embedder.dimension
        onnx_ok = True
    except Exception:
        pass

    # Check Ollama
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            ollama_ok = resp.status_code == 200
    except Exception:
        pass

    # Index
    index_size = 0
    try:
        index_size = get_retriever().size
    except Exception:
        pass

    status = "healthy"
    if not ollama_ok or not onnx_ok:
        status = "degraded" if (ollama_ok or onnx_ok) else "unhealthy"

    return HealthResponse(
        status=status,
        ollama=ollama_ok,
        onnx=onnx_ok,
        index_size=index_size,
    )


# ---------------------------------------------------------------------------
# GET /stats
# ---------------------------------------------------------------------------


@app.get("/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    """Return index and configuration statistics."""
    from app.services.embedding_service import get_embedding_service
    from app.services.retrieval_service import get_retriever

    retriever = get_retriever()
    embedder = get_embedding_service()

    return StatsResponse(
        index_size=retriever.size,
        embedding_dimension=embedder.dimension,
        ollama_model=settings.ollama_model,
        confidence_threshold=settings.confidence_threshold,
        max_retries=settings.max_retries,
    )
