"""
LangGraph agent state definition.

This TypedDict is the shared data structure that flows through every
node in the graph.  Each node reads fields it needs and returns a
partial dict with the fields it updates.
"""

from __future__ import annotations

from typing import Literal, TypedDict


class AgentState(TypedDict, total=False):
    """Shared state for the agentic RAG workflow."""

    # ── Input ─────────────────────────────────────────────────
    query: str
    """Original user query."""

    # ── Intent ────────────────────────────────────────────────
    intent: Literal["qa", "chitchat", "summarize"]
    """Classified intent of the query."""

    intent_confidence: float
    """Confidence of the intent classifier (0–1)."""

    # ── Retrieval ─────────────────────────────────────────────
    refined_query: str
    """Search query (may differ from ``query`` after refinement)."""

    contexts: list[str]
    """Retrieved document chunks."""

    context_scores: list[float]
    """Similarity scores for each retrieved chunk."""

    # ── Generation ────────────────────────────────────────────
    answer: str
    """Generated answer text."""

    # ── Self-critique ─────────────────────────────────────────
    confidence: float
    """Aggregate self-critique score (0–1)."""

    critique_feedback: str
    """Textual explanation from the critic."""

    critique_relevance: float
    critique_groundedness: float
    critique_completeness: float

    # ── Control flow ──────────────────────────────────────────
    retry_count: int
    """Number of re-retrieval attempts so far."""

    max_retries: int
    """Maximum allowed re-retrieval attempts."""

    final: bool
    """True when the answer should be returned as-is."""
