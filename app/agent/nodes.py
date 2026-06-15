"""
LangGraph node functions.

Each function receives the shared ``AgentState`` and returns a partial
dict with the fields it updates.  Nodes are pure functions of state,
making them easy to test in isolation.
"""

from __future__ import annotations

from app.models.state import AgentState
from app.services.llm_service import get_llm_service
from app.services.retrieval_service import get_retriever
from app.config import settings
from app.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Node: Classify Intent
# ---------------------------------------------------------------------------


def classify_intent(state: AgentState) -> dict:
    """Classify the user's query into qa / chitchat / summarize."""
    llm = get_llm_service()
    result = llm.classify_intent(state["query"])

    return {
        "intent": result.intent,
        "intent_confidence": result.confidence,
        "refined_query": state["query"],  # initialise to original query
        "retry_count": 0,
        "max_retries": settings.max_retries,
        "final": False,
    }


# ---------------------------------------------------------------------------
# Node: Retrieve Context
# ---------------------------------------------------------------------------


def retrieve_context(state: AgentState) -> dict:
    """Embed the (possibly refined) query and retrieve top-K chunks from FAISS."""
    retriever = get_retriever()

    search_query = state.get("refined_query") or state["query"]
    results = retriever.search(search_query, top_k=settings.top_k)

    contexts = [r.text for r in results]
    scores = [r.score for r in results]

    log.info(
        "context_retrieved",
        query=search_query[:80],
        num_results=len(results),
        top_score=round(scores[0], 4) if scores else 0.0,
    )

    return {
        "contexts": contexts,
        "context_scores": scores,
    }


# ---------------------------------------------------------------------------
# Node: Generate Answer
# ---------------------------------------------------------------------------


def generate_answer(state: AgentState) -> dict:
    """Generate a RAG-grounded answer using the retrieved contexts."""
    llm = get_llm_service()
    answer = llm.generate_answer(state["query"], state.get("contexts", []))
    return {"answer": answer}


# ---------------------------------------------------------------------------
# Node: Direct Response (chitchat)
# ---------------------------------------------------------------------------


def direct_response(state: AgentState) -> dict:
    """Generate a casual response without any retrieval."""
    llm = get_llm_service()
    answer = llm.direct_response(state["query"])
    return {
        "answer": answer,
        "confidence": 1.0,
        "contexts": [],
        "context_scores": [],
        "critique_feedback": "",
        "retry_count": 0,
        "final": True,
    }


# ---------------------------------------------------------------------------
# Node: Self-Critique
# ---------------------------------------------------------------------------


def self_critique(state: AgentState) -> dict:
    """Score the generated answer and decide whether to accept or retry."""
    llm = get_llm_service()

    result = llm.critique_answer(
        query=state["query"],
        answer=state["answer"],
        contexts=state.get("contexts", []),
    )

    retry_count = state.get("retry_count", 0) + 1

    return {
        "confidence": result.score,
        "critique_feedback": result.feedback,
        "critique_relevance": result.relevance,
        "critique_groundedness": result.groundedness,
        "critique_completeness": result.completeness,
        "retry_count": retry_count,
    }


# ---------------------------------------------------------------------------
# Node: Refine Query
# ---------------------------------------------------------------------------


def refine_query(state: AgentState) -> dict:
    """Generate a better search query based on critique feedback."""
    llm = get_llm_service()

    refined = llm.refine_query(
        original_query=state["query"],
        critique_feedback=state.get("critique_feedback", ""),
    )

    log.info(
        "query_refined_for_retry",
        retry=state.get("retry_count", 0),
        refined_query=refined[:80],
    )

    return {"refined_query": refined}


# ---------------------------------------------------------------------------
# Routing functions (used as conditional edges)
# ---------------------------------------------------------------------------


def route_by_intent(state: AgentState) -> str:
    """Route based on classified intent."""
    intent = state.get("intent", "qa")
    if intent == "chitchat":
        return "chitchat"
    # Both "qa" and "summarize" go through retrieval
    return "retrieve"


def route_by_confidence(state: AgentState) -> str:
    """Route based on self-critique score and retry budget."""
    confidence = state.get("confidence", 0.0)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", settings.max_retries)

    if confidence >= settings.confidence_threshold:
        return "accept"
    if retry_count < max_retries:
        return "retry"
    return "give_up"
