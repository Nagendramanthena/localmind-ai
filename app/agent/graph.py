"""
LangGraph StateGraph definition.

Wires together all agent nodes with conditional edges to implement:

  query → classify → [chitchat? → direct_response]
                      [qa/summarize → retrieve → generate → critique]
                                                              ↓
                                                  [accept → END]
                                                  [retry  → refine → retrieve ↺]
                                                  [give_up → END]
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.models.state import AgentState
from app.agent.nodes import (
    classify_intent,
    retrieve_context,
    generate_answer,
    direct_response,
    self_critique,
    refine_query,
    route_by_intent,
    route_by_confidence,
)


def build_graph() -> StateGraph:
    """
    Construct and compile the agentic RAG graph.

    Returns a compiled ``StateGraph`` ready for ``.invoke()``.
    """
    builder = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("direct_response", direct_response)
    builder.add_node("self_critique", self_critique)
    builder.add_node("refine_query", refine_query)

    # ── Entry edge ────────────────────────────────────────────
    builder.add_edge(START, "classify_intent")

    # ── Intent routing ────────────────────────────────────────
    builder.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "retrieve": "retrieve_context",
            "chitchat": "direct_response",
        },
    )

    # ── Retrieval → Generation → Critique pipeline ────────────
    builder.add_edge("retrieve_context", "generate_answer")
    builder.add_edge("generate_answer", "self_critique")

    # ── Critique routing ──────────────────────────────────────
    builder.add_conditional_edges(
        "self_critique",
        route_by_confidence,
        {
            "accept": END,
            "retry": "refine_query",
            "give_up": END,
        },
    )

    # ── Retry loop: refine → re-retrieve ──────────────────────
    builder.add_edge("refine_query", "retrieve_context")

    # ── Chitchat exits directly ───────────────────────────────
    builder.add_edge("direct_response", END)

    # ── Compile ───────────────────────────────────────────────
    graph = builder.compile()
    return graph


# Module-level compiled graph singleton.
agent_graph = build_graph()
