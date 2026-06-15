"""
LLM service wrapping Ollama via langchain-ollama.

Provides structured methods for intent classification, RAG answer
generation, self-critique scoring, and query refinement — all running
against a locally hosted model with zero cloud dependency.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class IntentResult:
    intent: str  # "qa" | "chitchat" | "summarize"
    confidence: float


@dataclass
class CritiqueResult:
    score: float  # 0–1 aggregate confidence
    relevance: float
    groundedness: float
    completeness: float
    feedback: str


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_instance: LLMService | None = None


def get_llm_service() -> LLMService:
    """Return (or lazily create) the singleton LLM service."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = LLMService()
    return _instance


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class LLMService:
    """
    High-level interface to the local Ollama LLM.

    Every method sends a carefully constructed prompt and parses the
    structured JSON response.  If parsing fails, sensible defaults are
    returned so the agent never crashes on malformed LLM output.
    """

    def __init__(self) -> None:
        self._llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
            format="json",  # request JSON output mode
        )
        # A second instance without JSON mode for free-text generation.
        self._llm_text = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.3,
        )
        log.info(
            "llm_service_ready",
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
        )

    # ------------------------------------------------------------------
    # Intent Classification
    # ------------------------------------------------------------------

    def classify_intent(self, query: str) -> IntentResult:
        """
        Classify the user query into one of: ``qa``, ``chitchat``, ``summarize``.

        Returns an ``IntentResult`` with the predicted intent and
        a confidence score (0–1).
        """
        from app.agent.prompts import INTENT_CLASSIFICATION_PROMPT

        messages = [
            SystemMessage(content=INTENT_CLASSIFICATION_PROMPT),
            HumanMessage(content=query),
        ]

        response = self._llm.invoke(messages)
        parsed = self._safe_parse_json(response.content)

        intent = parsed.get("intent", "qa")
        if intent not in {"qa", "chitchat", "summarize"}:
            intent = "qa"

        confidence = float(parsed.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        log.info("intent_classified", query=query[:80], intent=intent, confidence=confidence)
        return IntentResult(intent=intent, confidence=confidence)

    # ------------------------------------------------------------------
    # Answer Generation (RAG)
    # ------------------------------------------------------------------

    def generate_answer(self, query: str, contexts: list[str]) -> str:
        """
        Generate a grounded answer using the retrieved context chunks.

        The model is instructed to only use information present in the
        provided contexts.
        """
        from app.agent.prompts import RAG_GENERATION_PROMPT

        context_block = "\n\n---\n\n".join(
            f"[Source {i + 1}]\n{ctx}" for i, ctx in enumerate(contexts)
        )

        system_prompt = RAG_GENERATION_PROMPT.format(contexts=context_block)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ]

        response = self._llm_text.invoke(messages)
        answer = response.content.strip()

        log.info("answer_generated", query=query[:80], answer_len=len(answer))
        return answer

    # ------------------------------------------------------------------
    # Direct Response (chitchat)
    # ------------------------------------------------------------------

    def direct_response(self, query: str) -> str:
        """Generate a direct conversational response without retrieval."""
        from app.agent.prompts import CHITCHAT_PROMPT

        messages = [
            SystemMessage(content=CHITCHAT_PROMPT),
            HumanMessage(content=query),
        ]

        response = self._llm_text.invoke(messages)
        return response.content.strip()

    # ------------------------------------------------------------------
    # Self-Critique
    # ------------------------------------------------------------------

    def critique_answer(
        self,
        query: str,
        answer: str,
        contexts: list[str],
    ) -> CritiqueResult:
        """
        Score the generated answer on relevance, groundedness, and
        completeness.  Returns a ``CritiqueResult`` with individual
        and aggregate scores plus textual feedback.
        """
        from app.agent.prompts import SELF_CRITIQUE_PROMPT

        context_block = "\n\n---\n\n".join(
            f"[Source {i + 1}]\n{ctx}" for i, ctx in enumerate(contexts)
        )

        prompt = SELF_CRITIQUE_PROMPT.format(
            query=query,
            answer=answer,
            contexts=context_block,
        )

        messages = [HumanMessage(content=prompt)]
        response = self._llm.invoke(messages)
        parsed = self._safe_parse_json(response.content)

        relevance = float(parsed.get("relevance", 0.5))
        groundedness = float(parsed.get("groundedness", 0.5))
        completeness = float(parsed.get("completeness", 0.5))
        feedback = parsed.get("feedback", "No feedback provided.")

        # Weighted aggregate
        score = 0.4 * relevance + 0.4 * groundedness + 0.2 * completeness
        score = max(0.0, min(1.0, score))

        log.info(
            "answer_critiqued",
            score=round(score, 3),
            relevance=round(relevance, 3),
            groundedness=round(groundedness, 3),
            completeness=round(completeness, 3),
        )

        return CritiqueResult(
            score=score,
            relevance=relevance,
            groundedness=groundedness,
            completeness=completeness,
            feedback=feedback,
        )

    # ------------------------------------------------------------------
    # Query Refinement
    # ------------------------------------------------------------------

    def refine_query(self, original_query: str, critique_feedback: str) -> str:
        """
        Generate a refined search query based on critique feedback.

        This is used when the self-critique loop determines the answer
        was insufficient and a re-retrieval is needed.
        """
        from app.agent.prompts import QUERY_REFINEMENT_PROMPT

        prompt = QUERY_REFINEMENT_PROMPT.format(
            original_query=original_query,
            feedback=critique_feedback,
        )

        messages = [HumanMessage(content=prompt)]
        response = self._llm.invoke(messages)
        parsed = self._safe_parse_json(response.content)

        refined = parsed.get("refined_query", original_query)
        log.info(
            "query_refined",
            original=original_query[:80],
            refined=refined[:80],
        )
        return refined

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_parse_json(text: str) -> dict:
        """
        Best-effort JSON extraction from LLM output.

        Handles cases where the model wraps JSON in markdown fences
        or adds prose around it.
        """
        text = text.strip()

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code fence
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding first { ... } block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        log.warning("json_parse_failed", raw_text=text[:200])
        return {}
