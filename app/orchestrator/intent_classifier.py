"""
Intent classification — routes user messages to appropriate agents using LLM.

Responsibilities:
  1. Parse user message and classify intent (which agent to invoke).
  2. Use ChatAnthropic with structured output to ask Claude which agent to route to.
  3. Fall back to keyword matching if LLM fails.

Public API:
    classify_intent(message: str) -> str
"""

from __future__ import annotations

import logging
import os
from typing import Literal

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel
from app.config import resolve_model_name

logger = logging.getLogger(__name__)


class _IntentResult(BaseModel):
    """Structured output schema for intent classification."""

    agent: Literal[
        "expense_analysis",
        "budget_planning",
        "goal_planning",
        "anomaly_detection",
        "health_assessment",
        "unknown",
    ]
    reasoning: str


def _keyword_fallback(message: str) -> str:
    """
    Fallback classification using simple keyword matching.
    Provides deterministic routing when LLM unavailable.
    """
    msg = message.lower()

    if "budget" in msg:
        return "budget_planning"
    elif any(kw in msg for kw in ["goal", "save", "saving", "fund", "deposit"]):
        return "goal_planning"
    elif any(kw in msg for kw in ["suspicious", "anomal"]):
        return "anomaly_detection"
    elif any(kw in msg for kw in ["health", "risk"]):
        return "health_assessment"
    elif any(kw in msg for kw in ["spend", "spending", "expense", "transaction", "categor"]):
        return "expense_analysis"

    return "unknown"


def _build_prompt(message: str) -> str:
    """Build the intent classification prompt."""
    return (
        "You are the routing brain of a personal finance AI assistant.\n\n"
        "Given the user's message, identify which single specialist agent should handle it:\n"
        '- "expense_analysis": user wants to see spending breakdown, transaction categories, or spending trends\n'
        '- "budget_planning": user wants to plan, adjust, or review spending limits/budgets\n'
        '- "goal_planning": user wants to set or track savings goals, deposits, or funds\n'
        '- "anomaly_detection": user wants to find suspicious, unexpected, or unusual transactions\n'
        '- "health_assessment": user wants a financial health check, risk score, or overall financial picture\n'
        '- "unknown": the message is not related to personal finance at all\n\n'
        "Respond with exactly one agent name and a brief one-sentence reasoning.\n\n"
        f"User message: {message}"
    )


def classify_intent(message: str) -> str:
    """
    Classify user intent and return the agent name to route to.

    Falls back to keyword matching on any LLM error, ensuring routing
    always succeeds even if the API is unavailable.
    """
    try:
        model_name = resolve_model_name(os.getenv("SMARTFIN_MODEL", "claude-haiku-4-5"))
        llm = ChatAnthropic(model=model_name)
        structured_llm = llm.with_structured_output(_IntentResult)

        prompt = _build_prompt(message)
        result: _IntentResult = structured_llm.invoke(prompt)

        logger.debug(
            "[intent_classifier] classified '%s' → %s (reasoning: %s)",
            message[:50],
            result.agent,
            result.reasoning,
        )
        return result.agent

    except Exception as exc:
        logger.warning(
            "[intent_classifier] LLM classification failed, falling back to keyword match: %s",
            exc,
        )
        fallback = _keyword_fallback(message)
        logger.debug("[intent_classifier] keyword fallback → %s", fallback)
        return fallback
