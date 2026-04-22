from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from app.state import TransactionCategory


logger = logging.getLogger(__name__)

MAX_RETRIES = 3
_INITIAL_BACKOFF = 2.0  # seconds; doubles on each subsequent retry
_LLM_TIMEOUT = 30       # seconds

# All valid spending categories
SUPPORTED_CATEGORIES = [cat.value for cat in TransactionCategory if cat.value != "income"]


class BudgetRequest(BaseModel):
    intent: str = Field(default="budget_planning")
    user_message: str
    monthly_income: Optional[float] = None
    categories_requested: List[str] = Field(default_factory=list)
    needs_clarification: bool = False


def extract_budget_request(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a structured budget-planning request from shared state via LLM.

    Responsibilities:
    - read latest user message
    - call LLM to normalize the request into structured fields
    - fall back to state["monthly_income"] if the user did not mention income
    """

    messages = state.get("messages", [])
    last_message = messages[-1].content if messages else ""

    # fallback income from state
    state_income = state.get("monthly_income")

    model_name = os.getenv("SMARTFIN_MODEL", "claude-haiku-4-5")

    try:
        llm = ChatAnthropic(model=model_name, temperature=0, timeout=_LLM_TIMEOUT)
        structured_llm = llm.with_structured_output(BudgetRequest)
    except Exception as exc:
        logger.warning("Failed to initialise LLM for budget extraction: %s", exc)
        return {
            "intent": "budget_planning",
            "user_message": last_message,
            "monthly_income": state_income,
            "categories_requested": [],
            "needs_clarification": state_income is None,
        }

    prompt = f"""
You are an information extraction assistant for a personal finance multi-agent system.

Your task is to extract a structured budget-planning request from the user's message.

Return:
- intent: always "budget_planning"
- user_message: the original user message
- monthly_income: extract a numeric monthly income only if explicitly stated in the message; otherwise null
- categories_requested: extract any spending categories the user specifically mentions wanting to plan. Only choose from: {', '.join(SUPPORTED_CATEGORIES)}. Leave empty list if no specific categories are mentioned.
- needs_clarification: true if the request lacks enough information for budget planning, especially when monthly income is unknown both in the message and external state

User message:
{last_message}

Known monthly income from state:
{state_income}
"""

    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = structured_llm.invoke(prompt)
            monthly_income = result.monthly_income if result.monthly_income is not None else state_income
            return {
                "intent": "budget_planning",
                "user_message": result.user_message,
                "monthly_income": monthly_income,
                "categories_requested": result.categories_requested,
                "needs_clarification": monthly_income is None,
            }
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                wait = _INITIAL_BACKOFF * (2 ** (attempt - 1))
                logger.warning(
                    "Budget extraction failed (attempt %d/%d), retrying in %.0fs: %s",
                    attempt, MAX_RETRIES, wait, exc,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "Budget extraction failed after %d attempts: %s",
                    MAX_RETRIES, last_exc,
                )

    return {
        "intent": "budget_planning",
        "user_message": last_message,
        "monthly_income": state_income,
        "categories_requested": [],
        "needs_clarification": state_income is None,
    }