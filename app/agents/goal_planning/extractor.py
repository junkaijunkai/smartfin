"""
Goal Planning extractor — uses Claude to understand the user's latest message
and extract structured financial goal information.

This module is intentionally similar in spirit to expense_analysis/categoriser.py:
- LLM is used for language understanding / field extraction
- deterministic financial calculations remain in tracker.py

Responsibilities:
1. Detect whether the user is expressing a financial goal intent.
2. Extract goal name, target amount, target date, and current saved amount.
3. Return structured output for the Goal Planning agent to consume.
4. Fall back gracefully if the LLM call fails.
5. Support mock mode for local testing without an Anthropic API key.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import date
from typing import Optional

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured-output schema
# ---------------------------------------------------------------------------

class GoalExtractionResult(BaseModel):
    """
    Structured result returned by the LLM.

    Fields:
    - is_goal_intent: whether the message is about a savings / financial goal
    - name: short goal name, e.g. "Laptop Fund"
    - target_amount: target amount the user wants to save
    - target_date: deadline / target completion date
    - current_amount: how much the user has already saved for this goal
    - missing_fields: required fields still missing for goal creation
    """
    is_goal_intent: bool = Field(
        description="True if the user is expressing or discussing a financial savings goal."
    )
    name: Optional[str] = Field(
        default=None,
        description="A concise goal name such as 'Laptop Fund' or 'Emergency Fund'."
    )
    target_amount: Optional[float] = Field(
        default=None,
        description="The target amount the user wants to accumulate."
    )
    target_date: Optional[date] = Field(
        default=None,
        description="The target date by which the user wants to reach the goal."
    )
    current_amount: Optional[float] = Field(
        default=None,
        description="The amount already saved toward this goal, if mentioned."
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Missing required fields, typically target_amount and/or target_date."
    )


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(user_message: str) -> str:
    """
    Build the prompt for goal extraction.

    We keep it explicit and schema-oriented so the LLM output is predictable.
    """
    return f"""
You are a financial planning assistant for a personal finance AI system.

Your task is to determine whether the user's message expresses a financial savings goal.
If yes, extract the relevant goal parameters.

Return structured data with these fields:
- is_goal_intent
- name
- target_amount
- target_date
- current_amount
- missing_fields

Rules:
1. A financial goal includes examples such as:
   - saving for a laptop
   - building an emergency fund
   - saving for a holiday / travel
   - saving for a house deposit
2. Required fields for creating a usable goal:
   - target_amount
   - target_date
3. If the user clearly has goal intent but omits one or more required fields:
   - set is_goal_intent = true
   - include the missing fields in missing_fields
4. If the message is not about a financial goal:
   - set is_goal_intent = false
   - leave other fields as null or empty
5. If current saved amount is not mentioned, leave current_amount as null.
6. Choose a short, natural goal name when possible.
7. Do NOT calculate monthly savings. Only extract information.

User message:
\"\"\"{user_message}\"\"\"
""".strip()


# ---------------------------------------------------------------------------
# Fallback extraction
# ---------------------------------------------------------------------------

_AMOUNT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)")
_DATE_PATTERN = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def _fallback_extract(user_message: str) -> GoalExtractionResult:
    """
    Lightweight fallback when LLM extraction fails.

    This fallback is intentionally simple:
    - detects rough goal intent using keywords
    - extracts one numeric amount if present
    - extracts YYYY-MM-DD date if present
    """
    msg = user_message.lower()

    goal_keywords = [
        "save", "saving", "goal", "fund", "deposit",
        "emergency", "laptop", "travel", "holiday", "house"
    ]
    is_goal_intent = any(keyword in msg for keyword in goal_keywords)

    extracted_amount: Optional[float] = None
    amount_match = _AMOUNT_PATTERN.search(msg)
    if amount_match:
        try:
            extracted_amount = float(amount_match.group(1))
        except ValueError:
            extracted_amount = None

    extracted_date: Optional[date] = None
    date_match = _DATE_PATTERN.search(user_message)
    if date_match:
        try:
            extracted_date = date.fromisoformat(date_match.group(1))
        except ValueError:
            extracted_date = None

    goal_name: Optional[str] = None
    if "laptop" in msg:
        goal_name = "Laptop Fund"
    elif "emergency" in msg:
        goal_name = "Emergency Fund"
    elif "travel" in msg or "holiday" in msg:
        goal_name = "Travel Fund"
    elif "house" in msg or "deposit" in msg:
        goal_name = "House Deposit Fund"
    elif is_goal_intent:
        goal_name = "Financial Goal"

    missing_fields: list[str] = []
    if is_goal_intent:
        if extracted_amount is None:
            missing_fields.append("target_amount")
        if extracted_date is None:
            missing_fields.append("target_date")

    return GoalExtractionResult(
        is_goal_intent=is_goal_intent,
        name=goal_name,
        target_amount=extracted_amount,
        target_date=extracted_date,
        current_amount=None,
        missing_fields=missing_fields,
    )


# ---------------------------------------------------------------------------
# Mock mode
# ---------------------------------------------------------------------------

def _mock_extract(user_message: str) -> GoalExtractionResult:
    """
    Mock extractor for local testing without an Anthropic API key.

    Strategy:
    - reuse fallback parsing
    - but always behave as if extraction succeeded
    - supports inputs like:
      'I want to save 8000 by 2027-06-01 for a laptop.'
    """
    return _fallback_extract(user_message)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_goal_from_message(user_message: str) -> tuple[GoalExtractionResult, bool]:
    """
    Extract structured goal information from a user message.

    Returns:
        (result, llm_succeeded)

    - result: structured extraction result
    - llm_succeeded: True if the result came from Claude or mock mode,
                     False if fallback was used after a real LLM failure
    """

    if not user_message.strip():
        return GoalExtractionResult(
            is_goal_intent=False,
            missing_fields=[],
        ), True

    # Mock mode for local testing
    # PowerShell:
    #   $env:SMARTFIN_MOCK_LLM="true"
    mock_mode = os.getenv("SMARTFIN_MOCK_LLM", "").lower() == "true"
    if mock_mode:
        logger.info("SMARTFIN_MOCK_LLM=true, using mock goal extractor.")
        return _mock_extract(user_message), True

    model_name = os.getenv("SMARTFIN_MODEL", "claude-sonnet-4-6")

    try:
        llm = ChatAnthropic(model=model_name)
        structured_llm = llm.with_structured_output(GoalExtractionResult)
        prompt = _build_prompt(user_message)
        result: GoalExtractionResult = structured_llm.invoke(prompt)
        return result, True
    except Exception as exc:
        logger.warning("Goal extraction failed, using fallback extractor: %s", exc)
        return _fallback_extract(user_message), False