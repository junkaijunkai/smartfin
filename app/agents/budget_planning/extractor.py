from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic


SUPPORTED_CATEGORIES = ["food", "transport", "housing", "entertainment"]


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

    llm = ChatAnthropic(
        model="claude-3-5-sonnet-latest",
        temperature=0,
    )

    structured_llm = llm.with_structured_output(BudgetRequest)

    prompt = f"""
You are an information extraction assistant for a personal finance multi-agent system.

Your task is to extract a structured budget-planning request from the user's message.

Return:
- intent: always "budget_planning"
- user_message: the original user message
- monthly_income: extract a numeric monthly income only if explicitly stated in the message; otherwise null
- categories_requested: only choose from {SUPPORTED_CATEGORIES}
- needs_clarification: true if the request lacks enough information for budget planning, especially when monthly income is unknown both in the message and external state

User message:
{last_message}

Known monthly income from state:
{state_income}
"""

    result = structured_llm.invoke(prompt)

    monthly_income = result.monthly_income if result.monthly_income is not None else state_income

    return {
        "intent": "budget_planning",
        "user_message": result.user_message,
        "monthly_income": monthly_income,
        "categories_requested": result.categories_requested,
        "needs_clarification": monthly_income is None,
    }