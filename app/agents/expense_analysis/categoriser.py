"""
Transaction categoriser — uses Claude to classify raw transactions in a single
batch call, returning the same list with the category field populated.
"""

from __future__ import annotations

import os

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel

from app.state import Transaction, TransactionCategory


# ---------------------------------------------------------------------------
# Structured-output schema (internal, not exported)
# ---------------------------------------------------------------------------

class _CategoryResult(BaseModel):
    transaction_id: str
    category: TransactionCategory


class _CategoryBatch(BaseModel):
    results: list[_CategoryResult]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def categorise_transactions(transactions: list[Transaction]) -> list[Transaction]:
    """
    Assign a TransactionCategory to every transaction in one LLM call.

    Transactions that the model cannot confidently classify default to OTHER.
    Returns a new list of Transaction objects with the category field set;
    the original list is not mutated.
    """
    if not transactions:
        return []

    model_name = os.getenv("SMARTFIN_MODEL", "claude-sonnet-4-6")
    llm = ChatAnthropic(model=model_name)
    structured_llm = llm.with_structured_output(_CategoryBatch)

    # Build a concise representation for each transaction
    lines = [
        f"id={t.id} | merchant={t.merchant} | description={t.description} | amount={t.amount:.2f}"
        for t in transactions
    ]
    category_values = ", ".join(c.value for c in TransactionCategory)
    transactions_block = "\n".join(lines)

    prompt = (
        f"You are a financial data analyst. Classify each transaction below into "
        f"exactly one of these categories: {category_values}.\n\n"
        "Rules:\n"
        "- Positive amount = expense, negative amount = income → use 'income'.\n"
        "- Use 'other' only when no category fits clearly.\n"
        "- Return a result for EVERY transaction id listed, in any order.\n\n"
        f"Transactions:\n{transactions_block}"
    )

    response: _CategoryBatch = structured_llm.invoke(prompt)

    category_map = {r.transaction_id: r.category for r in response.results}

    return [
        t.model_copy(update={"category": category_map.get(t.id, TransactionCategory.OTHER)})
        for t in transactions
    ]
