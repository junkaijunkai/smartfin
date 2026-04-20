"""
Expense Analysis agent — LangGraph node entry point.

Responsibilities:
  1. Categorise raw transactions via LLM (categoriser.py).
  2. Compute 30-day spending trends (analyser.py).
  3. Write results back to AppState.
  4. Set pending_confirmation so the graph pauses for HITL review before
     downstream agents consume the categorised data.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from app.state import AppState, SpendingTrend
from app.agents.expense_analysis.categoriser import categorise_transactions
from app.agents.expense_analysis.analyser import compute_spending_trends
from app.tools.transaction_store import save_analysis


def _build_result(
    categorised: list,
    trends: list[SpendingTrend],
    llm_succeeded: bool,
) -> dict:
    """Build the HITL confirmation payload and return dict for state."""
    trend_lines: list[str] = []
    for t in trends:
        if t.deviation_pct is None:
            trend_lines.append(
                f"  {t.category.value:<15} £{t.current_period_total:>8.2f}"
                f" No data for prev period "
            )
        else:
            sign = "+" if t.deviation_pct >= 0 else "-"
            trend_lines.append(
                f"  {t.category.value:<15} £{t.current_period_total:>8.2f}"
                f"  ({sign}{t.deviation_pct:.1f}% vs prev period)"
            )

    pending_confirmation = {
        "action": "approve_expense_analysis",
        "agent": "expense_analysis",
        "summary": (
            f"Categorised {len(categorised)} transactions across "
            f"{len(trends)} spending categories."
        ),
        "details": trend_lines,
        "categorisation_confidence": "llm" if llm_succeeded else "fallback_keywords",
    }

    return {
        "categorised_transactions": categorised,
        "spending_trends": trends,
        "pending_confirmation": pending_confirmation,
    }


def run(state: AppState, config: RunnableConfig | None = None) -> dict:
    """
    LangGraph node function for expense analysis with incremental processing.

    Incremental logic:
      1. If state["categorised_transactions"] exists, use it as base.
      2. Filter state["transactions"] by ID — only process new ones.
      3. Call LLM only on new transactions, merge with existing.
      4. Recalculate trends on merged set, persist, return HITL payload.

    Reads:   state["transactions"], state["categorised_transactions"]
    Writes:  categorised_transactions, spending_trends, pending_confirmation
    """
    new_transactions = state.get("transactions") or []
    existing_categorised = state.get("categorised_transactions") or []

    # Filter to only process transactions not already categorised
    existing_ids = {t.id for t in existing_categorised}
    to_categorise = [t for t in new_transactions if t.id not in existing_ids]

    if not to_categorise:
        # No new transactions — just recompute trends on existing data
        if not existing_categorised:
            # Empty input, return empty with HITL payload
            return _build_result([], [], llm_succeeded=True)
        trends = compute_spending_trends(existing_categorised)
        thread_id = (config or {}).get("configurable", {}).get("thread_id")
        if thread_id:
            save_analysis(thread_id, existing_categorised, trends)
        return _build_result(existing_categorised, trends, llm_succeeded=True)

    # Categorise only new transactions
    newly_categorised, llm_succeeded = categorise_transactions(to_categorise)

    if not newly_categorised:
        # LLM failed on new transactions
        if not existing_categorised:
            # No data anywhere, return empty with HITL payload
            return _build_result([], [], llm_succeeded=False)
        # Return existing data as-is
        trends = compute_spending_trends(existing_categorised)
        return _build_result(existing_categorised, trends, llm_succeeded=False)

    # Merge existing + newly categorised, recompute trends on full set
    merged = existing_categorised + newly_categorised
    trends = compute_spending_trends(merged)

    # Persist merged results
    thread_id = (config or {}).get("configurable", {}).get("thread_id")
    if thread_id:
        save_analysis(thread_id, merged, trends)

    return _build_result(merged, trends, llm_succeeded)
