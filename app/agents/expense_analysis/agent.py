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

from app.state import AppState
from app.agents.expense_analysis.categoriser import categorise_transactions
from app.agents.expense_analysis.analyser import compute_spending_trends


def run(state: AppState) -> dict:
    """
    LangGraph node function for expense analysis.

    Reads:   state["transactions"], state["monthly_income"]
    Writes:  categorised_transactions, spending_trends, pending_confirmation
    """
    transactions = state.get("transactions") or []

    # ------------------------------------------------------------------
    # Step 1: Categorise
    # ------------------------------------------------------------------
    categorised = categorise_transactions(transactions)

    # ------------------------------------------------------------------
    # Step 2: Compute trends
    # ------------------------------------------------------------------
    trends = compute_spending_trends(categorised)

    # ------------------------------------------------------------------
    # Step 3: Build HITL confirmation payload
    #
    # The graph is compiled with interrupt_before=[NODE_CONFIRM], so setting
    # pending_confirmation here will cause the graph to pause after this node
    # and before confirm_node runs. The UI reads this payload, presents it to
    # the user, then calls resume_with_confirmation() to continue.
    # ------------------------------------------------------------------
    trend_lines: list[str] = []
    for t in trends:
        sign = "+" if t.deviation_pct >= 0 else ""
        if t.deviation_pct is None:
            trend_lines.append(
                f"  {t.category.value:<15} £{t.current_period_total:>8.2f}"
                f" No data for prev period ")
        else:
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
        # confirmed is intentionally absent — route_after_agent checks for
        # its absence to decide whether a HITL pause is needed.
    }

    return {
        "categorised_transactions": categorised,
        "spending_trends": trends,
        "pending_confirmation": pending_confirmation,
    }
