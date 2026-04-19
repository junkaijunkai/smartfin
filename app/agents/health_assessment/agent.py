"""
Financial Health and Risk Assessment agent — LangGraph node entry point.

Reads categorised_transactions (populated by expense_analysis) and falls back
to raw transactions when invoked standalone. Also consumes spending_trends and
monthly_income from state.

Writes:
  health_summary  — HealthSummary object
  alerts          — appends any new Alert objects to the existing list
"""

from __future__ import annotations

from app.agents.health_assessment.assessor import assess_health
from app.state import AppState


def run(state: AppState) -> dict:
    """
    LangGraph node entry point for financial health assessment.

    Prefers categorised_transactions (richer category data from expense_analysis).
    Falls back to raw transactions when invoked standalone or when
    expense_analysis has not yet run.
    """
    transactions = state.get("categorised_transactions") or state.get("transactions") or []
    monthly_income = state.get("monthly_income") or 0.0
    spending_trends = state.get("spending_trends") or []

    health_summary, new_alerts = assess_health(
        transactions=transactions,
        monthly_income=monthly_income,
        spending_trends=spending_trends,
    )

    # Merge new alerts into any alerts already present in state
    existing_alerts = list(state.get("alerts") or [])
    merged_alerts = existing_alerts + new_alerts

    print(
        f"[health_assessment] Assessed {len(transactions)} transactions. "
        f"Rating: {health_summary.rating.value}. "
        f"DTI: {health_summary.debt_to_income_ratio:.0%}, "
        f"Reserves: {health_summary.liquid_reserve_months:.1f} months. "
        f"Generated {len(new_alerts)} alert(s)."
    )

    return {
        "health_summary": health_summary,
        "alerts": merged_alerts,
    }
