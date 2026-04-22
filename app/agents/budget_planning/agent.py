from __future__ import annotations

import calendar
from datetime import date, datetime
from typing import Any, Dict

from langchain_core.messages import AIMessage

from app.agents.budget_planning.extractor import extract_budget_request
from app.agents.budget_planning.planner import (
    calculate_monthly_spending,
    evaluate_budget_progress,
    generate_budget_allocations,
    generate_budget_warnings,
)
from app.state import BudgetAllocation, TransactionCategory
from app.guardrails.output_validator import validate_budget_output


def _build_ai_message(
    allocation_list: list,
    warnings: list,
    progress: dict,
    monthly_income: float,
) -> str:
    lines = [f"Here's your budget plan based on a monthly income of £{monthly_income:.2f}:\n"]

    if allocation_list:
        lines.append("Budget allocations:")
        for alloc in sorted(allocation_list, key=lambda a: a.allocated_amount, reverse=True):
            status = progress.get(alloc.category.value, {}).get("status", "on_track")
            status_icon = {"exceeded": "❌", "near_limit": "⚠", "on_track": "✓"}.get(status, "")
            lines.append(
                f"  {status_icon} {alloc.category.value.capitalize():<15} "
                f"£{alloc.allocated_amount:>7.2f} budget  |  "
                f"£{alloc.spent_amount:>7.2f} spent  |  "
                f"£{alloc.remaining:>7.2f} remaining"
            )

    if warnings:
        lines.append("\nWarnings:")
        for w in warnings:
            severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(w.get("severity", ""), "•")
            lines.append(f"  {severity_icon} {w.get('message', '')}")
    else:
        lines.append("\nAll categories are within budget. Keep it up!")

    return "\n".join(lines)

def budget_planning_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node entry point for Budget Planning Agent.

    Responsibilities:
    1. Read input from shared state
    2. Use extractor to parse/normalize the user's request
    3. Call planner functions for local computation
    4. Write structured outputs back into state
    """
    extracted = extract_budget_request(state)
    if extracted.get("needs_clarification"):
        state["budget_request"] = extracted
        state["budget_summary"] = "More information is needed before generating a budget plan."
        state["budget_warnings"] = []
        state["budget_progress"] = {}
        state["output_validation_result"] = {
            "valid": True,
            "errors": [],
            "sanitized_output": None,
        }
        state["security_events"] = [{
            "source": "budget_planning",
            "event_type": "clarification_required",
            "reason": "monthly_income_missing",
        }]
        state["pending_confirmation"] = {
            "action": "clarify_budget_planning",
            "agent": "budget_planning",
            "summary": "Monthly income is required to generate a budget plan.",
            "details": [
                "Please provide your monthly income so I can calculate budget allocations.",
            ],
        }
        return state

    monthly_income = extracted.get("monthly_income")
    categories_requested = extracted.get("categories_requested", [])

    # Use LLM-categorised transactions for consistency with expense_analysis
    categorised = state.get("categorised_transactions") or []
    # Fallback to raw transactions if not yet categorised
    if not categorised:
        categorised = state.get("transactions", [])

    expense_analysis = state.get("expense_analysis", {}) or {}
    if not isinstance(expense_analysis, dict):
        raise ValueError("expense_analysis must be a dictionary")

    category_monthly_avg = expense_analysis.get("category_monthly_avg", {}) or {}
    category_trends = expense_analysis.get("category_trends", {}) or {}

    if not isinstance(category_monthly_avg, dict):
        raise ValueError("expense_analysis.category_monthly_avg must be a dictionary")
    if not isinstance(category_trends, dict):
        raise ValueError("expense_analysis.category_trends must be a dictionary")

    # If user requested specific categories, filter to only those
    if categories_requested:
        category_monthly_avg = {
            cat: amt for cat, amt in category_monthly_avg.items()
            if cat in categories_requested
        }
        category_trends = {
            cat: trend for cat, trend in category_trends.items()
            if cat in categories_requested
        }

    raw_existing = state.get("budget_allocations") or []
    existing_budget: Dict[str, float] = {}
    for alloc in raw_existing:
        if isinstance(alloc, BudgetAllocation):
            existing_budget[alloc.category.value] = alloc.allocated_amount

    current_date_str = state.get("current_date")
    if current_date_str:
        try:
            current_date = datetime.strptime(current_date_str, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("current_date must be in YYYY-MM-DD format") from exc
    else:
        current_date = datetime.today()

    current_day = current_date.day
    days_in_month = calendar.monthrange(current_date.year, current_date.month)[1]

    budget_allocations = generate_budget_allocations(
        monthly_income=monthly_income,
        category_monthly_avg=category_monthly_avg,
        category_trends=category_trends,
        existing_budget=existing_budget,
    )

    # Calculate actual spending from LLM-categorised transactions for consistency
    actual_spending = calculate_monthly_spending(categorised)

    progress = evaluate_budget_progress(
        budget_allocations=budget_allocations,
        actual_spending=actual_spending,
        current_day=current_day,
        days_in_month=days_in_month,
    )

    warnings = generate_budget_warnings(progress)

    period_start = date(current_date.year, current_date.month, 1)
    period_end = date(current_date.year, current_date.month, days_in_month)

    allocation_list: list[BudgetAllocation] = []
    for cat, amount in budget_allocations.items():
        if cat == TransactionCategory.INCOME.value:
            continue
        try:
            category_enum = TransactionCategory(cat)
        except ValueError:
            continue
        allocation_list.append(
            BudgetAllocation(
                category=category_enum,
                allocated_amount=amount,
                spent_amount=actual_spending.get(cat, 0.0),
                period_start=period_start,
                period_end=period_end,
            )
        )

    warning_count = len(warnings)
    summary = (
        f"Budget planning completed for {len(budget_allocations)} categories. "
        f"{warning_count} warning(s) generated."
    )

    ai_message = _build_ai_message(allocation_list, warnings, progress, monthly_income)

    candidate_output = {
        "budget_allocations": allocation_list,
        "budget_progress": progress,
        "budget_warnings": warnings,
        "budget_summary": summary,
        "budget_request": extracted,
    }

    validation_result = validate_budget_output(candidate_output)

    if not validation_result["valid"]:
        state["budget_summary"] = "Budget planning failed output validation."
        state["budget_warnings"] = []
        state["budget_progress"] = {}
        state["budget_request"] = extracted
        state["output_validation_result"] = validation_result
        state["security_events"] = [{
            "source": "budget_planning",
            "event_type": "output_validation_failed",
            "errors": validation_result["errors"],
        }]
        return state

    state["budget_allocations"] = allocation_list
    state["budget_progress"] = progress
    state["budget_warnings"] = warnings
    state["budget_summary"] = summary
    state["budget_request"] = extracted
    state["output_validation_result"] = validation_result
    state["messages"] = [AIMessage(content=ai_message)]

    return state
