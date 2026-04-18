from __future__ import annotations

import calendar
from datetime import datetime
from typing import Any, Dict

from app.agents.budget_planning.planner import (
    calculate_monthly_spending,
    evaluate_budget_progress,
    generate_budget_allocations,
    generate_budget_warnings,
)


def budget_planning_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for Budget Planning Agent.
    Reads input from shared state, performs budget planning,
    and writes results back into state.
    """

    # === 1. 读取输入 ===
    monthly_income = state.get("monthly_income")
    transactions = state.get("transactions", [])

    expense_analysis = state.get("expense_analysis", {}) or {}
    if not isinstance(expense_analysis, dict):
        raise ValueError("expense_analysis must be a dictionary")

    category_monthly_avg = expense_analysis.get("category_monthly_avg", {}) or {}
    category_trends = expense_analysis.get("category_trends", {}) or {}

    if not isinstance(category_monthly_avg, dict):
        raise ValueError("expense_analysis.category_monthly_avg must be a dictionary")

    if not isinstance(category_trends, dict):
        raise ValueError("expense_analysis.category_trends must be a dictionary")

    existing_budget = state.get("budget_allocations", {}) or {}
    current_date_str = state.get("current_date")

    # === 2. 处理日期 ===
    if current_date_str:
        try:
            current_date = datetime.strptime(current_date_str, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("current_date must be in YYYY-MM-DD format") from exc
    else:
        current_date = datetime.today()

    current_day = current_date.day
    days_in_month = calendar.monthrange(current_date.year, current_date.month)[1]

    # === 3. 调用 planner 层 ===
    budget_allocations = generate_budget_allocations(
        monthly_income=monthly_income,
        category_monthly_avg=category_monthly_avg,
        category_trends=category_trends,
        existing_budget=existing_budget,
    )

    actual_spending = calculate_monthly_spending(transactions)

    progress = evaluate_budget_progress(
        budget_allocations=budget_allocations,
        actual_spending=actual_spending,
        current_day=current_day,
        days_in_month=days_in_month,
    )

    warnings = generate_budget_warnings(progress)

    # === 4. 写回 state（LangGraph 关键点） ===
    state["budget_allocations"] = budget_allocations
    state["budget_progress"] = progress
    state["budget_warnings"] = warnings

    return state