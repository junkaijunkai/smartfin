from __future__ import annotations

import calendar
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional


def generate_budget_allocations(
    monthly_income: Optional[float],
    category_monthly_avg: Dict[str, float],
    category_trends: Dict[str, str],
    existing_budget: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Generate or update category-level monthly budget allocations.

    Rules:
    1. If an existing budget is available for a category, keep it.
    2. Otherwise, derive budget from historical monthly average and trend:
       - fixed   -> avg * 1.00
       - stable  -> avg * 1.05
       - rising  -> avg * 1.10
       - volatile-> avg * 0.95
       - unknown -> avg * 1.05
    3. If total generated budget exceeds 90% of monthly income,
       scale all categories proportionally.

    Args:
        monthly_income: User's monthly income. Can be None.
        category_monthly_avg: Historical monthly average spending by category.
        category_trends: Trend labels from Expense Analysis Agent.
        existing_budget: Previously confirmed budget allocations.

    Returns:
        Dict[str, float]: Final monthly budget allocations by category.
    """
    existing_budget = existing_budget or {}
    allocations: Dict[str, float] = {}

    for category, avg_spend in category_monthly_avg.items():
        avg_spend = max(float(avg_spend), 0.0)

        if category in existing_budget:
            allocations[category] = round(max(float(existing_budget[category]), 0.0), 2)
            continue

        trend = category_trends.get(category, "stable").lower()

        if trend == "fixed":
            budget = avg_spend
        elif trend == "rising":
            budget = avg_spend * 1.10
        elif trend == "volatile":
            budget = avg_spend * 0.95
        else:  # stable / unknown
            budget = avg_spend * 1.05

        allocations[category] = round(max(budget, 0.0), 2)

    # Preserve categories that exist only in existing_budget
    for category, value in existing_budget.items():
        if category not in allocations:
            allocations[category] = round(max(float(value), 0.0), 2)

    total_budget = sum(allocations.values())

    # Cap total budget at 90% of monthly income to leave room for savings/flexibility
    if monthly_income is not None:
        monthly_income = max(float(monthly_income), 0.0)
        max_budget = monthly_income * 0.90

        if total_budget > max_budget and total_budget > 0:
            scale = max_budget / total_budget
            for category in allocations:
                allocations[category] = round(allocations[category] * scale, 2)

    return allocations


def calculate_monthly_spending(
    transactions: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Aggregate actual spending by category from the current month's transactions.

    Expected transaction format:
    {
        "date": "2026-04-01",
        "category": "food",
        "amount": 12.5
    }

    Notes:
    - Negative or zero amounts are ignored.
    - Missing category is mapped to 'uncategorized'.

    Args:
        transactions: List of transaction records for the current period.

    Returns:
        Dict[str, float]: Actual spending by category.
    """
    spending: Dict[str, float] = defaultdict(float)

    for tx in transactions:
        try:
            amount = float(tx.get("amount", 0.0))
        except (TypeError, ValueError):
            continue

        if amount <= 0:
            continue

        category = str(tx.get("category", "uncategorized")).strip().lower()
        if not category:
            category = "uncategorized"

        spending[category] += amount

    return {category: round(amount, 2) for category, amount in spending.items()}


def evaluate_budget_progress(
    budget_allocations: Dict[str, float],
    actual_spending: Dict[str, float],
    current_day: int,
    days_in_month: int
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate in-period progress against budget allocations.

    Status rules:
    - exceeded: usage_ratio >= 1.0
    - near_limit: usage_ratio > expected_ratio_by_today + 0.10
    - on_track: otherwise

    Args:
        budget_allocations: Planned budget by category.
        actual_spending: Actual current-period spending by category.
        current_day: Current day of month.
        days_in_month: Total days in month.

    Returns:
        Dict[str, Dict[str, Any]]: Per-category budget progress details.
    """
    progress: Dict[str, Dict[str, Any]] = {}

    if days_in_month <= 0:
        raise ValueError("days_in_month must be greater than 0")

    current_day = max(1, min(current_day, days_in_month))
    expected_ratio = current_day / days_in_month

    all_categories = set(budget_allocations.keys()) | set(actual_spending.keys())

    for category in all_categories:
        budget = float(budget_allocations.get(category, 0.0))
        spent = float(actual_spending.get(category, 0.0))

        if budget <= 0:
            # If there is spending but no budget, treat it as exceeded.
            if spent > 0:
                usage_ratio = 1.0
                remaining = round(-spent, 2)
                status = "exceeded"
            else:
                usage_ratio = 0.0
                remaining = 0.0
                status = "on_track"
        else:
            usage_ratio = spent / budget
            remaining = budget - spent

            if usage_ratio >= 1.0:
                status = "exceeded"
            elif usage_ratio > expected_ratio + 0.10:
                status = "near_limit"
            else:
                status = "on_track"

        progress[category] = {
            "spent": round(spent, 2),
            "budget": round(budget, 2),
            "remaining": round(remaining, 2),
            "usage_ratio": round(usage_ratio, 3),
            "expected_ratio_by_today": round(expected_ratio, 3),
            "status": status
        }

    return progress


def generate_budget_warnings(
    progress: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Generate warnings based on budget progress.

    Warning levels:
    - high:
        a) already exceeded budget
        b) usage_ratio > expected_ratio + 0.25
    - medium:
        usage_ratio > expected_ratio + 0.15
    - low:
        usage_ratio >= 0.80

    Args:
        progress: Output from evaluate_budget_progress().

    Returns:
        List[Dict[str, Any]]: Structured warning messages.
    """
    warnings: List[Dict[str, Any]] = []

    for category, item in progress.items():
        usage_ratio = float(item.get("usage_ratio", 0.0))
        expected_ratio = float(item.get("expected_ratio_by_today", 0.0))
        status = item.get("status", "on_track")

        if status == "exceeded" or usage_ratio >= 1.0:
            warnings.append({
                "category": category,
                "severity": "high",
                "message": f"Spending in '{category}' has exceeded the monthly budget."
            })
        elif usage_ratio > expected_ratio + 0.25:
            warnings.append({
                "category": category,
                "severity": "high",
                "message": f"Spending in '{category}' is far ahead of schedule and is likely to exceed the monthly budget."
            })
        elif usage_ratio > expected_ratio + 0.15:
            warnings.append({
                "category": category,
                "severity": "medium",
                "message": f"Spending in '{category}' is ahead of the expected monthly pace."
            })
        elif usage_ratio >= 0.80:
            warnings.append({
                "category": category,
                "severity": "low",
                "message": f"Spending in '{category}' is approaching its monthly budget limit."
            })

    return warnings


def run_budget_planning_agent(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the full Budget Planning Agent pipeline.

    Expected input structure:
    {
        "monthly_income": 5000,
        "transactions": [...],
        "expense_analysis": {
            "category_monthly_avg": {...},
            "category_trends": {...}
        },
        "existing_budget": {...},
        "current_date": "2026-04-16"
    }

    Returns:
    {
        "budget_allocations": {...},
        "progress": {...},
        "warnings": [...],
        "summary": "..."
    }
    """
    monthly_income = input_data.get("monthly_income")
    transactions = input_data.get("transactions", [])
    expense_analysis = input_data.get("expense_analysis", {})
    existing_budget = input_data.get("existing_budget", {}) or {}
    current_date_str = input_data.get("current_date")

    category_monthly_avg = expense_analysis.get("category_monthly_avg", {}) or {}
    category_trends = expense_analysis.get("category_trends", {}) or {}

    if not isinstance(category_monthly_avg, dict):
        raise ValueError("expense_analysis.category_monthly_avg must be a dictionary")
    if not isinstance(category_trends, dict):
        raise ValueError("expense_analysis.category_trends must be a dictionary")

    # Parse current date
    if current_date_str:
        try:
            current_date = datetime.strptime(current_date_str, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("current_date must be in YYYY-MM-DD format") from exc
    else:
        current_date = datetime.today()

    current_day = current_date.day
    days_in_month = calendar.monthrange(current_date.year, current_date.month)[1]

    # 1. Generate / update budget allocations
    budget_allocations = generate_budget_allocations(
        monthly_income=monthly_income,
        category_monthly_avg=category_monthly_avg,
        category_trends=category_trends,
        existing_budget=existing_budget
    )

    # 2. Calculate actual spending
    actual_spending = calculate_monthly_spending(transactions)

    # 3. Evaluate progress
    progress = evaluate_budget_progress(
        budget_allocations=budget_allocations,
        actual_spending=actual_spending,
        current_day=current_day,
        days_in_month=days_in_month
    )

    # 4. Generate warnings
    warnings = generate_budget_warnings(progress)

    # 5. Build summary
    on_track_count = sum(1 for v in progress.values() if v["status"] == "on_track")
    near_limit_count = sum(1 for v in progress.values() if v["status"] == "near_limit")
    exceeded_count = sum(1 for v in progress.values() if v["status"] == "exceeded")

    summary = (
        f"Budget planning completed for {len(progress)} categories. "
        f"{on_track_count} categories are on track, "
        f"{near_limit_count} categories are near the budget limit, and "
        f"{exceeded_count} categories have exceeded budget."
    )

    return {
        "budget_allocations": budget_allocations,
        "progress": progress,
        "warnings": warnings,
        "summary": summary
    }