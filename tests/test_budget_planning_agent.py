import pytest

from app.budget_planning_agent import (
    generate_budget_allocations,
    calculate_monthly_spending,
    evaluate_budget_progress,
    generate_budget_warnings,
    run_budget_planning_agent,
)


def test_generate_budget_allocations_basic():
    category_monthly_avg = {
        "food": 500,
        "transport": 200,
        "housing": 1200,
    }
    category_trends = {
        "food": "stable",
        "transport": "rising",
        "housing": "fixed",
    }

    result = generate_budget_allocations(
        monthly_income=5000,
        category_monthly_avg=category_monthly_avg,
        category_trends=category_trends,
        existing_budget=None,
    )

    assert result["food"] == 525.00
    assert result["transport"] == 220.00
    assert result["housing"] == 1200.00


def test_generate_budget_allocations_keep_existing_budget():
    category_monthly_avg = {
        "food": 500,
        "transport": 200,
    }
    category_trends = {
        "food": "stable",
        "transport": "rising",
    }
    existing_budget = {
        "food": 600
    }

    result = generate_budget_allocations(
        monthly_income=5000,
        category_monthly_avg=category_monthly_avg,
        category_trends=category_trends,
        existing_budget=existing_budget,
    )

    assert result["food"] == 600.00
    assert result["transport"] == 220.00


def test_generate_budget_allocations_scale_down_when_exceed_income_limit():
    category_monthly_avg = {
        "housing": 2500,
        "food": 1000,
        "transport": 500,
    }
    category_trends = {
        "housing": "fixed",
        "food": "stable",
        "transport": "stable",
    }

    result = generate_budget_allocations(
        monthly_income=3000,
        category_monthly_avg=category_monthly_avg,
        category_trends=category_trends,
        existing_budget=None,
    )

    total_budget = sum(result.values())
    assert total_budget <= 2700.0 + 0.1  # 90% of 3000


def test_calculate_monthly_spending_basic():
    transactions = [
        {"date": "2026-04-01", "category": "food", "amount": 20},
        {"date": "2026-04-02", "category": "food", "amount": 30.5},
        {"date": "2026-04-03", "category": "transport", "amount": 15},
    ]

    result = calculate_monthly_spending(transactions)

    assert result["food"] == 50.5
    assert result["transport"] == 15.0


def test_calculate_monthly_spending_ignore_invalid_and_non_positive_values():
    transactions = [
        {"date": "2026-04-01", "category": "food", "amount": 20},
        {"date": "2026-04-02", "category": "food", "amount": 0},
        {"date": "2026-04-03", "category": "food", "amount": -5},
        {"date": "2026-04-04", "category": "food", "amount": "invalid"},
        {"date": "2026-04-05", "amount": 10},  # no category
    ]

    result = calculate_monthly_spending(transactions)

    assert result["food"] == 20.0
    assert result["uncategorized"] == 10.0


def test_evaluate_budget_progress_on_track():
    budget_allocations = {
        "food": 600
    }
    actual_spending = {
        "food": 250
    }

    result = evaluate_budget_progress(
        budget_allocations=budget_allocations,
        actual_spending=actual_spending,
        current_day=15,
        days_in_month=30,
    )

    assert result["food"]["spent"] == 250.0
    assert result["food"]["remaining"] == 350.0
    assert result["food"]["usage_ratio"] == round(250 / 600, 3)
    assert result["food"]["status"] == "on_track"


def test_evaluate_budget_progress_near_limit():
    budget_allocations = {
        "food": 600
    }
    actual_spending = {
        "food": 400
    }

    result = evaluate_budget_progress(
        budget_allocations=budget_allocations,
        actual_spending=actual_spending,
        current_day=15,
        days_in_month=30,
    )

    assert result["food"]["status"] == "near_limit"


def test_evaluate_budget_progress_exceeded():
    budget_allocations = {
        "food": 600
    }
    actual_spending = {
        "food": 700
    }

    result = evaluate_budget_progress(
        budget_allocations=budget_allocations,
        actual_spending=actual_spending,
        current_day=15,
        days_in_month=30,
    )

    assert result["food"]["status"] == "exceeded"
    assert result["food"]["remaining"] == -100.0


def test_evaluate_budget_progress_spending_without_budget():
    budget_allocations = {}
    actual_spending = {
        "misc": 100
    }

    result = evaluate_budget_progress(
        budget_allocations=budget_allocations,
        actual_spending=actual_spending,
        current_day=10,
        days_in_month=30,
    )

    assert result["misc"]["status"] == "exceeded"


def test_evaluate_budget_progress_invalid_days_in_month():
    with pytest.raises(ValueError):
        evaluate_budget_progress(
            budget_allocations={"food": 100},
            actual_spending={"food": 50},
            current_day=1,
            days_in_month=0,
        )


def test_generate_budget_warnings_low_medium_high():
    progress = {
        "food": {
            "usage_ratio": 0.82,
            "expected_ratio_by_today": 0.75,
            "status": "on_track",
        },
        "transport": {
            "usage_ratio": 0.75,
            "expected_ratio_by_today": 0.50,
            "status": "near_limit",
        },
        "entertainment": {
            "usage_ratio": 1.05,
            "expected_ratio_by_today": 0.50,
            "status": "exceeded",
        },
    }

    warnings = generate_budget_warnings(progress)

    assert len(warnings) == 3

    severity_map = {w["category"]: w["severity"] for w in warnings}

    assert severity_map["food"] == "low"
    assert severity_map["transport"] in ["medium", "high"]
    assert severity_map["entertainment"] == "high"


def test_run_budget_planning_agent_end_to_end():
    input_data = {
        "monthly_income": 5000,
        "transactions": [
            {"date": "2026-04-01", "category": "food", "amount": 50},
            {"date": "2026-04-03", "category": "food", "amount": 80},
            {"date": "2026-04-04", "category": "entertainment", "amount": 200},
            {"date": "2026-04-10", "category": "housing", "amount": 1200},
            {"date": "2026-04-12", "category": "transport", "amount": 60},
        ],
        "expense_analysis": {
            "category_monthly_avg": {
                "food": 600,
                "transport": 200,
                "housing": 1200,
                "entertainment": 250,
            },
            "category_trends": {
                "food": "stable",
                "transport": "stable",
                "housing": "fixed",
                "entertainment": "rising",
            },
        },
        "existing_budget": {},
        "current_date": "2026-04-16",
    }

    result = run_budget_planning_agent(input_data)

    assert "budget_allocations" in result
    assert "progress" in result
    assert "warnings" in result
    assert "summary" in result

    assert "food" in result["budget_allocations"]
    assert "food" in result["progress"]
    assert isinstance(result["warnings"], list)
    assert isinstance(result["summary"], str)


def test_run_budget_planning_agent_invalid_current_date():
    input_data = {
        "monthly_income": 5000,
        "transactions": [],
        "expense_analysis": {
            "category_monthly_avg": {"food": 500},
            "category_trends": {"food": "stable"},
        },
        "existing_budget": {},
        "current_date": "16-04-2026",  # invalid format
    }

    with pytest.raises(ValueError):
        run_budget_planning_agent(input_data)


def test_run_budget_planning_agent_invalid_expense_analysis_structure():
    input_data = {
        "monthly_income": 5000,
        "transactions": [],
        "expense_analysis": {
            "category_monthly_avg": ["food", 500],  # invalid
            "category_trends": {"food": "stable"},
        },
        "existing_budget": {},
        "current_date": "2026-04-16",
    }

    with pytest.raises(ValueError):
        run_budget_planning_agent(input_data)