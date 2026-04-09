"""
Unit tests for the Expense Analysis agent.

Categoriser tests use unittest.mock to patch ChatAnthropic so no real API
calls are made.  Analyser tests are pure Python — no mocking needed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.state import (
    AppState,
    Transaction,
    TransactionCategory,
)
from app.agents.expense_analysis.analyser import PERIOD_DAYS, compute_spending_trends
from app.agents.expense_analysis.categoriser import categorise_transactions
from app.agents.expense_analysis.agent import run as expense_analysis_run


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _tx(
    id_: str,
    days_ago: float,
    amount: float,
    category: TransactionCategory = TransactionCategory.OTHER,
) -> Transaction:
    """Create a Transaction with a date relative to now."""
    return Transaction(
        id=id_,
        date=datetime.now(tz=timezone.utc) - timedelta(days=days_ago),
        amount=amount,
        description="test transaction",
        merchant="Test Merchant",
        category=category,
    )


def _mock_chain(id_category_pairs: list[tuple[str, TransactionCategory]]) -> MagicMock:
    """Return a mock structured-output chain that yields the given classifications."""
    results = [
        SimpleNamespace(transaction_id=id_, category=cat)
        for id_, cat in id_category_pairs
    ]
    chain = MagicMock()
    chain.invoke.return_value = SimpleNamespace(results=results)
    return chain


# ---------------------------------------------------------------------------
# Analyser tests  (pure Python, no LLM)
# ---------------------------------------------------------------------------


class TestComputeSpendingTrends:

    def test_basic_deviation_calculation(self):
        txs = [
            _tx("a1", 5, 100.0, TransactionCategory.FOOD),   # current period
            _tx("a2", 35, 80.0, TransactionCategory.FOOD),   # previous period
        ]
        trends = compute_spending_trends(txs)
        food = next(t for t in trends if t.category == TransactionCategory.FOOD)
        assert food.current_period_total == 100.0
        assert food.previous_period_total == 80.0
        assert food.deviation_pct == pytest.approx(25.0)

    def test_decrease_gives_negative_deviation(self):
        txs = [
            _tx("b1", 5, 60.0, TransactionCategory.TRANSPORT),   # current
            _tx("b2", 35, 100.0, TransactionCategory.TRANSPORT),  # previous
        ]
        trends = compute_spending_trends(txs)
        t = trends[0]
        assert t.deviation_pct == pytest.approx(-40.0)

    def test_no_previous_period_returns_none(self):
        """Category appearing only in the current period → deviation_pct is None."""
        txs = [_tx("c1", 5, 50.0, TransactionCategory.HEALTHCARE)]
        trends = compute_spending_trends(txs)
        assert len(trends) == 1
        assert trends[0].deviation_pct is None

    def test_no_current_period_returns_none(self):
        """Category only in the previous period also has deviation_pct = None."""
        txs = [_tx("d1", 45, 50.0, TransactionCategory.EDUCATION)]
        trends = compute_spending_trends(txs)
        assert trends[0].deviation_pct is None

    def test_income_excluded_from_trends(self):
        """Negative amounts (income) must not appear in spending trends."""
        txs = [_tx("e1", 5, -3000.0, TransactionCategory.INCOME)]
        assert compute_spending_trends(txs) == []

    def test_sorted_by_current_period_descending(self):
        txs = [
            _tx("f1", 5,  50.0, TransactionCategory.FOOD),
            _tx("f2", 5, 200.0, TransactionCategory.HOUSING),
            _tx("f3", 5, 100.0, TransactionCategory.TRANSPORT),
        ]
        trends = compute_spending_trends(txs)
        totals = [t.current_period_total for t in trends]
        assert totals == sorted(totals, reverse=True)

    def test_empty_input(self):
        assert compute_spending_trends([]) == []

    def test_boundary_transaction_at_period_edge(self):
        """A transaction exactly at PERIOD_DAYS ago falls in the current period."""
        txs = [_tx("g1", PERIOD_DAYS - 0.01, 75.0, TransactionCategory.SHOPPING)]
        trends = compute_spending_trends(txs)
        assert trends[0].current_period_total == 75.0
        assert trends[0].previous_period_total == 0.0


# ---------------------------------------------------------------------------
# Categoriser tests  (mock ChatAnthropic)
# ---------------------------------------------------------------------------


class TestCategoriseTransactions:

    def test_categories_are_assigned(self):
        txs = [
            Transaction(id="x1", date=datetime.now(tz=timezone.utc), amount=50.0,
                        description="Tesco shop", merchant="Tesco"),
            Transaction(id="x2", date=datetime.now(tz=timezone.utc), amount=20.0,
                        description="Bus pass", merchant="TfL"),
        ]
        chain = _mock_chain([
            ("x1", TransactionCategory.FOOD),
            ("x2", TransactionCategory.TRANSPORT),
        ])
        with patch("app.agents.expense_analysis.categoriser.ChatAnthropic") as MockLLM:
            MockLLM.return_value.with_structured_output.return_value = chain
            result, llm_ok = categorise_transactions(txs)

        assert result[0].category == TransactionCategory.FOOD
        assert result[1].category == TransactionCategory.TRANSPORT
        assert llm_ok is True

    def test_missing_id_defaults_to_other(self):
        """If the LLM omits a transaction id, that transaction falls back to OTHER."""
        txs = [
            Transaction(id="y1", date=datetime.now(tz=timezone.utc), amount=30.0,
                        description="Mystery charge", merchant="Unknown"),
        ]
        chain = _mock_chain([])  # LLM returns no results
        with patch("app.agents.expense_analysis.categoriser.ChatAnthropic") as MockLLM:
            MockLLM.return_value.with_structured_output.return_value = chain
            result, _ = categorise_transactions(txs)

        assert result[0].category == TransactionCategory.OTHER

    def test_empty_input_returns_empty(self):
        result, llm_ok = categorise_transactions([])
        assert result == []
        assert llm_ok is True

    def test_original_transactions_not_mutated(self):
        """categorise_transactions must return new objects; originals stay unchanged."""
        tx = Transaction(id="z1", date=datetime.now(tz=timezone.utc), amount=10.0,
                         description="Coffee", merchant="Cafe")
        chain = _mock_chain([("z1", TransactionCategory.FOOD)])
        with patch("app.agents.expense_analysis.categoriser.ChatAnthropic") as MockLLM:
            MockLLM.return_value.with_structured_output.return_value = chain
            result, _ = categorise_transactions([tx])

        assert tx.category == TransactionCategory.OTHER   # original untouched
        assert result[0].category == TransactionCategory.FOOD

    def test_llm_called_once_for_any_batch_size(self):
        """Only one LLM call is made regardless of how many transactions are passed."""
        txs = [
            Transaction(id=f"m{i}", date=datetime.now(tz=timezone.utc), amount=10.0,
                        description="tx", merchant="M")
            for i in range(10)
        ]
        chain = _mock_chain([(f"m{i}", TransactionCategory.OTHER) for i in range(10)])
        with patch("app.agents.expense_analysis.categoriser.ChatAnthropic") as MockLLM:
            MockLLM.return_value.with_structured_output.return_value = chain
            categorise_transactions(txs)

        chain.invoke.assert_called_once()

    def test_retries_on_failure_then_succeeds(self):
        """LLM succeeds on the second attempt; invoke is called twice."""
        txs = [Transaction(id="r1", date=datetime.now(tz=timezone.utc), amount=10.0,
                           description="Coffee", merchant="Cafe")]
        chain = MagicMock()
        chain.invoke.side_effect = [
            RuntimeError("transient error"),   # attempt 1 fails
            SimpleNamespace(results=[          # attempt 2 succeeds
                SimpleNamespace(transaction_id="r1", category=TransactionCategory.FOOD)
            ]),
        ]
        with patch("app.agents.expense_analysis.categoriser.ChatAnthropic") as MockLLM:
            MockLLM.return_value.with_structured_output.return_value = chain
            with patch("app.agents.expense_analysis.categoriser.time.sleep"):
                result, llm_ok = categorise_transactions(txs)

        assert llm_ok is True
        assert result[0].category == TransactionCategory.FOOD
        assert chain.invoke.call_count == 2

    def test_falls_back_after_max_retries(self):
        """All LLM attempts fail → keyword fallback used, llm_ok=False."""
        txs = [Transaction(id="fb1", date=datetime.now(tz=timezone.utc), amount=50.0,
                           description="Weekly grocery shop", merchant="Tesco")]
        chain = MagicMock()
        chain.invoke.side_effect = RuntimeError("API down")

        with patch("app.agents.expense_analysis.categoriser.ChatAnthropic") as MockLLM:
            MockLLM.return_value.with_structured_output.return_value = chain
            with patch("app.agents.expense_analysis.categoriser.time.sleep"):
                result, llm_ok = categorise_transactions(txs)

        assert llm_ok is False
        assert chain.invoke.call_count == 3  # MAX_RETRIES
        assert result[0].category == TransactionCategory.FOOD  # keyword matched "tesco"

    def test_fallback_flags_income_by_negative_amount(self):
        """Fallback must classify negative amounts as income regardless of description."""
        txs = [Transaction(id="inc1", date=datetime.now(tz=timezone.utc), amount=-3200.0,
                           description="Monthly salary", merchant="Employer Ltd")]
        chain = MagicMock()
        chain.invoke.side_effect = RuntimeError("API down")

        with patch("app.agents.expense_analysis.categoriser.ChatAnthropic") as MockLLM:
            MockLLM.return_value.with_structured_output.return_value = chain
            with patch("app.agents.expense_analysis.categoriser.time.sleep"):
                result, llm_ok = categorise_transactions(txs)

        assert llm_ok is False
        assert result[0].category == TransactionCategory.INCOME


# ---------------------------------------------------------------------------
# Agent node test  (end-to-end with mocked LLM)
# ---------------------------------------------------------------------------


class TestExpenseAnalysisAgentNode:

    def _make_state(self, transactions: list[Transaction]) -> AppState:
        return {
            "messages": [],
            "transactions": transactions,
            "monthly_income": 3000.0,
            "categorised_transactions": [],
            "spending_trends": [],
            "budget_allocations": [],
            "goals": [],
            "anomaly_flags": [],
            "health_summary": None,
            "alerts": [],
            "pending_confirmation": None,
            "active_agent": "expense_analysis",
            "agents_queue": [],
        }

    def test_writes_required_state_fields(self):
        txs = [
            _tx("e1", 5, 100.0, TransactionCategory.FOOD),
            _tx("e2", 35, 80.0, TransactionCategory.FOOD),
        ]
        chain = _mock_chain([
            ("e1", TransactionCategory.FOOD),
            ("e2", TransactionCategory.FOOD),
        ])
        with patch("app.agents.expense_analysis.categoriser.ChatAnthropic") as MockLLM:
            MockLLM.return_value.with_structured_output.return_value = chain
            result = expense_analysis_run(self._make_state(txs))

        assert len(result["categorised_transactions"]) == 2
        assert len(result["spending_trends"]) == 1
        assert result["pending_confirmation"]["action"] == "approve_expense_analysis"
        assert result["pending_confirmation"]["categorisation_confidence"] == "llm"

    def test_hitl_payload_has_no_confirmed_key(self):
        """
        pending_confirmation must NOT include a 'confirmed' key so that
        route_after_agent correctly detects a pending HITL pause.
        """
        txs = [_tx("h1", 5, 50.0, TransactionCategory.SHOPPING)]
        chain = _mock_chain([("h1", TransactionCategory.SHOPPING)])
        with patch("app.agents.expense_analysis.categoriser.ChatAnthropic") as MockLLM:
            MockLLM.return_value.with_structured_output.return_value = chain
            result = expense_analysis_run(self._make_state(txs))

        assert "confirmed" not in result["pending_confirmation"]

    def test_empty_transactions_still_returns_valid_state(self):
        chain = _mock_chain([])
        with patch("app.agents.expense_analysis.categoriser.ChatAnthropic") as MockLLM:
            MockLLM.return_value.with_structured_output.return_value = chain
            result = expense_analysis_run(self._make_state([]))

        assert result["categorised_transactions"] == []
        assert result["spending_trends"] == []
        assert result["pending_confirmation"] is not None
