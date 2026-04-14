"""Unit tests for anomaly detection — detector.py and agent.py."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from app.agents.anomaly_detection.agent import run
from app.agents.anomaly_detection.detector import (
    FREQUENCY_THRESHOLD,
    MIN_SAMPLE_SIZE,
    _detect_unusual_amounts,
    _detect_unusual_frequency,
    _llm_verify_candidates,
    detect_anomalies,
)
from app.state import AnomalyFlag, AnomalyType, Transaction, TransactionCategory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _txn(
    amount: float,
    category: TransactionCategory = TransactionCategory.FOOD,
    merchant: str = "TestMerchant",
    days_ago: float = 1,
    txn_id: str | None = None,
) -> Transaction:
    return Transaction(
        id=txn_id or str(uuid.uuid4()),
        date=datetime.now(tz=timezone.utc) - timedelta(days=days_ago),
        amount=amount,
        description="test transaction",
        merchant=merchant,
        category=category,
    )


def _food_batch(amounts: list[float]) -> list[Transaction]:
    """Return FOOD transactions with given amounts, spread one per day."""
    return [_txn(a, days_ago=i + 1) for i, a in enumerate(amounts)]


def _merchant_txns(
    count: int,
    merchant: str = "Coffee Shop",
    start_days_ago: int = 6,
) -> list[Transaction]:
    """Return `count` expense transactions to the same merchant, one per day."""
    return [
        _txn(5.0, merchant=merchant, days_ago=start_days_ago - i, txn_id=f"{merchant}-{i}")
        for i in range(count)
    ]


def _passthrough_llm(candidates, transactions):
    """Mock for _llm_verify_candidates that approves all statistical candidates."""
    return candidates, True


# ---------------------------------------------------------------------------
# TestDetectUnusualAmounts
# ---------------------------------------------------------------------------


class TestDetectUnusualAmounts:
    def test_flags_outlier_above_fence(self):
        """A transaction far above Q3 + 1.5*IQR is flagged as UNUSUAL_AMOUNT.

        Batch [10, 11, 12, 13, 14, 200]:
          q1=11, q3=14, iqr=3, fence=18.5 → 200 is flagged.
        """
        outlier = _txn(200.0, txn_id="outlier")
        txns = _food_batch([10, 11, 12, 13, 14]) + [outlier]
        flags = _detect_unusual_amounts(txns)
        assert any(f.transaction_id == "outlier" for f in flags)

    def test_anomaly_type_is_unusual_amount(self):
        outlier = _txn(200.0, txn_id="outlier")
        txns = _food_batch([10, 11, 12, 13, 14]) + [outlier]
        flags = _detect_unusual_amounts(txns)
        assert all(f.anomaly_type == AnomalyType.UNUSUAL_AMOUNT for f in flags)

    def test_no_flag_when_all_amounts_normal(self):
        """Transactions within the fence produce no flags."""
        txns = _food_batch([10, 11, 12, 13, 14, 15])
        assert _detect_unusual_amounts(txns) == []

    def test_skips_category_below_min_sample_size(self):
        """Fewer than MIN_SAMPLE_SIZE transactions per category → no detection."""
        txns = _food_batch([10, 200, 300])  # only 3 entries
        assert len(txns) < MIN_SAMPLE_SIZE
        assert _detect_unusual_amounts(txns) == []

    def test_income_excluded(self):
        """Transactions with amount <= 0 (income) are not evaluated."""
        txns = [_txn(-5000.0, txn_id=f"income-{i}") for i in range(6)]
        assert _detect_unusual_amounts(txns) == []

    def test_explanation_mentions_category(self):
        outlier = _txn(500.0, category=TransactionCategory.SHOPPING, txn_id="outlier")
        normals = [_txn(20.0, category=TransactionCategory.SHOPPING, days_ago=i + 2) for i in range(5)]
        flags = _detect_unusual_amounts(normals + [outlier])
        assert flags, "expected at least one flag"
        assert "shopping" in flags[0].explanation.lower()

    def test_categories_evaluated_independently(self):
        """An amount that is an outlier in FOOD should not flag the same amount
        in TRANSPORT where it is a normal value."""
        food_outlier = _txn(200.0, category=TransactionCategory.FOOD, txn_id="food-outlier")
        food_normals = [
            _txn(a, category=TransactionCategory.FOOD, days_ago=i + 1)
            for i, a in enumerate([10, 11, 12, 13, 14])
        ]
        # TRANSPORT batch where 200 is the normal value — no outlier
        transport = [
            _txn(200.0, category=TransactionCategory.TRANSPORT, txn_id=f"t-{i}", days_ago=i + 1)
            for i in range(6)
        ]
        flags = _detect_unusual_amounts(food_normals + [food_outlier] + transport)
        flagged_ids = {f.transaction_id for f in flags}
        assert "food-outlier" in flagged_ids
        assert not any(fid.startswith("t-") for fid in flagged_ids)


# ---------------------------------------------------------------------------
# TestDetectUnusualFrequency
# ---------------------------------------------------------------------------


class TestDetectUnusualFrequency:
    def test_flags_excess_visits_in_window(self):
        """More than FREQUENCY_THRESHOLD visits within the window → excess flagged."""
        txns = _merchant_txns(FREQUENCY_THRESHOLD + 1)
        flags = _detect_unusual_frequency(txns)
        assert len(flags) == 1
        assert flags[0].anomaly_type == AnomalyType.UNUSUAL_FREQUENCY

    def test_no_flag_at_exactly_threshold(self):
        """Exactly FREQUENCY_THRESHOLD visits → no flag."""
        txns = _merchant_txns(FREQUENCY_THRESHOLD)
        assert _detect_unusual_frequency(txns) == []

    def test_no_flag_below_threshold(self):
        txns = _merchant_txns(FREQUENCY_THRESHOLD - 1)
        assert _detect_unusual_frequency(txns) == []

    def test_income_excluded_from_frequency(self):
        """Negative-amount (income) entries do not count toward frequency."""
        txns = [_txn(-5.0, merchant="Employer", days_ago=i) for i in range(FREQUENCY_THRESHOLD + 2)]
        assert _detect_unusual_frequency(txns) == []

    def test_excess_not_double_flagged(self):
        """A transaction appearing in multiple rolling windows is flagged only once."""
        txns = _merchant_txns(FREQUENCY_THRESHOLD + 2)
        flags = _detect_unusual_frequency(txns)
        flagged_ids = [f.transaction_id for f in flags]
        assert len(flagged_ids) == len(set(flagged_ids))

    def test_merchant_comparison_case_insensitive(self):
        """'Starbucks' and 'starbucks' are treated as the same merchant."""
        upper = [_txn(5.0, merchant="Starbucks", days_ago=6 - i, txn_id=f"upper-{i}") for i in range(3)]
        lower = [_txn(5.0, merchant="starbucks", days_ago=2 - i, txn_id=f"lower-{i}") for i in range(3)]
        flags = _detect_unusual_frequency(upper + lower)
        assert len(flags) >= 1

    def test_visits_outside_window_not_grouped(self):
        """Transactions separated by more than the window are not combined."""
        # 3 visits in week 1, 3 visits 3 weeks later — no window exceeds threshold
        week1 = [_txn(5.0, merchant="Cafe", days_ago=21 - i, txn_id=f"w1-{i}") for i in range(3)]
        week3 = [_txn(5.0, merchant="Cafe", days_ago=2 - i, txn_id=f"w3-{i}") for i in range(3)]
        assert _detect_unusual_frequency(week1 + week3) == []


# ---------------------------------------------------------------------------
# TestDetectAnomalies (public API)
# ---------------------------------------------------------------------------


class TestDetectAnomalies:
    def test_empty_input_returns_empty(self):
        assert detect_anomalies([]) == []

    @patch(
        "app.agents.anomaly_detection.detector._llm_verify_candidates",
        side_effect=_passthrough_llm,
    )
    def test_combines_amount_and_frequency_flags(self, _mock):
        """Both detectors contribute to the combined result."""
        outlier = _txn(500.0, category=TransactionCategory.FOOD, txn_id="outlier")
        food_normals = _food_batch([10, 11, 12, 13, 14])
        freq_txns = _merchant_txns(FREQUENCY_THRESHOLD + 1, merchant="Cafe")
        flags = detect_anomalies(food_normals + [outlier] + freq_txns)
        types = {f.anomaly_type for f in flags}
        assert AnomalyType.UNUSUAL_AMOUNT in types
        assert AnomalyType.UNUSUAL_FREQUENCY in types

    @patch(
        "app.agents.anomaly_detection.detector._llm_verify_candidates",
        side_effect=_passthrough_llm,
    )
    def test_result_sorted_by_id_and_type(self, _mock):
        """Output is deterministically sorted by (transaction_id, anomaly_type)."""
        outlier = _txn(500.0, txn_id="zzz-outlier")
        txns = _food_batch([10, 11, 12, 13, 14]) + [outlier]
        flags = detect_anomalies(txns)
        keys = [(f.transaction_id, f.anomaly_type) for f in flags]
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# TestAnomalyDetectionAgentNode
# ---------------------------------------------------------------------------


class TestAnomalyDetectionAgentNode:
    def test_writes_anomaly_flags_to_state(self):
        result = run({"transactions": []})
        assert "anomaly_flags" in result
        assert isinstance(result["anomaly_flags"], list)

    def test_empty_state_returns_empty_flags(self):
        assert run({})["anomaly_flags"] == []

    @patch(
        "app.agents.anomaly_detection.detector._llm_verify_candidates",
        side_effect=_passthrough_llm,
    )
    def test_prefers_categorised_transactions(self, _mock):
        """When both keys are present, categorised_transactions takes priority."""
        outlier = _txn(500.0, category=TransactionCategory.FOOD, txn_id="outlier")
        food_normals = _food_batch([10, 11, 12, 13, 14])
        state = {
            "transactions": [],
            "categorised_transactions": food_normals + [outlier],
        }
        result = run(state)
        assert any(f.transaction_id == "outlier" for f in result["anomaly_flags"])

    @patch(
        "app.agents.anomaly_detection.detector._llm_verify_candidates",
        side_effect=_passthrough_llm,
    )
    def test_falls_back_to_raw_transactions(self, _mock):
        """When categorised_transactions is absent, raw transactions are used."""
        outlier = _txn(500.0, category=TransactionCategory.FOOD, txn_id="outlier")
        state = {"transactions": _food_batch([10, 11, 12, 13, 14]) + [outlier]}
        result = run(state)
        assert any(f.transaction_id == "outlier" for f in result["anomaly_flags"])

    @patch(
        "app.agents.anomaly_detection.detector._llm_verify_candidates",
        side_effect=_passthrough_llm,
    )
    def test_falls_back_when_categorised_is_empty_list(self, _mock):
        """An empty categorised_transactions triggers fallback to raw transactions."""
        outlier = _txn(500.0, category=TransactionCategory.FOOD, txn_id="outlier")
        state = {
            "transactions": _food_batch([10, 11, 12, 13, 14]) + [outlier],
            "categorised_transactions": [],
        }
        result = run(state)
        assert any(f.transaction_id == "outlier" for f in result["anomaly_flags"])


# ---------------------------------------------------------------------------
# TestLLMVerifyCandidates
# ---------------------------------------------------------------------------


class TestLLMVerifyCandidates:
    def _flag(self, txn_id: str, anomaly_type: AnomalyType = AnomalyType.UNUSUAL_AMOUNT) -> AnomalyFlag:
        return AnomalyFlag(
            transaction_id=txn_id,
            anomaly_type=anomaly_type,
            explanation="statistical reason",
        )

    def test_empty_candidates_skips_llm(self):
        """No LLM call when the candidate list is empty."""
        with patch("app.agents.anomaly_detection.detector.ChatAnthropic") as mock_llm:
            result, ok = _llm_verify_candidates([], [])
        assert result == []
        assert ok is True
        mock_llm.assert_not_called()

    def test_llm_confirms_candidate_and_replaces_explanation(self):
        """LLM is_anomaly=True → flag kept with LLM explanation overriding statistical one."""
        outlier = _txn(500.0, txn_id="outlier")
        flag = self._flag("outlier")

        verdict_batch = MagicMock()
        verdict_batch.results = [
            MagicMock(transaction_id="outlier", is_anomaly=True, explanation="LLM verdict")
        ]

        with patch("app.agents.anomaly_detection.detector.ChatAnthropic") as MockLLM:
            MockLLM.return_value.with_structured_output.return_value.invoke.return_value = verdict_batch
            result, ok = _llm_verify_candidates([flag], [outlier])

        assert ok is True
        assert len(result) == 1
        assert result[0].explanation == "LLM verdict"

    def test_llm_rejects_candidate(self):
        """LLM is_anomaly=False → flag is dropped (false positive filtered out)."""
        outlier = _txn(500.0, txn_id="outlier")
        flag = self._flag("outlier")

        verdict_batch = MagicMock()
        verdict_batch.results = [
            MagicMock(transaction_id="outlier", is_anomaly=False, explanation="Plausible purchase")
        ]

        with patch("app.agents.anomaly_detection.detector.ChatAnthropic") as MockLLM:
            MockLLM.return_value.with_structured_output.return_value.invoke.return_value = verdict_batch
            result, ok = _llm_verify_candidates([flag], [outlier])

        assert ok is True
        assert result == []

    def test_llm_failure_falls_back_to_statistical_candidates(self):
        """LLM exception → llm_succeeded=False and all statistical candidates returned."""
        outlier = _txn(500.0, txn_id="outlier")
        flag = self._flag("outlier")

        with patch("app.agents.anomaly_detection.detector.ChatAnthropic") as MockLLM:
            # 模拟API调用失败，抛出异常
            MockLLM.return_value.with_structured_output.return_value.invoke.side_effect = RuntimeError("API down")
            result, ok = _llm_verify_candidates([flag], [outlier])

        assert ok is False
        # 正常应该降级
        assert result == [flag]

    def test_missing_verdict_keeps_flag_conservatively(self):
        """If the LLM omits a transaction id, the flag is kept (conservative fallback)."""
        outlier = _txn(500.0, txn_id="outlier")
        flag = self._flag("outlier")

        verdict_batch = MagicMock()
        verdict_batch.results = []  # LLM returned no verdict for this id

        with patch("app.agents.anomaly_detection.detector.ChatAnthropic") as MockLLM:
            MockLLM.return_value.with_structured_output.return_value.invoke.return_value = verdict_batch
            result, ok = _llm_verify_candidates([flag], [outlier])

        assert ok is True
        assert len(result) == 1
        assert result[0].transaction_id == "outlier"
        assert result[0].explanation == "statistical reason"
