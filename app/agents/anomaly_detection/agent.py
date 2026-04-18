"""
Anomaly Detection agent — LangGraph node entry point.

Reads categorised_transactions from state (populated by expense_analysis) and
falls back to raw transactions when invoked as a standalone service.
Writes anomaly_flags and anomaly_explanation to state. No HITL — detection is
diagnostic/read-only.
"""

from __future__ import annotations

from app.agents.anomaly_detection.extractor import extract_and_detect
from app.state import AppState


def run(state: AppState) -> dict:
    """
    LangGraph node entry point for anomaly detection.

    Prefers categorised_transactions (richer category data from expense_analysis).
    Falls back to raw transactions when invoked standalone or before expense_analysis
    has run.

    Delegates to extract_and_detect() which runs statistical detection and generates
    natural-language explanations via LLM.
    """
    transactions = state.get("categorised_transactions") or state.get("transactions") or []
    messages = state.get("messages", [])

    anomaly_flags, anomaly_explanation = extract_and_detect(messages, transactions)

    print(
        f"[anomaly_detection] Scanned {len(transactions)} transactions, "
        f"found {len(anomaly_flags)} anomaly flag(s)."
    )

    return {
        "anomaly_flags": anomaly_flags,
        "anomaly_explanation": anomaly_explanation,
    }
