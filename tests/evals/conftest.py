"""Shared fixtures for deepeval LLM evaluation tests."""

from __future__ import annotations

import importlib.util
import os
from datetime import datetime, timedelta, timezone

import pytest
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

try:
    from deepeval.models.base_model import DeepEvalBaseLLM
except ModuleNotFoundError:
    class DeepEvalBaseLLM:  # type: ignore[no-redef]
        pass

from app.state import Transaction, TransactionCategory

load_dotenv()


def _eval_dependencies_available() -> tuple[bool, str]:
    if importlib.util.find_spec("deepeval") is None:
        return False, "deepeval is not installed"

    if not os.getenv("ANTHROPIC_API_KEY"):
        return False, "ANTHROPIC_API_KEY is not set"

    return True, ""


def pytest_ignore_collect(collection_path, config: pytest.Config) -> bool:
    normalized_path = str(collection_path).replace("\\", "/")
    if "/tests/evals/" not in normalized_path:
        return False

    available, _ = _eval_dependencies_available()
    return not available


def pytest_report_header(config: pytest.Config) -> str | None:
    available, reason = _eval_dependencies_available()
    if available:
        return None

    return f"eval tests not collected: {reason}"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "eval: deepeval LLM evaluation tests (exclude with -m 'not eval')"
    )


class ClaudeJudge(DeepEvalBaseLLM):
    """Thin wrapper so deepeval uses Claude as its judge model instead of OpenAI."""

    def load_model(self) -> ChatAnthropic:
        return ChatAnthropic(model="claude-haiku-4-5")

    def generate(self, prompt: str, *args, **kwargs) -> str:
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        result = await self.model.ainvoke(prompt)
        return result.content

    def get_model_name(self) -> str:
        return "claude-haiku-4-5"


@pytest.fixture(scope="session")
def judge() -> ClaudeJudge:
    return ClaudeJudge()


def _txn(
    txn_id: str,
    merchant: str,
    description: str,
    amount: float,
    category: TransactionCategory = TransactionCategory.OTHER,
    days_ago: int = 5,
) -> Transaction:
    dt = datetime.now(tz=timezone.utc) - timedelta(days=days_ago)
    return Transaction(
        id=txn_id,
        date=dt,
        amount=amount,
        description=description,
        merchant=merchant,
        category=category,
    )
