from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate


logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-haiku-4-5"
_MODEL_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "config" / "model_registry.json"
_EXTRA_LOG_FIELDS = ("agent", "model", "thread_id", "guardrail", "event")

LANGSMITH_PROMPTS: dict[str, str] = {
    "intent_classifier":        "intent-classifier:v1",
    "expense_categoriser":      "expense-categoriser:v1",
    "anomaly_explainer":        "anomaly-explainer:v1",
    "budget_request_extractor": "budget-request-extractor:v1",
    "goal_extractor":           "goal-extractor-main:v1",
    "health_advisory":          "health-advisory:v2",
}


@lru_cache(maxsize=None)
def _get_local_prompt(name: str) -> ChatPromptTemplate:
    templates: dict[str, ChatPromptTemplate] = {
        "intent_classifier": ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are the routing brain of a personal finance AI assistant. "
                    "Choose exactly one agent from: expense_analysis, budget_planning, goal_planning, anomaly_detection, health_assessment, unknown.",
                ),
                (
                    "human",
                    "User message: {message}",
                ),
            ]
        ),
        "expense_categoriser": ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Categorise each transaction into exactly one allowed category. Allowed categories: {category_values}. "
                    "Return structured results keyed by transaction_id.",
                ),
                (
                    "human",
                    "Transactions:\n{transactions_text}",
                ),
            ]
        ),
        "anomaly_explainer": ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Explain flagged financial transactions in plain English using only the supplied statistical reasons.",
                ),
                (
                    "human",
                    "Flagged transactions:\n{flagged_transactions_text}",
                ),
            ]
        ),
        "budget_request_extractor": ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Extract a structured budget planning request. Supported categories: {supported_categories}. "
                    "Use the provided state income only when the user did not mention income.",
                ),
                (
                    "human",
                    "Latest user message: {last_message}\nState monthly income: {state_income}",
                ),
            ]
        ),
        "goal_extractor": ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Today's date is {today}. Determine whether the user is expressing a financial savings goal. "
                    "Extract is_goal_intent, name, target_amount, target_date, current_amount, and missing_fields. "
                    "Resolve relative dates to concrete future dates when possible.",
                ),
                (
                    "human",
                    "User message: {user_message}",
                ),
            ]
        ),
        "health_advisory": ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Generate 2-4 plain-English financial health observations grounded only in the provided metrics and trends.",
                ),
                (
                    "human",
                    "Rating: {rating}\nDTI: {dti}\nReserve months: {reserve_months}\n"
                    "Income concentration risk: {concentration_risk}\nOverspending: {overspending}\n"
                    "Monthly income: {monthly_income}\nSpending trends:\n{trends_text}",
                ),
            ]
        ),
    }
    return templates[name]


@lru_cache(maxsize=None)
def get_prompt(name: str):
    try:
        from langsmith import Client

        return Client().pull_prompt(LANGSMITH_PROMPTS[name])
    except Exception as exc:
        logger.warning("Falling back to local prompt '%s': %s", name, exc)
        return _get_local_prompt(name)


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def load_model_registry() -> dict[str, Any]:
    if _MODEL_REGISTRY_PATH.exists():
        with _MODEL_REGISTRY_PATH.open(encoding="utf-8") as handle:
            return json.load(handle)

    return {
        "schema_version": 1,
        "default_alias": "default",
        "approved_models": {
            "default": {
                "provider": "anthropic",
                "model": _DEFAULT_MODEL,
                "version": "fallback",
                "stage": "prod",
            }
        },
    }


def get_default_model_name() -> str:
    registry = load_model_registry()
    default_alias = registry.get("default_alias", "default")
    approved_models = registry.get("approved_models", {})
    default_entry = approved_models.get(default_alias, {})
    return default_entry.get("model", _DEFAULT_MODEL)


def is_model_approved(model_name: str) -> bool:
    registry = load_model_registry()
    approved_models = registry.get("approved_models", {})
    return any(entry.get("model") == model_name for entry in approved_models.values())


def resolve_model_name(
    requested_model: str | None = None,
    *,
    strict: bool | None = None,
) -> str:
    registry = load_model_registry()
    approved_models = registry.get("approved_models", {})
    strict_mode = _is_truthy(os.getenv("SMARTFIN_ENFORCE_APPROVED_MODELS")) if strict is None else strict

    if not requested_model:
        return get_default_model_name()

    if requested_model in approved_models:
        return approved_models[requested_model].get("model", get_default_model_name())

    if is_model_approved(requested_model):
        return requested_model

    if strict_mode:
        fallback_model = get_default_model_name()
        logger.warning(
            "Requested model '%s' is not approved; falling back to '%s'.",
            requested_model,
            fallback_model,
        )
        return fallback_model

    return requested_model


def get_monitoring_settings() -> dict[str, Any]:
    return {
        "langsmith_tracing": _is_truthy(os.getenv("LANGCHAIN_TRACING_V2")),
        "langsmith_project": os.getenv("LANGCHAIN_PROJECT", "smartfin"),
        "log_level": os.getenv("LOG_LEVEL", "INFO").upper(),
        "log_format": os.getenv("SMARTFIN_LOG_FORMAT", "plain").lower(),
    }


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for field in _EXTRA_LOG_FIELDS:
            if hasattr(record, field):
                payload[field] = getattr(record, field)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def configure_logging(*, force: bool = False) -> logging.Logger:
    settings = get_monitoring_settings()
    level_name = settings["log_level"]
    level = getattr(logging, level_name, logging.INFO)
    root_logger = logging.getLogger()

    if root_logger.handlers and not force:
        root_logger.setLevel(level)
        return root_logger

    handler = logging.StreamHandler()
    if settings["log_format"] == "json":
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )

    root_logger.handlers = [handler]
    root_logger.setLevel(level)
    return root_logger
