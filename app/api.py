from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.config import configure_logging, get_default_model_name, get_monitoring_settings
from app.orchestrator import app_graph
from app.state import Transaction, TransactionCategory

load_dotenv()
configure_logging()
logger = logging.getLogger(__name__)
FIXTURE_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "sample_transactions.json"


class TransactionPayload(BaseModel):
    id: str
    date: str
    amount: float
    description: str
    merchant: str
    category: str = "other"
    location: str | None = None


class AnalyzeRequest(BaseModel):
    message: str
    thread_id: str | None = None
    monthly_income: float = 3200.0
    current_date: str | None = None
    use_sample_data: bool = True
    transactions: list[TransactionPayload] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    thread_id: str
    active_agent: str | None = None
    assistant_message: str | None = None
    pending_confirmation: dict | None = None
    alerts: list[dict] = Field(default_factory=list)
    summary: dict = Field(default_factory=dict)


app = FastAPI(title="SmartFin Backend", version="0.1.0")


def _safe_category(raw_value: str) -> TransactionCategory:
    try:
        return TransactionCategory(raw_value)
    except ValueError:
        return TransactionCategory.OTHER


def load_demo_transactions() -> list[Transaction]:
    with FIXTURE_PATH.open(encoding="utf-8") as handle:
        raw_transactions = json.load(handle)

    return [
        Transaction(
            id=item["id"],
            date=item["date"],
            amount=item["amount"],
            description=item["description"],
            merchant=item["merchant"],
            category=_safe_category(item.get("category", "other")),
            location=item.get("location"),
        )
        for item in raw_transactions
    ]


def parse_transactions(items: list[TransactionPayload]) -> list[Transaction]:
    return [
        Transaction(
            id=item.id,
            date=item.date,
            amount=item.amount,
            description=item.description,
            merchant=item.merchant,
            category=_safe_category(item.category),
            location=item.location,
        )
        for item in items
    ]


def extract_assistant_message(messages: list) -> str | None:
    for message in reversed(messages or []):
        if getattr(message, "type", None) == "ai":
            return getattr(message, "content", None)
    return None


def serialize_alerts(alerts: list) -> list[dict]:
    serialized: list[dict] = []
    for alert in alerts or []:
        if hasattr(alert, "model_dump"):
            serialized.append(alert.model_dump(mode="json"))
        else:
            serialized.append(dict(alert))
    return serialized


def summarize_state(state: dict) -> dict:
    health_summary = state.get("health_summary")
    health_rating = None
    if health_summary is not None:
        rating = getattr(health_summary, "rating", None)
        health_rating = getattr(rating, "value", rating)

    return {
        "categorised_transaction_count": len(state.get("categorised_transactions", [])),
        "spending_trend_count": len(state.get("spending_trends", [])),
        "goal_count": len(state.get("goals", [])),
        "anomaly_flag_count": len(state.get("anomaly_flags", [])),
        "health_rating": health_rating,
    }


def build_response(thread_id: str, state: dict) -> AnalyzeResponse:
    return AnalyzeResponse(
        thread_id=thread_id,
        active_agent=state.get("active_agent"),
        assistant_message=extract_assistant_message(state.get("messages", [])),
        pending_confirmation=state.get("pending_confirmation"),
        alerts=serialize_alerts(state.get("alerts", [])),
        summary=summarize_state(state),
    )


@app.get("/")
def root() -> dict:
    return {
        "service": "smartfin-backend",
        "status": "ok",
        "default_model": get_default_model_name(),
    }


@app.get("/health")
def health() -> dict:
    return {
        "service": "smartfin-backend",
        "status": "ok",
        "default_model": get_default_model_name(),
        "monitoring": get_monitoring_settings(),
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    thread_id = request.thread_id or str(uuid4())
    transactions = parse_transactions(request.transactions)
    if not transactions and request.use_sample_data:
        transactions = load_demo_transactions()

    initial_state = {
        "messages": [HumanMessage(content=request.message)],
        "monthly_income": request.monthly_income,
        "goals": [],
        "current_date": request.current_date or date.today().isoformat(),
    }
    if transactions:
        initial_state["transactions"] = transactions

    try:
        state = app_graph.invoke(initial_state, {"configurable": {"thread_id": thread_id}})
    except Exception as exc:
        logger.exception("SmartFin backend request failed.")
        raise HTTPException(status_code=500, detail="SmartFin backend request failed.") from exc

    return build_response(thread_id, state)
