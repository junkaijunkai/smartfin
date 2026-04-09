"""
Shared AppState — the central contract between all agents in the SmartFin system.

All agents read from and write to this state object via the LangGraph Supervisor.
Changes to this schema must be coordinated across the whole team.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Annotated


from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TransactionCategory(str, Enum):
    FOOD = "food"
    TRANSPORT = "transport"
    HOUSING = "housing"
    ENTERTAINMENT = "entertainment"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    SHOPPING = "shopping"
    UTILITIES = "utilities"
    INCOME = "income"
    SAVINGS = "savings"
    OTHER = "other"


class AnomalyType(str, Enum):
    UNUSUAL_LOCATION = "unusual_location"
    UNUSUAL_AMOUNT = "unusual_amount"
    UNUSUAL_FREQUENCY = "unusual_frequency"
    UNUSUAL_TIME = "unusual_time"


class HealthRating(str, Enum):
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Core data models
# ---------------------------------------------------------------------------


class Transaction(BaseModel):
    id: str
    date: datetime
    amount: float = Field(description="Positive = expense, negative = income")
    description: str
    merchant: str
    category: TransactionCategory = TransactionCategory.OTHER
    location: str | None = None


class BudgetAllocation(BaseModel):
    category: TransactionCategory
    allocated_amount: float
    spent_amount: float = 0.0
    period_start: date
    period_end: date

    @property
    def remaining(self) -> float:
        return self.allocated_amount - self.spent_amount

    @property
    def utilisation_rate(self) -> float:
        if self.allocated_amount == 0:
            return 0.0
        return self.spent_amount / self.allocated_amount


class FinancialGoal(BaseModel):
    id: str
    name: str
    target_amount: float
    current_amount: float = 0.0
    target_date: date
    required_monthly_saving: float = 0.0
    on_track: bool = True


class AnomalyFlag(BaseModel):
    transaction_id: str
    anomaly_type: AnomalyType
    explanation: str
    flagged_at: datetime = Field(default_factory=datetime.utcnow)


class SpendingTrend(BaseModel):
    category: TransactionCategory
    current_period_total: float
    previous_period_total: float
    deviation_pct: float | None


class HealthSummary(BaseModel):
    rating: HealthRating
    debt_to_income_ratio: float
    liquid_reserve_months: float
    income_concentration_risk: bool
    sustained_overspending: bool
    observations: list[str] = Field(default_factory=list)
    assessed_at: datetime = Field(default_factory=datetime.utcnow)


class Alert(BaseModel):
    id: str
    severity: AlertSeverity
    source_agent: str
    message: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    acknowledged: bool = False


# ---------------------------------------------------------------------------
# LangGraph shared state
# ---------------------------------------------------------------------------


class AppState(TypedDict):
    # Conversation messages (managed by LangGraph add_messages reducer)
    messages: Annotated[list, add_messages]

    # --- Raw input data ---
    transactions: list[Transaction]
    monthly_income: float

    # --- Expense Analysis Agent outputs ---
    categorised_transactions: list[Transaction]
    spending_trends: list[SpendingTrend]

    # --- Budget Planning Agent outputs ---
    budget_allocations: list[BudgetAllocation]

    # --- Financial Goal Planning Agent outputs ---
    goals: list[FinancialGoal]

    # --- Transaction Anomaly Detection Agent outputs ---
    anomaly_flags: list[AnomalyFlag]

    # --- Financial Health and Risk Assessment Agent outputs ---
    health_summary: HealthSummary | None

    # --- Orchestrator-managed ---
    alerts: list[Alert]
    pending_confirmation: dict | None  # HITL(Human In The Loop): payload awaiting user confirmation
    active_agent: str | None    
    agents_queue: list[str] # 记录agent执行队列，从顶部移出，从尾部添加，等队列为空时才END
