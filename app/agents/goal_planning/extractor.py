"""
Goal Planning extractor — uses Claude to understand the user's latest message
and extract structured financial goal information.

Design principle:
- LLM is used only for language understanding / field extraction
- deterministic financial calculations remain in tracker.py
- testing concerns should stay in the test layer, not in production code
"""

from __future__ import annotations

import calendar
import inspect
import logging
import os
import re
import time
from datetime import date
from dateutil.relativedelta import relativedelta
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field
from app.config import resolve_model_name, get_prompt

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
_INITIAL_BACKOFF = 2.0  # seconds; doubles on each subsequent retry
_LLM_TIMEOUT = 30       # seconds


# ---------------------------------------------------------------------------
# Structured-output schema
# ---------------------------------------------------------------------------

class GoalExtractionResult(BaseModel):
    """
    结构化提取结果。

    字段说明：
    - is_goal_intent: 用户是否表达了一个“财务目标 / 储蓄目标”
    - name: 目标名称，例如 "Laptop Fund"
    - target_amount: 目标金额
    - target_date: 目标截止日期
    - current_amount: 当前已存金额（如果用户提到了）
    - missing_fields: 创建目标仍缺失的必要字段
    """
    is_goal_intent: bool = Field(
        description="True if the user is expressing or discussing a financial savings goal."
    )
    name: Optional[str] = Field(
        default=None,
        description="A concise goal name such as 'Laptop Fund' or 'Emergency Fund'."
    )
    target_amount: Optional[float] = Field(
        default=None,
        description="The target amount the user wants to accumulate."
    )
    target_date: Optional[date] = Field(
        default=None,
        description="The target date by which the user wants to reach the goal."
    )
    current_amount: Optional[float] = Field(
        default=None,
        description="The amount already saved toward this goal, if mentioned."
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Missing required fields, typically target_amount and/or target_date."
    )


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Fallback extraction
# ---------------------------------------------------------------------------

# 提取数字金额，支持货币符号前缀和千位逗号，例如 $8,000 / £1,200.50 / 640000
_CURRENCY_PREFIX = re.compile(r"[$£€¥₹]")
_AMOUNT_PATTERN = re.compile(r"(\d[\d,]*(?:\.\d+)?)")

# 仅支持简单的 YYYY-MM-DD 格式日期
_DATE_PATTERN = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def _extract_relative_date(msg_lower: str, today: date) -> Optional[date]:
    """Resolve common relative date expressions to a concrete date."""
    if re.search(r"end of next year|by next year|by the end of next year", msg_lower):
        return date(today.year + 1, 12, 31)
    if re.search(r"end of (this )?year|end of year|by year.?s? end|by end of year", msg_lower):
        return date(today.year, 12, 31)
    m = re.search(r"in (\d+) months?", msg_lower)
    if m:
        target = today + relativedelta(months=int(m.group(1)))
        return date(target.year, target.month, calendar.monthrange(target.year, target.month)[1])
    if re.search(r"next month", msg_lower):
        target = today + relativedelta(months=1)
        return date(target.year, target.month, calendar.monthrange(target.year, target.month)[1])
    month_names = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }
    m = re.search(r"by (" + "|".join(month_names) + r")", msg_lower)
    if m:
        month_num = month_names[m.group(1)]
        year = today.year if month_num > today.month else today.year + 1
        return date(year, month_num, calendar.monthrange(year, month_num)[1])
    return None


def _derive_goal_name(msg: str, is_goal_intent: bool) -> Optional[str]:
    if "laptop" in msg:
        return "Laptop Fund"
    if "emergency" in msg:
        return "Emergency Fund"
    if "travel" in msg or "holiday" in msg:
        return "Travel Fund"
    if "house" in msg or "deposit" in msg:
        return "House Deposit Fund"
    if is_goal_intent:
        return "Financial Goal"
    return None


def _goal_name_slug(value: Optional[str]) -> str:
    if not value:
        return ""
    normalized = re.sub(r"\b(fund|goal|savings|saving)\b", "", value.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _normalize_extraction(result: GoalExtractionResult, user_message: str) -> GoalExtractionResult:
    msg = user_message.lower()
    canonical_name = _derive_goal_name(msg, result.is_goal_intent)
    name = result.name

    if not result.is_goal_intent:
        name = None
        missing_fields: list[str] = []
    else:
        if canonical_name:
            name_slug = _goal_name_slug(name)
            canonical_slug = _goal_name_slug(canonical_name)
            if not name_slug or name_slug == canonical_slug or name_slug == "financial":
                name = canonical_name

        missing_fields = list(dict.fromkeys(result.missing_fields))
        if result.target_amount is None and "target_amount" not in missing_fields:
            missing_fields.append("target_amount")
        if result.target_date is None and "target_date" not in missing_fields:
            missing_fields.append("target_date")

    return result.model_copy(update={"name": name, "missing_fields": missing_fields})


def _fallback_extract(user_message: str, today: Optional[date] = None) -> GoalExtractionResult:
    """
    当 LLM 调用失败时使用的轻量级兜底逻辑。

    这个 fallback 的目标不是“非常聪明”，而是：
    1. 能识别大致 goal intent
    2. 尽量提取金额和日期
    3. 给 agent 提供一个可继续处理的结构化结果
    """
    msg = user_message.lower()
    today = today or date.today()

    # 一组非常简单的关键词，用于粗粒度判断是否像是”财务目标”
    goal_keywords = [
        "save", "saving", "savings", "goal", "fund", "deposit",
        "emergency", "laptop", "travel", "holiday", "house",
        "accumulate", "set aside", "put aside",
    ]
    is_goal_intent = any(keyword in msg for keyword in goal_keywords)

    # ------------------------------------------------------------------
    # 先提取日期（ISO 格式优先，再尝试相对日期表达）
    # 这样后面提取金额时，可以先把日期字符串从文本里去掉，
    # 避免把 2027-06-01 中的年份 2027 误识别成 target_amount
    # ------------------------------------------------------------------
    extracted_date: Optional[date] = None
    message_without_date = user_message

    date_match = _DATE_PATTERN.search(user_message)
    if date_match:
        try:
            date_str = date_match.group(1)
            extracted_date = date.fromisoformat(date_str)
            message_without_date = user_message.replace(date_str, "")  # 去掉日期部分，避免干扰金额提取
        except ValueError:
            extracted_date = None

    if extracted_date is None:
        extracted_date = _extract_relative_date(msg, today)

    # ------------------------------------------------------------------
    # 再提取金额
    # 先去掉货币符号，再去掉千位逗号，避免 $64,0000 被截断为 64
    # 注意：使用”去掉日期后的文本”，避免年份被误识别为金额
    # ------------------------------------------------------------------
    extracted_amount: Optional[float] = None
    normalized = _CURRENCY_PREFIX.sub("", message_without_date)
    amount_match = _AMOUNT_PATTERN.search(normalized)
    if amount_match:
        try:
            extracted_amount = float(amount_match.group(1).replace(",", ""))
        except ValueError:
            extracted_amount = None

    goal_name = _derive_goal_name(msg, is_goal_intent)

    missing_fields: list[str] = []
    if is_goal_intent:
        if extracted_amount is None:
            missing_fields.append("target_amount")
        if extracted_date is None:
            missing_fields.append("target_date")

    return GoalExtractionResult(
        is_goal_intent=is_goal_intent,
        name=goal_name,
        target_amount=extracted_amount,
        target_date=extracted_date,
        current_amount=None,
        missing_fields=missing_fields,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_goal_from_message(
    user_message: str,
    today: date | None = None,
) -> tuple[GoalExtractionResult, bool]:
    """
    从用户消息中提取结构化的 goal 信息。

    返回：
        (result, llm_succeeded)

    - result: 结构化提取结果
    - llm_succeeded:
        True  -> 真实 LLM 路径成功
        False -> LLM 调用失败，退回 fallback

    注意：
    - 空消息不算失败，直接返回一个”非 goal intent”的结果
    - 测试时不要在这里写 mock 逻辑，应该在 pytest 中 patch ChatAnthropic
    """

    # 空输入时，直接返回一个”没有 goal intent”的结果
    if not user_message.strip():
        return GoalExtractionResult(
            is_goal_intent=False,
            missing_fields=[],
        ), True

    today = today or date.today()

    # 允许通过环境变量覆盖模型名，但不再在生产代码里放 mock mode
    model_name = resolve_model_name(os.getenv("SMARTFIN_MODEL", "claude-haiku-4-5"))

    try:
        chat_signature = inspect.signature(ChatAnthropic)
        if "timeout" in chat_signature.parameters:
            llm = ChatAnthropic(model=model_name, timeout=_LLM_TIMEOUT)
        else:
            llm = ChatAnthropic(model=model_name)
        structured_llm = llm.with_structured_output(GoalExtractionResult)
    except Exception as exc:
        logger.warning("Failed to initialise LLM for goal extraction: %s", exc)
        return _fallback_extract(user_message, today=today), False
    
    # build prompt
    messages = get_prompt("goal_extractor").format_messages(
        today=today.isoformat(),
        user_message=user_message,
    )
    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result: GoalExtractionResult = structured_llm.invoke(messages)
            return _normalize_extraction(result, user_message), True
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                wait = _INITIAL_BACKOFF * (2 ** (attempt - 1))
                logger.warning(
                    "Goal extraction failed (attempt %d/%d), retrying in %.0fs: %s",
                    attempt, MAX_RETRIES, wait, exc,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "Goal extraction failed after %d attempts: %s",
                    MAX_RETRIES, last_exc,
                )

    return _fallback_extract(user_message, today=today), False