"""
Goal Planning extractor — uses Claude to understand the user's latest message
and extract structured financial goal information.

Design principle:
- LLM is used only for language understanding / field extraction
- deterministic financial calculations remain in tracker.py
- testing concerns should stay in the test layer, not in production code
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import date
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

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

def _build_prompt(user_message: str) -> str:
    """
    构造给 LLM 的提示词。

    这里尽量写得明确、结构化，方便模型稳定地产出 GoalExtractionResult。
    """
    return f"""
You are a financial planning assistant for a personal finance AI system.

Your task is to determine whether the user's message expresses a financial savings goal.
If yes, extract the relevant goal parameters.

Return structured data with these fields:
- is_goal_intent
- name
- target_amount
- target_date
- current_amount
- missing_fields

Rules:
1. A financial goal includes examples such as:
   - saving for a laptop
   - building an emergency fund
   - saving for a holiday / travel
   - saving for a house deposit
2. Required fields for creating a usable goal:
   - target_amount
   - target_date
3. If the user clearly has goal intent but omits one or more required fields:
   - set is_goal_intent = true
   - include the missing fields in missing_fields
4. If the message is not about a financial goal:
   - set is_goal_intent = false
   - leave other fields as null or empty
5. If current saved amount is not mentioned, leave current_amount as null.
6. Choose a short, natural goal name when possible.
7. Do NOT calculate monthly savings. Only extract information.
8. IMPORTANT — Relative and partial date expressions MUST be resolved to a concrete date
   using today's date from the system message. Do NOT mark target_date as missing simply
   because the user used a relative expression. Examples:
   - "end of this month"  → last day of the current month
   - "by April"           → last day of April of the nearest future year
   - "next month"         → last day of next month
   - "by end of year"     → December 31 of the current year
   - "in 3 months"        → today's date plus 3 months
   Only mark target_date as missing when the user provides NO date information at all.

User message:
\"\"\"{user_message}\"\"\"
""".strip()


# ---------------------------------------------------------------------------
# Fallback extraction
# ---------------------------------------------------------------------------

# 提取数字金额，例如 8000 / 1200.50
_AMOUNT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)")

# 仅支持简单的 YYYY-MM-DD 格式日期
_DATE_PATTERN = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def _fallback_extract(user_message: str) -> GoalExtractionResult:
    """
    当 LLM 调用失败时使用的轻量级兜底逻辑。

    这个 fallback 的目标不是“非常聪明”，而是：
    1. 能识别大致 goal intent
    2. 尽量提取金额和日期
    3. 给 agent 提供一个可继续处理的结构化结果
    """
    msg = user_message.lower()

    # 一组非常简单的关键词，用于粗粒度判断是否像是“财务目标”
    goal_keywords = [
        "save", "saving", "goal", "fund", "deposit",
        "emergency", "laptop", "travel", "holiday", "house"
    ]
    is_goal_intent = any(keyword in msg for keyword in goal_keywords)

    # ------------------------------------------------------------------
    # 先提取日期
    # 这样后面提取金额时，可以先把日期字符串从文本里去掉，
    # 避免把 2027-06-01 中的年份 2027 误识别成 target_amount
    # ------------------------------------------------------------------
    extracted_date: Optional[date] = None
    date_match = _DATE_PATTERN.search(user_message)
    message_without_date = user_message

    if date_match:
        try:
            date_str = date_match.group(1)
            extracted_date = date.fromisoformat(date_str)

            # 从原始消息中去掉日期片段，再做金额匹配
            message_without_date = user_message.replace(date_str, " ")
        except ValueError:
            extracted_date = None

    # ------------------------------------------------------------------
    # 再提取金额
    # 注意：这里使用“去掉日期后的文本”来做匹配，
    # 就不会把日期中的年份误当成金额了
    # ------------------------------------------------------------------
    extracted_amount: Optional[float] = None
    amount_match = _AMOUNT_PATTERN.search(message_without_date)
    if amount_match:
        try:
            extracted_amount = float(amount_match.group(1))
        except ValueError:
            extracted_amount = None

    # 根据关键词给一个比较自然的目标名称
    goal_name: Optional[str] = None
    if "laptop" in msg:
        goal_name = "Laptop Fund"
    elif "emergency" in msg:
        goal_name = "Emergency Fund"
    elif "travel" in msg or "holiday" in msg:
        goal_name = "Travel Fund"
    elif "house" in msg or "deposit" in msg:
        goal_name = "House Deposit Fund"
    elif is_goal_intent:
        # 如果看起来像 goal intent，但识别不出具体类别
        goal_name = "Financial Goal"

    # 如果用户表达了 goal intent，但缺少必要字段，就记录缺失项
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
    model_name = os.getenv("SMARTFIN_MODEL", "claude-haiku-4-5")

    try:
        llm = ChatAnthropic(model=model_name, timeout=_LLM_TIMEOUT)
        structured_llm = llm.with_structured_output(GoalExtractionResult)
    except Exception as exc:
        logger.warning("Failed to initialise LLM for goal extraction: %s", exc)
        return _fallback_extract(user_message), False

    system_msg = SystemMessage(
        content=(
            f"Today's date is {today.isoformat()}. "
            "When the user mentions a date without a year, "
            "always infer the nearest future date relative to today. "
            "Never resolve an ambiguous date to a date in the past."
        )
    )
    human_msg = HumanMessage(content=_build_prompt(user_message))
    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result: GoalExtractionResult = structured_llm.invoke([system_msg, human_msg])
            return result, True
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

    return _fallback_extract(user_message), False