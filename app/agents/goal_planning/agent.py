"""
Goal Planning agent — LangGraph node entry point.

Responsibilities:
1. Read the latest user message from AppState.
2. Use extractor.py (LLM-based) to detect / extract a financial goal from natural language.
3. If enough information is available, create a new FinancialGoal.
4. Reuse tracker.py for deterministic required monthly saving calculation.
5. Compare required monthly saving against current monthly surplus.
6. Return updated goals plus a HITL confirmation payload.

Design principle:
- LLM handles language understanding only.
- tracker.py remains the deterministic financial calculation tool.
"""

from __future__ import annotations

import inspect
import logging
from datetime import date
from uuid import uuid4

from app.state import AppState, FinancialGoal
from app.agents.goal_planning.tracker import calculate_required_monthly_saving
from app.agents.goal_planning.extractor import extract_goal_from_message, GoalExtractionResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation layer
# ---------------------------------------------------------------------------


def _validate_extraction_consistency(extraction: GoalExtractionResult) -> tuple[bool, str]:
    """
    验证提取结果的一致性：missing_fields 应该准确反映缺失的字段。

    Returns:
        (is_valid, error_message)
    """
    if not extraction.is_goal_intent:
        return True, ""

    actual_missing = []
    if extraction.target_amount is None:
        actual_missing.append("target_amount")
    if extraction.target_date is None:
        actual_missing.append("target_date")

    # 检查 missing_fields 与实际缺失字段是否一致
    expected_missing = set(extraction.missing_fields)
    actual_missing_set = set(actual_missing)

    if expected_missing != actual_missing_set:
        error = (
            f"Extraction inconsistency: missing_fields={list(expected_missing)} "
            f"but actual missing fields={actual_missing}. "
            f"This may indicate a bug in the extractor or LLM output."
        )
        logger.warning(error)
        return False, error

    return True, ""


def _validate_goal_creation_fields(extraction: GoalExtractionResult) -> tuple[bool, str]:
    """
    验证 Goal 创建所需的字段完整性。

    Goal 创建前的必要条件：
    - is_goal_intent = True
    - missing_fields = [] (空列表，所有必要字段都存在)
    - target_amount 不为 None 且 > 0
    - target_date 不为 None 且在未来

    Returns:
        (is_valid, error_message)
    """
    if not extraction.is_goal_intent:
        return False, "Not a goal intent"

    if extraction.missing_fields:
        return False, f"Missing required fields: {', '.join(extraction.missing_fields)}"

    if extraction.target_amount is None:
        return False, "target_amount is None despite missing_fields being empty"

    if extraction.target_amount <= 0:
        return (
            False,
            f"target_amount must be positive, got {extraction.target_amount}"
        )

    if extraction.target_date is None:
        return False, "target_date is None despite missing_fields being empty"

    if extraction.target_date <= date.today():
        return (
            False,
            f"target_date must be in the future, got {extraction.target_date}"
        )

    return True, ""


def _get_latest_message_text(state: AppState) -> str:
    """
    Extract the latest message text from LangGraph state.

    Messages are typically HumanMessage / AIMessage objects.
    We only need the content text here.
    """
    messages = state.get("messages", []) or []
    if not messages:
        return ""

    last_message = messages[-1]
    content = getattr(last_message, "content", None)

    if isinstance(content, str):
        return content

    return str(last_message)


def _build_goal_from_extraction(extraction: GoalExtractionResult) -> tuple[FinancialGoal | None, str]:
    """
    Convert the structured extraction result into a FinancialGoal object.

    验证所有必要字段后才创建 Goal 对象。

    Returns:
        (goal_object, error_message)
        - goal_object: 创建成功时返回 FinancialGoal，失败时返回 None
        - error_message: 验证失败时的错误信息，成功时为空字符串
    """
    # 第一步：验证提取结果的一致性
    is_consistent, consistency_error = _validate_extraction_consistency(extraction)
    if not is_consistent:
        return None, consistency_error

    # 第二步：验证 Goal 创建所需的所有字段
    is_valid, validation_error = _validate_goal_creation_fields(extraction)
    if not is_valid:
        return None, validation_error

    # 第三步：所有验证通过，创建 Goal 对象
    try:
        goal = FinancialGoal(
            id=f"goal-{uuid4().hex[:8]}",
            name=extraction.name or "Financial Goal",
            target_amount=float(extraction.target_amount),
            current_amount=float(extraction.current_amount or 0.0),
            target_date=extraction.target_date,
        )
        return goal, ""
    except (ValueError, TypeError) as e:
        error_msg = f"Failed to create FinancialGoal: {str(e)}"
        logger.error(error_msg)
        return None, error_msg


def run(state: AppState) -> dict:
    """
    Main entry point for the Goal Planning agent.

    Reads:
    - state["messages"]
    - state["goals"]
    - state["monthly_income"]
    - state["budget_allocations"]

    Writes:
    - goals
    - pending_confirmation
    """
    # ------------------------------------------------------------------
    # Step 1: Read current state
    # ------------------------------------------------------------------
    goals = list(state.get("goals") or [])
    monthly_income = state.get("monthly_income") or 0.0
    budget_allocations = state.get("budget_allocations") or []

    # 已花预算总额
    total_spent = sum(allocation.spent_amount for allocation in budget_allocations)

    # 当前月可支配结余
    monthly_surplus = monthly_income - total_spent

    # ------------------------------------------------------------------
    # Step 2: Extract user goal intent from latest message
    # ------------------------------------------------------------------
    latest_message = _get_latest_message_text(state)
    current_date_str = state.get("current_date")
    today = date.fromisoformat(current_date_str) if current_date_str else date.today()
    extract_signature = inspect.signature(extract_goal_from_message)
    if "today" in extract_signature.parameters:
        extraction, llm_succeeded = extract_goal_from_message(latest_message, today=today)
    else:
        extraction, llm_succeeded = extract_goal_from_message(latest_message)

    # ------------------------------------------------------------------
    # Step 3: If user is expressing a new goal, handle creation logic
    # ------------------------------------------------------------------
    creation_detail_lines: list[str] = []
    new_goal_added = False

    if extraction.is_goal_intent:
        # 目标意图明确，但信息不足 → 先提示补充，不创建
        if extraction.missing_fields:
            pending_confirmation = {
                "action": "clarify_goal_planning",
                "agent": "goal_planning",
                "summary": "I detected a financial goal request, but some required information is missing.",
                "details": [
                    f"Detected goal name: {extraction.name or 'N/A'}",
                    f"Missing fields: {', '.join(extraction.missing_fields)}",
                    "Please confirm or provide the missing details before continuing.",
                ],
                "goal_extraction_confidence": "llm" if llm_succeeded else "fallback",
            }

            return {
                "goals": goals,
                "pending_confirmation": pending_confirmation,
            }

        # 信息完整 → 尝试创建新目标
        new_goal, creation_error = _build_goal_from_extraction(extraction)

        if new_goal is None:
            # 创建失败（例如数据不一致或验证失败）
            logger.error(
                f"Failed to create goal from extraction: {creation_error}. "
                f"Extraction: {extraction}"
            )
            pending_confirmation = {
                "action": "clarify_goal_planning",
                "agent": "goal_planning",
                "summary": "Goal extraction validation failed.",
                "details": [
                    f"Error: {creation_error}",
                    "This may indicate a data issue. Please try again with clearer information.",
                ],
                "goal_extraction_confidence": "llm" if llm_succeeded else "fallback",
            }

            return {
                "goals": goals,
                "pending_confirmation": pending_confirmation,
            }

        # 创建成功 → 加入现有 goals
        goals.append(new_goal)
        new_goal_added = True

        creation_detail_lines.append(
            f"Created new goal: {new_goal.name} "
            f"(target={new_goal.target_amount:.2f}, "
            f"current={new_goal.current_amount:.2f}, "
            f"target_date={new_goal.target_date.isoformat()})"
        )

    # ------------------------------------------------------------------
    # Step 4: Evaluate all goals using tracker.py
    # ------------------------------------------------------------------
    updated_goals: list[FinancialGoal] = []
    detail_lines: list[str] = []

    for goal in goals:
        required_monthly_saving = calculate_required_monthly_saving(goal)
        on_track = required_monthly_saving <= monthly_surplus

        updated_goal = goal.model_copy(
            update={
                "required_monthly_saving": required_monthly_saving,
                "on_track": on_track,
            }
        )

        updated_goals.append(updated_goal)

        status_text = "on track" if on_track else "behind schedule"
        detail_lines.append(
            f"{goal.name}: need to save {required_monthly_saving:.2f}/month, status = {status_text}"
        )

    # ------------------------------------------------------------------
    # Step 5: Build HITL confirmation payload
    # ------------------------------------------------------------------
    if new_goal_added:
        summary = (
            f"Created and evaluated {len(updated_goals)} goal(s). "
            f"Monthly surplus = {monthly_surplus:.2f}."
        )
    else:
        summary = (
            f"Evaluated {len(updated_goals)} financial goal(s). "
            f"Monthly surplus = {monthly_surplus:.2f}."
        )

    pending_confirmation = {
        "action": "approve_goal_planning",
        "agent": "goal_planning",
        "summary": summary,
        "details": creation_detail_lines + detail_lines,
        "goal_extraction_confidence": "llm" if llm_succeeded else "fallback",
        # confirmed 字段故意不加，保持和当前 router.py 逻辑一致
    }

    return {
        "goals": updated_goals,
        "pending_confirmation": pending_confirmation,
    }