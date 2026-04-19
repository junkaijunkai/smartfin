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

from uuid import uuid4

from app.state import AppState, FinancialGoal
from app.agents.goal_planning.tracker import calculate_required_monthly_saving
from app.agents.goal_planning.extractor import extract_goal_from_message


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


def _build_goal_from_extraction(extraction) -> FinancialGoal:
    """
    Convert the structured extraction result into a FinancialGoal object.

    Assumes required fields already exist.
    """
    return FinancialGoal(
        id=f"goal-{uuid4().hex[:8]}",
        name=extraction.name or "Financial Goal",
        target_amount=float(extraction.target_amount),
        current_amount=float(extraction.current_amount or 0.0),
        target_date=extraction.target_date,
    )


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

        # 信息完整 → 创建新目标并加入现有 goals
        new_goal = _build_goal_from_extraction(extraction)
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

# # 引入类型定义，表示整个多 agent 系统共享的状态对象
# from app.state import AppState, FinancialGoal

# # 引入你刚刚已经写好并测试通过的计算函数
# from app.agents.goal_planning.tracker import calculate_required_monthly_saving


# def run(state: AppState) -> dict:
#     """
#     Goal Planning agent 的主入口函数。

#     功能：
#     1. 从共享状态 AppState 中读取 goals、monthly_income、budget_allocations
#     2. 计算当前月可支配结余 monthly_surplus
#     3. 为每个 financial goal 计算每月所需储蓄 required_monthly_saving
#     4. 判断该 goal 是否按当前结余水平能够按时完成（on_track）
#     5. 返回更新后的 goals，并生成 pending_confirmation 供 HITL 审核

#     读取：
#     - state["goals"]
#     - state["monthly_income"]
#     - state["budget_allocations"]

#     写回：
#     - goals
#     - pending_confirmation
#     """

#     # 从 state 中读取目标列表；如果为空则用空列表兜底
#     goals = state.get("goals") or []

#     # 从 state 中读取月收入；如果没有值则默认为 0
#     monthly_income = state.get("monthly_income") or 0.0

#     # 从 state 中读取预算分配列表；如果为空则用空列表兜底
#     budget_allocations = state.get("budget_allocations") or []

#     # 计算当前预算中已经花出去的总金额
#     total_spent = sum(allocation.spent_amount for allocation in budget_allocations)

#     # 当前月可支配结余 = 月收入 - 已花预算
#     monthly_surplus = monthly_income - total_spent

#     # 用来存放更新后的 goal 列表
#     updated_goals: list[FinancialGoal] = []

#     # 用来生成给用户看的简要说明
#     detail_lines: list[str] = []

#     # 逐个处理每个 financial goal
#     for goal in goals:
#         # 调用 tracker 中的函数，计算该目标每月需要存多少钱
#         required_monthly_saving = calculate_required_monthly_saving(goal)

#         # 判断该目标在当前月结余水平下是否可按时完成
#         on_track = required_monthly_saving <= monthly_surplus

#         # 复制一个新的 goal 对象，并更新字段
#         # model_copy 是 Pydantic v2 推荐写法，避免直接修改原对象
#         updated_goal = goal.model_copy(
#             update={
#                 "required_monthly_saving": required_monthly_saving,
#                 "on_track": on_track,
#             }
#         )

#         # 放入新的 goal 列表中
#         updated_goals.append(updated_goal)

#         # 组织一条给前端 / 用户确认界面展示的说明文字
#         status_text = "on track" if on_track else "behind schedule"
#         detail_lines.append(
#             f"{goal.name}: need to save {required_monthly_saving:.2f}/month, status = {status_text}"
#         )

#     # 构造 HITL（Human-in-the-Loop）确认信息
#     # 这个结构和 expense_analysis agent 的设计风格保持一致
#     pending_confirmation = {
#         "action": "approve_goal_planning",
#         "agent": "goal_planning",
#         "summary": f"Evaluated {len(updated_goals)} financial goals. Monthly surplus = {monthly_surplus:.2f}.",
#         "details": detail_lines,
#         # 不要主动加 confirmed 字段
#         # router.py 会根据 confirmed 是否存在来判断是否需要暂停确认
#     }

#     # 返回写回 state 的内容
#     return {
#         "goals": updated_goals,
#         "pending_confirmation": pending_confirmation,
#     }