# 引入日期工具，用来构造未来/过去日期
from datetime import date, timedelta

# 引入项目里定义好的数据模型和共享状态类型
from app.state import AppState, FinancialGoal, BudgetAllocation, TransactionCategory

# 引入你要测试的 tracker 函数
from app.agents.goal_planning.tracker import (
    calculate_months_remaining,
    calculate_required_monthly_saving,
)

# 引入 goal planning agent 的主入口函数
from app.agents.goal_planning.agent import run as goal_planning_run


# ---------------------------------------------------------------------------
# tracker.py 测试
# ---------------------------------------------------------------------------

def test_calculate_months_remaining_future_date():
    """
    测试：如果目标日期在未来，函数应返回一个大于等于 1 的月份数。
    """
    # 构造一个 90 天后的日期（大约 3 个月）
    target_date = date.today() + timedelta(days=90)

    # 调用函数
    months = calculate_months_remaining(target_date)

    # 断言结果至少为 1，并且这里预期大约是 3
    assert months >= 1
    assert months == 3


def test_calculate_months_remaining_past_date():
    """
    测试：如果目标日期已经过去，函数也应返回 1，避免后续除以 0。
    """
    # 构造一个过去的日期
    target_date = date.today() - timedelta(days=10)

    # 调用函数
    months = calculate_months_remaining(target_date)

    # 断言返回 1
    assert months == 1


def test_calculate_required_monthly_saving_basic_case():
    """
    测试：一个正常目标下，每月所需储蓄金额是否计算正确。
    """
    # 构造一个测试目标：
    # 目标金额 1200，当前已有 0，截止日期约 4 个月后
    goal = FinancialGoal(
        id="g1",
        name="Emergency Fund",
        target_amount=1200.0,
        current_amount=0.0,
        target_date=date.today() + timedelta(days=120),
    )

    # 调用函数
    required = calculate_required_monthly_saving(goal)

    # 1200 / 4 = 300
    assert required == 300.0


def test_calculate_required_monthly_saving_when_goal_already_completed():
    """
    测试：如果当前金额已经达到或超过目标金额，每月所需储蓄应为 0。
    """
    goal = FinancialGoal(
        id="g2",
        name="Completed Goal",
        target_amount=1000.0,
        current_amount=1200.0,
        target_date=date.today() + timedelta(days=60),
    )

    required = calculate_required_monthly_saving(goal)

    assert required == 0.0


# ---------------------------------------------------------------------------
# agent.py 测试
# ---------------------------------------------------------------------------

def _make_state(
    goals: list[FinancialGoal],
    monthly_income: float,
    budget_allocations: list[BudgetAllocation],
) -> AppState:
    """
    构造一个最小可用的 AppState，供 goal planning agent 测试使用。
    """
    return {
        "messages": [],
        "transactions": [],
        "monthly_income": monthly_income,
        "categorised_transactions": [],
        "spending_trends": [],
        "budget_allocations": budget_allocations,
        "goals": goals,
        "anomaly_flags": [],
        "health_summary": None,
        "alerts": [],
        "pending_confirmation": None,
        "active_agent": "goal_planning",
        "agents_queue": [],
    }


def test_goal_planning_agent_updates_goal_fields():
    """
    测试：agent 运行后，应正确更新 goals 中的
    required_monthly_saving 和 on_track 字段。
    """
    # 构造一个目标：4个月后要存到 1200，目前 0
    goal = FinancialGoal(
        id="g1",
        name="Emergency Fund",
        target_amount=1200.0,
        current_amount=0.0,
        target_date=date.today() + timedelta(days=120),
    )

    # 构造预算：本月收入 2000，当前已花 500，所以可结余 1500
    budget = BudgetAllocation(
        category=TransactionCategory.FOOD,
        allocated_amount=600.0,
        spent_amount=500.0,
        period_start=date.today().replace(day=1),
        period_end=date.today() + timedelta(days=30),
    )

    state = _make_state(
        goals=[goal],
        monthly_income=2000.0,
        budget_allocations=[budget],
    )

    # 调用 agent
    result = goal_planning_run(state)

    # 取出更新后的 goal
    updated_goals = result["goals"]
    assert len(updated_goals) == 1

    updated_goal = updated_goals[0]

    # 1200 / 4 = 300
    assert updated_goal.required_monthly_saving == 300.0

    # monthly_surplus = 2000 - 500 = 1500，所以可以完成
    assert updated_goal.on_track is True


def test_goal_planning_agent_sets_pending_confirmation():
    """
    测试：agent 运行后，应设置 pending_confirmation，
    且不应主动写入 confirmed 字段。
    """
    goal = FinancialGoal(
        id="g2",
        name="Laptop Fund",
        target_amount=2400.0,
        current_amount=0.0,
        target_date=date.today() + timedelta(days=120),
    )

    budget = BudgetAllocation(
        category=TransactionCategory.SHOPPING,
        allocated_amount=800.0,
        spent_amount=700.0,
        period_start=date.today().replace(day=1),
        period_end=date.today() + timedelta(days=30),
    )

    state = _make_state(
        goals=[goal],
        monthly_income=1500.0,
        budget_allocations=[budget],
    )

    result = goal_planning_run(state)

    pending_confirmation = result["pending_confirmation"]

    # 确认字段存在
    assert pending_confirmation is not None

    # 检查 action / agent 是否正确
    assert pending_confirmation["action"] == "approve_goal_planning"
    assert pending_confirmation["agent"] == "goal_planning"

    # 不应主动带 confirmed 字段，否则会影响 router 的 HITL 判断
    assert "confirmed" not in pending_confirmation


def test_goal_planning_agent_marks_goal_behind_schedule_when_surplus_not_enough():
    """
    测试：如果月结余不足，goal 应标记为 on_track = False。
    """
    goal = FinancialGoal(
        id="g3",
        name="Travel Fund",
        target_amount=5000.0,
        current_amount=0.0,
        target_date=date.today() + timedelta(days=60),
    )

    # 收入 1000，已花 900，可结余只有 100
    budget = BudgetAllocation(
        category=TransactionCategory.ENTERTAINMENT,
        allocated_amount=1000.0,
        spent_amount=900.0,
        period_start=date.today().replace(day=1),
        period_end=date.today() + timedelta(days=30),
    )

    state = _make_state(
        goals=[goal],
        monthly_income=1000.0,
        budget_allocations=[budget],
    )

    result = goal_planning_run(state)

    updated_goal = result["goals"][0]

    # 由于目标金额大、时间短，每月所需储蓄会明显高于 100
    assert updated_goal.on_track is False