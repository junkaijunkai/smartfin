# 引入日期工具，用来构造未来 / 过去日期
from datetime import date, timedelta

# 引入项目里的数据模型和共享状态类型
from app.state import AppState, FinancialGoal, BudgetAllocation, TransactionCategory

# 引入你要测试的 tracker 函数
from app.agents.goal_planning.tracker import (
    calculate_months_remaining,
    calculate_required_monthly_saving,
)

# 这里同时引入 agent 模块本身，方便 monkeypatch 其中的函数
import app.agents.goal_planning.agent as goal_agent_module

# 引入 goal planning agent 的主入口函数
from app.agents.goal_planning.agent import run as goal_planning_run

# 引入 extractor 模块，方便直接测试 extractor 的内部逻辑
import app.agents.goal_planning.extractor as extractor_module
from app.agents.goal_planning.extractor import (
    GoalExtractionResult,
    extract_goal_from_message,
)


# ---------------------------------------------------------------------------
# tracker.py 测试
# ---------------------------------------------------------------------------

def test_calculate_months_remaining_future_date():
    """
    测试：如果目标日期在未来，函数应返回一个大于等于 1 的月份数。
    """
    target_date = date.today() + timedelta(days=90)

    months = calculate_months_remaining(target_date)

    assert months >= 1
    assert months == 3


def test_calculate_months_remaining_past_date():
    """
    测试：如果目标日期已经过去，函数也应返回 1，避免后续除以 0。
    """
    target_date = date.today() - timedelta(days=10)

    months = calculate_months_remaining(target_date)

    assert months == 1


def test_calculate_required_monthly_saving_basic_case():
    """
    测试：一个正常目标下，每月所需储蓄金额是否计算正确。
    """
    goal = FinancialGoal(
        id="g1",
        name="Emergency Fund",
        target_amount=1200.0,
        current_amount=0.0,
        target_date=date.today() + timedelta(days=120),
    )

    required = calculate_required_monthly_saving(goal)

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
# extractor.py 测试
# ---------------------------------------------------------------------------



def test_fallback_extract_complete_laptop_goal():
    """
    测试：fallback 能识别完整的 laptop goal，
    包括 goal intent、名称、金额、日期和 missing_fields。
    """
    result = extractor_module._fallback_extract(
        "I want to save 8000 by 2027-06-01 for a laptop."
    )

    assert result.is_goal_intent is True
    assert result.name == "Laptop Fund"
    assert result.target_amount == 8000.0
    assert result.target_date == date(2027, 6, 1)
    assert result.current_amount is None
    assert result.missing_fields == []


def test_fallback_extract_missing_amount():
    """
    测试：如果只有日期没有金额，fallback 应标记缺少 target_amount。
    """
    result = extractor_module._fallback_extract(
        "I want to save for a laptop by 2027-06-01."
    )

    assert result.is_goal_intent is True
    assert result.name == "Laptop Fund"
    assert result.target_amount is None
    assert result.target_date == date(2027, 6, 1)
    assert "target_amount" in result.missing_fields
    assert "target_date" not in result.missing_fields


def test_fallback_extract_missing_date():
    """
    测试：如果只有金额没有日期，fallback 应标记缺少 target_date。
    """
    result = extractor_module._fallback_extract(
        "I want to save 5000 for travel."
    )

    assert result.is_goal_intent is True
    assert result.name == "Travel Fund"
    assert result.target_amount == 5000.0
    assert result.target_date is None
    assert "target_date" in result.missing_fields
    assert "target_amount" not in result.missing_fields


def test_fallback_extract_non_goal_message():
    """
    测试：普通聊天内容不应被识别为 financial goal intent。
    """
    result = extractor_module._fallback_extract(
        "Hello, how are you today?"
    )

    assert result.is_goal_intent is False
    assert result.name is None
    assert result.target_amount is None
    assert result.target_date is None
    assert result.missing_fields == []


def test_fallback_extract_emergency_goal_name():
    """
    测试：emergency 关键词应映射成 Emergency Fund。
    """
    result = extractor_module._fallback_extract(
        "I want to build an emergency fund of 10000 by 2027-12-31."
    )

    assert result.is_goal_intent is True
    assert result.name == "Emergency Fund"


def test_extract_goal_from_message_empty_input():
    """
    测试：空输入应直接返回非 goal intent，
    并且 llm_succeeded 记为 True（因为这不是异常场景）。
    """
    result, llm_succeeded = extract_goal_from_message("   ")

    assert llm_succeeded is True
    assert result.is_goal_intent is False
    assert result.missing_fields == []


def test_extract_goal_from_message_llm_success(monkeypatch):
    """
    测试：当 LLM 正常返回结构化结果时，
    extract_goal_from_message 应返回该结果，并标记 llm_succeeded=True。
    """

    expected_result = GoalExtractionResult(
        is_goal_intent=True,
        name="Laptop Fund",
        target_amount=8000.0,
        target_date=date(2027, 6, 1),
        current_amount=1000.0,
        missing_fields=[],
    )

    class FakeStructuredLLM:
        def invoke(self, messages):
            # messages is a list of SystemMessage/HumanMessage objects
            combined = " ".join(str(getattr(m, "content", m)) for m in messages)
            assert "I want to save 8000" in combined
            return expected_result

    class FakeChatAnthropic:
        def __init__(self, model: str, **kwargs):
            # 检查模型名参数是否被正确传入
            assert isinstance(model, str)

        def with_structured_output(self, schema):
            assert schema is GoalExtractionResult
            return FakeStructuredLLM()

    monkeypatch.setattr(extractor_module, "ChatAnthropic", FakeChatAnthropic)

    result, llm_succeeded = extract_goal_from_message(
        "I want to save 8000 by 2027-06-01 for a laptop."
    )

    assert llm_succeeded is True
    assert result == expected_result


def test_extract_goal_from_message_llm_failure_uses_fallback(monkeypatch):
    """
    测试：当 LLM 调用抛异常时，应自动退回 fallback，
    并标记 llm_succeeded=False。
    """

    class FakeChatAnthropic:
        def __init__(self, model: str):
            pass

        def with_structured_output(self, schema):
            raise RuntimeError("LLM is unavailable")

    monkeypatch.setattr(extractor_module, "ChatAnthropic", FakeChatAnthropic)

    result, llm_succeeded = extract_goal_from_message(
        "I want to save 8000 by 2027-06-01 for a laptop."
    )

    assert llm_succeeded is False
    assert result.is_goal_intent is True
    assert result.name == "Laptop Fund"
    assert result.target_amount == 8000.0
    assert result.target_date == date(2027, 6, 1)
    assert result.missing_fields == []


# ---------------------------------------------------------------------------
# agent.py 测试
# ---------------------------------------------------------------------------

def _make_state(
    goals: list[FinancialGoal],
    monthly_income: float,
    budget_allocations: list[BudgetAllocation],
    messages: list | None = None,
) -> AppState:
    """
    构造一个最小可用的 AppState，供 goal planning agent 测试使用。
    """
    return {
        "messages": messages or [],
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


class DummyMessage:
    """
    一个简单的测试消息对象，用来模拟 LangChain 的 HumanMessage / AIMessage。
    """
    def __init__(self, content):
        self.content = content

    def __str__(self):
        return f"DummyMessage(content={self.content})"


def test_get_latest_message_text_returns_empty_when_no_messages():
    """
    测试：当 state 中没有 messages 时，应返回空字符串。
    """
    state = _make_state(goals=[], monthly_income=0.0, budget_allocations=[], messages=[])

    result = goal_agent_module._get_latest_message_text(state)

    assert result == ""


def test_get_latest_message_text_returns_string_content():
    """
    测试：当最后一条 message 的 content 是字符串时，应直接返回该字符串。
    """
    state = _make_state(
        goals=[],
        monthly_income=0.0,
        budget_allocations=[],
        messages=[DummyMessage("I want to save 5000 for travel.")],
    )

    result = goal_agent_module._get_latest_message_text(state)

    assert result == "I want to save 5000 for travel."


def test_get_latest_message_text_falls_back_to_str_for_non_string_content():
    """
    测试：当 message.content 不是字符串时，应回退到 str(message)。
    """
    state = _make_state(
        goals=[],
        monthly_income=0.0,
        budget_allocations=[],
        messages=[DummyMessage({"text": "structured content"})],
    )

    result = goal_agent_module._get_latest_message_text(state)

    assert result == "DummyMessage(content={'text': 'structured content'})"


def test_build_goal_from_extraction_uses_defaults():
    """
    测试：_build_goal_from_extraction 应把结构化结果转成 FinancialGoal，
    并在 name 缺失时使用默认名称。
    """
    extraction = GoalExtractionResult(
        is_goal_intent=True,
        name=None,
        target_amount=3000.0,
        target_date=date(2027, 1, 1),
        current_amount=None,
        missing_fields=[],
    )

    goal, error = goal_agent_module._build_goal_from_extraction(extraction)

    assert goal is not None
    assert error == ""
    assert goal.id.startswith("goal-")
    assert goal.name == "Financial Goal"
    assert goal.target_amount == 3000.0
    assert goal.current_amount == 0.0
    assert goal.target_date == date(2027, 1, 1)


def test_goal_planning_agent_updates_goal_fields(monkeypatch):
    """
    测试：agent 运行后，应正确更新 goals 中的
    required_monthly_saving 和 on_track 字段。
    """
    # 这里 mock extractor，让测试只关注 agent 本身逻辑
    monkeypatch.setattr(
        goal_agent_module,
        "extract_goal_from_message",
        lambda _, **kwargs: (
            GoalExtractionResult(
                is_goal_intent=False,
                missing_fields=[],
            ),
            True,
        ),
    )

    goal = FinancialGoal(
        id="g1",
        name="Emergency Fund",
        target_amount=1200.0,
        current_amount=0.0,
        target_date=date.today() + timedelta(days=120),
    )

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

    result = goal_planning_run(state)

    updated_goals = result["goals"]
    assert len(updated_goals) == 1

    updated_goal = updated_goals[0]

    assert updated_goal.required_monthly_saving == 300.0
    assert updated_goal.on_track is True


def test_goal_planning_agent_sets_pending_confirmation(monkeypatch):
    """
    测试：agent 运行后，应设置 pending_confirmation，
    且不应主动写入 confirmed 字段。
    """
    monkeypatch.setattr(
        goal_agent_module,
        "extract_goal_from_message",
        lambda _, **kwargs: (
            GoalExtractionResult(
                is_goal_intent=False,
                missing_fields=[],
            ),
            True,
        ),
    )

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

    assert pending_confirmation is not None
    assert pending_confirmation["action"] == "approve_goal_planning"
    assert pending_confirmation["agent"] == "goal_planning"
    assert "confirmed" not in pending_confirmation


def test_goal_planning_agent_marks_goal_behind_schedule_when_surplus_not_enough(monkeypatch):
    """
    测试：如果月结余不足，goal 应标记为 on_track = False。
    """
    monkeypatch.setattr(
        goal_agent_module,
        "extract_goal_from_message",
        lambda _, **kwargs: (
            GoalExtractionResult(
                is_goal_intent=False,
                missing_fields=[],
            ),
            True,
        ),
    )

    goal = FinancialGoal(
        id="g3",
        name="Travel Fund",
        target_amount=5000.0,
        current_amount=0.0,
        target_date=date.today() + timedelta(days=60),
    )

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

    assert updated_goal.on_track is False


def test_goal_planning_agent_returns_clarification_when_required_fields_missing(monkeypatch):
    """
    测试：如果识别到了 goal intent，但缺少必要字段，
    agent 应返回 clarify_goal_planning，而不是直接创建目标。
    """
    monkeypatch.setattr(
        goal_agent_module,
        "extract_goal_from_message",
        lambda _, **kwargs: (
            GoalExtractionResult(
                is_goal_intent=True,
                name="Laptop Fund",
                target_amount=None,
                target_date=None,
                current_amount=None,
                missing_fields=["target_amount", "target_date"],
            ),
            False,
        ),
    )

    state = _make_state(
        goals=[],
        monthly_income=2000.0,
        budget_allocations=[],
        messages=[DummyMessage("I want to save for a laptop.")],
    )

    result = goal_planning_run(state)

    assert result["goals"] == []
    pending_confirmation = result["pending_confirmation"]

    assert pending_confirmation["action"] == "clarify_goal_planning"
    assert pending_confirmation["agent"] == "goal_planning"
    assert pending_confirmation["goal_extraction_confidence"] == "fallback"
    assert "Missing fields: target_amount, target_date" in pending_confirmation["details"][1]


def test_goal_planning_agent_creates_new_goal_when_extraction_is_complete(monkeypatch):
    """
    测试：如果 extractor 返回完整 goal 信息，
    agent 应创建一个新目标并进行评估。
    """
    monkeypatch.setattr(
        goal_agent_module,
        "extract_goal_from_message",
        lambda _, **kwargs: (
            GoalExtractionResult(
                is_goal_intent=True,
                name="Laptop Fund",
                target_amount=8000.0,
                target_date=date.today() + timedelta(days=120),
                current_amount=1000.0,
                missing_fields=[],
            ),
            True,
        ),
    )

    budget = BudgetAllocation(
        category=TransactionCategory.SHOPPING,
        allocated_amount=600.0,
        spent_amount=500.0,
        period_start=date.today().replace(day=1),
        period_end=date.today() + timedelta(days=30),
    )

    state = _make_state(
        goals=[],
        monthly_income=3000.0,
        budget_allocations=[budget],
        messages=[DummyMessage("I want to save 8000 by next year for a laptop.")],
    )

    result = goal_planning_run(state)

    updated_goals = result["goals"]
    pending_confirmation = result["pending_confirmation"]

    assert len(updated_goals) == 1
    assert updated_goals[0].name == "Laptop Fund"
    assert updated_goals[0].target_amount == 8000.0
    assert updated_goals[0].current_amount == 1000.0

    assert pending_confirmation["action"] == "approve_goal_planning"
    assert pending_confirmation["agent"] == "goal_planning"
    assert pending_confirmation["goal_extraction_confidence"] == "llm"
    assert pending_confirmation["summary"].startswith("Created and evaluated 1 goal(s).")
    assert "Created new goal: Laptop Fund" in pending_confirmation["details"][0]


def test_goal_planning_agent_uses_fallback_confidence_for_non_goal_intent(monkeypatch):
    """
    测试：即使不是 goal intent，只要 extractor 返回 llm_succeeded=False，
    pending_confirmation 中也应体现 fallback 置信来源。
    """
    monkeypatch.setattr(
        goal_agent_module,
        "extract_goal_from_message",
        lambda _, **kwargs: (
            GoalExtractionResult(
                is_goal_intent=False,
                missing_fields=[],
            ),
            False,
        ),
    )

    goal = FinancialGoal(
        id="g4",
        name="Emergency Fund",
        target_amount=1200.0,
        current_amount=0.0,
        target_date=date.today() + timedelta(days=120),
    )

    state = _make_state(
        goals=[goal],
        monthly_income=2000.0,
        budget_allocations=[],
        messages=[DummyMessage("hello")],
    )

    result = goal_planning_run(state)

    assert result["pending_confirmation"]["goal_extraction_confidence"] == "fallback"


# ---------------------------------------------------------------------------
# Validation layer tests
# ---------------------------------------------------------------------------


def test_validate_extraction_consistency_passes_when_consistent():
    """
    测试：当 missing_fields 与实际缺失字段一致时，验证应通过。
    """
    extraction = GoalExtractionResult(
        is_goal_intent=True,
        name="Laptop Fund",
        target_amount=5000.0,
        target_date=date.today() + timedelta(days=180),
        current_amount=None,
        missing_fields=[],
    )

    is_valid, error = goal_agent_module._validate_extraction_consistency(extraction)

    assert is_valid is True
    assert error == ""


def test_validate_extraction_consistency_fails_when_missing_target_amount_not_recorded():
    """
    测试：当 target_amount 为 None 但 missing_fields 中没有记录时，验证应失败。
    """
    extraction = GoalExtractionResult(
        is_goal_intent=True,
        name="Laptop Fund",
        target_amount=None,  # ← 实际缺失
        target_date=date.today() + timedelta(days=180),
        current_amount=None,
        missing_fields=[],  # ← 但没有记录缺失
    )

    is_valid, error = goal_agent_module._validate_extraction_consistency(extraction)

    assert is_valid is False
    assert "Extraction inconsistency" in error
    assert "target_amount" in error


def test_validate_extraction_consistency_fails_when_missing_target_date_not_recorded():
    """
    测试：当 target_date 为 None 但 missing_fields 中没有记录时，验证应失败。
    """
    extraction = GoalExtractionResult(
        is_goal_intent=True,
        name="Laptop Fund",
        target_amount=5000.0,
        target_date=None,  # ← 实际缺失
        current_amount=None,
        missing_fields=[],  # ← 但没有记录缺失
    )

    is_valid, error = goal_agent_module._validate_extraction_consistency(extraction)

    assert is_valid is False
    assert "Extraction inconsistency" in error
    assert "target_date" in error


def test_validate_extraction_consistency_ignores_non_goal_intent():
    """
    测试：当 is_goal_intent=False 时，missing_fields 不需要与字段一致。
    """
    extraction = GoalExtractionResult(
        is_goal_intent=False,  # ← 不是 goal intent
        name=None,
        target_amount=None,
        target_date=None,
        current_amount=None,
        missing_fields=[],  # ← 即使缺失多个字段也不计入
    )

    is_valid, error = goal_agent_module._validate_extraction_consistency(extraction)

    assert is_valid is True
    assert error == ""


def test_validate_goal_creation_fields_passes_when_all_valid():
    """
    测试：所有必要字段都存在且有效时，验证通过。
    """
    extraction = GoalExtractionResult(
        is_goal_intent=True,
        name="Laptop Fund",
        target_amount=5000.0,
        target_date=date.today() + timedelta(days=180),
        current_amount=1000.0,
        missing_fields=[],
    )

    is_valid, error = goal_agent_module._validate_goal_creation_fields(extraction)

    assert is_valid is True
    assert error == ""


def test_validate_goal_creation_fields_fails_when_target_amount_zero():
    """
    测试：target_amount 为 0 时，验证应失败。
    """
    extraction = GoalExtractionResult(
        is_goal_intent=True,
        name="Laptop Fund",
        target_amount=0.0,  # ← 无效：必须 > 0
        target_date=date.today() + timedelta(days=180),
        current_amount=None,
        missing_fields=[],
    )

    is_valid, error = goal_agent_module._validate_goal_creation_fields(extraction)

    assert is_valid is False
    assert "must be positive" in error


def test_validate_goal_creation_fields_fails_when_target_amount_negative():
    """
    测试：target_amount 为负数时，验证应失败。
    """
    extraction = GoalExtractionResult(
        is_goal_intent=True,
        name="Laptop Fund",
        target_amount=-5000.0,  # ← 无效：负数
        target_date=date.today() + timedelta(days=180),
        current_amount=None,
        missing_fields=[],
    )

    is_valid, error = goal_agent_module._validate_goal_creation_fields(extraction)

    assert is_valid is False
    assert "must be positive" in error


def test_validate_goal_creation_fields_fails_when_target_date_in_past():
    """
    测试：target_date 在过去时，验证应失败。
    """
    extraction = GoalExtractionResult(
        is_goal_intent=True,
        name="Laptop Fund",
        target_amount=5000.0,
        target_date=date.today() - timedelta(days=10),  # ← 无效：在过去
        current_amount=None,
        missing_fields=[],
    )

    is_valid, error = goal_agent_module._validate_goal_creation_fields(extraction)

    assert is_valid is False
    assert "must be in the future" in error


def test_validate_goal_creation_fields_fails_when_target_date_is_today():
    """
    测试：target_date 是今天时，验证应失败（目标必须在未来）。
    """
    extraction = GoalExtractionResult(
        is_goal_intent=True,
        name="Laptop Fund",
        target_amount=5000.0,
        target_date=date.today(),  # ← 无效：必须在未来
        current_amount=None,
        missing_fields=[],
    )

    is_valid, error = goal_agent_module._validate_goal_creation_fields(extraction)

    assert is_valid is False
    assert "must be in the future" in error


def test_validate_goal_creation_fields_fails_when_not_goal_intent():
    """
    测试：is_goal_intent=False 时，不能创建 goal，验证应失败。
    """
    extraction = GoalExtractionResult(
        is_goal_intent=False,  # ← 不是 goal intent
        name="Laptop Fund",
        target_amount=5000.0,
        target_date=date.today() + timedelta(days=180),
        current_amount=None,
        missing_fields=[],
    )

    is_valid, error = goal_agent_module._validate_goal_creation_fields(extraction)

    assert is_valid is False
    assert "Not a goal intent" in error


def test_validate_goal_creation_fields_fails_when_missing_fields_not_empty():
    """
    测试：missing_fields 不为空时，不能创建 goal，验证应失败。
    """
    extraction = GoalExtractionResult(
        is_goal_intent=True,
        name="Laptop Fund",
        target_amount=5000.0,
        target_date=date.today() + timedelta(days=180),
        current_amount=None,
        missing_fields=["target_amount"],  # ← 还有缺失字段
    )

    is_valid, error = goal_agent_module._validate_goal_creation_fields(extraction)

    assert is_valid is False
    assert "Missing required fields" in error
    assert "target_amount" in error


def test_build_goal_from_extraction_fails_when_data_inconsistent(monkeypatch):
    """
    测试：当提取结果不一致时，_build_goal_from_extraction 应返回 (None, error_msg)。
    """
    extraction = GoalExtractionResult(
        is_goal_intent=True,
        name="Laptop Fund",
        target_amount=None,  # ← 缺失但未记录
        target_date=date.today() + timedelta(days=180),
        current_amount=None,
        missing_fields=[],
    )

    goal, error = goal_agent_module._build_goal_from_extraction(extraction)

    assert goal is None
    assert "Extraction inconsistency" in error


def test_build_goal_from_extraction_fails_when_validation_fails(monkeypatch):
    """
    测试：当字段验证失败时，_build_goal_from_extraction 应返回 (None, error_msg)。
    """
    extraction = GoalExtractionResult(
        is_goal_intent=True,
        name="Laptop Fund",
        target_amount=0.0,  # ← 无效：必须 > 0
        target_date=date.today() + timedelta(days=180),
        current_amount=None,
        missing_fields=[],
    )

    goal, error = goal_agent_module._build_goal_from_extraction(extraction)

    assert goal is None
    assert "must be positive" in error


def test_build_goal_from_extraction_succeeds_when_all_valid(monkeypatch):
    """
    测试：当所有验证通过时，_build_goal_from_extraction 应返回有效的 FinancialGoal。
    """
    extraction = GoalExtractionResult(
        is_goal_intent=True,
        name="Laptop Fund",
        target_amount=8000.0,
        target_date=date.today() + timedelta(days=180),
        current_amount=2000.0,
        missing_fields=[],
    )

    goal, error = goal_agent_module._build_goal_from_extraction(extraction)

    assert goal is not None
    assert error == ""
    assert goal.name == "Laptop Fund"
    assert goal.target_amount == 8000.0
    assert goal.current_amount == 2000.0
    assert goal.target_date == date.today() + timedelta(days=180)


def test_build_goal_from_extraction_uses_fallback_name_when_none(monkeypatch):
    """
    测试：当 extraction.name 为 None 时，应使用默认名称 "Financial Goal"。
    """
    extraction = GoalExtractionResult(
        is_goal_intent=True,
        name=None,  # ← 名称缺失
        target_amount=5000.0,
        target_date=date.today() + timedelta(days=180),
        current_amount=None,
        missing_fields=[],
    )

    goal, error = goal_agent_module._build_goal_from_extraction(extraction)

    assert goal is not None
    assert error == ""
    assert goal.name == "Financial Goal"


def test_goal_planning_agent_handles_validation_failure_gracefully(monkeypatch):
    """
    测试：当 goal 创建验证失败时，agent 应返回 clarify_goal_planning，
    而不是抛异常。
    """
    # Mock extractor 返回数据不一致的结果
    monkeypatch.setattr(
        goal_agent_module,
        "extract_goal_from_message",
        lambda _, **kwargs: (
            GoalExtractionResult(
                is_goal_intent=True,
                name="Laptop Fund",
                target_amount=None,  # ← 缺失但未记录
                target_date=date.today() + timedelta(days=180),
                current_amount=None,
                missing_fields=[],
            ),
            True,
        ),
    )

    state = _make_state(
        goals=[],
        monthly_income=3000.0,
        budget_allocations=[],
    )

    result = goal_planning_run(state)

    # 应返回 clarify 而不是创建目标
    assert result["goals"] == []
    assert result["pending_confirmation"]["action"] == "clarify_goal_planning"
    assert "Extraction inconsistency" in result["pending_confirmation"]["details"][0]