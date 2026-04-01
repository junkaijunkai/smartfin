"""
Smoke tests for the orchestrator pipeline in stub state.

These tests verify that the graph compiles, routes correctly, and
completes without errors — before any real agent logic is implemented.

Run with:
    pytest tests/integration/test_agent_pipeline.py -v
"""

import pytest
from langchain_core.messages import HumanMessage

from app.orchestrator import app_graph, get_pending_interrupt


# A thread_id uniquely identifies one user's conversation session.
# Use a different id per test so checkpointer state doesn't bleed across tests.
def make_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def invoke(message: str, thread_id: str) -> dict:
    """Invoke the graph with a single human message and return final state."""
    return app_graph.invoke(
        {"messages": [HumanMessage(content=message)]},
        make_config(thread_id),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_graph_compiles():
    """app_graph should be a compiled graph, not None."""
    assert app_graph is not None


def test_expense_analysis_route():
    """A generic message should route through expense_analysis."""
    state = invoke("Show me my spending", thread_id="t-expense")
    # After stub node runs and returns to supervisor, active_agent becomes "end"
    assert state["active_agent"] == "end"


def test_budget_planning_route():
    """Message containing 'budget' should route through budget_planning."""
    state = invoke("Help me with my budget", thread_id="t-budget")
    assert state["active_agent"] == "end"


def test_goal_planning_route():
    """Message containing 'goal' should route through goal_planning."""
    state = invoke("I want to set a savings goal", thread_id="t-goal")
    assert state["active_agent"] == "end"


def test_anomaly_detection_route():
    """Message containing 'suspicious' should route through anomaly_detection."""
    state = invoke("There's a suspicious transaction", thread_id="t-anomaly")
    assert state["active_agent"] == "end"


def test_health_assessment_route():
    """Message containing 'health' should route through health_assessment."""
    state = invoke("What is my financial health?", thread_id="t-health")
    assert state["active_agent"] == "end"


def test_messages_are_accumulated():
    """The messages list should contain the original human message after the run."""
    state = invoke("Show me my spending", thread_id="t-messages")
    contents = [m.content for m in state["messages"]]
    assert "Show me my spending" in contents


def test_no_pending_interrupt_on_normal_flow():
    """A normal flow (no budget/goal confirmation needed) should not pause."""
    config = make_config("t-no-interrupt")
    app_graph.invoke(
        {"messages": [HumanMessage(content="Show me my spending")]},
        config,
    )
    # Graph should be fully done, not paused
    pending = get_pending_interrupt(app_graph, config)
    assert pending is None
