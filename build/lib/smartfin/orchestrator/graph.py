"""
LangGraph Supervisor graph — wires all nodes together and compiles the graph.

Responsibilities:
  - Define each node (supervisor + 5 specialist agents + HITL confirm node).
  - Add edges and conditional edges using the router functions.
  - Compile with a checkpointer so state is persisted after every step,
    enabling HITL interrupts and session resumption.

Each specialist agent node is a STUB here (returns state unchanged).
Replace the stub functions with real imports as each agent is implemented.
"""

from langgraph.graph import END, StateGraph

from smartfin.state import AppState
from smartfin.orchestrator.checkpoints import memory_checkpointer
from smartfin.orchestrator.router import (
    NODE_ANOMALY_DETECTION,
    NODE_BUDGET_PLANNING,
    NODE_CONFIRM,
    NODE_EXPENSE_ANALYSIS,
    NODE_GOAL_PLANNING,
    NODE_HEALTH_ASSESSMENT,
    NODE_SUPERVISOR,
    route_after_agent,
    route_to_agent,
)


# ---------------------------------------------------------------------------
# Node implementations (stubs — replace with real agent imports)
# ---------------------------------------------------------------------------

def supervisor_node(state: AppState) -> dict:
    """
    Reads the latest user message and decides which specialist agent to invoke.

    In the real implementation this will call an LLM with a system prompt
    that describes each agent's capability, then write the chosen agent name
    into state["active_agent"].

    Stub behaviour: routes to expense_analysis for any input, then ends.
    """
    # TODO: replace with LLM-based intent classification
    messages = state.get("messages", [])
    last_message = messages[-1].content if messages else ""

    # Simple keyword-based stub routing
    if "budget" in last_message.lower():
        active = "budget_planning"
    elif "goal" in last_message.lower():
        active = "goal_planning"
    elif "anomal" in last_message.lower() or "suspicious" in last_message.lower():
        active = "anomaly_detection"
    elif "health" in last_message.lower() or "risk" in last_message.lower():
        active = "health_assessment"
    elif state.get("active_agent") and state["active_agent"] != "end":
        # An agent just ran — decide if we need another or we're done
        active = "end"
    else:
        active = "expense_analysis"

    return {"active_agent": active}


def expense_analysis_node(state: AppState) -> dict:
    """Stub — to be replaced by smartfin.agents.expense_analysis.agent"""
    # TODO: from smartfin.agents.expense_analysis.agent import run
    #       return run(state)
    print("[stub] expense_analysis_node called")
    return {}


def budget_planning_node(state: AppState) -> dict:
    """Stub — to be replaced by smartfin.agents.budget_planning.agent"""
    print("[stub] budget_planning_node called")
    return {}


def goal_planning_node(state: AppState) -> dict:
    """Stub — to be replaced by smartfin.agents.goal_planning.agent"""
    print("[stub] goal_planning_node called")
    return {}


def anomaly_detection_node(state: AppState) -> dict:
    """Stub — to be replaced by smartfin.agents.anomaly_detection.agent"""
    print("[stub] anomaly_detection_node called")
    return {}


def health_assessment_node(state: AppState) -> dict:
    """Stub — to be replaced by smartfin.agents.health_assessment.agent"""
    print("[stub] health_assessment_node called")
    return {}


def confirm_node(state: AppState) -> dict:
    """
    HITL confirmation node.

    The graph is compiled with interrupt_before=[NODE_CONFIRM], so execution
    pauses BEFORE this node runs.  The UI reads state["pending_confirmation"],
    presents it to the user, then calls resume_with_confirmation() from
    checkpoints.py to continue.

    When this node finally executes, pending_confirmation["confirmed"] is
    already set by the resume call, so we just clear the pending payload.
    """
    confirmation = state.get("pending_confirmation", {})
    if confirmation.get("confirmed"):
        print("[confirm] User approved the pending action.")
    else:
        print("[confirm] User rejected the pending action.")

    # Clear the pending confirmation so the next cycle starts clean
    return {"pending_confirmation": None}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    """
    Construct and compile the SmartFin StateGraph.

    Returns a compiled graph ready to invoke:
        app = build_graph()
        config = {"configurable": {"thread_id": "user-123"}}
        result = app.invoke({"messages": [HumanMessage(content="...")]}, config)
    """
    builder = StateGraph(AppState)

    # --- Register nodes ---
    builder.add_node(NODE_SUPERVISOR, supervisor_node)
    builder.add_node(NODE_EXPENSE_ANALYSIS, expense_analysis_node)
    builder.add_node(NODE_BUDGET_PLANNING, budget_planning_node)
    builder.add_node(NODE_GOAL_PLANNING, goal_planning_node)
    builder.add_node(NODE_ANOMALY_DETECTION, anomaly_detection_node)
    builder.add_node(NODE_HEALTH_ASSESSMENT, health_assessment_node)
    builder.add_node(NODE_CONFIRM, confirm_node)

    # --- Entry point ---
    builder.set_entry_point(NODE_SUPERVISOR)

    # --- Supervisor routes conditionally to one of the specialist agents ---
    builder.add_conditional_edges(
        NODE_SUPERVISOR,
        route_to_agent,
        {
            NODE_EXPENSE_ANALYSIS:  NODE_EXPENSE_ANALYSIS,
            NODE_BUDGET_PLANNING:   NODE_BUDGET_PLANNING,
            NODE_GOAL_PLANNING:     NODE_GOAL_PLANNING,
            NODE_ANOMALY_DETECTION: NODE_ANOMALY_DETECTION,
            NODE_HEALTH_ASSESSMENT: NODE_HEALTH_ASSESSMENT,
            NODE_CONFIRM:           NODE_CONFIRM,
            END:                    END,
        },
    )

    # --- After each specialist agent, go back to supervisor (or confirm) ---
    for agent_node in [
        NODE_EXPENSE_ANALYSIS,
        NODE_BUDGET_PLANNING,
        NODE_GOAL_PLANNING,
        NODE_ANOMALY_DETECTION,
        NODE_HEALTH_ASSESSMENT,
    ]:
        builder.add_conditional_edges(
            agent_node,
            route_after_agent,
            {
                NODE_CONFIRM:    NODE_CONFIRM,
                NODE_SUPERVISOR: NODE_SUPERVISOR,
            },
        )

    # --- After confirm node, always return to supervisor ---
    builder.add_edge(NODE_CONFIRM, NODE_SUPERVISOR)

    # --- Compile with checkpointer for state persistence and HITL support ---
    graph = builder.compile(
        checkpointer=memory_checkpointer,
        interrupt_before=[NODE_CONFIRM],  # pause before confirm node for HITL
    )

    return graph
