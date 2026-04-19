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

from app.state import AppState
from app.orchestrator.checkpoints import memory_checkpointer
from app.orchestrator.router import (
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

    Stub behaviour: maintain a list of agents to run. 
    On each invocation, pop the top agent, and return to supervsior to check the queue state.
    Return end when the queue is empty.
    """

    # TODO: replace with LLM-based intent classification
    queue = list(state.get("agents_queue", []))
    
    # queue里有待跑agent
    if queue:
        active_agent = queue.pop(0)
        return {"active_agent": active_agent, "agents_queue": queue}
    
    # queue为空，但active_agent不为空（还是上一个agent）
    if state.get("active_agent")!=None and state.get("active_agent")!="end":
        return {"active_agent": "end", "agents_queue": []}

    # queue为空，根据message从头计划agent队列
    messages = state.get("messages", [])
    last_message = messages[-1].content if messages else ""
    msg = last_message.lower() 

    if "budget" in msg:
        planned = ["expense_analysis", "budget_planning"]
    #elif "goal" in msg:
        #planned = ["expense_analysis", "goal_planning"]
    elif any(kw in msg for kw in ["goal", "save", "saving", "fund", "deposit"]):
        planned = ["expense_analysis", "goal_planning"]
    elif any(kw in msg for kw in ["suspicious", "anomal"]):
        planned = ["anomaly_detection"]  # standalone; uses categorised_transactions if already in state
    elif any(kw in msg for kw in ["health", "risk"]):
        planned = ["expense_analysis", "health_assessment"]
    else:
        planned = ["expense_analysis", "anomaly_detection"]  # default: always chain anomaly after expense analysis

    active_agent = planned.pop(0)
    return {"active_agent": active_agent, "agents_queue": planned} # 记录状态，交给route_to_agent处理


def expense_analysis_node(state: AppState) -> dict:
    from app.agents.expense_analysis.agent import run
    return run(state) 


def budget_planning_node(state: AppState) -> dict:
    """Stub — to be replaced by smartfin.agents.budget_planning.agent"""
    print("[stub] budget_planning_node called")
    return {}


# def goal_planning_node(state: AppState) -> dict:
#     """Stub — to be replaced by smartfin.agents.goal_planning.agent"""
#     print("[stub] goal_planning_node called")
#     return {}
def goal_planning_node(state: AppState) -> dict:
    """
    Goal Planning node.

    Calls the real Goal Planning agent, which:
    - extracts goal information from the user's message
    - creates/evaluates goals
    - returns pending_confirmation for HITL
    """
    from app.agents.goal_planning.agent import run
    return run(state)


def anomaly_detection_node(state: AppState) -> dict:
    from app.agents.anomaly_detection.agent import run
    return run(state) # anomaly_detection can be called standalone or after expense_analysis


def health_assessment_node(state: AppState) -> dict:
    from app.agents.health_assessment.agent import run
    return run(state)


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
