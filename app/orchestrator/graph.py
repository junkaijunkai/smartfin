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
from langchain_core.runnables import RunnableConfig

from app.state import AppState
from app.orchestrator.checkpoints import memory_checkpointer
from app.orchestrator.intent_classifier import classify_intent
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
from app.tools.transaction_store import load_analysis


# ---------------------------------------------------------------------------
# Node implementations (stubs — replace with real agent imports)
# ---------------------------------------------------------------------------


def _resolve_analysis_cache(state: AppState, config: dict | None) -> dict:
    """
    Determine whether cached categorised data is available and inject it into state.

    Priority order:
      1. categorised_transactions already in state → no-op
      2. Cache file exists on disk → load and return as state update
      3. Neither → return empty dict
    """
    if state.get("categorised_transactions"):
        return {}

    thread_id = (config or {}).get("configurable", {}).get("thread_id")
    if thread_id:
        result = load_analysis(thread_id)
        if result:
            categorised, trends = result
            return {
                "categorised_transactions": categorised,
                "spending_trends": trends,
            }

    return {}


def supervisor_node(state: AppState, config: RunnableConfig | None = None) -> dict:
    """
    Routes user requests to appropriate worker agents based on intent and data availability.

    Five routing scenarios:
      1. state["categorised_transactions"] exists → worker (session cache)
      2. Disk cache exists + no new transactions → inject cache + worker
      3. Disk cache exists + new transactions → inject cache + expense_analysis(incremental) + worker
      4. No data anywhere → prompt user, don't call worker
      5. New transactions + no cache → expense_analysis(full) + worker
    """
    # --- Queue processing: if agents_queue has items, continue consuming ---
    queue = list(state.get("agents_queue", []))
    if queue:
        active_agent = queue.pop(0)
        return {"active_agent": active_agent, "agents_queue": queue}

    # --- Clear active_agent if previous agent finished ---
    if state.get("active_agent") not in (None, "end"):
        return {"active_agent": "end", "agents_queue": []}

    # --- Fresh routing: classify user intent using LLM ---
    messages = state.get("messages", [])
    last_message = messages[-1].content if messages else ""

    # Use LLM to classify intent; falls back to keyword matching on error
    agent_name = classify_intent(last_message)
    worker_agents = [agent_name]

    # --- Data availability check ---
    has_categorised = bool(state.get("categorised_transactions"))
    has_new_txns = bool(state.get("transactions"))

    cache_update: dict = {}
    planned: list = []

    # Scenario 1: Already have categorised in state (multi-call within same session)
    if has_categorised:
        planned = worker_agents

    else:
        disk_cache = _resolve_analysis_cache(state, config)

        if disk_cache and has_new_txns:
            # Scenario 3: Cache hit + new transactions → inject cache + incremental analysis
            # Avoid double-adding expense_analysis if that's already the intended worker
            if worker_agents[0] == "expense_analysis":
                planned = worker_agents
            else:
                planned = ["expense_analysis"] + worker_agents
            cache_update = disk_cache

        elif disk_cache and not has_new_txns:
            # Scenario 2: Cache hit, no new transactions → use cached data directly
            planned = worker_agents
            cache_update = disk_cache

        elif not disk_cache and has_new_txns:
            # Scenario 5: No cache, new transactions → full analysis needed
            # Avoid double-adding expense_analysis if that's already the intended worker
            if worker_agents[0] == "expense_analysis":
                planned = worker_agents
            else:
                planned = ["expense_analysis"] + worker_agents
            cache_update = {}

        else:
            # Scenario 4: No data anywhere → prompt user to provide transactions
            from langchain_core.messages import AIMessage

            prompt = (
                "I need transaction data to help you. "
                "Please provide your recent transactions so I can get started."
            )
            return {
                "active_agent": "end",
                "agents_queue": [],
                "messages": [AIMessage(content=prompt)],
            }

    active_agent = planned.pop(0)
    result = {"active_agent": active_agent, "agents_queue": planned}
    result.update(cache_update)
    return result


def expense_analysis_node(state: AppState) -> dict:
    from app.agents.expense_analysis.agent import run
    return run(state) 


def budget_planning_node(state: AppState) -> dict:
    from app.agents.budget_planning.agent import budget_planning_node as run_budget_planning
    return run_budget_planning(state)


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

    Two scenarios:
      1. User approved/rejected: just clear pending_confirmation
      2. User provided clarification: re-route to supervisor for fresh routing
         decision based on the new user message

    When user provides clarification (e.g., "I want to save $8000 by June 2027"
    in response to missing fields prompt), active_agent is set to None to
    trigger supervisor re-routing on the next cycle.
    """
    confirmation = state.get("pending_confirmation", {})
    action = confirmation.get("action")

    if confirmation.get("confirmed"):
        print(f"[confirm] User approved action: {action}")

        # If user provided clarification (indicated by active_agent being None),
        # the supervisor will route to the appropriate agent with the new message
        if state.get("active_agent") is None and len(state.get("messages", [])) > 0:
            print("[confirm] User provided clarification, supervisor will re-route.")
            return {"pending_confirmation": None}

        print("[confirm] User approved the pending action.")
    else:
        print(f"[confirm] User rejected the pending action: {action}")

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
