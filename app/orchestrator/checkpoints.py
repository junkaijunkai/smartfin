"""
Checkpointer setup and Human-in-the-Loop (HITL) utilities.

Two distinct concepts live here:

1. Checkpointer  — LangGraph's built-in state persistence layer.
   After every node execution, LangGraph serialises the full AppState
   and stores it via the checkpointer.  This lets us:
     - Resume a paused graph (e.g. after HITL interrupt) without losing state.
     - Replay or inspect any past execution step.

   In production you would swap MemorySaver for SqliteSaver or a
   Postgres-backed checkpointer.  For development / demo, MemorySaver
   (in-process dict) is sufficient.

2. HITL helpers — thin wrappers that the UI / CLI layer calls to
   resume a graph that has been paused by interrupt_before.
   The graph itself declares WHICH nodes trigger an interrupt
   (see graph.py); these helpers handle the resume flow.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# ---------------------------------------------------------------------------
# Checkpointer instance
# ---------------------------------------------------------------------------
# A single shared instance is fine for in-process use.
# Import this wherever you need to compile or resume the graph.

# Register all Pydantic state models so LangGraph's JsonPlusSerializer doesn't
# emit a WARNING for each object when deserializing checkpointed state.
_STATE_MODULE = "app.state"
_serde = JsonPlusSerializer(
    allowed_msgpack_modules=[
        (_STATE_MODULE, "Transaction"),
        (_STATE_MODULE, "BudgetAllocation"),
        (_STATE_MODULE, "FinancialGoal"),
        (_STATE_MODULE, "AnomalyFlag"),
        (_STATE_MODULE, "SpendingTrend"),
        (_STATE_MODULE, "HealthSummary"),
        (_STATE_MODULE, "Alert"),
        (_STATE_MODULE, "TransactionCategory"),
        (_STATE_MODULE, "AnomalyType"),
        (_STATE_MODULE, "HealthRating"),
        (_STATE_MODULE, "AlertSeverity"),
    ]
)

memory_checkpointer = MemorySaver(serde=_serde)


# ---------------------------------------------------------------------------
# HITL helpers
# ---------------------------------------------------------------------------

def get_pending_interrupt(graph, config: dict) -> dict | None:
    """
    Return the current AppState if the graph is paused at an interrupt,
    or None if it has already finished.

    Usage:
        state = get_pending_interrupt(app_graph, config)
        if state:
            print("Waiting for user confirmation:", state["pending_confirmation"])
    """
    snapshot = graph.get_state(config)
    # snapshot.next is a tuple of node names that are about to execute.
    # If it's non-empty the graph is paused and waiting.
    if snapshot.next:
        return snapshot.values
    return None


def resume_with_confirmation(graph, config: dict, confirmed: bool, user_message: str | None = None) -> dict:
    """
    Resume a paused graph after the user has accepted, rejected, or provided
    additional information for the pending action.

    LangGraph resumes by calling graph.invoke(state_update, config).
    Passing None as the first argument means "use the existing checkpointed
    state"; we only need to patch the fields that changed.

    Args:
        graph:       The compiled StateGraph (returned by build_graph()).
        config:      The same thread config dict that was used to start the run.
                     Must contain {"configurable": {"thread_id": "..."}}.
        confirmed:   True = user approved the pending action,
                     False = user rejected/cancelled it.
        user_message: (Optional) When action is "clarify_*", user can provide
                     supplementary information here. The message is appended
                     to the messages list and the graph re-routes to the
                     original agent with new context.

    Returns:
        The final AppState dict after the graph resumes and finishes.

    Example usage for clarification:
        >>> # User rejected missing fields prompt and now provides clarification
        >>> resume_with_confirmation(
        ...     graph, config,
        ...     confirmed=True,  # confirmed=True indicates "user provided info"
        ...     user_message="I want to save $8000 by June 2027"
        ... )
    """
    from langchain_core.messages import HumanMessage

    # Patch the fields that communicate the user's decision back to the graph
    update = {
        "pending_confirmation": {"confirmed": confirmed},
    }

    # If user provided clarification, append it to messages and reset active_agent
    # so the supervisor re-routes to the appropriate agent with new context
    if user_message and confirmed:
        messages = list((update.get("messages") or []) or [])
        messages.append(HumanMessage(content=user_message))
        update["messages"] = messages

        # Reset active_agent so supervisor makes a fresh routing decision
        # based on the new user message
        update["active_agent"] = None

    return graph.invoke(update, config)
