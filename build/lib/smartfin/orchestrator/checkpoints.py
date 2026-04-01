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

# ---------------------------------------------------------------------------
# Checkpointer instance
# ---------------------------------------------------------------------------
# A single shared instance is fine for in-process use.
# Import this wherever you need to compile or resume the graph.
memory_checkpointer = MemorySaver()


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


def resume_with_confirmation(graph, config: dict, confirmed: bool) -> dict:
    """
    Resume a paused graph after the user has accepted or rejected the
    pending action.

    LangGraph resumes by calling graph.invoke(state_update, config).
    Passing None as the first argument means "use the existing checkpointed
    state"; we only need to patch the fields that changed.

    Args:
        graph:     The compiled StateGraph (returned by build_graph()).
        config:    The same thread config dict that was used to start the run.
                   Must contain {"configurable": {"thread_id": "..."}}.
        confirmed: True = user approved the pending action,
                   False = user rejected it.

    Returns:
        The final AppState dict after the graph resumes and finishes.
    """
    # Patch only the fields that communicate the user's decision back to the graph.
    update = {
        "pending_confirmation": {"confirmed": confirmed},
    }
    return graph.invoke(update, config)
