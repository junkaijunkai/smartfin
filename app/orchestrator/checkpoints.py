"""
Checkpointer setup and Human-in-the-Loop (HITL) utilities.

Two distinct concepts live here:

1. Checkpointer  — LangGraph's built-in state persistence layer.
   After every node execution, LangGraph serialises the full AppState
   and stores it via the checkpointer.  This lets us:
     - Resume a paused graph (e.g. after HITL interrupt) without losing state.
     - Replay or inspect any past execution step.
     - Rewind to an earlier checkpoint and re-run (used by the UI's
       edit-and-resend feature).

   We persist checkpoints in a local SQLite file under .smartfin/
   so sessions survive server restarts.

2. HITL helpers — thin wrappers that the UI / CLI layer calls to
   resume a graph that has been paused by interrupt_before.
   The graph itself declares WHICH nodes trigger an interrupt
   (see graph.py); these helpers handle the resume flow.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

# ---------------------------------------------------------------------------
# Checkpointer instance
# ---------------------------------------------------------------------------

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

_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _REPO_ROOT / ".smartfin"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DB_PATH = DATA_DIR / "chatbot.db"

# check_same_thread=False because Streamlit serves each rerun on a different
# worker thread; the sqlite3 driver's per-thread default would raise.
_conn = sqlite3.connect(str(CHECKPOINT_DB_PATH), check_same_thread=False)
memory_checkpointer = SqliteSaver(_conn, serde=_serde)
memory_checkpointer.setup()
# SqliteSaver.setup() flips the DB to WAL mode. The .db-shm memory-mapped file
# WAL uses is flaky on some virtualised / bind-mounted filesystems (we've seen
# SIGBUS on concurrent access). Rollback journal ("DELETE" mode) is safer for
# single-process Streamlit and still correct.
_conn.execute("PRAGMA journal_mode=DELETE")
_conn.commit()


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
