"""
SmartFin minimalist chat UI.

Visualises the LangGraph supervisor pipeline by streaming each node's
update to its own chat bubble. HITL pauses (set by an agent via
pending_confirmation) render as an Approve/Reject card.

Run with:  streamlit run ui/app.py
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage

REPO_ROOT = Path(__file__).resolve().parents[1]
# Streamlit launches this file as __main__, so the package root isn't
# implicitly on sys.path — add it before importing app.*
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.orchestrator import app_graph, get_pending_interrupt  # noqa: E402
from app.state import Transaction  # noqa: E402

SAMPLE_TXNS_PATH = REPO_ROOT / "tests" / "fixtures" / "sample_transactions.json"

AGENT_META: dict[str, tuple[str, str]] = {
    "supervisor":         ("🧭", "Supervisor"),
    "expense_analysis":   ("📊", "Expense Analysis"),
    "budget_planning":    ("💰", "Budget Planning"),
    "goal_planning":      ("🎯", "Goal Planning"),
    "anomaly_detection":  ("🚨", "Anomaly Detection"),
    "health_assessment":  ("❤️", "Health Assessment"),
    "confirm":            ("✋", "HITL Confirm"),
}

STUB_NODES = {"budget_planning", "goal_planning", "anomaly_detection", "health_assessment"}


# ---------------------------------------------------------------------------
# Page setup + session state
# ---------------------------------------------------------------------------

st.set_page_config(page_title="SmartFin Chat", page_icon="💬", layout="wide")

st.session_state.setdefault("thread_id", f"ui-{uuid.uuid4().hex[:8]}")
st.session_state.setdefault("trace", [])
st.session_state.setdefault("monthly_income", 3200.0)
st.session_state.setdefault("use_sample", True)
st.session_state.setdefault("pending_prompt", None)


def graph_config() -> dict:
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def load_sample_transactions() -> list[Transaction]:
    data = json.loads(SAMPLE_TXNS_PATH.read_text())
    return [Transaction(**row) for row in data]


# ---------------------------------------------------------------------------
# Trace rendering
# ---------------------------------------------------------------------------

def summarise_update(node: str, update: dict) -> str:
    if node == "supervisor":
        active = update.get("active_agent")
        queue = update.get("agents_queue") or []
        if active == "end":
            return "All planned agents completed → ending graph."
        parts = [f"Dispatching → `{active}`"]
        if queue:
            parts.append("Remaining queue: " + " → ".join(f"`{a}`" for a in queue))
        return "\n\n".join(parts)

    if node == "expense_analysis":
        cats = update.get("categorised_transactions") or []
        trends = update.get("spending_trends") or []
        pending = update.get("pending_confirmation")
        parts = [f"Categorised **{len(cats)}** transactions across **{len(trends)}** categories."]
        if pending:
            parts.append(f"⏸ Awaiting user confirmation — _{pending.get('summary', '')}_")
        return "\n\n".join(parts)

    if node in STUB_NODES:
        return "_(stub agent — returns no output yet)_"

    if node == "confirm":
        return "Confirmation processed → handing back to supervisor."

    return "_(no output)_"


def render_entry(entry: dict) -> None:
    role = entry["role"]
    if role == "agent":
        emoji, name = AGENT_META.get(entry["agent"], ("🤖", entry["agent"]))
        with st.chat_message("assistant", avatar=emoji):
            st.markdown(f"**{name}**  \n`{entry['agent']}`")
            st.markdown(entry["content"])
    else:
        with st.chat_message(role):
            st.markdown(entry["content"])


def record_and_render(entry: dict) -> None:
    st.session_state.trace.append(entry)
    render_entry(entry)


# ---------------------------------------------------------------------------
# Graph invocation
# ---------------------------------------------------------------------------

def stream_graph(inputs: dict | None) -> None:
    try:
        for chunk in app_graph.stream(inputs, graph_config(), stream_mode="updates"):
            for node, update in chunk.items():
                if update is None:
                    continue
                record_and_render({
                    "role": "agent",
                    "agent": node,
                    "content": summarise_update(node, update),
                })
    except Exception as exc:
        record_and_render({
            "role": "agent",
            "agent": "supervisor",
            "content": f"❌ **Error during graph execution:** `{type(exc).__name__}: {exc}`",
        })


def run_turn(user_msg: str) -> None:
    record_and_render({"role": "user", "content": user_msg})
    inputs = {
        "messages": [HumanMessage(content=user_msg)],
        "transactions": load_sample_transactions() if st.session_state.use_sample else [],
        "monthly_income": st.session_state.monthly_income,
    }
    stream_graph(inputs)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("💬 SmartFin")
    st.caption(f"Thread: `{st.session_state.thread_id}`")

    if st.button("🔄 New session", use_container_width=True):
        st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:8]}"
        st.session_state.trace = []
        st.session_state.pending_prompt = None
        st.rerun()

    st.divider()
    st.session_state.monthly_income = st.number_input(
        "Monthly income (£)",
        value=float(st.session_state.monthly_income),
        step=100.0,
        min_value=0.0,
    )
    st.session_state.use_sample = st.checkbox(
        "Load 28 sample transactions",
        value=st.session_state.use_sample,
        help="Loads tests/fixtures/sample_transactions.json into the graph state.",
    )

    st.divider()
    st.caption("Quick prompts")
    quick_prompts = {
        "📊 Analyse spending":   "Analyse my spending",
        "💰 Budget help":         "Help me plan a budget",
        "🎯 Savings goal":        "I want to set a savings goal",
        "🚨 Suspicious txns":     "Flag any suspicious transactions",
        "❤️ Financial health":    "How is my financial health?",
    }
    for label, prompt in quick_prompts.items():
        if st.button(label, use_container_width=True, key=f"qp-{label}"):
            st.session_state.pending_prompt = prompt

    st.divider()
    with st.expander("How the flow works"):
        st.markdown(
            "1. **Supervisor** classifies the message and queues specialist agents.\n"
            "2. Each agent runs in turn and may set a `pending_confirmation`.\n"
            "3. If set, the graph **pauses before** the `confirm` node — you approve or reject.\n"
            "4. After confirmation, the next queued agent runs.\n"
            "5. When the queue empties, the graph terminates.\n\n"
            "Only `expense_analysis` is implemented today; the other four are stubs."
        )


# ---------------------------------------------------------------------------
# Main chat
# ---------------------------------------------------------------------------

st.title("SmartFin Multi-Agent Chat")
st.caption(
    "Type a message. Watch the supervisor dispatch specialist agents through the LangGraph pipeline, "
    "with a human-in-the-loop checkpoint after expense analysis."
)

for entry in st.session_state.trace:
    render_entry(entry)

# Quick-prompt button click is consumed in the same run via pending_prompt.
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    run_turn(prompt)

user_msg = st.chat_input("Ask SmartFin… (e.g. 'Analyse my spending')")
if user_msg:
    run_turn(user_msg)


# ---------------------------------------------------------------------------
# HITL pause card
# ---------------------------------------------------------------------------

paused_state = get_pending_interrupt(app_graph, graph_config())
pending_payload = (paused_state or {}).get("pending_confirmation") if paused_state else None
is_paused = bool(pending_payload) and "confirmed" not in pending_payload

if is_paused:
    st.divider()
    st.subheader("✋ Human-in-the-loop confirmation")
    st.markdown(
        f"**Agent:** `{pending_payload.get('agent', '?')}`  \n"
        f"**Action:** `{pending_payload.get('action', '?')}`"
    )
    st.info(pending_payload.get("summary", ""))

    confidence = pending_payload.get("categorisation_confidence")
    if confidence:
        st.caption(f"Categorisation confidence: `{confidence}`")

    details = pending_payload.get("details") or []
    if details:
        with st.expander(f"Details ({len(details)} lines)"):
            st.code("\n".join(details), language="text")

    col_ok, col_no = st.columns(2)
    approved = col_ok.button("✅ Approve & continue", type="primary", use_container_width=True)
    rejected = col_no.button("❌ Reject", use_container_width=True)
    if approved or rejected:
        record_and_render({
            "role": "user",
            "content": "✅ Approved — continuing." if approved else "❌ Rejected — continuing anyway.",
        })
        stream_graph({"pending_confirmation": {"confirmed": bool(approved)}})
        st.rerun()


# ---------------------------------------------------------------------------
# Final-state panel (only when not paused and we have results to show)
# ---------------------------------------------------------------------------

else:
    snapshot = app_graph.get_state(graph_config())
    final = snapshot.values if snapshot else None
    trends = (final or {}).get("spending_trends") or []
    cats = (final or {}).get("categorised_transactions") or []

    if trends or cats:
        with st.expander(f"📊 Final state — {len(cats)} txns, {len(trends)} categories", expanded=False):
            if trends:
                st.markdown("**Spending trends (last 30 days vs prior 30):**")
                rows = []
                max_total = max((t.current_period_total for t in trends), default=1.0) or 1.0
                for t in trends:
                    dev = f"{t.deviation_pct:+.1f}%" if t.deviation_pct is not None else "—"
                    bar = "█" * max(1, int(20 * t.current_period_total / max_total))
                    rows.append(
                        f"- `{t.category.value:<14}` £{t.current_period_total:>8.2f}  {bar}  _{dev}_"
                    )
                st.markdown("\n".join(rows))

            if cats:
                st.markdown(f"**Categorised transactions:** {len(cats)} total")
                with st.expander("Show transaction list"):
                    for txn in cats[:50]:
                        st.markdown(
                            f"- `{txn.date.date()}` £{txn.amount:>7.2f}  "
                            f"**{txn.category.value}**  — {txn.merchant} — {txn.description}"
                        )
                    if len(cats) > 50:
                        st.caption(f"…and {len(cats) - 50} more.")
