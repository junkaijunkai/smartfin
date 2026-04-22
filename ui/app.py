"""
SmartFin minimalist chat UI.

Streams each LangGraph node update to its own chat bubble. When an agent sets
pending_confirmation, the graph pauses before the confirm node and the UI
renders an Approve/Reject card. Free-text while paused is treated as a
clarification message and resumes the graph with the new context.

Run with:  streamlit run ui/app.py
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import date
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

REPO_ROOT = Path(__file__).resolve().parents[1]
# Streamlit launches this file as __main__, so the package root isn't
# implicitly on sys.path — add it before importing app.*
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from app.orchestrator import (  # noqa: E402
    app_graph,
    get_pending_interrupt,
)
from app.state import Transaction, TransactionCategory  # noqa: E402

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


# ---------------------------------------------------------------------------
# Page setup + session state
# ---------------------------------------------------------------------------

st.set_page_config(page_title="SmartFin Chat", page_icon="💬", layout="wide")

st.session_state.setdefault("thread_id", f"ui-{uuid.uuid4().hex[:8]}")
st.session_state.setdefault("trace", [])
st.session_state.setdefault("monthly_income", 3200.0)
st.session_state.setdefault("use_sample", True)
st.session_state.setdefault("pending_prompt", None)
st.session_state.setdefault("transactions_sent", False)


def graph_config() -> dict:
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def load_sample_transactions() -> list[Transaction]:
    data = json.loads(SAMPLE_TXNS_PATH.read_text())
    return [
        Transaction(
            id=row["id"],
            date=row["date"],
            amount=row["amount"],
            description=row["description"],
            merchant=row["merchant"],
            category=TransactionCategory(row["category"]),
            location=row.get("location"),
        )
        for row in data
    ]


# ---------------------------------------------------------------------------
# Per-node update summaries
# ---------------------------------------------------------------------------

def _latest_ai_text(update: dict) -> str | None:
    msgs = update.get("messages") or []
    for msg in reversed(msgs):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
            return msg.content
    return None


def summarise_update(node: str, update: dict) -> str:
    if node == "supervisor":
        # Supervisor may surface a prompt (unknown intent / no data)
        ai = _latest_ai_text(update)
        if ai:
            return ai

        active = update.get("active_agent")
        queue = update.get("agents_queue") or []
        if active == "end" or active is None:
            return "All planned work complete — ending graph."
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

    if node == "budget_planning":
        pending = update.get("pending_confirmation")
        if pending and pending.get("action", "").startswith("clarify"):
            return f"⏸ Needs clarification — _{pending.get('summary', '')}_"

        allocs = update.get("budget_allocations") or []
        warnings = update.get("budget_warnings") or []
        summary = update.get("budget_summary") or ""
        lines = [f"**{len(allocs)}** budget allocation(s), **{len(warnings)}** warning(s)."]
        if summary:
            lines.append(f"_{summary}_")
        ai = _latest_ai_text(update)
        if ai:
            lines.append(f"```text\n{ai}\n```")
        return "\n\n".join(lines)

    if node == "goal_planning":
        pending = update.get("pending_confirmation") or {}
        goals = update.get("goals") or []
        action = pending.get("action", "")
        if action.startswith("clarify"):
            parts = [f"⏸ Needs clarification — _{pending.get('summary', '')}_"]
            for d in pending.get("details", []):
                parts.append(f"- {d}")
            return "\n".join(parts)
        parts = [f"Tracking **{len(goals)}** goal(s). _{pending.get('summary', '')}_"]
        for d in pending.get("details", []):
            parts.append(f"- {d}")
        return "\n".join(parts)

    if node == "anomaly_detection":
        flags = update.get("anomaly_flags") or []
        explanation = update.get("anomaly_explanation") or ""
        parts = [f"Scanned transactions — **{len(flags)}** anomaly flag(s)."]
        if explanation:
            parts.append(explanation)
        return "\n\n".join(parts)

    if node == "health_assessment":
        hs = update.get("health_summary")
        alerts = update.get("alerts") or []
        if hs is None:
            return "_(no health summary produced)_"
        rating = hs.rating.value if hasattr(hs.rating, "value") else str(hs.rating)
        lines = [
            f"**Rating:** `{rating.upper()}`",
            f"**DTI:** {hs.debt_to_income_ratio:.0%}  ·  "
            f"**Reserves:** {hs.liquid_reserve_months:.1f} months",
        ]
        if hs.income_concentration_risk:
            lines.append("⚠ Income concentration risk")
        if hs.sustained_overspending:
            lines.append("⚠ Sustained overspending")
        if hs.observations:
            lines.append("**Observations:**")
            for obs in hs.observations:
                lines.append(f"- {obs}")
        if alerts:
            lines.append(f"**Alerts:** {len(alerts)} total")
        return "\n".join(lines)

    if node == "confirm":
        return "Confirmation processed → handing back to supervisor."

    return "_(no output)_"


# ---------------------------------------------------------------------------
# Trace rendering
# ---------------------------------------------------------------------------

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
                if update is None or node.startswith("__"):
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
    """Fresh turn — user typed a message with no pending HITL."""
    record_and_render({"role": "user", "content": user_msg})

    inputs: dict = {
        "messages": [HumanMessage(content=user_msg)],
        "monthly_income": st.session_state.monthly_income,
        "current_date": date.today().isoformat(),
    }
    # Transactions are persisted by the backend checkpointer + disk cache,
    # so we only send them on the first turn of a session.
    if not st.session_state.transactions_sent and st.session_state.use_sample:
        inputs["transactions"] = load_sample_transactions()
        inputs["goals"] = []
        st.session_state.transactions_sent = True

    stream_graph(inputs)


def run_clarification(user_msg: str) -> None:
    """User typed free-text while HITL-paused — treat as clarification.

    Equivalent to resume_with_confirmation(confirmed=True, user_message=...)
    but streams per-node updates so the UI shows progress.
    """
    record_and_render({"role": "user", "content": user_msg})
    update = {
        "pending_confirmation": {"confirmed": True},
        "messages": [HumanMessage(content=user_msg)],
        "active_agent": None,  # force supervisor to re-classify with new message
    }
    stream_graph(update)


def run_resume(confirmed: bool) -> None:
    """User clicked Approve or Reject on the HITL card."""
    record_and_render({
        "role": "user",
        "content": "✅ Approved — continuing." if confirmed else "❌ Rejected — continuing.",
    })
    update = {"pending_confirmation": {"confirmed": confirmed}}
    for chunk in app_graph.stream(update, graph_config(), stream_mode="updates"):
        for node, delta in chunk.items():
            if delta is None or node.startswith("__"):
                continue
            record_and_render({
                "role": "agent",
                "agent": node,
                "content": summarise_update(node, delta),
            })


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
        st.session_state.transactions_sent = False
        st.rerun()

    st.divider()
    st.session_state.monthly_income = st.number_input(
        "Monthly income (£)",
        value=float(st.session_state.monthly_income),
        step=100.0,
        min_value=0.0,
    )
    st.session_state.use_sample = st.checkbox(
        "Load 28 sample transactions on first turn",
        value=st.session_state.use_sample,
        help="Loads tests/fixtures/sample_transactions.json into the graph state.",
    )

    st.divider()
    st.caption("Quick prompts")
    quick_prompts = {
        "📊 Analyse spending":   "Analyse my spending",
        "💰 Budget help":         "Help me plan a budget",
        "🎯 Savings goal":        "I want to save £8000 for an emergency fund by June 2027",
        "🚨 Suspicious txns":     "Flag any suspicious transactions",
        "❤️ Financial health":    "How is my financial health?",
    }
    for label, prompt in quick_prompts.items():
        if st.button(label, use_container_width=True, key=f"qp-{label}"):
            st.session_state.pending_prompt = prompt

    st.divider()
    with st.expander("How the flow works"):
        st.markdown(
            "1. **Supervisor** classifies intent and queues specialist agents.\n"
            "2. Each agent runs and may set a `pending_confirmation`.\n"
            "3. The graph **pauses before** the `confirm` node — approve, reject, or type a clarification.\n"
            "4. Clarifications append a new user message and re-route through the supervisor.\n"
            "5. When the queue empties, the graph terminates."
        )


# ---------------------------------------------------------------------------
# Main chat
# ---------------------------------------------------------------------------

st.title("SmartFin Multi-Agent Chat")
st.caption(
    "Type a message. Watch the supervisor dispatch specialist agents through the LangGraph pipeline, "
    "with human-in-the-loop checkpoints after sensitive actions."
)

for entry in st.session_state.trace:
    render_entry(entry)

paused_state = get_pending_interrupt(app_graph, graph_config())
pending_payload = (paused_state or {}).get("pending_confirmation") if paused_state else None
is_paused = bool(pending_payload) and "confirmed" not in pending_payload

# Quick-prompt click is consumed in the same run via pending_prompt.
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    if is_paused:
        run_clarification(prompt)
    else:
        run_turn(prompt)
    st.rerun()

placeholder = (
    "Provide clarification or details…"
    if is_paused
    else "Ask SmartFin… (e.g. 'Analyse my spending')"
)
user_msg = st.chat_input(placeholder)
if user_msg:
    if is_paused:
        run_clarification(user_msg)
    else:
        run_turn(user_msg)
    st.rerun()


# ---------------------------------------------------------------------------
# HITL pause card
# ---------------------------------------------------------------------------

if is_paused:
    st.divider()
    st.subheader("✋ Human-in-the-loop confirmation")
    st.markdown(
        f"**Agent:** `{pending_payload.get('agent', '?')}`  \n"
        f"**Action:** `{pending_payload.get('action', '?')}`"
    )
    st.info(pending_payload.get("summary", ""))

    confidence = (
        pending_payload.get("categorisation_confidence")
        or pending_payload.get("goal_extraction_confidence")
    )
    if confidence:
        st.caption(f"Confidence: `{confidence}`")

    details = pending_payload.get("details") or []
    if details:
        with st.expander(f"Details ({len(details)} lines)", expanded=True):
            for line in details:
                st.markdown(f"- {line}")

    col_ok, col_no = st.columns(2)
    approved = col_ok.button("✅ Approve & continue", type="primary", use_container_width=True)
    rejected = col_no.button("❌ Reject", use_container_width=True)
    if approved or rejected:
        run_resume(confirmed=bool(approved))
        st.rerun()

    st.caption(
        "Tip: type a clarification in the chat box above instead of clicking to "
        "refine the request — the graph will re-route with your new message."
    )


# ---------------------------------------------------------------------------
# Final-state panel
# ---------------------------------------------------------------------------

else:
    snapshot = app_graph.get_state(graph_config())
    final = (snapshot.values if snapshot else None) or {}

    trends = final.get("spending_trends") or []
    cats = final.get("categorised_transactions") or []
    allocations = final.get("budget_allocations") or []
    goals = final.get("goals") or []
    anomaly_flags = final.get("anomaly_flags") or []
    health = final.get("health_summary")
    alerts = final.get("alerts") or []

    has_any = any([trends, cats, allocations, goals, anomaly_flags, health, alerts])
    if not has_any:
        st.stop()

    st.divider()
    st.subheader("📋 Current session state")

    cols = st.columns(3)
    cols[0].metric("Transactions", len(cats))
    cols[1].metric("Goals", len(goals))
    cols[2].metric("Alerts", len(alerts))

    if trends:
        with st.expander(f"📊 Spending trends ({len(trends)} categories)", expanded=False):
            max_total = max((t.current_period_total for t in trends), default=1.0) or 1.0
            for t in trends:
                dev = f"{t.deviation_pct:+.1f}%" if t.deviation_pct is not None else "—"
                bar = "█" * max(1, int(20 * t.current_period_total / max_total))
                st.markdown(
                    f"- `{t.category.value:<14}` £{t.current_period_total:>8.2f}  {bar}  _{dev}_"
                )

    if allocations:
        with st.expander(f"💰 Budget allocations ({len(allocations)} categories)", expanded=False):
            progress = final.get("budget_progress") or {}
            for a in sorted(allocations, key=lambda x: x.allocated_amount, reverse=True):
                status = progress.get(a.category.value, {}).get("status", "on_track")
                icon = {"exceeded": "❌", "near_limit": "⚠", "on_track": "✓"}.get(status, "•")
                st.markdown(
                    f"- {icon} `{a.category.value:<14}` "
                    f"budget £{a.allocated_amount:.2f}  ·  "
                    f"spent £{a.spent_amount:.2f}  ·  "
                    f"remaining £{a.remaining:.2f}"
                )
            warnings = final.get("budget_warnings") or []
            if warnings:
                st.markdown("**Warnings:**")
                for w in warnings:
                    sev_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                        w.get("severity", ""), "•"
                    )
                    st.markdown(f"- {sev_icon} {w.get('message', '')}")

    if goals:
        with st.expander(f"🎯 Goals ({len(goals)})", expanded=False):
            for g in goals:
                state_icon = "✓" if g.on_track else "⚠"
                st.markdown(
                    f"- {state_icon} **{g.name}** — "
                    f"target £{g.target_amount:.2f} by {g.target_date.isoformat()}  ·  "
                    f"saved £{g.current_amount:.2f}  ·  "
                    f"need £{g.required_monthly_saving:.2f}/month"
                )

    if anomaly_flags or final.get("anomaly_explanation"):
        with st.expander(f"🚨 Anomalies ({len(anomaly_flags)} flag(s))", expanded=False):
            explanation = final.get("anomaly_explanation")
            if explanation:
                st.markdown(explanation)
            for f in anomaly_flags[:20]:
                st.markdown(
                    f"- `{f.transaction_id}` — **{f.anomaly_type.value}** — {f.explanation}"
                )

    if health:
        with st.expander(f"❤️ Health summary ({health.rating.value.upper()})", expanded=False):
            st.markdown(
                f"- DTI: `{health.debt_to_income_ratio:.0%}`\n"
                f"- Reserves: `{health.liquid_reserve_months:.1f}` months\n"
                f"- Income concentration risk: `{health.income_concentration_risk}`\n"
                f"- Sustained overspending: `{health.sustained_overspending}`"
            )
            for obs in health.observations:
                st.markdown(f"  • {obs}")

    if alerts:
        with st.expander(f"🔔 Alerts ({len(alerts)})", expanded=False):
            for a in alerts[-20:]:
                sev = a.severity.value if hasattr(a.severity, "value") else str(a.severity)
                st.markdown(f"- `[{sev.upper()}]` {a.message}  _(from {a.source_agent})_")

    if cats:
        with st.expander(f"🧾 Categorised transactions ({len(cats)})", expanded=False):
            for txn in cats[:50]:
                st.markdown(
                    f"- `{txn.date.date()}` £{txn.amount:>7.2f}  "
                    f"**{txn.category.value}**  — {txn.merchant} — {txn.description}"
                )
            if len(cats) > 50:
                st.caption(f"…and {len(cats) - 50} more.")
