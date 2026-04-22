"""
SmartFin chat UI with persistent sessions and edit-and-resend.

Session management:
  - Sessions live in .smartfin/chat_ui.db (see ui/sessions.py).
  - Graph checkpoints live in .smartfin/chatbot.db (SqliteSaver).
  - Both survive server restarts; the sidebar lists every prior chat.

Edit and resend:
  - Each user bubble has an ✏️ button.
  - Clicking it turns the bubble into a text area with Save / Cancel.
  - On save, the graph rewinds to the checkpoint captured just before
    that message was sent, invokes with the edited text, and re-streams
    the agent bubbles that follow. The trace in SQLite is truncated and
    rewritten.

Run with:  streamlit run ui/app.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from app.orchestrator import app_graph, get_pending_interrupt  # noqa: E402
from app.state import Transaction, TransactionCategory  # noqa: E402
from ui import sessions as sess  # noqa: E402

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

st.session_state.setdefault("monthly_income", 3200.0)
st.session_state.setdefault("use_sample", True)
st.session_state.setdefault("pending_prompt", None)
st.session_state.setdefault("editing_index", None)


def ensure_current_session() -> str:
    """Return a valid current thread_id, creating/picking one if needed."""
    tid = st.session_state.get("thread_id")
    if tid and sess.get_session(tid):
        return tid

    existing = sess.list_sessions()
    if existing:
        tid = existing[0]["thread_id"]
    else:
        tid = sess.create_session("New chat")

    st.session_state.thread_id = tid
    st.session_state.editing_index = None
    return tid


def graph_config(thread_id: str | None = None, checkpoint_id: str | None = None) -> dict:
    cfg: dict = {"configurable": {"thread_id": thread_id or st.session_state.thread_id}}
    if checkpoint_id:
        cfg["configurable"]["checkpoint_id"] = checkpoint_id
    return cfg


def current_checkpoint_id() -> str | None:
    """Checkpoint id at the tip of the current thread (None if empty)."""
    snap = app_graph.get_state(graph_config())
    if not snap or not snap.config:
        return None
    return snap.config.get("configurable", {}).get("checkpoint_id")


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


def relative_time(ts: float) -> str:
    delta = time.time() - ts
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    return datetime.fromtimestamp(ts).strftime("%d %b")


# ---------------------------------------------------------------------------
# Trace helpers (SQLite is the source of truth)
# ---------------------------------------------------------------------------

def get_trace() -> list[dict]:
    return sess.load_trace(st.session_state.thread_id)


def persist_trace(trace: list[dict]) -> None:
    sess.save_trace(st.session_state.thread_id, trace)
    sess.touch_session(st.session_state.thread_id)


def append_trace(entry: dict) -> None:
    trace = get_trace()
    trace.append(entry)
    persist_trace(trace)


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
        ai = _latest_ai_text(update)
        if ai:
            return ai
        active = update.get("active_agent")
        queue = update.get("agents_queue") or []
        if active in ("end", None):
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
# Graph invocation
# ---------------------------------------------------------------------------

def _turn_inputs(user_msg: str) -> dict:
    inputs: dict = {
        "messages": [HumanMessage(content=user_msg)],
        "monthly_income": st.session_state.monthly_income,
        "current_date": date.today().isoformat(),
    }
    if st.session_state.use_sample:
        # Backend dedupes by transaction id, so re-sending is cheap and keeps
        # forked/rewound threads correct without us tracking send state.
        inputs["transactions"] = load_sample_transactions()
        inputs["goals"] = []
    return inputs


def stream_into_trace(
    inputs: dict | None,
    config: dict | None = None,
    trace: list[dict] | None = None,
) -> list[dict]:
    """Stream graph updates, appending each node's summary to `trace`.

    Persists after each chunk so reloads show partial progress.
    """
    cfg = config or graph_config()
    if trace is None:
        trace = get_trace()

    try:
        for chunk in app_graph.stream(inputs, cfg, stream_mode="updates"):
            for node, update in chunk.items():
                if update is None or node.startswith("__"):
                    continue
                trace.append({
                    "role": "agent",
                    "agent": node,
                    "content": summarise_update(node, update),
                })
                persist_trace(trace)
    except Exception as exc:
        trace.append({
            "role": "agent",
            "agent": "supervisor",
            "content": f"❌ **Error during graph execution:** `{type(exc).__name__}: {exc}`",
        })
        persist_trace(trace)
    return trace


def run_turn(user_msg: str) -> None:
    """Fresh turn — user typed a message with no pending HITL."""
    cp_before = current_checkpoint_id()
    trace = get_trace()
    trace.append({"role": "user", "content": user_msg, "checkpoint_id": cp_before})

    # Auto-title from the first user message if the session is still named "New chat"
    info = sess.get_session(st.session_state.thread_id)
    if info and info["title"] == "New chat":
        sess.rename_session(st.session_state.thread_id, sess.auto_title(user_msg))

    persist_trace(trace)
    stream_into_trace(_turn_inputs(user_msg), trace=trace)


def run_clarification(user_msg: str) -> None:
    """User typed free-text while HITL-paused — clarification resume."""
    trace = get_trace()
    trace.append({"role": "user", "content": user_msg})
    persist_trace(trace)
    update = {
        "pending_confirmation": {"confirmed": True},
        "messages": [HumanMessage(content=user_msg)],
        "active_agent": None,
    }
    stream_into_trace(update, trace=trace)


def run_resume(confirmed: bool) -> None:
    """User clicked Approve or Reject on the HITL card."""
    trace = get_trace()
    trace.append({
        "role": "user",
        "content": "✅ Approved — continuing." if confirmed else "❌ Rejected — continuing.",
    })
    persist_trace(trace)
    stream_into_trace({"pending_confirmation": {"confirmed": confirmed}}, trace=trace)


def run_edit_resend(index: int, new_text: str) -> None:
    """Rewind to the checkpoint before trace[index] and resend the edited text."""
    trace = get_trace()
    if index < 0 or index >= len(trace):
        return
    entry = trace[index]
    if entry.get("role") != "user":
        return

    checkpoint_id = entry.get("checkpoint_id")
    tid = st.session_state.thread_id

    # Truncate trace to exclude the edited message and everything after.
    trimmed = trace[:index]

    if checkpoint_id is None:
        # First message in the thread — no prior checkpoint. Wipe graph state
        # for this thread_id so the invoke below starts from a clean slate.
        try:
            app_graph.checkpointer.delete_thread(tid)
        except Exception:
            pass
        cfg = graph_config()
    else:
        cfg = graph_config(checkpoint_id=checkpoint_id)

    # Re-record the (edited) user entry with the same parent checkpoint_id
    trimmed.append({"role": "user", "content": new_text, "checkpoint_id": checkpoint_id})
    persist_trace(trimmed)

    # If the user edited the very first message, refresh the auto-title too.
    if index == 0:
        info = sess.get_session(tid)
        if info and info["title"].startswith(("New chat", "Untitled")):
            sess.rename_session(tid, sess.auto_title(new_text))

    stream_into_trace(_turn_inputs(new_text), config=cfg, trace=trimmed)


# ---------------------------------------------------------------------------
# Sidebar — sessions
# ---------------------------------------------------------------------------

ensure_current_session()

with st.sidebar:
    st.title("💬 SmartFin")

    if st.button("＋ New chat", type="primary", use_container_width=True):
        new_tid = sess.create_session("New chat")
        st.session_state.thread_id = new_tid
        st.session_state.editing_index = None
        st.rerun()

    st.divider()
    st.caption("Chats")

    for s in sess.list_sessions():
        is_current = s["thread_id"] == st.session_state.thread_id
        label = f"{'▸ ' if is_current else ''}{s['title']}"
        cols = st.columns([5, 1])
        if cols[0].button(
            label,
            key=f"pick-{s['thread_id']}",
            use_container_width=True,
            type="secondary" if not is_current else "primary",
            help=relative_time(s["last_activity_at"]),
        ):
            st.session_state.thread_id = s["thread_id"]
            st.session_state.editing_index = None
            st.rerun()

        with cols[1].popover("⋯", use_container_width=True):
            new_title = st.text_input(
                "Rename",
                value=s["title"],
                key=f"rename-input-{s['thread_id']}",
            )
            c1, c2 = st.columns(2)
            if c1.button("Save", key=f"rename-save-{s['thread_id']}", use_container_width=True):
                sess.rename_session(s["thread_id"], new_title)
                st.rerun()
            if c2.button("Delete", key=f"del-{s['thread_id']}", use_container_width=True):
                try:
                    app_graph.checkpointer.delete_thread(s["thread_id"])
                except Exception:
                    pass
                sess.delete_session(s["thread_id"])
                if s["thread_id"] == st.session_state.thread_id:
                    st.session_state.thread_id = None  # ensure_current_session picks a new one
                st.rerun()

    st.divider()
    st.session_state.monthly_income = st.number_input(
        "Monthly income (£)",
        value=float(st.session_state.monthly_income),
        step=100.0,
        min_value=0.0,
    )
    st.session_state.use_sample = st.checkbox(
        "Send 28 sample transactions each turn",
        value=st.session_state.use_sample,
        help="Loads tests/fixtures/sample_transactions.json into the graph state. "
             "Backend dedupes by transaction id.",
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
            "5. Edit ✏️ on any user bubble to rewind the graph and re-run from that point.\n"
            "6. Sessions + checkpoints persist to `.smartfin/*.db`."
        )


# ---------------------------------------------------------------------------
# Main chat
# ---------------------------------------------------------------------------

info = sess.get_session(st.session_state.thread_id)
st.title(info["title"] if info else "SmartFin")
st.caption(
    f"`{st.session_state.thread_id}` · "
    "Type a message. Agents stream into the chat; HITL pauses surface as an Approve/Reject card below."
)


def render_user_entry(entry: dict, index: int) -> None:
    with st.chat_message("user"):
        if st.session_state.editing_index == index:
            new_text = st.text_area(
                "Edit message",
                value=entry["content"],
                key=f"edit-ta-{index}",
                label_visibility="collapsed",
            )
            c1, c2 = st.columns([1, 1])
            if c1.button("💾 Save & resend", key=f"edit-save-{index}", type="primary"):
                st.session_state.editing_index = None
                run_edit_resend(index, new_text)
                st.rerun()
            if c2.button("Cancel", key=f"edit-cancel-{index}"):
                st.session_state.editing_index = None
                st.rerun()
        else:
            cols = st.columns([10, 1])
            cols[0].markdown(entry["content"])
            if cols[1].button("✏️", key=f"edit-btn-{index}", help="Edit and resend"):
                st.session_state.editing_index = index
                st.rerun()


def render_agent_entry(entry: dict) -> None:
    emoji, name = AGENT_META.get(entry.get("agent", ""), ("🤖", entry.get("agent", "")))
    with st.chat_message("assistant", avatar=emoji):
        st.markdown(f"**{name}**  \n`{entry.get('agent', '')}`")
        st.markdown(entry["content"])


trace = get_trace()
for i, entry in enumerate(trace):
    if entry["role"] == "user":
        render_user_entry(entry, i)
    else:
        render_agent_entry(entry)


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

paused_state = get_pending_interrupt(app_graph, graph_config())
pending_payload = (paused_state or {}).get("pending_confirmation") if paused_state else None
is_paused = bool(pending_payload) and "confirmed" not in pending_payload

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

    if not any([trends, cats, allocations, goals, anomaly_flags, health, alerts]):
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
