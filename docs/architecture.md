## Architecture

### Shared State (`app/state.py`)

The central `AppState` TypedDict is the data contract between all agents. Key fields:

- **Input:** `transactions: list[Transaction]`, `monthly_income: float`
- **Agent outputs:** `categorised_transactions`, `spending_trends`, `budget_allocations`, `goals`, `anomaly_flags`, `anomaly_explanation`, `health_summary`, `alerts`
- **Orchestration:** `active_agent`, `agents_queue`, `pending_confirmation` (HITL payload)
- `messages` uses `Annotated[list, add_messages]` for LangGraph's auto-reducer

Pydantic models: `Transaction`, `BudgetAllocation`, `FinancialGoal`, `AnomalyFlag`,  `SpendingTrend`, `HealthSummary`, `Alert`.

### Orchestrator (`app/orchestrator/`)

LangGraph `StateGraph` with a supervisor + 5 specialist agent nodes + a HITL confirm node.

**Flow:**

1. `SUPERVISOR` classifies user intent and populates `agents_queue`
2. `route_to_agent()` (conditional edge) dispatches to the next agent in the queue
3. Each agent node updates `AppState` fields and optionally sets `pending_confirmation`
4. `route_after_agent()` checks `pending_confirmation`; if set, routes to `CONFIRM` (HITL pause), else back to `SUPERVISOR`
5. When the queue is exhausted, SUPERVISOR sets `active_agent = "end"` → graph terminates

**Keyword routing (extracted from user input via the invocation of LLM in each Worker Agent):**

Examples of keywords:
- `"budget"` → [expense_analysis, budget_planning]
- `"goal"` → [expense_analysis, goal_planning]
- `"suspicious"` / `"anomal"` → [expense_analysis, anomaly_detection]
- `"health"` / `"risk"` → [expense_analysis, health_assessment]
- default → [expense_analysis]

**HITL:** Graph is compiled with `interrupt_before=[NODE_CONFIRM]`. Use `get_pending_interrupt(app_graph, config)` to detect a paused state and `resume_with_confirmation(app_graph, config, confirmed=True/False)` to resume.

**Checkpointer:** `MemorySaver` (in-process only; replace with `SqliteSaver` / Postgres for production). Sessions are isolated by `thread_id` in the config.


### Other stubs

- `app/tools/transaction_loader.py`, `app/tools/notifications.py`
- `app/guardrails/input_filter.py`, `app/guardrails/output_validator.py`
- `tests/fixtures/sample_transactions.json`