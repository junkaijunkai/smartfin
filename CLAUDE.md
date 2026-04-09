# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**SmartFin** — a multi-agent AI co-pilot for personal financial management. Built with LangGraph (supervisor pattern), LangChain, Anthropic Claude API, and a Streamlit frontend.

## Repository Etiquette

### Step 1: Sync the latest code

Before starting any work, run `git pull --rebase` to sync the latest code
from remote.

### Step 2: Create a branch

Always create a new branch before making changes. Never work directly on
`main`.

Branch structure:

- `main` — production only, never commit directly
- `dev` — integration branch, all feature branches are cut from here
- Work branches, always branched from `dev`

Work Branch naming convention: `<type>/<description>`

- `feature/` — new functionality
- `fix/` — bug fixes
- `hotfix/` — urgent production fixes
- `chore/` — maintenance, dependencies, config
- `docs/` — documentation only

Example: `feature/user-oauth`, `fix/login-timeout`

### Step 3: Make changes

Always verify you are on the correct branch (`git branch`) before editing
any files.

### Step 4: Commit & Push

Follow Conventional Commits format for all commit messages:
`<type>(<scope>): <description>`

Types must match the branch type prefix (e.g. a `fix/` branch uses `fix:`
commits).

Never push to `main`. Always push to the current feature branch.

## Commands

### Setup

```bash
cp .env.example .env          # fill in ANTHROPIC_API_KEY, LANGCHAIN_API_KEY
pip install -r requirements.txt
# or: uv pip install -r requirements.txt
```

### Run tests

```bash
pytest tests/integration/test_agent_pipeline.py -v          # integration smoke tests (8 passing)
pytest tests/ -v                                             # full suite (most unit tests are stubs)
pytest tests/ --cov=app --cov-report=term-missing           # with coverage (auto-added via pyproject.toml)
pytest tests/integration/test_agent_pipeline.py::test_graph_compiles -v  # single test
```

### Run UI

```bash
streamlit run ui/app.py
```

### Install as editable package

```bash
pip install -e .
```

## Architecture

### Shared State (`app/state.py`)

The central `AppState` TypedDict is the data contract between all agents. Key fields:

- **Input:** `transactions: list[Transaction]`, `monthly_income: float`
- **Agent outputs:** `categorised_transactions`, `spending_trends`, `budget_allocations`, `goals`, `anomaly_flags`, `health_summary`, `alerts`
- **Orchestration:** `active_agent`, `agents_queue`, `pending_confirmation` (HITL payload)
- `messages` uses `Annotated[list, add_messages]` for LangGraph's auto-reducer

Pydantic models: `Transaction`, `BudgetAllocation`, `FinancialGoal`, `AnomalyFlag`, `SpendingTrend`, `HealthSummary`, `Alert`.

### Orchestrator (`app/orchestrator/`)

LangGraph `StateGraph` with a supervisor + 5 specialist agent nodes + a HITL confirm node.

**Flow:**

1. `SUPERVISOR` classifies user intent and populates `agents_queue`
2. `route_to_agent()` (conditional edge) dispatches to the next agent in the queue
3. Each agent node updates `AppState` fields and optionally sets `pending_confirmation`
4. `route_after_agent()` checks `pending_confirmation`; if set, routes to `CONFIRM` (HITL pause), else back to `SUPERVISOR`
5. When the queue is exhausted, SUPERVISOR sets `active_agent = "end"` → graph terminates

**Keyword routing (current, to be replaced with LLM-based intent):**

- `"budget"` → [expense_analysis, budget_planning]
- `"goal"` → [expense_analysis, goal_planning]
- `"suspicious"` / `"anomal"` → [expense_analysis, anomaly_detection]
- `"health"` / `"risk"` → [expense_analysis, health_assessment]
- default → [expense_analysis]

**HITL:** Graph is compiled with `interrupt_before=[NODE_CONFIRM]`. Use `get_pending_interrupt(app_graph, config)` to detect a paused state and `resume_with_confirmation(app_graph, config, confirmed=True/False)` to resume.

**Checkpointer:** `MemorySaver` (in-process only; replace with `SqliteSaver` / Postgres for production). Sessions are isolated by `thread_id` in the config.

### Public API

```python
from app.orchestrator import app_graph, get_pending_interrupt, resume_with_confirmation
config = {"configurable": {"thread_id": "user-123"}}
result = app_graph.invoke({"messages": [HumanMessage("Analyze my spending")]}, config)
```

### 5 Specialist Agents (`app/agents/`)

All are currently **stub files**. Each agent has an `agent.py` (LangGraph node entry) and a logic module:

- `expense_analysis/` — transaction categorisation + trend analysis (**implement first**; feeds all downstream agents)
- `budget_planning/` — budget allocation & overspend warnings
- `goal_planning/` — financial goal tracking
- `anomaly_detection/` — suspicious transaction detection
- `health_assessment/` — debt-to-income, reserve months, risk rating

### UI (`ui/`)

Streamlit multi-page app. All pages are stubs. Entry point: `ui/app.py`.

### Other stubs

- `app/tools/transaction_loader.py`, `app/tools/notifications.py`
- `app/guardrails/input_filter.py`, `app/guardrails/output_validator.py`
- `tests/fixtures/sample_transactions.json`

## Environment Variables (`.env.example`)

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Claude API access |
| `LANGCHAIN_API_KEY` | LangSmith tracing |
| `LANGCHAIN_TRACING_V2` | Enable LangSmith (set `true`) |
| `LANGCHAIN_PROJECT` | LangSmith project name (`smartfin`) |
| `SMARTFIN_MODEL` | Model to use (default: `claude-sonnet-4-6`) |
