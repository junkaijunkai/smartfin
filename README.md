# 💰 SmartFin

> **An Agentic AI Multi-Agent Co-Pilot for Personal Financial Management**

SmartFin unifies personal finance into a single, coordinated AI system — connecting day-to-day spending behaviour with long-term financial wellbeing through proactive, context-aware guidance.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧾 **Expense Analysis** | Auto-categorises transactions and surfaces 30-day spending trends |
| 📊 **Budget Planning** | Generates monthly allocations and warns before limits are breached |
| 🎯 **Goal Tracking** | Calculates required savings rates and monitors progress toward long-term goals |
| 🚨 **Anomaly Detection** | Flags suspicious transactions by location, amount, frequency, and timing |
| 🏥 **Health Assessment** | Rates financial health via debt-to-income, liquidity, and overspending indicators |
| 🤝 **Human-in-the-Loop** | Pauses at critical decisions for explicit user confirmation before acting |

---

## 🏗️ Architecture

SmartFin is built on a **LangGraph Supervisor** pattern: one orchestrator routes user intent to a pipeline of specialist agents that share a single typed state object.

```
User Message
     │
     ▼
┌────────────┐     agents_queue      ┌─────────────────────┐
│ Supervisor │ ───────────────────▶  │  Specialist Agents  │
│  (Router)  │ ◀─── back to queue ── │                     │
└────────────┘                       │ • Expense Analysis  │
     │                               │ • Budget Planning   │
     │ END (queue empty)             │ • Goal Planning     │
     ▼                               │ • Anomaly Detection │
  Response                           │ • Health Assessment │
                                     └─────────────────────┘
                                              │
                                    pending_confirmation?
                                              │
                                              ▼
                                     ┌──────────────┐
                                     │  HITL Pause  │
                                     │ (user review)│
                                     └──────────────┘
```

### Shared State (`app/state.py`)

All agents communicate through a single `AppState` TypedDict — the contract that makes agent outputs composable:

```
transactions ──▶ categorised_transactions ──▶ spending_trends
                                          ──▶ budget_allocations
                                          ──▶ anomaly_flags
                                          ──▶ health_summary
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Agent Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) (Supervisor pattern) |
| **LLM** | [Anthropic Claude](https://www.anthropic.com/) (`claude-sonnet-4-6` by default) |
| **LLM Framework** | [LangChain](https://www.langchain.com/) |
| **Observability** | [LangSmith](https://smith.langchain.com/) |
| **UI** | [Streamlit](https://streamlit.io/) |
| **Data Validation** | [Pydantic v2](https://docs.pydantic.dev/) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/)

### Installation

```bash
git clone <repo-url>
cd finance-agent-app

# Install dependencies
pip install -r requirements.txt
# or with uv:
uv pip install -r requirements.txt

# Configure environment
cp .env.example .env
# → Fill in ANTHROPIC_API_KEY (and optionally LANGCHAIN_API_KEY)
```

### Run the UI

```bash
streamlit run ui/app.py
```

### Run the Backend API

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

### Run Tests

```bash
# Integration smoke tests (orchestrator pipeline)
pytest tests/integration/test_agent_pipeline.py -v

# Full suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=app --cov-report=term-missing
```

### Run LLMSecOps Checks

```bash
python scripts/llmsecops_ci.py
```

For the CI-oriented LLMSecOps design, see `docs/llmsecops_pipeline.md`.

### Run with Docker

```bash
docker compose up --build
```

Services:

- frontend: `http://localhost:8501`
- backend: `http://localhost:8000`

---

## 📁 Project Structure

```
finance-agent-app/
├── app/
│   ├── state.py                    # Shared AppState — central data contract
│   ├── orchestrator/
│   │   ├── graph.py                # LangGraph StateGraph assembly
│   │   ├── router.py               # Conditional edge routing functions
│   │   ├── checkpoints.py          # MemorySaver + HITL helpers
│   │   └── __init__.py             # Public API (app_graph, resume_with_confirmation)
│   ├── agents/
│   │   ├── expense_analysis/       # ✅ Implemented — categoriser + trend analyser
│   │   ├── budget_planning/        # 🚧 Stub
│   │   ├── goal_planning/          # 🚧 Stub
│   │   ├── anomaly_detection/      # 🚧 Stub
│   │   └── health_assessment/      # 🚧 Stub
│   ├── tools/
│   │   ├── transaction_loader.py   # 🚧 Stub
│   │   └── notifications.py        # 🚧 Stub
│   └── guardrails/
│       ├── input_filter.py         # 🚧 Stub
│       └── output_validator.py     # 🚧 Stub
├── ui/                             # 🚧 Streamlit pages (stubs)
├── tests/
│   ├── integration/                # ✅ Orchestrator pipeline smoke tests (8 passing)
│   ├── unit/                       # 🚧 Unit test stubs
│   └── fixtures/
└── pyproject.toml
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | Claude API access |
| `LANGCHAIN_API_KEY` | Optional | LangSmith tracing |
| `LANGCHAIN_TRACING_V2` | Optional | Enable LangSmith (`true`) |
| `LANGCHAIN_PROJECT` | Optional | LangSmith project name (`smartfin`) |
| `SMARTFIN_MODEL` | Optional | Claude model ID or approved alias (default: `claude-haiku-4-5`) |
| `SMARTFIN_ENFORCE_APPROVED_MODELS` | Optional | When `true`, unapproved model IDs fall back to the registry default |
| `SMARTFIN_LOG_FORMAT` | Optional | `plain` or `json` logging output |
| `SMARTFIN_BACKEND_URL` | Optional | Frontend-to-backend base URL for the Streamlit container/UI |

---

## 👥 Team

Rong Shu · Shi Zihan · Wen Qi · Xie Linhan · Zhang Junkai · Zhang Yuchen

---

## 📄 License

This project is developed for academic purposes.
