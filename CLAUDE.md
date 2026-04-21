# CLAUDE.md

**SmartFin** — A multi-agent AI co-pilot for personal financial management, built with LangGraph (supervisor pattern), LangChain, Anthropic Claude API, and Streamlit.

---

## Quick Start

```bash
git pull --rebase
cp .env.example .env        # fill in ANTHROPIC_API_KEY, LANGCHAIN_API_KEY
# create venv
uv init
uv venv
# activate venv
# for Windows Powershell
.venv/Scripts/activate.ps1 
# for Windows Bash
.venv/Scripts/activate
# for Linux/MacOS
source .venv/bin/activate
# install dependencies
uv pip install -r requirements.txt # or uv sync
# run UI
streamlit run ui/app.py (currently is a stub)
```

---

## Directory Index

| Topic | Path |
|---|---|
| Git workflow, branch naming, commit format | `CONTRIBUTING.md` |
| Environment variables | `.env.example` |
| Architecture, AppState, routing logic, HITL | `docs/architecture.md` |
| Agent responsibilities and implementation notes | `app/agents/README.md` |
| Orchestrator entry point and public API | `app/orchestrator/README.md` |
| UI structure and page descriptions | `ui/README.md` |
| Test commands and coverage | `tests/README.md` |

---

## Hard Rules (read before any task)

1. Always run `git pull --rebase` before starting work
2. Never commit or push directly to `main`
3. All changes must be branched off `dev`
4. Branch type and commit type must match → see `CONTRIBUTING.md`
5. Always activate venv before any code run
