# Orchestrator: LangGraph Supervisor Pattern

The orchestrator module wires together all specialist agents under a single supervisor node and manages their execution flow, state persistence, and Human-in-the-Loop (HITL) confirmation.

## Architecture Overview

```
                    SUPERVISOR NODE
                (Intent classification)
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   expense_analysis  budget_planning  goal_planning  ...
        │                │                │
        └────────────────┼────────────────┘
                         │
                   CONFIRM NODE (HITL)
                         │
                    Back to SUPERVISOR
```

## Key Concepts

### Supervisor Node
- Reads user intent from the latest message
- Maintains an agent queue to chain multiple analyses in one turn
- Routes to the appropriate specialist agent or confirm node
- Currently uses keyword-based heuristics; TODO: replace with LLM-based classification

### Agent Nodes
Each specialist agent (expense_analysis, budget_planning, etc.) is a LangGraph node entry point that:
- Reads relevant fields from AppState
- Performs analysis
- Writes results back to AppState
- Optionally sets `pending_confirmation` to trigger HITL review

### HITL Confirmation
The graph is compiled with `interrupt_before=[NODE_CONFIRM]`, pausing execution before the confirm node runs. This allows the UI to:
1. Read `state["pending_confirmation"]` payload
2. Present it to the user for review
3. Call `resume_with_confirmation(...)` to continue

### Checkpointer
- Persists full AppState after every node execution
- Enables session resumption and state replay
- Currently uses MemorySaver (in-memory); swap for SqliteSaver/Postgres in production

## Public API

### Core Graph Object

```python
from app.orchestrator import app_graph

config = {"configurable": {"thread_id": "user-123"}}
result = app_graph.invoke(
    {"messages": [HumanMessage(content="Analyze my spending")]}, 
    config
)
```

### HITL Helpers

```python
from app.orchestrator import get_pending_interrupt, resume_with_confirmation

# Check if graph is paused at interrupt
state = get_pending_interrupt(app_graph, config)
if state:
    print("Pending confirmation:", state["pending_confirmation"])
    
    # User reviews and decides
    resume_with_confirmation(app_graph, config, confirmed=True)
```

## Module Files

| File | Responsibility |
|------|-----------------|
| `graph.py` | StateGraph definition, node wiring, compilation |
| `router.py` | Routing logic (supervisor → agent, agent → confirm/supervisor) |
| `checkpoints.py` | State persistence, HITL helpers |
| `__init__.py` | Public API exports |

## Extending the Orchestrator

To add a new specialist agent:

1. Create the agent module (e.g., `app/agents/new_agent/agent.py`)
2. Add a node function in `graph.py`:
   ```python
   def new_agent_node(state: AppState) -> dict:
       from app.agents.new_agent.agent import run
       return run(state)
   ```
3. Register the node in `build_graph()`:
   ```python
   builder.add_node(NODE_NEW_AGENT, new_agent_node)
   ```
4. Update routing logic in `router.py` to include the new agent
5. Export the updated `app_graph` from `__init__.py`
