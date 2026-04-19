# Agent Responsibilities and Implementation Notes

## Specialist Agents (`app/agents/`)

Each agent has an `agent.py` (LangGraph node entry) and a logic module:

- `expense_analysis/` — transaction categorisation + trend analysis (**implement first**; feeds all downstream agents)
- `budget_planning/` — budget allocation & overspend warnings
- `goal_planning/` — financial goal tracking
- `anomaly_detection/` — suspicious transaction detection
- `health_assessment/` — debt-to-income, reserve months, risk rating
