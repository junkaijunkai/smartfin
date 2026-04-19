# Agent Compatibility Analysis Report
**Date**: 2026-04-19  
**Branch**: dev (after latest pull)

## Summary
✅ **All unit tests pass**: 82 tests (15 budget, 7 goal, 60 health)  
⚠️ **Critical Issue Found**: Data structure mismatch between `budget_planning` and `goal_planning` agents

---

## Test Results
- ✅ `test_budget_planning.py`: 15 passed
- ✅ `test_goal_planning.py`: 7 passed  
- ✅ `test_health_assessment.py`: 60 passed

---

## Critical Incompatibility: budget_allocations Data Type

### Issue
**Goal Planning Agent expects**: `budget_allocations: list[BudgetAllocation]`
**Budget Planning Agent provides**: `budget_allocations: Dict[str, float]`

### Evidence

#### Goal Planning (app/agents/goal_planning/agent.py:80-83)
```python
budget_allocations = state.get("budget_allocations") or []

# Expects objects with .spent_amount attribute
total_spent = sum(allocation.spent_amount for allocation in budget_allocations)
```

**Problem**: If this receives a `Dict[str, float]` from budget_planning, line 83 will crash with:
```
TypeError: 'float' object has no attribute 'spent_amount'
```

#### Budget Planning (app/agents/budget_planning/agent.py:74)
```python
state["budget_allocations"] = budget_allocations  # Dict[str, float]
```

**Returns**: A dictionary like `{"food": 525.0, "transport": 220.0, ...}`

#### AppState Definition (app/state.py:152)
```python
budget_allocations: list[BudgetAllocation]
```

**Expects**: A list of `BudgetAllocation` objects, not a dict.

#### Test Evidence (tests/unit/test_goal_planning.py:106)
```python
budget_allocations: budget_allocations,  # type: list[BudgetAllocation]
```

#### Budget Planning Test (tests/unit/test_budget_planning.py:258)
```python
assert "food" in new_state["budget_allocations"]  # Dict indexing
```

---

## Why Tests Don't Catch This

1. **Budget Planning tests** use mock data that never goes through goal planning
2. **Goal Planning tests** manually construct `BudgetAllocation` objects in the test setup, never relying on budget_planning's output
3. **No integration test** runs budget_planning → goal_planning in sequence

---

## Data Flow Analysis

### Normal Orchestration Path (from graph.py supervisor logic)
1. **expense_analysis** → `categorised_transactions`, `spending_trends`
2. **budget_planning** → `budget_allocations` ⚠️ **Wrong type!**
3. **goal_planning** → tries to read `budget_allocations` and crashes

### Other Agents (No Issues Found)
- **health_assessment** ✅ Reads `transactions` (fallback) or `categorised_transactions` (preferred)
- **anomaly_detection** ✅ Reads `transactions` (fallback) or `categorised_transactions` (preferred)

---

## Recommended Fix

**Option 1**: Change budget_planning to return `list[BudgetAllocation]` (proper fix)
- Modify `generate_budget_allocations()` to return BudgetAllocation objects
- Update all budget_planning logic to work with objects
- Impact: Medium (requires refactoring budget_planning.planner.py)

**Option 2**: Change goal_planning to read from dict (quick workaround)
- Accept `Dict[str, float]` and convert to spending amount differently
- Impact: Low (just change how goal_planning interprets the dict)

**Recommended**: **Option 1** for type consistency with AppState contract

---

## Other Observations

### ✅ Positive Findings
1. All agents properly handle missing state fields with fallbacks
2. Health assessment has good fallback chain: `categorised_transactions` → `transactions`
3. Anomaly detection properly handles both standalone and post-expense_analysis invocations
4. HITL (Human-in-the-Loop) confirmation handling is consistent across agents

### ⚠️ Minor Issues (Not Critical)
1. Goal Planning's goal extraction (extractor.py) is not mocked in tests, could fail silently without API key
2. Budget Planning doesn't set `pending_confirmation`, so it won't pause for HITL review (may be intentional)

---

## Conclusion
The agent implementations are well-designed and mostly compatible. However, **the budget_allocations data type mismatch must be fixed before running end-to-end flows through the orchestrator**. This will cause a runtime TypeError when budget_planning → goal_planning is executed.

