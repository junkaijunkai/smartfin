---
description: Run the SmartFin LLMSecOps CI-equivalent checks locally
---
1. Ensure your Python environment is activated.
2. Install project dependencies with `pip install -r requirements.txt`.
3. Install CI-only quality and security tools with `pip install ruff bandit pip-audit`.
4. Run lint checks with `ruff check app tests scripts`.
5. Run unit tests with `python -m pytest tests/unit -v`.
6. Run integration tests with `python -m pytest tests/integration -v`.
7. Run security tests with `python -m pytest tests/security -v`.
8. Run LLMSecOps policy checks with `python scripts/llmsecops_ci.py`.
9. Run static security scanning with `bandit -q -r app scripts`.
10. Run dependency vulnerability scanning with `pip-audit -r requirements.txt`.
11. If any step fails, fix the issue before opening or updating a pull request.
