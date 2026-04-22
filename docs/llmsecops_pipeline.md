# SmartFin MLSecOps / LLMSecOps Pipeline

## Scope

This baseline implements CI-oriented MLSecOps controls for SmartFin.

It covers:

- CI orchestration
- container packaging for frontend and backend services
- automated unit, integration, and AI security testing
- model versioning and approval gates
- monitoring and logging standards

It does not cover:

- production deployment automation
- runtime CD approvals
- infrastructure provisioning

## Tooling

- GitHub Actions for CI orchestration
- Docker and Docker Compose for local packaging and environment parity
- Pytest for unit, integration, and security tests
- Ruff for linting
- Bandit for static Python security scanning
- pip-audit for dependency vulnerability scanning
- LangSmith for LLM tracing and monitoring
- structured application logging via `app.config.configure_logging()`
- deterministic guardrails in `app.guardrails`
- approved model registry in `config/model_registry.json`

## CI Workflow

The CI workflow is defined in `.github/workflows/ci-llmsecops.yml`.

The workflow runs these stages on push and pull request:

1. install Python dependencies
2. run lint checks with Ruff
3. run unit tests
4. run integration tests
5. run security tests
6. run deterministic LLMSecOps policy checks from `scripts/llmsecops_ci.py`
7. run Bandit for static code security analysis
8. run pip-audit for dependency vulnerability detection
9. build the backend Docker image
10. build the frontend Docker image

## Container Topology

SmartFin currently ships as two containers:

- `backend`: FastAPI service exposing `/health` and `/analyze`
- `frontend`: Streamlit app that calls the backend over `SMARTFIN_BACKEND_URL`

Local orchestration is defined in `docker-compose.yml`.

Container artifacts:

- `Dockerfile.backend`
- `Dockerfile.frontend`
- `.dockerignore`

## Automated Testing Strategy

### Unit tests

Validate deterministic business logic, parsers, routing helpers, and configuration behavior.

### Integration tests

Validate orchestrator behavior, agent chaining, HITL interruptions, and shared-state compatibility.

### AI security tests

Validate:

- prompt injection detection
- system prompt and secret exfiltration blocking
- unsafe output redaction
- model registry compliance

Security tests live under `tests/security/`.

The additional CI policy runner in `scripts/llmsecops_ci.py` gives a fast fail path for common LLM security regressions without requiring external model calls.

## Model Versioning

Approved models are tracked in `config/model_registry.json`.

The registry provides:

- approved aliases
- exact provider/model identifiers
- lifecycle stage
- owner
- risk tier
- intended use cases

Runtime model selection is resolved through `app.config.resolve_model_name()`.

Recommended workflow for changing a model:

1. update `config/model_registry.json`
2. reference the approved alias or exact model in `SMARTFIN_MODEL`
3. run `python scripts/llmsecops_ci.py`
4. run the full test suite
5. merge only after CI passes

## Monitoring

SmartFin uses LangSmith-compatible environment variables for request tracing:

- `LANGCHAIN_TRACING_V2`
- `LANGCHAIN_API_KEY`
- `LANGCHAIN_PROJECT`

Recommended monitored signals:

- request volume by agent
- prompt injection blocks by rule
- fallback rate from LLM to deterministic logic
- model selection changes over time
- anomaly explanation generation failures
- HITL interruption frequency
- test and security check failure trends

## Logging

Logging is configured through `app.config.configure_logging()`.

Supported controls:

- `LOG_LEVEL`
- `SMARTFIN_LOG_FORMAT`

Recommended production-style fields:

- timestamp
- level
- logger
- event
- agent
- model
- thread_id
- guardrail

Sensitive content should not be logged raw. Guardrail hits should log rule identifiers and event metadata, not secrets.

## Local Developer Workflow

Use `.windsurf/workflows/llmsecops-ci.md` to run the CI-equivalent checks locally before opening a pull request.

## Deferred CD Work

CD is intentionally deferred.

When CD is introduced later, recommended additions are:

- environment-specific model approval policies
- signed artifact promotion
- manual approval for model alias changes
- post-deploy smoke evaluation against security prompts
- rollback on guardrail or monitoring regression
