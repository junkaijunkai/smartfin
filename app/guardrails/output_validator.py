from __future__ import annotations

import re
from dataclasses import dataclass


_OUTPUT_PATTERNS: dict[str, re.Pattern[str]] = {
    "anthropic_key": re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}", re.IGNORECASE),
    "generic_api_key": re.compile(r"api[_ -]?key\s*[:=]\s*[A-Za-z0-9_-]{12,}", re.IGNORECASE),
    "bearer_token": re.compile(r"bearer\s+[A-Za-z0-9._-]{12,}", re.IGNORECASE),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
}


@dataclass(frozen=True)
class OutputValidationResult:
    allowed: bool
    sanitized_text: str
    matched_rules: tuple[str, ...]


def validate_output(text: str) -> OutputValidationResult:
    sanitized = text
    matched_rules: list[str] = []

    for rule_name, pattern in _OUTPUT_PATTERNS.items():
        if pattern.search(sanitized):
            matched_rules.append(rule_name)
            sanitized = pattern.sub("[REDACTED]", sanitized)

    return OutputValidationResult(
        allowed=not matched_rules,
        sanitized_text=sanitized,
        matched_rules=tuple(matched_rules),
    )
