from __future__ import annotations

import re
from dataclasses import dataclass


_RULE_PATTERNS: dict[str, tuple[re.Pattern[str], str]] = {
    "prompt_injection": (
        re.compile(
            r"ignore (all|any|the)? ?(previous|prior|above) instructions|"
            r"disregard (all|the) instructions|"
            r"bypass (the )?(guardrails|safety)|"
            r"jailbreak",
            re.IGNORECASE,
        ),
        "The request appears to contain prompt-injection instructions.",
    ),
    "system_prompt_exfiltration": (
        re.compile(
            r"reveal .*system prompt|show .*developer message|print .*hidden prompt|"
            r"leak .*instruction|display .*internal prompt",
            re.IGNORECASE,
        ),
        "The request asks for hidden prompts or internal instructions.",
    ),
    "secret_exfiltration": (
        re.compile(
            r"print .*api key|reveal .*api key|show .*token|dump .*secret|"
            r"exfiltrat(e|ion).*secret|return .*credential",
            re.IGNORECASE,
        ),
        "The request appears to seek secrets or credentials.",
    ),
}


@dataclass(frozen=True)
class InputGuardrailResult:
    allowed: bool
    sanitized_text: str
    matched_rules: tuple[str, ...]
    reason: str


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def scan_input(text: str) -> InputGuardrailResult:
    normalized = _normalize(text)
    matched_rules: list[str] = []
    reasons: list[str] = []

    for rule_name, (pattern, reason) in _RULE_PATTERNS.items():
        if pattern.search(normalized):
            matched_rules.append(rule_name)
            reasons.append(reason)

    if matched_rules:
        return InputGuardrailResult(
            allowed=False,
            sanitized_text=normalized,
            matched_rules=tuple(matched_rules),
            reason=" ".join(reasons),
        )

    return InputGuardrailResult(
        allowed=True,
        sanitized_text=normalized,
        matched_rules=(),
        reason="",
    )
