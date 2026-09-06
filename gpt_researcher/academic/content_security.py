from __future__ import annotations

import re


PROMPT_INJECTION_PATTERNS = (
    re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior)\s+instructions?\b", re.I),
    re.compile(r"\b(?:system|developer)\s+(?:message|prompt)\b", re.I),
    re.compile(r"\b(?:reveal|exfiltrate|print)\s+(?:the\s+)?(?:secret|api key|token)\b", re.I),
    re.compile(r"<\|(?:system|assistant|developer)\|>", re.I),
)


def detect_untrusted_content_warnings(text: str) -> list[str]:
    """Flag instruction-like document content without deleting source evidence."""
    if any(pattern.search(text) for pattern in PROMPT_INJECTION_PATTERNS):
        return ["prompt_injection_pattern"]
    return []
