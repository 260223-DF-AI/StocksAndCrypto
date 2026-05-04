"""
ResearchFlow — PII Masking Middleware

Scans user inputs and agent outputs for personally identifiable
information and redacts it before processing or returning.
"""

import re


# Patterns to detect — extend as needed
PII_PATTERNS: dict[str, str] = {
    "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
}


def mask_pii(text: str) -> str:
    """Replace detected PII patterns with redaction placeholders.

    Returns a string of the same shape with each match swapped for a
    typed redaction token like [REDACTED_EMAIL] / [REDACTED_PHONE].
    """
    if not text:
        return text
    redacted = text
    for label, pattern in PII_PATTERNS.items():
        redacted = re.sub(pattern, f"[REDACTED_{label.upper()}]", redacted)
    return redacted