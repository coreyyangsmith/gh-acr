"""Classify hard pipeline / LLM exceptions into stable failure categories."""

from __future__ import annotations

# Ordered: first match wins. Patterns mirror resilient_invoke fatal/retryable heuristics.
_CATEGORY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "token_limit",
        (
            "context_length",
            "maximum context length",
            "token limit",
            "context window",
            "too many tokens",
            "max_tokens",
            "maximum tokens",
        ),
    ),
    (
        "credits",
        (
            "insufficient credits",
            "payment required",
            "402",
            "purchase more",
            "out of credits",
        ),
    ),
    (
        "auth",
        (
            "authentication",
            "unauthorized",
            "invalid api key",
            "incorrect api key",
            "permission denied",
            "forbidden",
            "401",
            "403",
        ),
    ),
    (
        "rate_limit",
        (
            "rate limit",
            "ratelimit",
            "too many requests",
            "429",
        ),
    ),
    (
        "timeout",
        (
            "timeout",
            "timed out",
            "408",
        ),
    ),
    (
        "connection",
        (
            "connection",
            "connect",
            "remote end closed",
            "broken pipe",
            "reset by peer",
            "api connection",
            "temporarily unavailable",
            "service unavailable",
            "502",
            "503",
            "504",
            "bad gateway",
            "gateway timeout",
            "overloaded",
        ),
    ),
    (
        "prep",
        (
            "clone",
            "checkout",
            "prepare",
            "context cache",
            "git ",
            "ensure_prepared",
        ),
    ),
)


def classify_failure(error: BaseException | str, *, prep: bool = False) -> str:
    """Return a stable category string for *error*.

    Parameters
    ----------
    error
        Exception or message string.
    prep
        If True, prefer ``prep`` when no more specific category matches
        (used for scenario prep/clone failures).
    """
    if prep:
        # Still allow more specific matches (e.g. connection during clone).
        pass

    name = type(error).__name__.lower() if isinstance(error, BaseException) else ""
    msg = str(error).lower()
    combined = f"{name} {msg}"

    for category, patterns in _CATEGORY_PATTERNS:
        if any(p in combined for p in patterns):
            return category

    if prep:
        return "prep"
    return "other"


__all__ = ["classify_failure"]
