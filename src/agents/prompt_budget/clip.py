"""Head/tail excerpt helpers for structure-aware evidence clipping."""

from __future__ import annotations

from typing import Any, Optional

from ..token_utils import count_tokens

OMIT_MARKER_FMT = (
    "\n[[GHACR_OMITTED block_id={block_id} tokens={tokens} reason=budget]]\n"
)
TRUNC_MARKER_FMT = (
    "\n[[GHACR_TRUNCATED block_id={block_id} dropped_tokens={dropped}]]\n"
)


def _as_id_list(ids: Any) -> list[Any]:
    try:
        return list(ids)
    except TypeError:  # pragma: no cover
        return ids


def head_tail_clip(
    text: str,
    *,
    encoder: Optional[Any],
    target_tokens: int,
    block_id: str,
) -> tuple[str, int, int, int]:
    """Clip *text* to at most *target_tokens*, keeping a balanced head and tail.

    Returns ``(clipped_text, tokens_before, tokens_after, dropped_tokens)``.
    When *target_tokens* <= 0, returns a compact omit marker.
    """
    before = count_tokens(encoder, text)
    if target_tokens <= 0:
        # Compact marker so fully-omitted blocks do not blow the budget.
        marker = f"[[GHACR_OMITTED block_id={block_id} tokens={before}]]"
        after = count_tokens(encoder, marker)
        # If even the marker is too costly, leave an empty placeholder.
        if after > 8:
            marker = f"[omitted:{before}]"
            after = count_tokens(encoder, marker)
        return marker, before, after, before

    if before <= target_tokens:
        return text, before, before, 0

    # Leave room for the truncation marker inside the target budget.
    marker_probe = TRUNC_MARKER_FMT.format(block_id=block_id, dropped=max(0, before - target_tokens))
    marker_tokens = max(1, count_tokens(encoder, marker_probe))
    content_budget = max(1, target_tokens - marker_tokens)

    if encoder is not None and hasattr(encoder, "encode") and hasattr(encoder, "decode"):
        try:
            ids = _as_id_list(encoder.encode(text))  # type: ignore[attr-defined]
            if len(ids) <= content_budget:
                return text, before, before, 0
            head_n = content_budget // 2
            tail_n = content_budget - head_n
            dropped = max(0, len(ids) - content_budget)
            if head_n <= 0:
                body = encoder.decode(ids[-content_budget:])  # type: ignore[attr-defined]
            elif tail_n <= 0:
                body = encoder.decode(ids[:content_budget])  # type: ignore[attr-defined]
            else:
                head = encoder.decode(ids[:head_n])  # type: ignore[attr-defined]
                tail = encoder.decode(ids[-tail_n:])  # type: ignore[attr-defined]
                body = (
                    f"{head}"
                    f"{TRUNC_MARKER_FMT.format(block_id=block_id, dropped=dropped)}"
                    f"{tail}"
                )
            after = count_tokens(encoder, body)
            # If marker push-over still exceeds target, keep pure head only.
            if after > target_tokens and content_budget > 0:
                body = encoder.decode(ids[: min(len(ids), target_tokens)])  # type: ignore[attr-defined]
                after = count_tokens(encoder, body)
            return body, before, after, max(0, before - after)
        except Exception:
            pass

    words = text.split()
    if len(words) <= content_budget:
        return text, before, before, 0
    head_n = content_budget // 2
    tail_n = content_budget - head_n
    dropped = max(0, len(words) - content_budget)
    if head_n <= 0:
        body = " ".join(words[-content_budget:])
    elif tail_n <= 0:
        body = " ".join(words[:content_budget])
    else:
        body = (
            " ".join(words[:head_n])
            + TRUNC_MARKER_FMT.format(block_id=block_id, dropped=dropped)
            + " ".join(words[-tail_n:])
        )
    after = count_tokens(encoder, body)
    if after > target_tokens:
        body = " ".join(words[:target_tokens])
        after = count_tokens(encoder, body)
    return body, before, after, max(0, before - after)


__all__ = ["OMIT_MARKER_FMT", "TRUNC_MARKER_FMT", "head_tail_clip"]
