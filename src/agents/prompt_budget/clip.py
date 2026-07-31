"""Head/tail excerpt helpers for structure-aware evidence clipping."""

from __future__ import annotations

from typing import Any, Optional

from ..token_utils import estimate_prompt_tokens

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


def _char_head_tail(text: str, content_budget_chars: int, block_id: str, dropped: int) -> str:
    """Clip by characters when tokenizer ids cannot reach the estimate target."""
    if content_budget_chars <= 0 or len(text) <= content_budget_chars:
        return text
    head_n = content_budget_chars // 2
    tail_n = content_budget_chars - head_n
    if head_n <= 0:
        return text[-content_budget_chars:]
    if tail_n <= 0:
        return text[:content_budget_chars]
    return (
        f"{text[:head_n]}"
        f"{TRUNC_MARKER_FMT.format(block_id=block_id, dropped=dropped)}"
        f"{text[-tail_n:]}"
    )


def head_tail_clip(
    text: str,
    *,
    encoder: Optional[Any],
    target_tokens: int,
    block_id: str,
) -> tuple[str, int, int, int]:
    """Clip *text* to at most *target_tokens*, keeping a balanced head and tail.

    Token sizes use ``estimate_prompt_tokens`` (max of encoder count and
    chars/4) so provider overcounts still trigger clipping.

    Returns ``(clipped_text, tokens_before, tokens_after, dropped_tokens)``.
    When *target_tokens* <= 0, returns a compact omit marker.
    """
    before = estimate_prompt_tokens(encoder, text)
    if target_tokens <= 0:
        # Compact marker so fully-omitted blocks do not blow the budget.
        marker = f"[[GHACR_OMITTED block_id={block_id} tokens={before}]]"
        after = estimate_prompt_tokens(encoder, marker)
        # If even the marker is too costly, leave an empty placeholder.
        if after > 8:
            marker = f"[omitted:{before}]"
            after = estimate_prompt_tokens(encoder, marker)
        return marker, before, after, before

    if before <= target_tokens:
        return text, before, before, 0

    # Leave room for the truncation marker inside the target budget.
    marker_probe = TRUNC_MARKER_FMT.format(
        block_id=block_id, dropped=max(0, before - target_tokens)
    )
    marker_tokens = max(1, estimate_prompt_tokens(encoder, marker_probe))
    content_budget = max(1, target_tokens - marker_tokens)

    if encoder is not None and hasattr(encoder, "encode") and hasattr(encoder, "decode"):
        try:
            ids = _as_id_list(encoder.encode(text))  # type: ignore[attr-defined]
            # When HF undercounts vs chars/4, id-clipping alone cannot shrink the
            # estimate; fall through to character clipping below.
            if len(ids) > content_budget:
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
                after = estimate_prompt_tokens(encoder, body)
                if after <= target_tokens:
                    return body, before, after, max(0, before - after)
                # Marker push-over or residual estimate overcount: try pure head.
                body = encoder.decode(ids[: min(len(ids), target_tokens)])  # type: ignore[attr-defined]
                after = estimate_prompt_tokens(encoder, body)
                if after <= target_tokens:
                    return body, before, after, max(0, before - after)
        except Exception:
            pass

    # Character / word fallback — required when encoder undercounts vs chars/4.
    # Target roughly 4 chars per estimated token.
    content_chars = max(1, content_budget * 4)
    dropped_est = max(0, before - target_tokens)
    body = _char_head_tail(text, content_chars, block_id, dropped_est)
    after = estimate_prompt_tokens(encoder, body)
    if after > target_tokens:
        # Binary-search character length until estimate fits.
        lo, hi = 1, max(1, len(text))
        best = text[:1]
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = (
                text[-mid:] if mid < len(text) else text
            )
            # Prefer balanced head/tail when mid is large enough for a marker.
            if mid < len(text):
                candidate = _char_head_tail(text, mid, block_id, dropped_est)
            est = estimate_prompt_tokens(encoder, candidate)
            if est <= target_tokens:
                best = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        body = best
        after = estimate_prompt_tokens(encoder, body)
    return body, before, after, max(0, before - after)


__all__ = ["OMIT_MARKER_FMT", "TRUNC_MARKER_FMT", "head_tail_clip"]
