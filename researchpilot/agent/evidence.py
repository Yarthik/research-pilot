"""Deciding which evidence the answer step sees.

The agent can retrieve more chunks than fit in the answer prompt (several sub-queries, a retry
round, web results). Simply keeping the first N would throw away everything found by a retry,
which defeats the point of retrying. Instead the whole pool is re-scored against the original
question with the cross-encoder and the best N are kept. Without a scorer, a fair split keeps
room for the newest round.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

log = logging.getLogger(__name__)
ScoreFn = Callable[[str, list[str]], list[float]]


def select_evidence(
    question: str,
    existing: list[dict],
    new: list[dict],
    cap: int,
    scorer: ScoreFn | None = None,
) -> list[dict]:
    pool = existing + new
    if len(pool) <= cap:
        return pool

    if scorer is not None:
        try:
            scores = scorer(question, [item["text"] for item in pool])
            order = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)
            return [pool[i] for i in order[:cap]]
        except Exception as exc:  # scoring is an optimisation; never fail the request over it
            log.warning("evidence scoring failed, using the fallback split: %s", exc)

    # Fallback: keep at least half of the slots (or all free slots) for the newest round.
    keep_new = min(len(new), max(cap // 2, cap - len(existing)))
    return existing[: cap - keep_new] + new[:keep_new]
