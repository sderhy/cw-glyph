"""Forced alignment of detected elements to a known truth string.

To fine-tune the glyph CNN on *real* hand-sent audio we need span-level labels,
but the livetest truth is only text. Given the detected dit/dah element stream
``S`` and the truth's concatenated dit/dah code ``T`` (with per-character
boundaries), a standard Needleman–Wunsch edit-distance alignment recovers, for
each truth character, which detected elements it owns — tolerating the missed /
spurious / misclassified elements that real detection produces.

Only *cleanly* aligned characters are harvested: a truth character is kept as a
positive span iff its detected elements form a contiguous run whose pattern
exactly equals the character's code (no substitution or indel touched it). That
run's region span is a guaranteed-correct real example of that glyph. Characters
around a detection error are skipped rather than mislabeled.
"""

from __future__ import annotations

from morse_char_recognizer.morse_table import unit_seconds  # noqa: F401  (re-export convenience)


def align_edit(observed: str, truth: str) -> list[tuple[int | None, int | None]]:
    """Needleman–Wunsch alignment of two dit/dah strings.

    Returns a list of ``(i, j)`` ops in order: a match/substitution pairs
    ``observed[i]`` with ``truth[j]``; an insertion is ``(i, None)`` (a spurious
    detected element); a deletion is ``(None, j)`` (a missed truth element).
    """
    n, m = len(observed), len(truth)
    # cost[i][j] = min edit cost aligning observed[:i] to truth[:j]
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i
    for j in range(1, m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = cost[i - 1][j - 1] + (0 if observed[i - 1] == truth[j - 1] else 1)
            cost[i][j] = min(sub, cost[i - 1][j] + 1, cost[i][j - 1] + 1)

    ops: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sub = cost[i - 1][j - 1] + (0 if observed[i - 1] == truth[j - 1] else 1)
            if cost[i][j] == sub:
                ops.append((i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            ops.append((i - 1, None))
            i -= 1
            continue
        ops.append((None, j - 1))
        j -= 1
    ops.reverse()
    return ops


def clean_char_spans(
    symbols: list[str],
    truth_tokens: list[tuple[str, str]],
) -> list[tuple[str, int, int]]:
    """Return ``(token, elem_start, elem_end)`` for each cleanly aligned character.

    ``truth_tokens`` is ``[(token, code), ...]`` with ``code`` the dit/dah string.
    ``elem_start:elem_end`` indexes into ``symbols`` (the detected element list).
    Only characters whose detected elements exactly reproduce their code (a
    contiguous all-match run) are returned.
    """
    observed = "".join(symbols)
    truth = "".join(code for _tok, code in truth_tokens)
    if not observed or not truth:
        return []

    # position in T -> truth-token index
    pos_token: list[int] = []
    for k, (_tok, code) in enumerate(truth_tokens):
        pos_token.extend([k] * len(code))

    ops = align_edit(observed, truth)
    # for each truth token, gather the observed element indices matched to it
    matched: dict[int, list[int]] = {}
    substituted: set[int] = set()
    for i, j in ops:
        if i is None or j is None:
            if j is not None:
                substituted.add(pos_token[j])  # a missed element in this token
            continue
        tok_k = pos_token[j]
        if observed[i] == truth[j]:
            matched.setdefault(tok_k, []).append(i)
        else:
            substituted.add(tok_k)  # a misclassified element in this token

    spans: list[tuple[str, int, int]] = []
    for k, (tok, code) in enumerate(truth_tokens):
        if k in substituted:
            continue
        idx = matched.get(k)
        if not idx or len(idx) != len(code):
            continue
        lo, hi = min(idx), max(idx)
        if hi - lo + 1 != len(code):
            continue  # not contiguous (an insertion split the run)
        if "".join(symbols[lo : hi + 1]) != code:
            continue
        spans.append((tok, lo, hi + 1))
    return spans
