"""V5 prospective stability observability: periodic token loops and word runs.

Diagnostics only. Neither function touches gradients, logits, sampling, or
decoding; both read an already generated continuation.

PERIODIC TOKEN LOOP (frozen algorithm ``v5_periodic_token_loop_v1``)
  Body: the generated continuation token IDs (prompt tokens never passed in).
  Exactly one trailing EOS is removed when the final token equals ``eos_id``;
  no other token is filtered (special tokens other than a final EOS are
  treated as ordinary tokens).
  For each candidate period p = 1..8, scan i = p..n-1 and form maximal runs of
  consecutive positions where body[i] == body[i-p]. A run of m matches that
  ends at position e defines the contiguous periodic region
  body[e-m-p+1 .. e] of covered length L = m + p (period p, any partial
  trailing cycle included in L). The region qualifies when
      floor(L / p) >= 8   (complete cycles)   and   L >= 32   (covered tokens).
  periodic_token_loop = any qualifying region exists (for any p, anywhere:
  suffix or interior). The reported detection is the qualifying region with
  the smallest p (the minimal qualifying period); ties by longest L, then by
  earliest start. Time O(8n), memory O(1) beyond the reported record.

PATHOLOGICAL WORD RUN (frozen algorithm ``v5_word_run_v1``)
  Text: decoded continuation (skip_special_tokens=True, so EOS is absent).
  Normalization: Unicode NFC; split on whitespace (``str.split()``); strip
  leading/trailing characters whose Unicode category starts with "P"
  (punctuation); casefold; discard items that become empty. No stemming, no
  fuzzy matching, no other character removal.
  pathological_word_run = longest run of consecutive identical normalized
  words >= 8.
"""

from __future__ import annotations

import unicodedata
from typing import Any

PERIODIC_ALGORITHM = "v5_periodic_token_loop_v1"
WORD_RUN_ALGORITHM = "v5_word_run_v1"
PERIODIC_THRESHOLDS = {"min_period": 1, "max_period": 8, "min_complete_cycles": 8, "min_covered_tokens": 32}
WORD_RUN_THRESHOLDS = {"min_consecutive_identical_words": 8}


def periodic_token_loop(token_ids: list[int], *, eos_id: int) -> dict[str, Any]:
    body = [int(x) for x in token_ids]
    eos_stripped = bool(body) and body[-1] == eos_id
    if eos_stripped:
        body = body[:-1]
    n = len(body)
    t = PERIODIC_THRESHOLDS
    best: tuple[int, int, int] | None = None  # (period, -covered, start)
    for p in range(t["min_period"], t["max_period"] + 1):
        run = 0
        for i in range(p, n + 1):
            if i < n and body[i] == body[i - p]:
                run += 1
                continue
            if run:
                covered, end = run + p, i - 1
                if covered // p >= t["min_complete_cycles"] and covered >= t["min_covered_tokens"]:
                    key = (p, -covered, end - covered + 1)
                    if best is None or key < best:
                        best = key
            run = 0
    detection = None
    if best is not None:
        p, neg_covered, start = best
        detection = {"period": p, "start_index": start, "covered_tokens": -neg_covered,
                     "complete_cycles": -neg_covered // p, "cycle_token_ids": body[start:start + p]}
    return {"algorithm": PERIODIC_ALGORITHM, "thresholds": dict(t), "body_tokens": n,
            "trailing_eos_excluded": eos_stripped,
            "periodic_token_loop": detection is not None, "detection": detection}


def normalize_words(text: str) -> list[str]:
    words = []
    for raw in unicodedata.normalize("NFC", text).split():
        start, end = 0, len(raw)
        while start < end and unicodedata.category(raw[start]).startswith("P"):
            start += 1
        while end > start and unicodedata.category(raw[end - 1]).startswith("P"):
            end -= 1
        word = raw[start:end].casefold()
        if word:
            words.append(word)
    return words


def word_run(text: str) -> dict[str, Any]:
    words = normalize_words(text)
    longest, word, run = 0, None, 0
    for i, w in enumerate(words):
        run = run + 1 if i and w == words[i - 1] else 1
        if run > longest:
            longest, word = run, w
    threshold = WORD_RUN_THRESHOLDS["min_consecutive_identical_words"]
    return {"algorithm": WORD_RUN_ALGORITHM, "thresholds": dict(WORD_RUN_THRESHOLDS),
            "normalized_word_count": len(words), "longest_identical_word_run": longest,
            "run_word": word, "pathological_word_run": longest >= threshold}


def v5_probe_predicates(new_ids: list[int], text: str, v4_metrics: dict[str, Any], *, eos_id: int) -> dict[str, Any]:
    """Every prospective V5 probe failure predicate; any True fails the probe."""
    periodic = periodic_token_loop(new_ids, eos_id=eos_id)
    words = word_run(text)
    failed = {
        "empty_generation": not new_ids,
        "not_eos": not v4_metrics["eos"],
        "hard_cap": bool(v4_metrics["hard_cap"]),
        "catastrophic_linguistic_collapse": bool(v4_metrics["catastrophic_linguistic_collapse"]),
        **{f"v4_signal_{name}": bool(value) for name, value in v4_metrics["signals"].items()},
        "periodic_token_loop": periodic["periodic_token_loop"],
        "pathological_word_run": words["pathological_word_run"],
    }
    return {"periodic": periodic, "word_run": words, "predicates": failed,
            "failed_predicates": [name for name, value in failed.items() if value],
            "probe_pass": not any(failed.values())}
