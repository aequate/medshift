"""Conservative linguistic collapse guard; never scores clinical quality."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

THRESHOLDS = {
    "min_tokens_for_statistical_guard": 32,
    "repeated_token_run": 8,
    "repeated_bigram_fraction": 0.65,
    "repeated_trigram_fraction": 0.50,
    "most_common_token_share": 0.25,
    "lexical_diversity": 0.15,
    "minimum_statistical_signals": 2,
}


def inspect_generation(text: str, token_ids: list[int], *, eos_id: int, cap: int, nonfinite: bool = False) -> dict[str, Any]:
    """Repeated 1-gram fraction is (tokens minus distinct token IDs) / tokens; empty is zero."""
    if cap < 1:
        raise ValueError("generation cap must be positive")
    tokens = [int(x) for x in token_ids]
    counts = Counter(tokens)
    n = len(tokens)
    repeated_unigram_fraction = (n - len(counts)) / n if n else 0.0
    run = longest = 0
    previous = None
    for token in tokens:
        run = run + 1 if token == previous else 1
        longest = max(longest, run)
        previous = token
    repeats = {}
    for size in (2, 3):
        grams = [tuple(tokens[i:i + size]) for i in range(max(0, n - size + 1))]
        repeats[size] = (len(grams) - len(set(grams))) / len(grams) if grams else 0.0
    words = re.findall(r"[\w'-]+", text, re.UNICODE)
    lexical = len({word.casefold() for word in words}) / len(words) if words else 0.0
    pattern = bool(
        re.search(r"(?:\.\s*){5,}|\b(?:T\s+){3}T\b|\b(?:Ac\s+){2}Ac\b|Acute(?:ute){3,}", text, re.IGNORECASE)
    )
    signals = {
        "repeated_token_run": longest >= THRESHOLDS["repeated_token_run"],
        "repeated_bigram": repeats[2] >= THRESHOLDS["repeated_bigram_fraction"],
        "repeated_trigram": repeats[3] >= THRESHOLDS["repeated_trigram_fraction"],
        "token_concentration": max(counts.values(), default=0) / max(1, n) >= THRESHOLDS["most_common_token_share"],
        "low_lexical_diversity": len(words) >= 16 and lexical <= THRESHOLDS["lexical_diversity"],
        "v3_text_pattern": pattern,
        "nonfinite": nonfinite,
    }
    statistical = sum(signals[key] for key in (
        "repeated_token_run", "repeated_bigram", "repeated_trigram", "token_concentration", "low_lexical_diversity"
    ))
    collapse = bool(nonfinite or pattern or (n >= THRESHOLDS["min_tokens_for_statistical_guard"] and statistical >= THRESHOLDS["minimum_statistical_signals"]))
    return {
        "generated_tokens": n, "eos": bool(tokens and tokens[-1] == eos_id),
        "hard_cap": n >= cap and not bool(tokens and tokens[-1] == eos_id),
        "longest_repeated_token_run": longest,
        "repeated_unigram_fraction": repeated_unigram_fraction,
        "repeated_bigram_fraction": repeats[2], "repeated_trigram_fraction": repeats[3],
        "most_common_token_share": max(counts.values(), default=0) / max(1, n),
        "lexical_diversity": lexical,
        "signals": signals, "catastrophic_linguistic_collapse": collapse,
    }
