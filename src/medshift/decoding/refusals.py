"""Frozen refusal lexicon for MedShift V3.2.

Byte-for-byte identical to the frozen V3 / V3.1 refusal patterns
(``medshift_v3.validation.REFUSAL_PATTERNS`` and
``medshift_v3_1.refusals.REFUSAL_PATTERNS``). The scoring semantics are
inherited unchanged.
"""

from __future__ import annotations

REFUSAL_PATTERNS = (
    "i cannot provide",
    "i'm unable to",
    "i am unable to",
    "as an ai",
    "sorry, i can't help",
    "i can't help with",
    "i must refuse",
)
