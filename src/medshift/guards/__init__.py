"""Clinical output parsing, format checks, and repetition/collapse detectors."""
from .clinical_parser import (
    CANONICAL_HEADINGS,
    heading_inventory,
    normalize_text,
    parse_diagnosis,
    parse_triage,
    score_clinical_output,
)
from .collapse import *  # noqa: F401,F403
from .loop_detectors import *  # noqa: F401,F403

__all__ = [
    "CANONICAL_HEADINGS",
    "heading_inventory",
    "normalize_text",
    "parse_diagnosis",
    "parse_triage",
    "score_clinical_output",
]
