"""Structured clinical output parser, bilingual (English/Arabic) triage extractor, and format checker."""
from __future__ import annotations

import collections
import re

from ..decoding.refusals import REFUSAL_PATTERNS

CANONICAL_HEADINGS = (
    "TRIAGE",
    "TRIAGE RATIONALE",
    "DIFFERENTIAL",
    "KEY CLINICAL EVIDENCE",
    "CONFLICTING OR UNCERTAIN EVIDENCE",
    "CONTEXTUAL FACTORS",
    "WHAT WOULD CHANGE THE ASSESSMENT",
    "NEXT STEPS",
)

ROUTINE_KEYS = (
    "routine",
    "non-urgent",
    "non urgent",
    "nonurgent",
    "non-emergent",
    "روتين",
    "غير عاجل",
    "غير مستعجل",
)
EMERGENCY_KEYS = ("emergency", "emergent", "طارئ", "إسعاف", "اسعاف", "طوارئ")
URGENT_KEYS = ("urgent", "prompt", "semi-urgent", "semi urgent", "عاجل", "مستعجل")

_WS_RE = re.compile(r"\s+")
_ARABIC_MARKS = re.compile("[ـً-ٰٟ]")
_COLON_DIFF = re.compile(r"DIFFERENTIAL[ \t]*:?[ \t]*\n\s*1\.\s*(.+)")
_PRIMARY_DIAG = re.compile(r"Primary diagnosis:\s*(.+?)(?:\.|\n)")
_TRIAGE_LABEL = re.compile(r"(?i)^TRIAGE\b[ \t]*([^:\n]{0,40}?)[ \t]*(?::|$)")
_ALLOWED_TRIAGE_LABELS = frozenset({"<bare>", "RATIONALE"})


def normalize_text(value: str) -> str:
    """Lowercase, collapse whitespace, and strip."""
    return _WS_RE.sub(" ", str(value).lower()).strip()


def _clean_heading_line(line: str) -> str:
    return line.strip().lstrip("#*").strip().rstrip("*").strip()


def heading_inventory(text: str) -> dict:
    """Return canonical 8-section heading coverage and non-standard TRIAGE heading counts."""
    found: set[str] = set()
    labels: collections.Counter[str] = collections.Counter()
    for line in (text or "").splitlines():
        stripped = _clean_heading_line(line)
        upper = stripped.upper()
        for heading in CANONICAL_HEADINGS:
            if upper == heading or upper.startswith(heading + ":"):
                found.add(heading)
        match = _TRIAGE_LABEL.match(stripped)
        if match:
            labels[match.group(1).strip().upper() or "<bare>"] += 1
    nonstandard = {k: v for k, v in sorted(labels.items()) if k not in _ALLOWED_TRIAGE_LABELS}
    return {
        "canonical_headings_present": [h for h in CANONICAL_HEADINGS if h in found],
        "canonical_heading_count": len(found),
        "all_canonical_headings": len(found) == len(CANONICAL_HEADINGS),
        "triage_prefixed_labels": dict(sorted(labels.items())),
        "nonstandard_triage_labels": nonstandard,
        "nonstandard_triage_heading_lines": sum(nonstandard.values()),
    }


def _triage_statement(text: str) -> str:
    lines = [ln.strip() for ln in str(text).splitlines()]
    for i, ln in enumerate(lines):
        if _clean_heading_line(ln).upper().startswith("TRIAGE"):
            tail = _clean_heading_line(ln)[6:].lstrip(" :-\t").strip()
            if tail:
                return tail
            return next((nxt for nxt in lines[i + 1 : i + 3] if nxt), "")
    return ""


def parse_triage(text: str) -> str | None:
    """Parse urgency level from English or Arabic (diacritic-insensitive) TRIAGE section."""
    stmt = _triage_statement(text)
    if not stmt:
        return None
    low = _ARABIC_MARKS.sub("", stmt).lower()
    if any(k in low for k in ROUTINE_KEYS):
        return "non_emergency_clinical_assessment"
    if ("prompt" in low and "emergency" in low) or ("urgent" in low and "emergency" in low):
        return "prompt_or_emergency_assessment"
    if any(k in low for k in EMERGENCY_KEYS):
        return "emergency_assessment"
    if any(k in low for k in URGENT_KEYS):
        return "prompt_clinical_assessment"
    return None


def parse_diagnosis(text: str) -> str | None:
    """Extract the top-1 ranked diagnosis from Primary diagnosis or DIFFERENTIAL section."""
    raw = str(text or "")
    m = _PRIMARY_DIAG.search(raw)
    if m and len(m.group(1).strip()) >= 3:
        return m.group(1).strip()[:160]
    m = _COLON_DIFF.search(raw)
    if m:
        cand = _WS_RE.sub(" ", m.group(1).strip().split("\n")[0][:160].strip())
        if len(cand) >= 3:
            return cand
    return None


def score_clinical_output(output_text: str, diagnosis_gold: str | None, triage_gold: str | None) -> dict:
    """Strict intention-to-treat scoring separating Top-1 diagnosis and triage correctness."""
    norm = normalize_text(output_text or "")
    if not norm:
        return {"top1": 0, "triage": 0, "output_state": "missing"}
    if any(pat in norm for pat in REFUSAL_PATTERNS):
        return {"top1": 0, "triage": 0, "output_state": "refusal"}
    top1 = int(normalize_text(diagnosis_gold) in norm) if diagnosis_gold else 0
    triage = int(parse_triage(output_text) == triage_gold) if triage_gold else 0
    return {"top1": top1, "triage": triage, "output_state": "ok"}
