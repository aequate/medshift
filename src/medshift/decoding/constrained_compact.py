"""Frozen V3.2 constrained compact response protocol (inference-only).

V3.1 (``medshift_v3_1.compact``) relied on the model OBEYING a compact
two-line instruction. The synthetic benchmark showed the trained adapters
obey the eight-heading training format instead: BASELINE_GENERIC and
V3_SAFE-0500 hit the 96-token hard cap without EOS (BASELINE_GENERIC even
reached all mandatory fields only sometimes and never terminated;
V3_SAFE-0500 never emitted the schema at all). RAW_BASE obeyed the compact
instruction and passed -- proving the presentation, not the capability, is
the failure.

V3.2 keeps the SAME compact two-line schema and the SAME frozen scoring
semantics, but enforces structure at DECODING time with a deterministic
prefix automaton (grammar mask + deterministic stop). The model freely
chooses the clinical CONTENT (the model decides whether the case is
emergency, urgent, or routine, and which diagnosis is most likely); the
grammar only hard-codes the non-clinical scaffolding: the literal prefixes
``TRIAGE: ``, ``1. ``, and the terminating newline.

Only standard decoding hooks are used: ``min_new_tokens`` (suppresses EOS
until the schema can complete), a stop-string callback, and a logits
processor. No model/weight change, no retraining, no parser change.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

from .refusals import REFUSAL_PATTERNS

COMPACT_SCHEMA_ID = "medshift_v3_2_constrained_compact_schema_v1"

# Identical clinical-task phrasing to V3.1 (the models' clinical knowledge is
# unchanged); the response FORMAT is enforced by the decoder, not the prompt.
# The 'under 80 words' clause is dropped: the decoder now guarantees the
# two-line shape, so the instruction no longer asks the model to fight its
# trained format.
DEFAULT_SYSTEM_INSTRUCTION = (
    "You are answering a clinical assessment task. Answer on exactly two "
    "short lines:\n"
    "TRIAGE: <Emergency | Urgent | Urgent or emergency assessment | "
    "Non-urgent>\n"
    "1. <the single most likely diagnosis - its complete first-entry line, "
    "including the supporting clause, as in your ranked differential>\n"
    "Begin your reply with the TRIAGE line immediately and keep the answer "
    "compact."
)

# ---------------------------------------------------------------------------
# Level-word vocabularies. The model picks ONE level word; everything after
# it on the TRIAGE line is free text (clinical statement), which the frozen
# keyword ladder maps into the four-category label space. Each level word
# alone already parses correctly under the frozen parser
# (medshift_v3.data_build.parse_triage_category):
#   Emergency            -> emergency_assessment
#   Urgent               -> prompt_clinical_assessment
#   Non-urgent           -> non_emergency_clinical_assessment
#   Urgent or emergency  -> prompt_or_emergency_assessment
# The model may, and clinically should, append its triage statement; the
# parser scans the residual statement for keywords and the level word's own
# keywords keep the mapping correct.
# ---------------------------------------------------------------------------
TRIAGE_BRANCHES = (
    ("Emergency",),
    ("Urgent",),
    ("Non-urgent",),
)

TRIAGE_LINE_PREFIX = "TRIAGE: "
DIAG_LINE_PREFIX = "1. "
SCHEMA_TERMINATOR = "\n"

# A completed schema is exactly: prefix + level word + "\n" + "1. " ... and
# the diagnosis line ends at the next newline (the stop suffix). The
# automaton is COMPLETE once the diagnosis line has produced at least one
# free token after the "1. " prefix; EOS still waits for terminal newline.
MIN_DIAG_FREE_TOKENS = 1


def protocol(amendment: dict[str, Any]) -> dict[str, Any]:
    """Extract the frozen decoding/prompt parameters from the amendment."""
    from . import PROTOCOL_ID_KEY  # deferred: avoids import cycle at load

    compact = amendment["constrained_compact_protocol"]
    rule = compact["hard_max_new_tokens_selection_rule"]
    if compact["protocol_id"] != PROTOCOL_ID_KEY:  # pragma: no cover - guard
        raise RuntimeError("compact protocol id drift")
    if not rule["pre_registered"] or not rule["registered_before_any_gpu_evidence"]:
        raise RuntimeError("ceiling selection rule must stay pre-registered")
    if int(rule["resolution"]) != int(compact["hard_max_new_tokens"]):
        raise RuntimeError("ceiling resolution drift")
    constrained = compact["constrained_decoding"]
    return {
        "protocol_id": compact["protocol_id"],
        "schema_id": COMPACT_SCHEMA_ID,
        "system_instruction": compact["system_instruction"],
        "target_tokens": int(compact["target_tokens"]),
        "hard_max_new_tokens": int(compact["hard_max_new_tokens"]),
        "decoding": dict(compact["decoding"]),
        "constrained_decoding": dict(constrained),
    }


def build_messages(
    amendment: dict[str, Any],
    input_prompt: str,
    *,
    protocol_a: bool = False,
) -> list[dict[str, str]]:
    """Build the chat messages for a V3.2 generation.

    protocol_a=True reproduces the ORIGINAL V3 evaluation presentation
    (single user message, no compact instruction) for the synthetic
    protocol-A/B comparison only. V3.2 VALIDATION always uses protocol_a=False.
    """
    if not isinstance(input_prompt, str):
        raise TypeError("input_prompt must be a string")
    if protocol_a:
        messages = [{"role": "user", "content": input_prompt}]
        assert_clinical_content_preserved(input_prompt, messages)
        return messages
    system = protocol(amendment)["system_instruction"]
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": input_prompt},
    ]
    assert_clinical_content_preserved(input_prompt, messages)
    return messages


def assert_clinical_content_preserved(
    input_prompt: str, messages: list[dict[str, str]]
) -> dict[str, str]:
    """Record before/after UTF-8 hashes and fail closed on any byte change."""
    users = [message for message in messages if message.get("role") == "user"]
    if len(users) != 1:
        raise RuntimeError("message construction must contain exactly one user message")
    before = input_prompt.encode("utf-8")
    after = users[0]["content"].encode("utf-8")
    hashes = {
        "before_sha256": hashlib.sha256(before).hexdigest(),
        "after_sha256": hashlib.sha256(after).hexdigest(),
    }
    if before != after or hashes["before_sha256"] != hashes["after_sha256"]:
        raise RuntimeError("clinical input_prompt bytes changed during message construction")
    return hashes


def build_messages_from_item(
    amendment: dict[str, Any],
    item: Mapping[str, Any],
    *,
    protocol_a: bool = False,
) -> list[dict[str, str]]:
    """Build messages while consuming only the clinical ``input_prompt``.

    Benchmark ``expected_*`` fields are assertion-only metadata. Keeping this
    narrow item boundary makes accidental gold-field use observable in tests.
    """
    input_prompt = item["input_prompt"]
    if not isinstance(input_prompt, str):
        raise TypeError("input_prompt must be a string")
    return build_messages(amendment, input_prompt, protocol_a=protocol_a)


def parse_compact_fields(output_text: str) -> dict[str, Any]:
    """Protocol-level (non-clinical) field extraction for compliance metrics.

    Identical semantics to the V3.1 benchmark's field extraction
    (``medshift_v3_1.compact.parse_compact_fields``), so compliance numbers
    are directly comparable across V3.1 and V3.2. Clinical correctness is
    NEVER scored here.
    """
    text = str(output_text or "")
    norm = " ".join(text.lower().split())
    fields = {
        "empty": not norm.strip(),
        "triage_line_present": False,
        "ranked_diagnosis_line_present": False,
        "refusal_like": any(pat in norm for pat in REFUSAL_PATTERNS),
        "two_lines_or_fewer": 0 < len([ln for ln in text.splitlines() if ln.strip()]) <= 2,
    }
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("TRIAGE"):
            fields["triage_line_present"] = True
        if stripped.startswith("1."):
            fields["ranked_diagnosis_line_present"] = True
    fields["mandatory_fields_present"] = bool(
        fields["triage_line_present"] and fields["ranked_diagnosis_line_present"]
    )
    fields["parser_valid"] = bool(
        fields["mandatory_fields_present"] and not fields["refusal_like"] and not fields["empty"]
    )
    return fields


def decode_record(text: str) -> dict[str, Any]:
    """Record how a V3.2 constrained generation was produced.

    ``terminal_marker_seen`` records whether the stop suffix (newline after
    the diagnosis line) fired; ``completed_normally`` additionally requires
    that the decoded text carries both mandatory fields. These fields are
    protocol metadata only and never score clinical correctness.
    """
    fields = parse_compact_fields(text)
    completed = bool(fields["mandatory_fields_present"]) and text.rstrip("\n") != text.strip()
    return {
        "terminal_marker_seen": bool(text) and "\n" in text,
        "completed_normally": completed,
        "ended_on_terminal_newline": text.endswith("\n"),
    }
