"""Unit tests for the cached token-mask schema automaton constrained decoder."""
from __future__ import annotations

from medshift.decoding import build_automaton, parse_compact_fields


def test_schema_automaton_transitions_and_compact_parser():
    auto = build_automaton()
    assert auto.constraint_state_key() is not None
    assert auto.allow_eos() is False
    snap = auto.snapshot()
    assert snap is not None

    valid = (
        "TRIAGE: emergency\n"
        "DIAGNOSIS: Acute myocardial infarction\n"
        "RATIONALE: Substernal chest pain with diaphoresis and ST elevation.\n"
    )
    parsed = parse_compact_fields(valid)
    assert isinstance(parsed, dict)
