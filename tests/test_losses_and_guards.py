"""Unit tests for next-token shifted causal CE, KL anchoring, clinical parser, and loop guards."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from medshift.guards import (
    CANONICAL_HEADINGS,
    heading_inventory,
    parse_diagnosis,
    parse_triage,
    periodic_token_loop,
    score_clinical_output,
    word_run,
)
from medshift.losses import causal_ce, causal_teacher_kl


def test_causal_ce_shifts_logits_by_one_token():
    torch.manual_seed(0)
    logits = torch.randn(1, 6, 12, requires_grad=True)
    labels = torch.tensor([[-100, -100, 3, 5, 7, 1]], dtype=torch.long)
    positions = torch.tensor([[False, False, True, True, True, True]])
    loss = causal_ce(logits, labels, positions)
    expected = F.cross_entropy(logits[:, :-1][positions[:, 1:]], labels[:, 1:][positions[:, 1:]])
    assert math.isclose(float(loss.detach()), float(expected.detach()), rel_tol=1e-6)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_causal_teacher_kl_is_zero_for_identical_logits():
    logits = torch.randn(1, 5, 10)
    labels = torch.tensor([[-100, 1, 2, 3, 4]], dtype=torch.long)
    positions = torch.tensor([[False, True, True, True, True]])
    kl = causal_teacher_kl(logits, logits.clone(), labels, positions)
    assert abs(float(kl)) < 1e-6


def test_clinical_parser_english_and_arabic_triage_and_headings():
    sample = "\n".join([
        "TRIAGE: Emergency assessment",
        "TRIAGE RATIONALE: Acute hemodynamic instability.",
        "DIFFERENTIAL:",
        "1. Acute myocardial infarction",
        "2. Pulmonary embolism",
        "3. Aortic dissection",
        "KEY CLINICAL EVIDENCE: Crushing chest pain.",
        "CONFLICTING OR UNCERTAIN EVIDENCE: Normal initial ECG.",
        "CONTEXTUAL FACTORS: Limited rural cath-lab access.",
        "WHAT WOULD CHANGE THE ASSESSMENT: Serial troponins.",
        "NEXT STEPS: Immediate transfer and aspirin.",
    ])
    inv = heading_inventory(sample)
    assert inv["all_canonical_headings"] is True
    assert inv["canonical_heading_count"] == len(CANONICAL_HEADINGS)
    assert inv["nonstandard_triage_heading_lines"] == 0
    assert parse_triage(sample) == "emergency_assessment"
    assert parse_diagnosis(sample) == "Acute myocardial infarction"

    scored = score_clinical_output(sample, "Acute myocardial infarction", "emergency_assessment")
    assert scored == {"top1": 1, "triage": 1, "output_state": "ok"}

    # Diacritic-insensitive Arabic plural emergency ("طوارئ")
    ar_sample = "TRIAGE: تقييم طَوارِئ فوري\nDIFFERENTIAL:\n1. احتشاء عضلة القلب"
    assert parse_triage(ar_sample) == "emergency_assessment"
    assert score_clinical_output("As an AI I cannot provide medical advice.", "AMI", "emergency_assessment")["output_state"] == "refusal"


def test_periodic_token_loop_and_word_run_detectors():
    looping = [10, 20, 30, 40] * 25
    res_loop = periodic_token_loop(looping, eos_id=999)
    assert res_loop["periodic_token_loop"] is True

    normal = list(range(1, 80))
    assert periodic_token_loop(normal, eos_id=999)["periodic_token_loop"] is False

    rep_text = " ".join(["urgent"] * 30)
    assert word_run(rep_text)["pathological_word_run"] is True
    assert word_run("Acute appendicitis requiring urgent surgical evaluation.")["pathological_word_run"] is False
