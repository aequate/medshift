"""Unit tests for statistical helpers and principal LoRA configuration."""
from __future__ import annotations

from medshift.statistics import normalize_suite, rate_cell
from medshift.training import MEDSHIFT_V1_CONFIG


def test_rate_cell_and_suite_normalization():
    formatted = rate_cell({"numerator": 187, "denominator": 260, "rate": 187 / 260})
    assert formatted == "187/260 (71.9%)"
    assert normalize_suite("geoshift") == "geoshift"


def test_principal_lora_config_matches_v1_spec():
    cfg = MEDSHIFT_V1_CONFIG.to_dict()
    assert cfg["base_model"] == "Qwen/Qwen3.5-9B"
    assert cfg["lora_r"] == 16 and cfg["lora_alpha"] == 32
    assert len(cfg["target_modules"]) == 7
    assert cfg["optimizer_steps"] == 500
    assert cfg["enable_thinking"] is False
