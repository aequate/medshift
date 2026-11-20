"""Principal LoRA fine-tuning configuration for MedShift V1 (Qwen3.5-9B)."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LoRAConfig:
    base_model: str = "Qwen/Qwen3.5-9B"
    base_revision: str = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    precision: str = "bf16"
    max_seq_len: int = 2048
    completion_only_loss: bool = True
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    micro_batch_size: int = 1
    grad_accum_steps: int = 8
    epochs: int = 2
    examples_per_arm: int = 2000
    optimizer_steps: int = 500
    seed: int = 42
    enable_thinking: bool = False
    max_new_tokens: int = 512

    def to_dict(self) -> dict:
        d = asdict(self)
        d["target_modules"] = list(self.target_modules)
        return d


MEDSHIFT_V1_CONFIG = LoRAConfig()
