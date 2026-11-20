"""V4 causal objectives, with all masks indexed by TARGET token position.

For target position t > 0, logit[t-1] predicts input_ids[t]. Each active
term is summed over its selected target positions and divided by the count
of those positions. An empty term is exactly zero. Per-example means are
then averaged across eight schedule exposures, preserving V3's intended
example weighting. Float32 accumulation is used for bfloat16 model logits.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def _selected(labels: torch.Tensor, positions: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or positions.shape != labels.shape:
        raise ValueError("logits, labels, and positions shape mismatch")
    if labels.shape[1] < 2 or positions.dtype != torch.bool:
        raise ValueError("causal sequence requires >=2 tokens and boolean mask")
    if bool((positions & (labels == IGNORE_INDEX)).any()):
        raise ValueError("mask includes ignored/padding token")
    if bool(positions[:, 0].any()):
        raise ValueError("first sequence token has no causal predecessor")
    return positions[:, 1:]


def causal_ce(
    logits: torch.Tensor, labels: torch.Tensor, positions: torch.Tensor, *, chunk_tokens: int = 64
) -> torch.Tensor:
    """mean CE(logits[b,t-1], labels[b,t]) over selected target positions."""
    selected = _selected(labels, positions, logits)
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be positive")
    count = int(selected.sum().item())
    if count == 0:
        return torch.zeros((), dtype=torch.float32, device=logits.device)
    total = torch.zeros((), dtype=torch.float32, device=logits.device)
    for start in range(0, selected.shape[1], chunk_tokens):
        stop = min(start + chunk_tokens, selected.shape[1])
        mask = selected[:, start:stop]
        if not bool(mask.any()):
            continue
        block = logits[:, start:stop, :][mask].float()
        target = labels[:, start + 1:stop + 1][mask]
        total = total + F.cross_entropy(block, target, reduction="sum")
    return total / count


def causal_teacher_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    positions: torch.Tensor,
    *,
    chunk_tokens: int = 64,
) -> torch.Tensor:
    """Mean KL(p_teacher[t-1] || p_student[t-1]) over selected target t."""
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("student/teacher logits shape mismatch")
    selected = _selected(labels, positions, student_logits)
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be positive")
    count = int(selected.sum().item())
    if count == 0:
        return torch.zeros((), dtype=torch.float32, device=student_logits.device)
    total = torch.zeros((), dtype=torch.float32, device=student_logits.device)
    teacher_logits = teacher_logits.detach()
    for start in range(0, selected.shape[1], chunk_tokens):
        stop = min(start + chunk_tokens, selected.shape[1])
        mask = selected[:, start:stop]
        if not bool(mask.any()):
            continue
        s = student_logits[:, start:stop, :][mask].float()
        t = teacher_logits[:, start:stop, :][mask].float()
        log_p = F.log_softmax(t, dim=-1)
        log_q = F.log_softmax(s, dim=-1)
        total = total + (log_p.exp() * (log_p - log_q)).sum()
    return total / count


def objective(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    assistant_mask: torch.Tensor,
    triage_mask: torch.Tensor,
    *,
    active: dict[str, bool],
    teacher_logits: torch.Tensor | None = None,
    lambda_r: float = 1.0,
    lambda_a: float = 0.5,
    lambda_t: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """V4 arm loss: shift + λr replay + λa anchor + λt triage.

    Caller sets role/eligibility-specific active terms from the frozen
    schedule. The anchor mask is the assistant mask, including EOS.
    """
    if bool((triage_mask & ~assistant_mask).any()):
        raise ValueError("triage mask leaves assistant span")
    if active.get("anchor") and teacher_logits is None:
        raise ValueError("active anchor requires frozen teacher logits")
    zero = torch.zeros((), dtype=torch.float32, device=student_logits.device)
    terms = {
        "shift": causal_ce(student_logits, labels, assistant_mask) if active.get("shift") else zero,
        "replay": causal_ce(student_logits, labels, assistant_mask) if active.get("replay") else zero,
        "anchor": causal_teacher_kl(student_logits, teacher_logits, labels, assistant_mask) if active.get("anchor") else zero,
        "triage": causal_ce(student_logits, labels, triage_mask) if active.get("triage") else zero,
    }
    total = terms["shift"] + lambda_r * terms["replay"] + lambda_a * terms["anchor"] + lambda_t * terms["triage"]
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("nonfinite V4 loss")
    return total, terms
