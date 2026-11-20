"""Cached, state-synchronized Transformers decoding hooks for V3.2."""

from __future__ import annotations

import copy
import numbers
import time
from typing import Any

from .schema_automaton import SchemaDivergenceError


class SchemaTokenConstraintCache:
    """Predecode tokenizer-valid IDs and memoize exact-state masks."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        vocab = tokenizer.get_vocab()
        added_vocab = tokenizer.get_added_vocab()
        valid_ids = {
            int(token_id)
            for token_id in (*vocab.values(), *added_vocab.values())
            if isinstance(token_id, numbers.Integral)
            and not isinstance(token_id, bool)
            and int(token_id) >= 0
        }
        if not valid_ids:
            raise RuntimeError("tokenizer exposes no valid non-negative integer token IDs")
        self.valid_token_ids = frozenset(valid_ids)
        self.sorted_valid_token_ids = tuple(sorted(valid_ids))
        self.valid_token_id_count = len(valid_ids)
        self.min_valid_token_id = self.sorted_valid_token_ids[0]
        self.max_valid_token_id = self.sorted_valid_token_ids[-1]
        self.required_score_vocab_size = self.max_valid_token_id + 1
        # Backward-compatible name. This is the minimum score width required to
        # represent every tokenizer-valid ID, not a count of model-head rows.
        self.vocab_size = self.required_score_vocab_size
        self.added_token_ids = frozenset(
            int(token_id)
            for token_id in added_vocab.values()
            if isinstance(token_id, numbers.Integral)
            and not isinstance(token_id, bool)
            and int(token_id) >= 0
        )
        self.valid_ids_contiguous = self.valid_token_ids == frozenset(
            range(self.required_score_vocab_size)
        )
        started = time.perf_counter()
        self.token_texts = {
            token_id: tokenizer.decode([token_id], skip_special_tokens=False)
            for token_id in self.sorted_valid_token_ids
        }
        self.vocab_decode_passes = 1
        self.vocab_decode_calls = self.valid_token_id_count
        self.vocab_decode_cpu_s = time.perf_counter() - started
        self._allowed: dict[tuple[int, tuple[Any, ...]], tuple[int, ...]] = {}
        self._newline_ids = tuple(
            token_id
            for token_id, text in self.token_texts.items()
            if "\n" in text.replace("\r", "\n")
        )
        self._bound_score_vocab_size: int | None = None
        self._validity_masks: dict[tuple[int, str], Any] = {}
        self.constraint_cache_hits = 0
        self.constraint_cache_misses = 0
        self.full_vocab_scans = 0
        self.constraint_lookup_cpu_s = 0.0
        self.cumulative_constraint_mask_cpu_s = 0.0

    def token_text(self, token_id: int) -> str:
        token_id = int(token_id)
        if token_id not in self.valid_token_ids:
            raise RuntimeError(f"token ID {token_id} is not valid for the frozen tokenizer")
        return self.token_texts[token_id]

    @property
    def newline_token_ids(self) -> tuple[int, ...]:
        return self._newline_ids

    @property
    def bound_score_vocab_size(self) -> int | None:
        return self._bound_score_vocab_size

    def bind_score_vocab(self, vocab: int) -> int:
        """Validate and immutably bind the model-head score width."""
        vocab = int(vocab)
        if vocab < self.required_score_vocab_size:
            raise RuntimeError(
                f"score vocab {vocab} cannot represent tokenizer ID "
                f"{self.max_valid_token_id}; need at least {self.required_score_vocab_size}"
            )
        if self._bound_score_vocab_size is None:
            self._bound_score_vocab_size = vocab
        elif vocab != self._bound_score_vocab_size:
            raise RuntimeError(
                "constraint cache score vocab mismatch: "
                f"bound {self._bound_score_vocab_size}, received {vocab}"
            )
        return vocab

    def validity_mask(self, vocab: int, device: Any) -> Any:
        """Return cached bool mask selecting tokenizer-valid model-head rows."""
        import torch

        vocab = self.bind_score_vocab(vocab)
        key = (vocab, str(device))
        mask = self._validity_masks.get(key)
        if mask is None:
            mask = torch.zeros(vocab, dtype=torch.bool, device=device)
            ids = torch.tensor(self.sorted_valid_token_ids, dtype=torch.long, device=device)
            mask[ids] = True
            self._validity_masks[key] = mask
        return mask

    @staticmethod
    def state_key(automaton: Any, vocab: int) -> tuple[int, tuple[Any, ...]]:
        return int(vocab), tuple(automaton.constraint_state_key())

    def compute_allowed_ids_uncached(
        self, automaton: Any, vocab: int | None = None
    ) -> tuple[int, ...]:
        """Reference definition: allowed iff ``automaton.can_emit(text)``."""
        limit = self.required_score_vocab_size if vocab is None else int(vocab)
        self.bind_score_vocab(limit)
        return tuple(
            token_id
            for token_id in self.sorted_valid_token_ids
            if automaton.can_emit(self.token_texts[token_id])
        )

    def allowed_ids(self, automaton: Any, vocab: int) -> tuple[int, ...]:
        """Return exact cached scaffold-token set for the full state key."""
        started = time.perf_counter()
        vocab = self.bind_score_vocab(vocab)
        key = self.state_key(automaton, vocab)
        if key in self._allowed:
            self.constraint_cache_hits += 1
            result = self._allowed[key]
        else:
            self.constraint_cache_misses += 1
            self.full_vocab_scans += 1
            result = self.compute_allowed_ids_uncached(automaton, vocab)
            self._allowed[key] = result
        self.constraint_lookup_cpu_s += time.perf_counter() - started
        return result

    def record_mask_cpu_time(self, elapsed: float) -> None:
        self.cumulative_constraint_mask_cpu_s += float(elapsed)

    def stats(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "valid_token_id_count": self.valid_token_id_count,
            "min_valid_token_id": self.min_valid_token_id,
            "max_valid_token_id": self.max_valid_token_id,
            "valid_ids_contiguous": self.valid_ids_contiguous,
            "added_token_id_count": len(self.added_token_ids),
            "required_score_vocab_size": self.required_score_vocab_size,
            "bound_score_vocab_size": self.bound_score_vocab_size,
            "model_only_token_id_count": (
                None
                if self.bound_score_vocab_size is None
                else self.bound_score_vocab_size - self.valid_token_id_count
            ),
            "vocab_decode_passes": self.vocab_decode_passes,
            "vocab_decode_calls": self.vocab_decode_calls,
            "vocab_decode_cpu_s": round(self.vocab_decode_cpu_s, 6),
            "constraint_cache_hits": self.constraint_cache_hits,
            "constraint_cache_misses": self.constraint_cache_misses,
            "full_vocab_scans": self.full_vocab_scans,
            "number_unique_constraint_states": len(self._allowed),
            "constraint_lookup_cpu_s": round(self.constraint_lookup_cpu_s, 6),
            "cumulative_constraint_mask_cpu_s": round(
                self.cumulative_constraint_mask_cpu_s, 6
            ),
        }


class SchemaGenerationKwargs(dict[str, Any]):
    """Normal ``model.generate`` kwargs with inspectable shared state."""

    def __init__(self, *args: Any, synchronizer: GenerationStateSynchronizer, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.synchronizer = synchronizer


class GenerationStateSynchronizer:
    """Feed each newly appended generation token into one automaton once."""

    def __init__(
        self,
        constraint_cache: SchemaTokenConstraintCache,
        automaton: Any,
        prompt_length: int,
        special_token_ids: set[int],
    ) -> None:
        if int(prompt_length) < 0:
            raise ValueError("prompt_length must be non-negative")
        self.constraint_cache = constraint_cache
        self.automaton = automaton
        self.prompt_length = int(prompt_length)
        self.special_token_ids = set(special_token_ids)
        self.consumed_generated_tokens = 0
        self.fed_token_ids: list[int] = []
        self.skipped_special_token_ids: list[int] = []
        self.emitted_text = ""

    def decode_token(self, token_id: int) -> str:
        return self.constraint_cache.token_text(int(token_id))

    @staticmethod
    def _row(input_ids: Any) -> Any:
        if len(input_ids) != 1:
            raise RuntimeError("V3.2 constrained decoding requires batch_size=1")
        return input_ids[0]

    @staticmethod
    def _truncate_at_terminal_newline(text: str) -> str:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        first = normalized.find("\n")
        if first < 0:
            return normalized
        second = normalized.find("\n", first + 1)
        return normalized if second < 0 else normalized[: second + 1]

    def sync(self, input_ids: Any) -> int:
        row = self._row(input_ids)
        generated = len(row) - self.prompt_length
        if generated < self.consumed_generated_tokens:
            raise RuntimeError("generated sequence shrank during synchronization")
        start = self.prompt_length + self.consumed_generated_tokens
        consumed_now = 0
        for position in range(start, len(row)):
            token_id = int(row[position])
            self.consumed_generated_tokens += 1
            consumed_now += 1
            if token_id in self.special_token_ids:
                self.skipped_special_token_ids.append(token_id)
                continue
            piece = self.decode_token(token_id)
            was_halted = bool(self.automaton.halt)
            self.automaton.feed(piece)
            self.fed_token_ids.append(token_id)
            if not was_halted:
                self.emitted_text += piece
            if self.automaton.halt:
                self.emitted_text = self._truncate_at_terminal_newline(self.emitted_text)
        return consumed_now

    def clone_accepts(self, token_id: int) -> bool:
        if (
            int(token_id) in self.special_token_ids
            or int(token_id) not in self.constraint_cache.valid_token_ids
        ):
            return False
        clone = copy.deepcopy(self.automaton)
        try:
            clone.feed(self.decode_token(int(token_id)))
        except SchemaDivergenceError:
            return False
        return True


def build_schema_processors(
    tokenizer: Any,
    automaton: Any,
    hard_cap: int,
    *,
    prompt_length: int,
    constraint_cache: SchemaTokenConstraintCache,
    min_new_tokens: int | None = None,
) -> SchemaGenerationKwargs:
    """Build synchronized hooks using one externally shared token cache."""
    import torch
    from transformers import (
        LogitsProcessor,
        LogitsProcessorList,
        StoppingCriteria,
        StoppingCriteriaList,
    )

    if constraint_cache.tokenizer is not tokenizer:
        raise RuntimeError("constraint cache tokenizer identity mismatch")
    eos_ids: set[int] = set()
    value = getattr(tokenizer, "eos_token_id", None)
    if value is not None:
        eos_ids.add(int(value))
    for token in ("<|im_end|>", "<|endoftext|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            eos_ids.add(int(token_id))

    eos_ids.intersection_update(constraint_cache.valid_token_ids)

    synchronizer = GenerationStateSynchronizer(
        constraint_cache, automaton, prompt_length, eos_ids
    )

    class _SchemaLogitsProcessor(LogitsProcessor):
        def __init__(self) -> None:
            self.synchronizer = synchronizer

        def __call__(self, input_ids, scores):  # type: ignore[override]
            started = time.perf_counter()
            synchronizer.sync(input_ids)
            vocab = int(scores.shape[-1])
            valid_mask = constraint_cache.validity_mask(vocab, scores.device)
            mask = torch.full_like(scores, float("-inf"))
            phase = automaton.phase
            if automaton.halt or phase == "complete":
                constraint_cache.record_mask_cpu_time(time.perf_counter() - started)
                return scores + mask
            if phase in ("prefix", "level_word", "diag_prefix"):
                allowed = constraint_cache.allowed_ids(automaton, vocab)
                if not allowed:
                    raise RuntimeError(f"no tokenizer token can advance scaffold phase {phase}")
                mask[:, allowed] = 0.0
            elif phase in ("triage_free", "diag_free"):
                mask[:, valid_mask] = 0.0
                # Precomputed newline subset only; never scan full vocabulary
                # in free-text phases.
                for token_id in constraint_cache.newline_token_ids:
                    if token_id < vocab and not synchronizer.clone_accepts(token_id):
                        mask[:, token_id] = float("-inf")
                for token_id in eos_ids:
                    if token_id < vocab:
                        mask[:, token_id] = float("-inf")
            constraint_cache.record_mask_cpu_time(time.perf_counter() - started)
            return scores + mask

    class _SchemaStopper(StoppingCriteria):
        def __init__(self) -> None:
            self.synchronizer = synchronizer

        def __call__(self, input_ids, scores, **kwargs) -> bool:  # type: ignore[override]
            synchronizer.sync(input_ids)
            return bool(automaton.halt)

    return SchemaGenerationKwargs(
        {
            "logits_processor": LogitsProcessorList([_SchemaLogitsProcessor()]),
            "stopping_criteria": StoppingCriteriaList([_SchemaStopper()]),
            "min_new_tokens": int(min_new_tokens if min_new_tokens is not None else 1),
        },
        synchronizer=synchronizer,
    )
