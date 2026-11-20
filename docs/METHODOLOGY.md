# MedShift Methodology & Experimental Design

## 1. Benchmark & Adjudication Design (`MedShiftBench`)

`MedShiftBench` links **260 clinical prompts** to **145 underlying clinical cores** across six structured suites:

* **GeoShift (30 cores / 60 prompts):** identical clinical presentation shifted across geographic epidemiology contexts.
* **Geography Control (20 cores / 40 prompts):** paired location changes where the optimal clinical decision should remain invariant.
* **LanguageShift (30 cores / 60 prompts):** matched English and Modern Standard Arabic (MSA) presentations.
* **Code-Switch (20 cores / 40 prompts):** English clinical anchor paired with constructed Arabic–English code-switching.
* **OxiShift (25 cores / 40 prompts):** pulse-oximetry reading conflict pairs plus normal controls.
* **General / Triage Control (20 cores / 20 prompts):** unpaired baseline clinical cases.

All 780 model outputs across Base, Generic LoRA, and MedShift LoRA were blinded prior to human adjudication by seven human raters, who independently confirmed or corrected every Top-1 diagnosis and triage label.

## 2. Training Configuration (`medshift.training.lora_config`)

* **Base model:** `Qwen/Qwen3.5-9B` (revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`)
* **Adaptation:** BF16 LoRA (`r = 16, α = 32`, dropout `0.05`, no bias) across all 7 attention and MLP projections (`q/k/v/o_proj`, `gate/up/down_proj`)
* **Objective:** completion-only next-token causal cross-entropy (`max_seq_len = 2048`), AdamW (`lr = 2e-4`, cosine decay, 3% warmup), effective batch size 8, 2,000 examples/arm, 2 epochs (500 optimizer steps), seed 42
* **Inference:** deterministic greedy decoding with thinking disabled (`enable_thinking=False`).

## 3. Statistical Protocol (`medshift.statistics`)

* **Strict Intention-to-Treat (ITT):** all 260 benchmark prompts enter the denominator for every arm; missing, refused, or unparseable outputs are scored as incorrect (`0`).
* **Separate Co-Primary Endpoints:** Diagnostic Top-1 accuracy and Triage accuracy are never combined into a composite score.
* **Inference:** item-paired bootstrap percentile 95% confidence intervals (10,000 replicates, seed 42), exact two-sided McNemar tests on discordant pairs (`n10, n01`), and two-endpoint Holm–Bonferroni multiplicity control (`α = 0.05`).

## 4. Decoding & Output Safety Infrastructure

* **Constrained Decoding (`medshift.decoding`):** deterministic schema automaton (`schema_automaton.py`) with cached exact vocabulary token masks (`decoding_glue.py`) enforcing structured clinical JSON/section schemas.
* **Bilingual Clinical Parsing & Collapse Guards (`medshift.guards`):**
  * `clinical_parser.py`: extracts Top-1 diagnosis and urgency level across English and diacritic-normalized Arabic (`TRIAGE` / `DIFFERENTIAL` canonical 8-section structure).
  * `collapse.py` & `loop_detectors.py`: detects periodic token loops, pathological word runs, and catastrophic repetition collapse during greedy generation.
* **KL-Anchored Replay Loss (`medshift.losses`):** exact next-token shifted (`logits[:, :-1]` vs `labels[:, 1:]`) causal cross-entropy with optional KL regularization toward a frozen reference teacher.
