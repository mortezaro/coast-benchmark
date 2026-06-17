# COAST Benchmark

COAST is a benchmark suite for collaborative auditory grounding. It tests whether audio and speech models can make grounded judgments about acoustic events, speech delivery, interaction structure, and longer event continuity.

The repository is organized as a data-evaluation artifact. The goal is not only to run models, but to show the checks used to decide whether a task is valid enough to report.

## What This Repository Contains

- Data builders for COAST task suites
- Human-packet generation utilities
- Statistical validation scripts
- Trivial-baseline checks: chance, random, majority, and always-first
- Robustness and leakage-check utilities
- Model evaluation runners for local/HF audio models and API audio models
- Example result tables from the compact V5 validation run
- Technical documentation describing task design and quality controls

Raw datasets, downloaded videos, cookies, credentials, generated audio folders, and private machine paths are intentionally excluded.

## Current Public Snapshot

The current compact snapshot is `COAST Human-Valid 14-Task Compact V5`.

- Total examples: 4,200
- Tier 1: 3 perceptual auditory grounding tasks
- Tier 2: 8 speech and interaction grounding tasks
- Tier 3: 3 long-context event grounding tasks
- Reporting standard: per-task first, per-tier second

Most tasks contain 312 examples. The explicit repair cue task contains 144 examples because high-precision real repair cues were rarer under the stricter human-validity rules.

## Task Groups

### Tier 1: Perceptual Auditory Grounding

- `acoustic_plausibility_ranking`
- `fine_grained_acoustic_contrast`
- `foreground_event_focus`

### Tier 2: Speech And Interaction Grounding

- `speaker_count_grounding`
- `delivery_style_grounding`
- `dyadic_matching`
- `turn_holding_continuation`
- `delivery_conditioned_continuation`
- `acknowledgment_cue_detection`
- `clarification_question_detection`
- `explicit_repair_cue_detection`

### Tier 3: Long-Context Event Grounding

- `event_window_grounding`
- `long_context_event_retrieval`
- `narrative_continuation_plausibility`

## Example Results

The example compact V5 result tables are in:

```text
examples/results/
```

These tables include task counts, validation baselines, model scores, and the best observed model per subtask. Failed or incomplete model runs are not counted as benchmark scores.

## Reproducibility Pattern

The standard workflow is:

1. Build candidate task suites from source metadata and audio.
2. Generate human-facing packets for sanity review.
3. Validate label distribution and trivial baselines.
4. Downsample or rebalance to a fixed compact subset.
5. Run model predictions against the fixed subset.
6. Aggregate per-task and per-tier results with baselines.

The important invariant is that model evaluation should point to a fixed validated subset, not resample a benchmark during scoring.

## Key Scripts

- `data_generation/build_coast_eval_subset.py`: create balanced validated subsets.
- `data_generation/build_coast_human_eval_packet.py`: build human-review packets.
- `data_generation/build_coast_human_valid_suite.py`: package human-valid selected tasks.
- `data_generation/build_coast_robustness_report.py`: run robustness and baseline diagnostics.
- `data_generation/build_coast_full_benchmark_results.py`: aggregate model results with baselines.
- `data_generation/run_coast_hf_benchmark.py`: run model evaluation.
- `src/dynsalmon/coast_grounding_eval.py`: scoring, bootstrap intervals, and comparison utilities.
- `src/dynsalmon/coast_hf_benchmark.py`: audio-model benchmark adapters.

## Documentation

Start with:

- `docs/COAST_TECHNICAL_EVALUATION_STANDARD.md`
- `docs/COAST_TASK_REVIEW_GUIDE_RC1.md`
- `docs/COAST_HUMAN_FEEDBACK_AUDIT_V2.md`
- `docs/COAST_COMPACT_V5_EVALUATION_INSTRUCTIONS.md`

These documents describe how the tasks were reframed, what human reviewers flagged, and how those critiques changed the release.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[eval]"
```

Some model backends require additional model-specific dependencies and access to gated checkpoints. Those failures should be reported separately from model scores.

## Minimal Table Refresh

```bash
python data_generation/build_coast_full_benchmark_results.py \
  --benchmark-root /path/to/model_outputs \
  --validated-root /path/to/validated_subsets \
  --output-dir /path/to/results_tables
```

