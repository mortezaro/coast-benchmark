# COAST Technical Evaluation Standard

COAST is a collaborative auditory grounding benchmark. The benchmark is intentionally built as a data-evaluation artifact first: every task is required to survive statistical sanity checks and human-facing clarity checks before it is used for model comparison.

This note is written for speech scientists, benchmark reviewers, and data evaluators who want to inspect whether the construction process is rigorous enough to trust.

## Core Principle

The benchmark should not reward shortcuts that a listener, a trivial classifier, or a fixed answer-position heuristic could exploit. A task is only retained when the audio judgment is meaningful, the answer space is balanced enough for fair comparison, and the instructions are understandable to humans.

## Release Unit

The current compact release is `COAST Human-Valid 14-Task Compact V5`.

- Total items: 4,200
- Tier 1: 3 tasks, 312 examples each
- Tier 2: 8 tasks, 7 tasks with 312 examples and one high-precision repair-cue task with 144 examples
- Tier 3: 3 tasks, 312 examples each
- Primary reporting: per-task first, per-tier second
- Avoided reporting: naive row-level pooling across all tasks

## Why Per-Task And Per-Tier Reporting Comes First

COAST tasks reuse some source clips across subtasks. That is useful for testing different forms of grounding over similar acoustic material, but it means rows are not globally independent. We therefore report:

1. task-level metrics
2. tier-level macro summaries
3. chance-normalized or cluster-aware aggregate analyses only when needed

This is deliberately conservative. It prevents one large or easy task from dominating the interpretation.

## Data Sources

The release uses real audio-derived examples from:

- VGGSound-style short acoustic event clips for Tier 1
- SpeakerVid-derived speech and interaction clips for Tier 2
- EPIC/Ego-style longer event windows for Tier 3

Synthetic labels are not used as final truth for the human-facing collaborative cue tasks. Transcripts can seed candidate extraction, but retained examples must remain audio-judgable and human-auditable.

## Task Families

### Tier 1: Perceptual Auditory Grounding

Tier 1 tests short-window acoustic grounding. The task should be answerable from sound, not from a leaked prompt.

- `acoustic_plausibility_ranking`: rank candidate clips by plausibility for an event prompt.
- `fine_grained_acoustic_contrast`: choose the closer acoustic match between two semantically related candidates.
- `foreground_event_focus`: decide whether a target sound is foreground, background, or absent.

### Tier 2: Speech And Interaction Grounding

Tier 2 tests speech delivery, turn structure, interaction scale, and low-level grounding cues.

- `speaker_count_grounding`: distinguish one dominant speaker from multiple/dyadic speech.
- `delivery_style_grounding`: choose the best vocal delivery style.
- `dyadic_matching`: choose which follow-up clip belongs to the same conversational scene.
- `turn_holding_continuation`: choose text that continues the same ongoing speaking turn.
- `delivery_conditioned_continuation`: choose text matching the speaker's delivery style.
- `acknowledgment_cue_detection`: detect short acknowledgments such as "okay", "yeah", or "mm-hmm".
- `clarification_question_detection`: detect explicit clarification questions.
- `explicit_repair_cue_detection`: detect explicit repair cues such as "what do you mean?" or "can you repeat that?"

### Tier 3: Long-Context Event Grounding

Tier 3 tests longer event windows and narrative/event continuity.

- `event_window_grounding`: choose the candidate window where a described event is central.
- `long_context_event_retrieval`: retrieve the best matching event window from a text query.
- `narrative_continuation_plausibility`: choose the most plausible continuation audio, avoiding the earlier text-leaky version.

## Statistical Validation Criteria

Every retained task is checked against trivial baselines and label distributions.

Required checks:

- `chance`: expected random performance from answer-space size
- `random`: empirical random baseline over repeated draws
- `majority`: always choose the most frequent label
- `always_first`: always choose the first option
- label/answer-position distribution
- task count and adaptive downsampling metadata
- bootstrap confidence intervals for model scores

Default validity rule:

```text
majority, random, and always-first should remain near chance.
If a trivial strategy exceeds chance by more than the configured tolerance,
the task is rebalanced, downsampled, or dropped.
```

For the compact V5 release the tolerance is `chance + 0.05`.

## Human Validity Criteria

The benchmark went through human packet review. The review exposed several problems in earlier versions:

- prompts that leaked the answer
- candidate clips with mismatched duration
- options that were duplicate or near-duplicate
- tasks whose theoretical label was too abstract for audio-only judgment
- short clips where humans lacked enough context
- examples where text could answer the task without listening

The release responds to those issues by requiring:

- concise human-readable prompts
- no answer leakage in the task prompt
- no duplicate options
- duration controls where task design requires fair comparison
- longer context for conversational and event-window tasks
- audio-first answerability
- human packet instructions for confidence, difficulty, comments, and audio-alone answerability

## Collaborative Cue Task Redesign

The first attempt at theory-native tasks used labels such as `acceptance_evidence_strength` and `escalation_threshold`. Human feedback showed these were too abstract and ambiguous for reliable automatic construction.

The final design uses observable cue tasks instead:

- acknowledgment cue present vs absent
- clarification question present vs absent
- explicit repair cue present vs absent

The theory mapping is kept in the paper framing, not forced into the raw label. This is a central data-quality choice: label what humans can hear, then interpret it theoretically.

## Leakage And Contamination Checks

The validation pipeline checks for:

- train/test leakage within a task
- duplicate identifiers inside each task
- source clip reuse across tasks
- answer-position collapse
- prompt/option leakage
- trivial-label collapse

Cross-task source reuse is documented rather than hidden. It is the reason COAST emphasizes task and tier metrics over naive pooled accuracy.

## Model Evaluation Protocol

The evaluator supports:

- contrastive audio-text models such as CLAP
- audio-language models such as Qwen2-Audio
- multimodal speech/audio models such as Phi-4-MM
- speech-to-speech / speech-token models such as CAST, TinyWave, and Moshi
- API audio models when quota permits

For each model:

1. evaluate the fixed validated subset
2. write per-task `predictions.jsonl`
3. write per-tier `summary.json`
4. write root `benchmark_summary.json`
5. aggregate per-task results with baselines into Markdown and CSV tables

The evaluator records model failures separately from benchmark scores. For example, an inaccessible gated dependency or an API rate limit is not reported as model performance.

## What Counts As A Clean Release

A clean release must include:

- fixed validated subset roots
- task metadata with counts and baselines
- human packet or human-review protocol
- per-task/pairwise model outputs
- per-tier summaries
- explicit dropped/blocked task notes
- reproducible builder commands
- no raw secrets, cookies, or licensed data blobs in GitHub

## Current Known Limitations

- Some public model families require gated dependencies.
- API models may hit request-per-day limits on the full 4,200-item release.
- `explicit_repair_cue_detection` has 144 examples rather than 312 because high-precision real repair cues were rarer under strict human-validity rules.
- Model scores should be interpreted task-by-task; global macro scores are secondary.

## Practical Reviewer Checklist

A reviewer should be able to verify:

- the data builders separate candidate generation from validation
- task metadata exposes count, label distribution, and baselines
- human feedback changed the task definitions
- no collapsed task is retained silently
- results tables include random, majority, and always-first baselines
- failed or incomplete model runs are not mixed into the leaderboard

That is the standard we tried to meet.
