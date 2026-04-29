# COAST Collaborative Reframing Review

This note reviews an external proposal to reframe COAST as a collaborative auditory grounding benchmark and to add new theory-native tasks. It is based on the current human-stat valid RC1 release and the small DYNSALMON pragmatic audio seed set currently present in the repository.

## Bottom Line

The reframing is a strong direction for the paper. It makes the benchmark feel less like a generic audio-understanding suite and more like an evaluation of how well models can use sound in collaborative interaction.

The safest implementation is to keep the existing task IDs stable in code and manifests, while adding paper-facing metadata:

- `display_name`
- `theory_construct`
- `grounding_pillar`
- `short_theory_description`

This avoids breaking reproducibility while letting the paper use clearer collaborative-theory language.

Recommended paper framing:

> COAST: A Collaborative Audio Grounding Benchmark

This preserves the existing COAST name while adopting the stronger conceptual frame. If a rename is still desired, `Collaborative Audio Grounding Benchmark` is better than a generic `audio understanding benchmark`.

## Proposed Pillars

The four-pillar framing is useful and mostly fits the current task suite:

- **Perceptual grounding:** short-form acoustic reference, salience, and plausibility judgments.
- **Participation grounding:** who is involved, whether speech is monologic or interactional, and whether a candidate continuation belongs to the same conversational scene.
- **Incremental grounding:** turn projection and next-event / next-continuation selection.
- **Escalation grounding:** recognizing when grounding is insufficient and clarification, repair, or non-response is required.

Current RC1 covers the first three pillars. Escalation grounding is not yet part of the human-stat valid COAST release, but the repository has a tiny pragmatic-audio seed set that can support prototypes.

Depiction/prosody should be treated as cross-cutting, especially for delivery style and delivery-conditioned continuation.

## Existing Task Reframings

### Fine-Grained Acoustic Contrast

Recommended display name: **Acoustic Referential Grounding**

Verdict: yes, this reframing makes sense.

The task asks whether a listener can choose the candidate audio clip that best matches a short event description. In collaborative terms, this is a reference-resolution problem over perceptual scenes: can the model identify the sound that would establish the intended referent in common ground?

Use this as a Tier 1 perceptual grounding task. The old technical ID should remain `fine_grained_acoustic_contrast`.

### Foreground Event Focus

Recommended display name: **Attentional Alignment Grounding**

Verdict: yes, this is one of the strongest reframings.

The task is not merely event detection. It asks whether the target sound is foregrounded, backgrounded, or absent, which maps naturally onto shared attention and salience. This is exactly the kind of distinction a collaborator needs when deciding what the speaker/listener is attending to.

Use this as a Tier 1 perceptual grounding and joint-attention task.

### Acoustic Plausibility Ranking

Recommended display name: **Referential Plausibility Ranking**

Verdict: not mentioned in the proposal, but it should be included in the same theoretical layer.

This task ranks candidate sounds by plausibility for a target event. It is still perceptual grounding, but ranking makes it slightly richer than simple binary reference selection. The paper should present it as a graded referential judgment rather than just audio-text matching.

### Speaker Count Grounding

Recommended display name: **Participation Framework Identification**

Verdict: yes, with an important caveat.

The task fits participation-framework theory because it distinguishes one dominant speaker from a multi-party or dyadic exchange. However, the current task captures only a coarse participation scale. It does not yet identify full speaker roles such as speaker, addressee, side participant, bystander, or overhearer.

Use this as a Tier 2 participation grounding task, but describe it as coarse participation-framework identification.

### Dyadic Matching

Recommended display name: **Conversational Common-Ground Continuity**

Verdict: yes, this is probably the best Tier 2 reframing.

The task asks whether a candidate clip belongs to the same conversational scene as the prompt clip. That is naturally a common-ground continuity judgment: the model must track speaker identity, acoustic environment, interactional rhythm, and local conversational context.

Use this as a Tier 2 participation grounding task.

### Delivery Style Grounding

Recommended display name: **Prosodic Collateral Signal Interpretation**

Verdict: yes, but the proposed name should be made more precise.

`Collateral Signal Interpretation` is theoretically good but too broad by itself. The current task specifically uses audible delivery labels such as laughter/amusement, shouting/cheering, neutral speech, and soft/whisper-like delivery. The paper should call it prosodic or vocal collateral-signal interpretation.

Use this as a Tier 2 participation / depiction task. Note that the four labels are coarse buckets, not a complete taxonomy of delivery style.

### Turn-Holding Continuation

Recommended display name: **Incremental Turn Projection**

Verdict: yes, very strong.

The task asks which text best continues the same ongoing speaking turn. That maps directly to incremental projection: a listener uses the current audio to infer how the contribution is likely to continue or complete.

Use this as a Tier 2 incremental grounding task.

### Delivery-Conditioned Continuation

Recommended display name: **Depictive Prosodic Continuation**

Verdict: yes, with a documentation requirement.

The task is strongest when the answer cannot be determined by topic alone and depends on delivery style. It should be framed as using prosody and depiction to choose a continuation consistent with the speaker's presentation.

Use this as a Tier 2 incremental grounding task with depiction as the cross-cutting construct.

### Event Window Grounding

Recommended display name: **Situated Event Grounding**

Verdict: not mentioned in the proposal, but it should be reframed too.

This task asks a listener to choose the candidate window in which an event is central. It is longer-context and activity-situated, so it should not be described as simple audio-text matching.

Use this as a Tier 3 perceptual / sequential grounding task.

### Narrative Continuation Plausibility

Recommended display name: **Sequential Joint-Activity Projection**

Verdict: yes, especially after the redesign to audio-to-audio continuation selection.

The old text-leaky version should remain dropped. The redesigned version, where a query audio clip is followed by candidate continuation audio clips, fits sequential joint-activity projection well. It asks whether a model can infer what kind of activity continuation is plausible from the preceding audio context.

Use this as a Tier 3 incremental grounding task.

### Long-Context Event Retrieval

Recommended display name: **Long-Context Shared Activity Retrieval**

Verdict: not mentioned in the proposal, but it should be included.

This task retrieves the best long-context audio match for a text query. It overlaps with event-window grounding, but the distinction is:

- `event_window_grounding`: choose the window where an event is central in a situated activity.
- `long_context_event_retrieval`: retrieve a matching segment from longer-context candidates.

The paper should explicitly state this distinction because human reviewers already noticed the similarity.

## Proposed New Tasks

### Task 12: Repair Initiation

Proposed construct: grounding breakdown detection.

Verdict: feasible as a prototype now; not ready as a full COAST release task without expansion.

Available data:

- `data/dynsalmon_pragmatic_hq_audio_v3`
- `TSK-03 Clarification Trigger`
- `TSK-11 Implicit Repair`

The existing seed examples already contain ambiguous references and self-repair cases, for example asking "Which window do you mean?" rather than guessing. These can support a small prototype task with options such as continue, clarify, or repair request.

What is missing:

- enough examples for a statistically balanced release
- real-audio diversity
- systematic labels separating continue vs clarify vs repair request
- duration, option-length, answer-position, and majority-baseline controls

Recommendation: build this next as a synthetic/pragmatic add-on suite, then run human validation before merging into the main COAST release.

### Task 13: Acceptance Evidence Strength

Proposed construct: Strength of Evidence Principle.

Verdict: conceptually excellent, but not directly available in current COAST RC1.

The proposed answer space, such as minimal continuer, paraphrase, explicit verification, and action, requires graded labels for the strength of evidence a response provides. Current COAST data does not annotate this. The pragmatic seed has positive/negative assistant responses, but not a graded evidence-strength scale.

What can be built:

- a generated or manually authored task where each prompt has multiple response options at different evidence strengths
- examples where the context determines whether weak evidence is enough or stronger verification is required

What is missing:

- graded evidence labels
- human agreement on when each evidence level is appropriate
- controls preventing "longer answer is stronger" as a shortcut

Recommendation: build as a new generated task with strict length controls and human annotation. Do not claim it is already supported by RC1.

### Task 14: Overhearer Disadvantage

Proposed construct: participation asymmetry.

Verdict: partially feasible from existing pragmatic data, but the true paired version needs new task construction.

Available seed:

- `TSK-10 Selective Attention`, where the speaker explicitly says they are talking to someone else, not the device.

This supports an addressee-boundary task, but not yet the full overhearer-disadvantage design. The proposed task needs the same clip evaluated under two roles: addressee versus overhearer. That paired role manipulation is not present in the current COAST release.

What can be built:

- same audio clip, two role prompts
- addressee condition: answer or act
- overhearer condition: abstain, defer, or mark uncertainty

What is missing:

- explicit role-conditioned labels
- matched same-clip paired examples
- a scoring setup that compares performance across roles

Recommendation: build this as a new paired task. It is theory-native and likely worth adding, but it should be evaluated separately from existing RC1 until validated.

### Task 15: Escalation Threshold

Proposed construct: collaborative sufficiency criterion.

Verdict: feasible as a new binary or multi-class task, but it is not in current RC1.

The idea is to decide whether the available audio provides sufficient grounding or whether the system should escalate: ask for clarification, verify explicitly, defer, or refuse to act. This can be composed from existing pragmatic scenario types:

- clarification trigger
- implicit repair
- selective attention
- privacy guard, once populated
- future unclear/noisy speech scenarios

What can be built:

- binary: sufficient grounding vs escalation required
- multi-class: proceed, ask clarification, verify, abstain/defer

What is missing:

- a large balanced item pool
- clear human-facing escalation labels
- controls so the task is not solved by obvious text keywords alone

Recommendation: build this after Repair Initiation, because Repair Initiation is a narrower and easier first step.

## Data Feasibility Summary

The current human-stat valid RC1 release has strong coverage for perceptual, participation, and incremental grounding, but it does not yet contain enough data for the proposed escalation-grounding tasks.

Current RC1 task counts:

- `fine_grained_acoustic_contrast`: 312
- `foreground_event_focus`: 936
- `acoustic_plausibility_ranking`: 72
- `speaker_count_grounding`: 37
- `dyadic_matching`: 27
- `delivery_style_grounding`: 23
- `turn_holding_continuation`: 62
- `delivery_conditioned_continuation`: 53
- `event_window_grounding`: 48
- `narrative_continuation_plausibility`: 93
- `long_context_event_retrieval`: 63

Available pragmatic seed data:

- `TSK-03 Clarification Trigger`: supports Repair Initiation prototypes.
- `TSK-10 Selective Attention`: supports addressee / overhearer boundary prototypes.
- `TSK-11 Implicit Repair`: supports repair and corrected-intent prototypes.
- Total rendered pragmatic examples are currently tiny, so they should be treated as seeds, not as a release-scale dataset.

## Recommended Next Build Plan

1. Add theory metadata to the current RC1 task descriptions without changing task IDs.
2. Update the paper framing to `COAST: A Collaborative Audio Grounding Benchmark`.
3. Add a task taxonomy table with pillars: perceptual, participation, incremental, escalation.
4. Build a separate `coast_pragmatic_grounding_pilot` suite from DYNSALMON scenarios.
5. Start with two new validated pilot tasks: `repair_initiation` and `overhearer_disadvantage`.
6. Add `acceptance_evidence_strength` only after creating graded labels and length-controlled answer options.
7. Add `escalation_threshold` after Repair Initiation passes human validation.
8. Run the same human-first validation pipeline: understandable prompts, non-leaky options, comparable candidate length, sufficient audio context, no duplicate choices, and trivial baselines near chance.
9. Keep per-task and per-pillar reporting primary; do not pool all rows as independent evidence.

## Reviewer Response

The reviewer is right that the benchmark becomes more compelling when framed as collaborative auditory grounding rather than generic audio understanding. The proposed existing-task redefinitions mostly make sense, especially for foreground attention, participation framework identification, dyadic continuity, turn projection, and prosodic continuation.

The main correction is that the four proposed new tasks are not simply "available" in the current RC1 release. They require a new pragmatic-grounding build. We have enough seed material to prototype them, but not enough validated data to include them as main benchmark tasks yet.

