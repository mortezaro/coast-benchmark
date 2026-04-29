# COAST Human-Stat Valid RC1 Task Review Guide

This guide describes the tasks in `coast_human_stat_valid_release_rc1_20260424`.

The release was built in this order:

1. Apply human-validity filters to the full task suites.
2. Run statistical validation on the human-filtered rows.
3. Adaptively downsample only when needed to remove trivial shortcuts.
4. Package the final rows for human evaluation.

The final packet contains 1,726 unique items across 11 tasks. Every included item passed the human-facing curation filters, and every included task passed the trivial-baseline checks used for this release.

## Shared Evaluation Protocol

For each item, a rater should provide:

- chosen answer
- confidence from 1 to 5
- difficulty from 1 to 5
- optional free-text comments

For blind human evaluation, share only:

- `items/`
- `task_instructions/`
- `items_manifest.csv`
- `responses_template.csv`

Keep private:

- `answer_key_private.csv`
- `curation_audit.csv`

## Shared Validity Criteria

The following criteria apply across the release:

- The task must be understandable to a human listener.
- The prompt must not reveal the answer.
- The prompt should use audio-facing wording whenever the benchmark input is audio.
- Answer options must not be duplicated.
- Candidate options should be comparable in format.
- Text options should not be so different in length that length becomes a shortcut.
- Candidate audio clips must be long enough for judgment.
- Candidate audio durations must be reasonably matched within an item.
- Query or prompt audio must provide enough context for the requested judgment.
- The task must require listening, not only reading the text.
- Missing, unreadable, or broken audio rows are excluded.
- MacOS sidecar files such as `._*.json` are ignored.
- Majority baseline must not exceed chance by more than 0.10.
- Always-first baseline must not exceed chance by more than 0.10.
- Random baseline must not exceed chance by more than 0.05.
- If a human-valid task is statistically imbalanced, adaptive downsampling is used to keep the largest passing subset.
- Tasks that still fail validation after adaptive downsampling are dropped.
- Per-task and per-tier reporting should be primary.
- Naive pooled row-level reporting should be avoided because some source clips are reused across tasks.

## Task 1: Fine-Grained Acoustic Contrast

Count in RC1: 312 items

Tier: Tier 1

Input:

- a short text prompt describing an audio event
- two candidate audio clips

Rater action:

- listen to both candidate clips
- choose the clip that better matches the described acoustic event

What it measures:

- local acoustic discrimination
- fine-grained audio-text grounding
- ability to distinguish acoustically similar or semantically nearby sounds

How it works:

- each item presents two candidate clips
- one candidate is the better acoustic match to the prompt
- the task is intentionally short-form and local, unlike long-context Tier 3 retrieval

Human-validity checks:

- prompt wording is cleaned to be audio-facing
- candidate clips must be present and listenable
- duplicate options are not allowed
- the task should not be answerable from text alone

Statistical checks:

- binary chance level is 0.50
- majority, always-first, and random baselines must stay near chance
- the final subset passed these checks without dropping the task

Reviewer notes:

- This task is closest to classic audio-text matching, but it is stricter because the candidates should be close enough that listening matters.
- Reviewers should flag examples where both clips are clearly unrelated, because that makes the task too easy.

## Task 2: Foreground Event Focus

Count in RC1: 936 items

Tier: Tier 1

Input:

- one query audio clip
- a target sound named in the prompt
- three label options: foreground, background, absent

Rater action:

- listen to the clip
- decide whether the target sound is the main audible focus, present but secondary, or absent

What it measures:

- salience-aware audio grounding
- ability to distinguish presence from foreground dominance
- robustness to mixtures and background sound

How it works:

- the prompt names a target sound
- the answer is not just whether the sound exists
- foreground means the target is dominant
- background means the target is audible but not dominant
- absent means the target is not meaningfully present

Human-validity checks:

- target sound is stated plainly
- label space is fixed and non-duplicated
- synthetic or mixed examples must remain clear enough for a human listener
- ambiguous or inaudible rows are filtered out where duration/audio checks fail

Statistical checks:

- three-way chance level is approximately 0.333
- label distribution is validated against majority and always-first shortcuts
- the final subset passed validation

Reviewer notes:

- This is not a dense source-separation task.
- It is a categorical salience judgment.
- Reviewers should mark low confidence when foreground/background is genuinely ambiguous.

## Task 3: Acoustic Plausibility Ranking

Count in RC1: 72 items

Tier: Tier 1

Input:

- a prompt naming a target sound
- three candidate audio clips

Rater action:

- rank the candidate clips from most plausible to least plausible for the target sound

What it measures:

- ranking-style audio-text grounding
- ability to identify the best match while also ordering weaker alternatives
- resistance to trivial one-obvious-negative shortcuts

How it works:

- each item has multiple candidate clips
- the answer key is a ranking, not just a single choice
- evaluation can use top-1 correctness and ranking-aware metrics

Human-validity checks:

- candidate clips must be long enough
- candidate durations must be reasonably matched
- prompts are cleaned to avoid benchmark boilerplate
- examples with overly mismatched candidate durations are filtered out

Statistical checks:

- chance depends on candidate count, usually 1/3 for top-1
- majority and always-first baselines are checked against chance
- the final subset passed validation

Reviewer notes:

- Some items may still include one candidate that is obviously wrong.
- That is acceptable only if the remaining candidates still require listening to rank.

## Task 4: Speaker Count Grounding

Count in RC1: 37 items

Tier: Tier 2

Input:

- one speech audio clip
- two text options: single dominant speaker, or multiple speakers/dyadic exchange

Rater action:

- listen to the clip
- decide whether the clip contains one dominant speaker or multiple speakers

What it measures:

- speech-scene grounding
- speaker/interlocutor scale detection
- distinction between monologue-like speech and multi-speaker interaction

How it works:

- the task is binary
- the rater should focus on audible speaker count and interaction pattern
- background noise should not count as a speaker

Human-validity checks:

- query audio must exist and be listenable
- prompt is simple and audio-facing
- label options are fixed and non-duplicated

Statistical checks:

- binary chance level is 0.50
- this task required adaptive downsampling after human filtering
- the retained subset is the largest passing subset found by the validator

Reviewer notes:

- This task has fewer examples because the human-valid filtered source pool was imbalanced.
- Reviewers should flag cases where a second speaker is too faint to judge.

## Task 5: Dyadic Matching

Count in RC1: 27 items

Tier: Tier 2

Input:

- a prompt speech clip
- two candidate follow-up clips

Rater action:

- listen to the prompt clip and both candidates
- choose the candidate most likely to come from the same conversational scene

What it measures:

- conversational scene matching
- room-tone and interaction-continuity reasoning
- broad dyadic context recognition

How it works:

- this is not pure speaker identification
- the best candidate should sound more compatible with the same scene or interaction context
- useful cues may include turn-taking rhythm, room tone, speaker energy, and local conversational style

Human-validity checks:

- prompt clip must be long enough
- candidate clips must be long enough
- candidate durations must be reasonably matched
- overly short or duration-mismatched items are removed

Statistical checks:

- binary chance level is 0.50
- this task required adaptive downsampling
- the final subset passed majority, always-first, and random-baseline checks

Reviewer notes:

- This is naturally difficult.
- Human difficulty ratings are especially important here.
- Reviewers should flag items where both candidates sound equally plausible.

## Task 6: Delivery Style Grounding

Count in RC1: 23 items

Tier: Tier 2

Input:

- one speech audio clip
- four coarse delivery-style options

Rater action:

- listen to the speech clip
- choose the broad delivery style that best matches the clip

What it measures:

- coarse paralinguistic grounding
- recognition of vocal delivery style
- distinction between neutral, amused, high-energy, and soft/whisper-like speech

How it works:

- the style labels are intentionally broad
- the task is not meant to cover every possible prosodic or emotional category
- the rater chooses the closest available bucket

Human-validity checks:

- query audio must be at least long enough to judge style
- options are fixed and non-duplicated
- prompt is audio-facing

Statistical checks:

- four-way chance level is 0.25
- this task required adaptive downsampling
- the final retained subset passed trivial-baseline validation

Reviewer notes:

- This task is smaller because the source labels were imbalanced after human filtering.
- Human comments are useful when a clip feels mixed between categories.

## Task 7: Turn-Holding Continuation

Count in RC1: 62 items

Tier: Tier 2

Input:

- a prompt speech clip
- two text continuation options

Rater action:

- listen to the prompt clip
- choose the text that best continues the same ongoing speaking turn

What it measures:

- local speech-context continuation
- ability to connect acoustic speech context with likely next text
- turn continuity rather than topic matching alone

How it works:

- the prompt audio provides local speaking context
- the correct option should sound like the same speaker continuing the same thought
- distractors may be plausible speech text but should fit the local turn less well

Human-validity checks:

- prompt audio must be long enough
- duplicate text options are not allowed
- known bad items from human review are manually excluded
- options should not be trivially distinguishable from formatting alone

Statistical checks:

- binary chance level is 0.50
- majority, always-first, and random baselines are checked
- the final subset passed validation

Reviewer notes:

- Reviewers should flag cases where the transcript appears to duplicate the prompt audio too directly.
- Confidence and difficulty ratings are important because this task can be subtle.

## Task 8: Delivery-Conditioned Continuation

Count in RC1: 53 items

Tier: Tier 2

Input:

- a prompt speech clip
- two text continuation options

Rater action:

- listen to the prompt clip
- choose the continuation text that best matches the speaker's delivery style

What it measures:

- style-conditioned audio-language continuation
- matching text continuation to vocal energy, pacing, emphasis, or affect
- grounding beyond topic alone

How it works:

- the task asks for delivery-style compatibility
- the semantically closest option is not always intended to be the answer
- the rater should consider how the text would sound if spoken in the current style

Human-validity checks:

- prompt clip must be long enough
- text options must not be too different in length
- duplicate options are removed
- prompt wording is cleaned

Statistical checks:

- binary chance level is 0.50
- retained rows passed baseline validation

Reviewer notes:

- This is partly subjective.
- Human confidence ratings help distinguish valid-but-hard examples from unclear ones.

## Task 9: Event Window Grounding

Count in RC1: 48 items

Tier: Tier 3

Input:

- a long-form event prompt
- multiple candidate audio clips

Rater action:

- listen to all candidates
- choose the clip where the named event is central or dominant

What it measures:

- long-form acoustic event grounding
- ability to identify the best local event window
- recognition of everyday action sounds

How it works:

- the target event should be the main event in the correct window
- candidates are drawn from long-form activity sources
- this is stricter than broad retrieval because centrality matters

Human-validity checks:

- candidate clips must be long enough
- candidate durations must be reasonably matched
- prompts are cleaned to explain the event directly
- many raw Tier 3 candidates were excluded for being too short

Statistical checks:

- chance depends on candidate count, usually 0.25
- final labels are balanced across answer positions
- the task passed validation

Reviewer notes:

- This is one of the best Tier 3 human tasks, but it is naturally difficult.
- Low confidence does not necessarily mean the task is invalid.

## Task 10: Narrative Continuation Plausibility

Count in RC1: 93 items

Tier: Tier 3

Input:

- a query audio clip
- candidate continuation audio clips

Rater action:

- listen to the query clip
- choose the continuation clip that most plausibly comes next

What it measures:

- audio-to-audio narrative continuation
- temporal and activity-sequence reasoning
- ability to infer plausible next events from sound

How it works:

- the old text-leaky version was removed
- the current version uses audio continuation candidates
- the rater should judge which continuation best fits the local activity sequence

Human-validity checks:

- query clip must be long enough
- candidate clips must be long enough
- candidate durations must be reasonably matched
- items with leaked text labels are not used

Statistical checks:

- chance depends on candidate count, usually 1/3
- answer positions are balanced
- the retained subset passed validation

Reviewer notes:

- This was the 11th task added back after redesign.
- It is included because the audio-to-audio version passed both human and statistical gates.

## Task 11: Long-Context Event Retrieval

Count in RC1: 63 items

Tier: Tier 3

Input:

- a text query naming an event
- multiple candidate audio clips

Rater action:

- choose the candidate clip that contains the queried event somewhere in the clip

What it measures:

- broad long-context event retrieval
- ability to find an event in longer or more complex activity audio
- retrieval under distractors

How it works:

- unlike event-window grounding, the queried event does not need to be the dominant event
- the correct candidate only needs to contain the queried event meaningfully
- candidates are drawn from long-form activity sources

Human-validity checks:

- candidate clips must be long enough
- candidate durations must be reasonably matched
- prompts are simple event queries
- many short raw candidates were excluded

Statistical checks:

- chance depends on candidate count, usually 0.25
- answer positions are balanced
- the final subset passed validation

Reviewer notes:

- This task is broader than event-window grounding.
- Reviewers should mark difficulty carefully, because the target event may be present but not dominant.

## Excluded Or Dropped Tasks

The following tasks are not included in RC1:

- `target_sound_grounding`: removed from the human-valid release because human review found the prompt/task framing too close to simple audio-text matching and sometimes unclear.
- `periodicity_aware_grounding`: removed because prompt wording could leak the answer from the event name.
- `onset_sharpness_grounding`: removed because humans found the sharp/diffuse distinction unclear on examples such as clapping or engine-like sounds.
- `audio_text_hard_negative_matching`: removed because prior options were duplicated or weak.
- `phonic_target_grounding`: removed because prompts contained video-centric and multilingual metadata artifacts.
- `speaker_count_aware_continuation`: removed because the relation between the audio and text options was unclear to human reviewers.
- `speech_activity_grounding`: dropped by statistical validation in earlier builds.
- `coarse_event_order`: dropped by statistical validation.
- `causal_event_order`: dropped by statistical validation in the hardened v2 pass.
- `speaker_event_overlap`: unscored because no usable task data is available.

## Reporting Guidance

Report results in this order:

1. per task
2. per tier
3. optional aggregate

Any aggregate should be secondary and ideally chance-normalized. Because source clips can be reused across tasks, row-level pooling should not be treated as independent evidence.
