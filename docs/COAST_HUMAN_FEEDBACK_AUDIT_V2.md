# COAST Human Feedback Audit V2

This note records the second human review pass after the first curated packet rebuild.

The key result is encouraging:

- several tasks now look meaningfully better to a human reviewer
- the benchmark is no longer failing uniformly at the human-facing level
- but a few conceptual distinctions still need to be sharpened

## Concrete Packet Changes After This Review

In response to this second human pass, the curated human-packet builder was tightened again:

- the flagged bad `turn_holding_continuation` example remains excluded
- `narrative_continuation_plausibility` is no longer accepted in its text-leaky form
- task instructions now explicitly explain:
  - why `event_window_grounding` and `long_context_event_retrieval` are different
  - that `delivery_style_grounding` uses four coarse buckets, not an exhaustive taxonomy
  - that `turn_holding_continuation` is about continuing the same thought, not just topical similarity
- the packet builder now tries to avoid reusing identical text-option sets within the same task when choosing examples

These updates do not make the human packet “finished,” but they do make the packet itself more self-explanatory and less repetitive.

## What Improved

The second human pass explicitly marked these as improved or good:

- `acoustic_plausibility_ranking`
  - much better than before
  - remaining issue: one obviously irrelevant distractor can still make the ranking too easy
- `delivery_conditioned_continuation`
  - much better
  - remaining issue: answer options should be closer in length and style
- `delivery_style_grounding`
  - acceptable
  - remaining issue: style taxonomy should be justified clearly
- `dyadic_matching`
  - liked by the human reviewer
  - remaining issue: increase example diversity across scenes
- `event_window_grounding`
  - strongly positive feedback
- `fine_grained_acoustic_contrast`
  - liked by the human reviewer
  - remaining issue: distinguish it more clearly from `event_window_grounding`
- `foreground_event_focus`
  - good overall
  - remaining issue: some synthetic mixes need clearer foreground/background rendering
- `speaker_count_grounding`
  - good overall
- `turn_holding_continuation`
  - liked conceptually
  - remaining issue: one selected example appears semantically mismatched

## Remaining Conceptual Problems

The second human pass also surfaced a few task-definition issues:

### 1. `event_window_grounding` vs `long_context_event_retrieval`

The reviewer found these too close.

The intended distinction should be:

- `event_window_grounding`
  - choose the clip where the queried event is central or dominant in that window
- `long_context_event_retrieval`
  - choose the clip that contains the queried event somewhere in the broader audio context, even if it is not dominant

If this distinction is not obvious to humans, the benchmark text and examples must make it clearer.

Current packet response:

- the human packet instructions now spell this out directly
- future full benchmark releases should keep these two tasks clearly separated in both wording and example selection

### 2. `turn_holding_continuation`

The reviewer liked the task concept, but one example still appears to include an option that is too close to the current prompt turn.

That means:

- the task concept survives
- the specific example selection still needs stricter curation

Current packet response:

- the known bad example was removed from the curated packet
- future versions should keep checking for “same topic but wrong turn” distractors, because those are the most misleading failure mode

### 3. `narrative_continuation_plausibility`

The reviewer still likes this task family and wants it back if possible.

That suggests:

- the task should not be abandoned
- but the current text-conditioned version should be dropped from the main human packet
- the replacement should be:
  - `query audio`
  - `2-3 continuation audio candidates`
  - matched duration windows
  - no lexical leakage from the prompt

Current packet response:

- the text-leaky version is no longer accepted as the main task format
- the main task should be rebuilt as `query audio -> continuation audio selection`
- a text version, if kept at all, should be treated as an appendix-only variant rather than part of the main human-valid suite

## Task Status After Second Human Pass

- `acoustic_plausibility_ranking`: improved, keep but continue reducing obviously irrelevant distractors
- `delivery_conditioned_continuation`: improved, keep but continue controlling option length and diversity
- `delivery_style_grounding`: acceptable, keep with explicit note that the four labels are coarse buckets
- `dyadic_matching`: strong concept, keep and increase source diversity in larger packets
- `event_window_grounding`: strong concept, keep
- `fine_grained_acoustic_contrast`: good, keep, but clarify distinction from event-window tasks
- `foreground_event_focus`: good, keep, but improve synthetic foreground/background clarity when mixing
- `long_context_event_retrieval`: keep, but only if the packet explains how it differs from event-window grounding
- `narrative_continuation_plausibility`: redesign as audio-continuation selection before treating it as human-valid
- `speaker_count_grounding`: good, keep
- `turn_holding_continuation`: strong concept, keep, but continue auditing distractor quality

## Updated Direction

After the second human pass, the right direction is:

- keep iterating on a **human-clean packet**
- continue excluding tasks that are still clearly leaky or unclear
- bring back promising tasks only after stricter item-level filtering

The benchmark is moving in the right direction, but the release bar should remain:

- structural validity
- human validity
- clear task differentiation
