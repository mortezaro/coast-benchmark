# COAST Compact V5 Full Benchmark Results

Primary metric is task-appropriate choice/top-1 accuracy. Baselines come from the validated compact subset metadata.

## Tier1
| Subtask | N | chance | random | majority | always_first | cast_0_7b_s2s | clap_htsat | moshiko_pytorch_bf16 | phi4_multimodal | qwen2_audio_7b | tinywave_expressive_spirit_lm_interleaved_librilight | tinywave_speech_base_2b | tinywave_speech_expressive_2b | Best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| acoustic_plausibility_ranking | 312 | 0.3333 | 0.3292 | 0.3750 | 0.2628 | 0.3558 | 0.5545 | 0.1859 | 0.3397 | 0.2628 | 0.4103 | 0.3686 | 0.3942 | clap_htsat (0.5545) |
| fine_grained_acoustic_contrast | 312 | 0.5000 | 0.4959 | 0.5288 | 0.4712 | 0.5769 | 0.5417 | 0.4712 | 0.4391 | 0.4712 | 0.4808 | 0.5096 | 0.4872 | cast_0_7b_s2s (0.5769) |
| foreground_event_focus | 312 | 0.3333 | 0.3363 | 0.3333 | 0.3333 | 0.3333 | 0.3237 | 0.3333 | 0.3301 | 0.3333 | 0.3333 | 0.3622 | 0.2788 | tinywave_speech_base_2b (0.3622) |

## Tier2
| Subtask | N | chance | random | majority | always_first | cast_0_7b_s2s | clap_htsat | moshiko_pytorch_bf16 | phi4_multimodal | qwen2_audio_7b | tinywave_expressive_spirit_lm_interleaved_librilight | tinywave_speech_base_2b | tinywave_speech_expressive_2b | Best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| speaker_count_grounding | 312 | 0.5000 | 0.5007 | 0.5449 | 0.4551 | 0.5321 | 0.5032 |  | 0.4551 | 0.4551 | 0.5449 | 0.4551 | 0.4551 | tinywave_expressive_spirit_lm_interleaved_librilight (0.5449) |
| delivery_style_grounding | 312 | 0.2500 | 0.2496 | 0.2500 | 0.2500 | 0.2628 | 0.3045 |  | 0.2500 | 0.2500 | 0.2500 | 0.2660 | 0.2532 | clap_htsat (0.3045) |
| dyadic_matching | 312 | 0.5000 | 0.4987 | 0.5096 | 0.4904 | 0.5321 | 0.4583 |  | 0.4712 | 0.4904 | 0.4679 | 0.4744 | 0.4936 | cast_0_7b_s2s (0.5321) |
| turn_holding_continuation | 312 | 0.5000 | 0.5002 | 0.5096 | 0.4904 | 0.3558 | 0.4776 |  | 0.4904 | 0.4904 | 0.6731 | 0.5224 | 0.4455 | tinywave_expressive_spirit_lm_interleaved_librilight (0.6731) |
| delivery_conditioned_continuation | 312 | 0.5000 | 0.5020 | 0.5032 | 0.5032 | 0.5801 | 0.4712 |  | 0.5032 | 0.5032 | 0.5833 | 0.5962 | 0.5897 | tinywave_speech_base_2b (0.5962) |
| acknowledgment_cue_detection | 312 | 0.5000 | 0.4970 | 0.5000 | 0.5000 | 0.5032 | 0.5096 |  | 0.5417 | 0.5000 | 0.5064 | 0.5000 | 0.5032 | phi4_multimodal (0.5417) |
| clarification_question_detection | 312 | 0.5000 | 0.4979 | 0.5000 | 0.5000 | 0.5000 | 0.5160 |  | 0.5545 | 0.5000 | 0.5000 | 0.4808 | 0.5064 | phi4_multimodal (0.5545) |
| explicit_repair_cue_detection | 144 | 0.5000 | 0.4997 | 0.5486 | 0.4514 | 0.5486 | 0.4792 |  | 0.5486 | 0.4514 | 0.5486 | 0.4236 | 0.4306 | cast_0_7b_s2s (0.5486) |

## Tier3
| Subtask | N | chance | random | majority | always_first | cast_0_7b_s2s | clap_htsat | moshiko_pytorch_bf16 | phi4_multimodal | qwen2_audio_7b | tinywave_expressive_spirit_lm_interleaved_librilight | tinywave_speech_base_2b | tinywave_speech_expressive_2b | Best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| event_window_grounding | 312 | 0.2500 | 0.2529 | 0.2500 | 0.2500 | 0.2692 | 0.4038 |  | 0.2436 | 0.2500 | 0.2628 | 0.2212 | 0.2692 | clap_htsat (0.4038) |
| long_context_event_retrieval | 312 | 0.2500 | 0.2489 | 0.2500 | 0.2500 | 0.2596 | 0.3846 |  | 0.2756 | 0.2500 | 0.2340 | 0.2083 | 0.2372 | clap_htsat (0.3846) |
| narrative_continuation_plausibility | 312 | 0.3333 | 0.3325 | 0.3333 | 0.3333 | 0.3109 | 0.3301 |  | 0.3333 | 0.3333 | 0.3590 | 0.3333 | 0.3558 | tinywave_expressive_spirit_lm_interleaved_librilight (0.3590) |
