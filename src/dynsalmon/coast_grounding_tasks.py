from __future__ import annotations

import json
import math
import random
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dynsalmon.coast import _ffmpeg_binary, write_jsonl


TASK_SPECS: dict[str, dict[str, str]] = {
    "task_1_target_sound_grounding": {
        "task_name": "target_sound_grounding",
        "prediction_type": "candidate_audio_classification",
        "conditioning": "audio_text",
        "current_primary_metric": "accuracy",
        "recommended_metrics": "accuracy, recall@k",
        "comparison_scope": "within_tier",
        "metric_note": "Current evaluation reports forced-choice correctness together with retrieval-style summaries.",
    },
    "task_2_fine_grained_acoustic_contrast": {
        "task_name": "fine_grained_acoustic_contrast",
        "prediction_type": "binary_audio_classification",
        "conditioning": "audio_text",
        "current_primary_metric": "accuracy",
        "recommended_metrics": "accuracy, macro_f1",
        "comparison_scope": "within_tier",
        "metric_note": "Use classification-style metrics; do not compare raw scores across tiers without normalization.",
    },
    "task_3_foreground_event_focus": {
        "task_name": "foreground_event_focus",
        "prediction_type": "label_classification",
        "conditioning": "audio_text",
        "current_primary_metric": "accuracy",
        "recommended_metrics": "accuracy, macro_f1",
        "comparison_scope": "within_tier",
        "metric_note": "Foreground/background labeling is currently categorical rather than temporally segmented.",
    },
    "task_4_acoustic_plausibility_ranking": {
        "task_name": "acoustic_plausibility_ranking",
        "prediction_type": "candidate_audio_ranking",
        "conditioning": "audio_text",
        "current_primary_metric": "top1_accuracy",
        "recommended_metrics": "mAP, ndcg, top1_accuracy",
        "comparison_scope": "within_tier",
        "metric_note": "Current evaluator reports top-ranked correctness together with ranking summaries such as NDCG and average precision.",
    },
    "task_5_periodicity_aware_grounding": {
        "task_name": "periodicity_aware_grounding",
        "prediction_type": "text_option_classification",
        "conditioning": "audio_text",
        "current_primary_metric": "accuracy",
        "recommended_metrics": "accuracy, macro_f1",
        "comparison_scope": "within_tier",
        "metric_note": "Task is designed to test periodic-vs-impulsive grounding under alias-resistant filtering.",
    },
    "task_6_onset_sharpness_grounding": {
        "task_name": "onset_sharpness_grounding",
        "prediction_type": "text_option_classification",
        "conditioning": "audio_text",
        "current_primary_metric": "accuracy",
        "recommended_metrics": "accuracy, macro_f1",
        "comparison_scope": "within_tier",
        "metric_note": "This is a coarse sharpness discrimination task, not a dense temporal onset benchmark.",
    },
    "task_7_audio_text_hard_negative_matching": {
        "task_name": "audio_text_hard_negative_matching",
        "prediction_type": "text_option_classification",
        "conditioning": "audio_text",
        "current_primary_metric": "accuracy",
        "recommended_metrics": "accuracy, macro_f1",
        "comparison_scope": "within_tier",
        "metric_note": "Best interpreted as acoustic-text grounding under plausible semantic distractors.",
    },
    "task_8_causal_event_order": {
        "task_name": "causal_event_order",
        "prediction_type": "candidate_audio_classification",
        "conditioning": "audio_text",
        "current_primary_metric": "accuracy",
        "recommended_metrics": "accuracy, order_consistency",
        "comparison_scope": "within_tier",
        "metric_note": "Sequence-level causal order is currently evaluated with forced-choice accuracy.",
    },
}


@dataclass(slots=True)
class GroundingSourceItem:
    source_item_id: str
    display_name: str
    oracle_audio_path: Path
    short_context_text: str
    rich_context_text: str
    periodicity: dict[str, Any]
    transient: dict[str, Any]
    task_id: str

    @property
    def event_phrase(self) -> str:
        return self.display_name.replace("_", " ")

    @property
    def group_key(self) -> str:
        parts = self.display_name.split("_")
        if len(parts) >= 2 and parts[0] == "people":
            return "people"
        return parts[0]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_source_pool(strict_suite_dir: Path) -> list[GroundingSourceItem]:
    pairwise_path = strict_suite_dir / "pairwise" / "all.jsonl"
    rows = _read_jsonl(pairwise_path)
    by_id: dict[str, GroundingSourceItem] = {}
    for row in rows:
        source_item_id = row["source_item_id"]
        if source_item_id in by_id:
            continue
        extra = row.get("extra", {})
        periodicity = row.get("control_notes", {}).get("periodicity") or extra.get("periodicity") or {}
        transient = extra.get("transient") or {}
        by_id[source_item_id] = GroundingSourceItem(
            source_item_id=source_item_id,
            display_name=row["display_name"],
            oracle_audio_path=Path(row["positive_sample"]["audio_path"]),
            short_context_text=row.get("short_context_text") or row["positive_sample"].get("short_context_text") or row["context_text"],
            rich_context_text=row.get("rich_context_text") or row["positive_sample"].get("rich_context_text") or row["context_text"],
            periodicity=periodicity,
            transient=transient,
            task_id=row["task_id"],
        )
    return sorted(by_id.values(), key=lambda item: item.source_item_id)


def _shared_token_score(a: GroundingSourceItem, b: GroundingSourceItem) -> float:
    a_tokens = set(a.display_name.split("_"))
    b_tokens = set(b.display_name.split("_"))
    score = float(len(a_tokens & b_tokens))
    if a.group_key == b.group_key:
        score += 2.0
    if a.display_name.startswith("people_") and b.display_name.startswith("people_"):
        score += 1.0
    return score


def _candidate_pool(
    target: GroundingSourceItem,
    items: list[GroundingSourceItem],
    *,
    exclude_ids: set[str] | None = None,
) -> list[GroundingSourceItem]:
    exclude_ids = exclude_ids or set()
    return [item for item in items if item.source_item_id != target.source_item_id and item.source_item_id not in exclude_ids]


def _choose_similar(
    target: GroundingSourceItem,
    items: list[GroundingSourceItem],
    rng: random.Random,
    *,
    k: int = 1,
    exclude_ids: set[str] | None = None,
) -> list[GroundingSourceItem]:
    pool = _candidate_pool(target, items, exclude_ids=exclude_ids)
    ranked = sorted(pool, key=lambda item: (-_shared_token_score(target, item), item.source_item_id))
    top = ranked[: max(k * 3, k)]
    rng.shuffle(top)
    return top[:k]


def _choose_dissimilar(
    target: GroundingSourceItem,
    items: list[GroundingSourceItem],
    rng: random.Random,
    *,
    k: int = 1,
    exclude_ids: set[str] | None = None,
) -> list[GroundingSourceItem]:
    pool = _candidate_pool(target, items, exclude_ids=exclude_ids)
    ranked = sorted(pool, key=lambda item: (_shared_token_score(target, item), item.source_item_id))
    top = ranked[: max(k * 3, k)]
    rng.shuffle(top)
    return top[:k]


def _probe_wav_duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def _run_ffmpeg(args: list[str]) -> None:
    subprocess.run([_ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def _mix_audio(
    output_path: Path,
    primary_path: Path,
    secondary_path: Path,
    *,
    primary_volume: float,
    secondary_volume: float,
    duration_sec: float | None = None,
    sample_rate: int = 16000,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filter_parts = [
        f"[0:a]volume={primary_volume}[a0]",
        f"[1:a]volume={secondary_volume}[a1]",
        "[a0][a1]amix=inputs=2:duration=longest[m]",
    ]
    final_label = "[m]"
    if duration_sec is not None:
        filter_parts.append(f"[m]atrim=0:{duration_sec:.6f}[o]")
        final_label = "[o]"
    _run_ffmpeg(
        [
            "-i",
            str(primary_path),
            "-i",
            str(secondary_path),
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            final_label,
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )


def _trim_audio(output_path: Path, input_path: Path, *, duration_sec: float, sample_rate: int = 16000) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "-t",
            f"{duration_sec:.6f}",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )


def _concat_audio(
    output_path: Path,
    first_path: Path,
    second_path: Path,
    *,
    trim_sec: float = 1.2,
    gap_sec: float = 0.25,
    sample_rate: int = 16000,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="coast_concat_") as temp_dir:
        silence_path = Path(temp_dir) / "silence.wav"
        _run_ffmpeg(
            [
                "-f",
                "lavfi",
                "-i",
                f"anullsrc=r={sample_rate}:cl=mono",
                "-t",
                f"{gap_sec:.6f}",
                "-c:a",
                "pcm_s16le",
                str(silence_path),
            ]
        )
        _run_ffmpeg(
            [
                "-i",
                str(first_path),
                "-i",
                str(silence_path),
                "-i",
                str(second_path),
                "-filter_complex",
                (
                    f"[0:a]atrim=0:{trim_sec:.6f},asetpts=PTS-STARTPTS[a0];"
                    f"[1:a]asetpts=PTS-STARTPTS[a1];"
                    f"[2:a]atrim=0:{trim_sec:.6f},asetpts=PTS-STARTPTS[a2];"
                    "[a0][a1][a2]concat=n=3:v=0:a=1[out]"
                ),
                "-map",
                "[out]",
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ]
        )


def _task_row_base(task_name: str, item_ids: list[str], prompt: str) -> dict[str, Any]:
    return {
        "task_name": task_name,
        "source_item_ids": item_ids,
        "prompt": prompt,
    }


def _build_task_1(items: list[GroundingSourceItem], rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        distractors = _choose_dissimilar(item, items, rng, k=min(3, max(0, len(items) - 1)))
        candidates = [item, *distractors]
        rng.shuffle(candidates)
        rows.append(
            {
                **_task_row_base("target_sound_grounding", [candidate.source_item_id for candidate in candidates], item.rich_context_text),
                "id": f"t1-{item.source_item_id}",
                "candidate_audio_paths": [str(candidate.oracle_audio_path) for candidate in candidates],
                "candidate_labels": [candidate.display_name for candidate in candidates],
                "gold_index": candidates.index(item),
            }
        )
    return rows


def _build_task_2(items: list[GroundingSourceItem], rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        similar = _choose_similar(item, items, rng, k=1)
        if not similar:
            continue
        candidates = [item, similar[0]]
        rng.shuffle(candidates)
        rows.append(
            {
                **_task_row_base("fine_grained_acoustic_contrast", [candidate.source_item_id for candidate in candidates], item.short_context_text),
                "id": f"t2-{item.source_item_id}",
                "candidate_audio_paths": [str(candidate.oracle_audio_path) for candidate in candidates],
                "candidate_labels": [candidate.display_name for candidate in candidates],
                "gold_index": candidates.index(item),
            }
        )
    return rows


def _build_task_3(items: list[GroundingSourceItem], rng: random.Random, audio_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        distractor_list = _choose_dissimilar(item, items, rng, k=1)
        if not distractor_list:
            continue
        distractor = distractor_list[0]
        duration_sec = min(_probe_wav_duration_sec(item.oracle_audio_path), _probe_wav_duration_sec(distractor.oracle_audio_path))
        for mode, volumes, label in [
            ("foreground", (1.0, 0.35), "foreground"),
            ("background", (0.25, 1.0), "background"),
            ("absent", (0.0, 1.0), "absent"),
        ]:
            output_path = audio_dir / "task3_foreground_focus" / f"{item.source_item_id}-{mode}.wav"
            if mode == "absent":
                _trim_audio(output_path, distractor.oracle_audio_path, duration_sec=duration_sec)
            else:
                _mix_audio(
                    output_path,
                    item.oracle_audio_path,
                    distractor.oracle_audio_path,
                    primary_volume=volumes[0],
                    secondary_volume=volumes[1],
                    duration_sec=duration_sec,
                )
            rows.append(
                {
                    **_task_row_base("foreground_event_focus", [item.source_item_id, distractor.source_item_id], item.rich_context_text),
                    "id": f"t3-{item.source_item_id}-{mode}",
                    "query_audio_path": str(output_path),
                    "label_space": ["foreground", "background", "absent"],
                    "gold_label": label,
                    "distractor_label": distractor.display_name,
                }
            )
    return rows


def _build_task_4(items: list[GroundingSourceItem], rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        similar = _choose_similar(item, items, rng, k=1)
        far = _choose_dissimilar(item, items, rng, k=1, exclude_ids={candidate.source_item_id for candidate in similar})
        if not similar or not far:
            continue
        candidates = [item, similar[0], far[0]]
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        gold_order = [shuffled.index(item), shuffled.index(similar[0]), shuffled.index(far[0])]
        rows.append(
            {
                **_task_row_base("acoustic_plausibility_ranking", [candidate.source_item_id for candidate in shuffled], item.rich_context_text),
                "id": f"t4-{item.source_item_id}",
                "candidate_audio_paths": [str(candidate.oracle_audio_path) for candidate in shuffled],
                "candidate_labels": [candidate.display_name for candidate in shuffled],
                "gold_ranking": gold_order,
            }
        )
    return rows


def _build_task_5(items: list[GroundingSourceItem]) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        periodic = bool(item.periodicity.get("is_periodic"))
        rows.append(
            {
                **_task_row_base("periodicity_aware_grounding", [item.source_item_id], item.rich_context_text),
                "id": f"t5-{item.source_item_id}",
                "query_audio_path": str(item.oracle_audio_path),
                "text_options": [
                    "A repetitive, rhythmic acoustic event.",
                    "A single abrupt impact-like event.",
                ],
                "gold_index": 0 if periodic else 1,
                "periodicity": item.periodicity,
            }
        )
    return rows


def _build_task_6(items: list[GroundingSourceItem]) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        sharp = bool(item.transient.get("is_sharp"))
        rows.append(
            {
                **_task_row_base(
                    "onset_sharpness_grounding",
                    [item.source_item_id],
                    "Focus on the onset profile of this clip rather than its semantic category.",
                ),
                "id": f"t6-{item.source_item_id}",
                "query_audio_path": str(item.oracle_audio_path),
                "text_options": [
                    "A sudden sharp-onset sound.",
                    "A diffuse or smeared-onset sound.",
                ],
                "gold_index": 0 if sharp else 1,
                "transient": item.transient,
                "balance_bucket": item.group_key,
            }
        )
    return rows


def _build_task_7(items: list[GroundingSourceItem], rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        negatives = _choose_similar(item, items, rng, k=min(3, max(0, len(items) - 1)))
        options = [item.rich_context_text, *[candidate.rich_context_text for candidate in negatives]]
        if len(options) < 2:
            continue
        indexed = list(enumerate(options))
        rng.shuffle(indexed)
        gold_index = next(index for index, (original_index, _) in enumerate(indexed) if original_index == 0)
        rows.append(
            {
                **_task_row_base("audio_text_hard_negative_matching", [item.source_item_id, *[candidate.source_item_id for candidate in negatives]], item.short_context_text),
                "id": f"t7-{item.source_item_id}",
                "query_audio_path": str(item.oracle_audio_path),
                "text_options": [text for _, text in indexed],
                "gold_index": gold_index,
            }
        )
    return rows


def _build_task_8(items: list[GroundingSourceItem], rng: random.Random, audio_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for index, first in enumerate(items):
        second_candidates = [item for item in items if item.source_item_id != first.source_item_id and item.group_key != first.group_key]
        if not second_candidates:
            continue
        second = second_candidates[index % len(second_candidates)]
        ab_path = audio_dir / "task8_causal_order" / f"{first.source_item_id}__then__{second.source_item_id}.wav"
        ba_path = audio_dir / "task8_causal_order" / f"{second.source_item_id}__then__{first.source_item_id}.wav"
        _concat_audio(ab_path, first.oracle_audio_path, second.oracle_audio_path)
        _concat_audio(ba_path, second.oracle_audio_path, first.oracle_audio_path)
        candidates = [ab_path, ba_path]
        labels = [f"{first.display_name}_then_{second.display_name}", f"{second.display_name}_then_{first.display_name}"]
        order = [0, 1]
        rng.shuffle(order)
        rows.append(
            {
                **_task_row_base(
                    "causal_event_order",
                    [first.source_item_id, second.source_item_id],
                    f"First {first.event_phrase}, then {second.event_phrase}.",
                ),
                "id": f"t8-{first.source_item_id}-{second.source_item_id}",
                "candidate_audio_paths": [str(candidates[idx]) for idx in order],
                "candidate_labels": [labels[idx] for idx in order],
                "gold_index": order.index(0),
                "balance_bucket": f"{first.group_key}__{second.group_key}",
            }
        )
    return rows


def build_and_write_coast_grounding_tasks(
    *,
    strict_suite_dir: Path,
    output_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    items = _load_source_pool(strict_suite_dir)
    rng = random.Random(seed)
    audio_dir = output_dir / "audio"
    source_pool_rows = [
        {
            "source_item_id": item.source_item_id,
            "display_name": item.display_name,
            "oracle_audio_path": str(item.oracle_audio_path),
            "short_context_text": item.short_context_text,
            "rich_context_text": item.rich_context_text,
            "periodicity": item.periodicity,
            "transient": item.transient,
            "task_id": item.task_id,
        }
        for item in items
    ]

    task_rows = {
        "task_1_target_sound_grounding": _build_task_1(items, rng),
        "task_2_fine_grained_acoustic_contrast": _build_task_2(items, rng),
        "task_3_foreground_event_focus": _build_task_3(items, rng, audio_dir),
        "task_4_acoustic_plausibility_ranking": _build_task_4(items, rng),
        "task_5_periodicity_aware_grounding": _build_task_5(items),
        "task_6_onset_sharpness_grounding": _build_task_6(items),
        "task_7_audio_text_hard_negative_matching": _build_task_7(items, rng),
        "task_8_causal_event_order": _build_task_8(items, rng, audio_dir),
    }

    counts: dict[str, int] = {}
    for task_name, rows in task_rows.items():
        counts[task_name] = len(rows)
        write_jsonl(output_dir / task_name / "all.jsonl", rows)
        task_metadata = {
            **TASK_SPECS[task_name],
            "output_path": str(output_dir / task_name / "all.jsonl"),
            "count": len(rows),
            "fields": sorted(rows[0].keys()) if rows else [],
        }
        with (output_dir / task_name / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(task_metadata, handle, indent=2, ensure_ascii=False)

    write_jsonl(output_dir / "source_pool.jsonl", source_pool_rows)

    metadata = {
        "suite": "COAST-Grounding",
        "seed": seed,
        "strict_suite_dir": str(strict_suite_dir),
        "source_items": len(items),
        "current_release_evaluation": {
            "default_primary_metric": "accuracy",
            "note": "Current packaged evaluator emphasizes forced-choice or top-1 correctness together with retrieval and ranking summaries where the task format supports them.",
            "cross_tier_comparison": "Do not compare raw task accuracy across tiers without normalization for task format and chance level.",
        },
        "counts": counts,
        "task_index": {
            task_name: {
                **TASK_SPECS[task_name],
                "path": str(output_dir / task_name / "all.jsonl"),
                "metadata_path": str(output_dir / task_name / "metadata.json"),
                "count": counts[task_name],
            }
            for task_name in task_rows
        },
        "source_pool_path": str(output_dir / "source_pool.jsonl"),
        "tasks": [
            "target_sound_grounding",
            "fine_grained_acoustic_contrast",
            "foreground_event_focus",
            "acoustic_plausibility_ranking",
            "periodicity_aware_grounding",
            "onset_sharpness_grounding",
            "audio_text_hard_negative_matching",
            "causal_event_order",
        ],
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    return metadata
