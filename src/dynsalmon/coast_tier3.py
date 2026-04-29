from __future__ import annotations

import csv
import json
import os
import random
import subprocess
import uuid
import wave
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from dynsalmon.coast import _ffmpeg_binary, write_jsonl


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg"}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _tokenize(text: str) -> list[str]:
    chars = []
    for char in text.lower():
        chars.append(char if char.isalnum() else " ")
    return [token for token in "".join(chars).split() if token]


def _find_media(media_root: Path, stem: str) -> Path | None:
    candidates = []
    for suffix in sorted(VIDEO_SUFFIXES | AUDIO_SUFFIXES):
        candidates.append(media_root / f"{stem}{suffix}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    patterns = [f"{stem}*", f"*{stem}*"]
    for pattern in patterns:
        for candidate in sorted(media_root.rglob(pattern)):
            if candidate.is_file() and candidate.suffix.lower() in VIDEO_SUFFIXES | AUDIO_SUFFIXES:
                return candidate
    return None


def _run_ffmpeg(args: list[str]) -> None:
    subprocess.run([_ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def _extract_audio(input_path: Path, output_path: Path, *, sample_rate: int = 16000) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )


def _concat_audio(parts: list[Path], output_path: Path, *, sample_rate: int = 16000) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    list_path = output_path.with_name(f"{output_path.stem}.{uuid.uuid4().hex}.concat.txt")
    list_path.write_text("".join(f"file '{part.as_posix()}'\n" for part in parts), encoding="utf-8")
    try:
        _run_ffmpeg(
            [
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ]
        )
    finally:
        list_path.unlink(missing_ok=True)


def _probe_wav_duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


@dataclass(slots=True)
class Tier3Item:
    source_item_id: str
    source_dataset: str
    split: str
    media_path: str
    audio_path: str
    context_short: str
    context_rich: str
    primary_event: str
    secondary_event: str
    event_sequence_text: str
    speaker_mode: str
    scenario_group: str
    duration_sec: float
    metadata: dict[str, Any]


def _item_to_row(item: Tier3Item) -> dict[str, Any]:
    payload = asdict(item)
    return payload


def _build_context(dataset_name: str, primary_event: str, secondary_event: str, speaker_mode: str) -> tuple[str, str]:
    short = f"A long-form audio clip involving {primary_event.replace('_', ' ')}."
    bits = [
        f"A long-form clip from {dataset_name}",
        f"where {primary_event.replace('_', ' ')} is central",
    ]
    if secondary_event:
        bits.append(f"and may co-occur with {secondary_event.replace('_', ' ')}")
    if speaker_mode != "unknown":
        bits.append(f"with a {speaker_mode.replace('_', ' ')} interaction pattern")
    rich = ". ".join(bits) + "."
    return short, rich


def _safe_event(*values: str) -> str:
    for value in values:
        if value and value.strip():
            return _normalize_text(value).replace(" ", "_").lower()
    return "unknown_event"


def _speaker_mode_from_text(text: str) -> str:
    tokens = set(_tokenize(text))
    if {"conversation", "dialogue", "talking", "speakers"} & tokens:
        return "multi_speaker"
    if {"narration", "monologue", "speaking", "speech"} & tokens:
        return "single_speaker"
    return "unknown"


def load_lfav_items(annotation_root: Path, media_root: Path) -> list[Tier3Item]:
    csv_candidates = sorted(annotation_root.glob("*.csv"))
    rows: list[Tier3Item] = []
    for csv_path in csv_candidates:
        split = csv_path.stem.split("_", 1)[0]
        for record in _read_csv(csv_path):
            video_id = record.get("video_id") or record.get("vid") or record.get("youtube_id") or record.get("id") or ""
            if not video_id:
                continue
            media_path = _find_media(media_root, video_id)
            if media_path is None:
                continue
            primary_event = _safe_event(
                record.get("audio_label", ""),
                record.get("visual_label", ""),
                record.get("label", ""),
            )
            secondary_event = _safe_event(record.get("visual_label", ""), record.get("audio_label", ""))
            sequence_text = _normalize_text(
                " then ".join(part for part in [record.get("audio_label", ""), record.get("visual_label", "")] if part)
            )
            short, rich = _build_context("LFAV", primary_event, secondary_event, _speaker_mode_from_text(sequence_text))
            rows.append(
                Tier3Item(
                    source_item_id=f"lfav-{split}-{video_id}",
                    source_dataset="LFAV",
                    split=split,
                    media_path=str(media_path),
                    audio_path="",
                    context_short=short,
                    context_rich=rich,
                    primary_event=primary_event,
                    secondary_event=secondary_event,
                    event_sequence_text=sequence_text or primary_event.replace("_", " "),
                    speaker_mode=_speaker_mode_from_text(sequence_text),
                    scenario_group="long_form",
                    duration_sec=0.0,
                    metadata={"annotation_file": str(csv_path), "record": record},
                )
            )
    return rows


def load_epic_items(annotation_root: Path, media_root: Path) -> list[Tier3Item]:
    rows: list[Tier3Item] = []
    media_index: dict[str, Path] = {}
    for candidate in media_root.rglob("*"):
        if candidate.is_file() and candidate.suffix.lower() in VIDEO_SUFFIXES | AUDIO_SUFFIXES:
            media_index[candidate.stem] = candidate
    csv_candidates = sorted(annotation_root.rglob("*.csv"))
    for csv_path in csv_candidates:
        name = csv_path.stem.lower()
        if not any(token in name for token in ["sound", "audio", "epic"]):
            continue
        for record in _read_csv(csv_path):
            narration_id = record.get("narration_id") or record.get("uid") or record.get("clip_id") or ""
            if not narration_id:
                continue
            media_path = media_index.get(str(narration_id))
            if media_path is None:
                continue
            primary_event = _safe_event(
                record.get("class", ""),
                record.get("narration", ""),
                record.get("verb", ""),
            )
            secondary_event = _safe_event(record.get("noun", ""), record.get("secondary_class", ""))
            sequence_text = _normalize_text(
                " ".join(
                    part for part in [record.get("narration", ""), record.get("verb", ""), record.get("noun", "")] if part
                )
            )
            dataset_name = "HD-EPIC" if "hd" in name else "EPIC"
            short, rich = _build_context(dataset_name, primary_event, secondary_event, _speaker_mode_from_text(sequence_text))
            rows.append(
                Tier3Item(
                    source_item_id=f"{dataset_name.lower()}-{narration_id}",
                    source_dataset=dataset_name,
                    split="validation" if "validation" in name else "test" if "test" in name else "train",
                    media_path=str(media_path),
                    audio_path="",
                    context_short=short,
                    context_rich=rich,
                    primary_event=primary_event,
                    secondary_event=secondary_event,
                    event_sequence_text=sequence_text or primary_event.replace("_", " "),
                    speaker_mode=_speaker_mode_from_text(sequence_text),
                    scenario_group="ego_kitchen",
                    duration_sec=0.0,
                    metadata={"annotation_file": str(csv_path), "record": record},
                )
            )
    return rows


def load_ego4d_items(annotation_root: Path, media_root: Path) -> list[Tier3Item]:
    rows: list[Tier3Item] = []
    jsonl_candidates = sorted(annotation_root.rglob("*.jsonl"))
    json_candidates = sorted(annotation_root.rglob("*.json"))
    for path in [*jsonl_candidates, *json_candidates]:
        name = path.stem.lower()
        if not any(token in name for token in ["ego4d", "social", "diar", "av", "forecast"]):
            continue
        records = _read_jsonl(path) if path.suffix == ".jsonl" else _read_json(path)
        if isinstance(records, dict):
            nested: list[dict[str, Any]] = []
            videos = records.get("videos")
            if isinstance(videos, list):
                for video in videos:
                    if not isinstance(video, dict):
                        continue
                    video_uid = str(video.get("video_uid") or video.get("clip_uid") or "")
                    split = str(video.get("split", records.get("split", "train")))
                    clips = video.get("clips")
                    if isinstance(clips, list):
                        for clip in clips:
                            if not isinstance(clip, dict):
                                continue
                            clip_record = dict(clip)
                            clip_record.setdefault("video_uid", video_uid)
                            clip_record.setdefault("split", split)
                            clip_record.setdefault("scenario", path.stem)
                            nested.append(clip_record)
                    else:
                        video_record = dict(video)
                        video_record.setdefault("split", split)
                        video_record.setdefault("scenario", path.stem)
                        nested.append(video_record)
            if not nested:
                for value in records.values():
                    if isinstance(value, list):
                        nested.extend(item for item in value if isinstance(item, dict))
            records = nested
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            clip_id = str(
                record.get("clip_uid")
                or record.get("source_clip_uid")
                or record.get("video_uid")
                or record.get("uid")
                or record.get("id")
                or ""
            )
            if not clip_id:
                continue
            media_path = _find_media(media_root, clip_id)
            if media_path is None:
                continue
            primary_event = _safe_event(
                record.get("text", ""),
                record.get("event", ""),
                record.get("narration_text", ""),
                record.get("label", ""),
            )
            secondary_event = _safe_event(record.get("secondary_event", ""), record.get("scenario", ""))
            sequence_text = _normalize_text(
                " ".join(
                    part
                    for part in [
                        str(record.get("text", "")),
                        str(record.get("event", "")),
                        str(record.get("scenario", "")),
                    ]
                    if part
                )
            )
            rows.append(
                Tier3Item(
                    source_item_id=f"ego4d-{clip_id}",
                    source_dataset="Ego4D",
                    split=str(record.get("split", "train")),
                    media_path=str(media_path),
                    audio_path="",
                    context_short=f"A long-form egocentric clip involving {primary_event.replace('_', ' ')}.",
                    context_rich=(
                        "A long-form Ego4D clip with conversation or activity context. "
                        f"Primary cue: {primary_event.replace('_', ' ')}. "
                        f"Secondary cue: {secondary_event.replace('_', ' ')}."
                    ),
                    primary_event=primary_event,
                    secondary_event=secondary_event,
                    event_sequence_text=sequence_text or primary_event.replace("_", " "),
                    speaker_mode=_speaker_mode_from_text(sequence_text),
                    scenario_group="ego_daily_life",
                    duration_sec=0.0,
                    metadata={"annotation_file": str(path), "record": record},
                )
            )
    return rows


def extract_tier3_audio(items: list[Tier3Item], output_dir: Path, *, sample_rate: int = 16000) -> tuple[list[Tier3Item], list[dict[str, Any]]]:
    extracted: list[Tier3Item] = []
    failures: list[dict[str, Any]] = []
    for item in items:
        media_path = Path(item.media_path)
        audio_path = output_dir / f"{item.source_item_id}.wav"
        try:
            if not audio_path.exists():
                if media_path.suffix.lower() in AUDIO_SUFFIXES:
                    _extract_audio(media_path, audio_path, sample_rate=sample_rate)
                else:
                    _extract_audio(media_path, audio_path, sample_rate=sample_rate)
            duration_sec = _probe_wav_duration_sec(audio_path)
            extracted.append(
                Tier3Item(
                    **{
                        **_item_to_row(item),
                        "audio_path": str(audio_path),
                        "duration_sec": duration_sec,
                    }
                )
            )
        except Exception as exc:
            if audio_path.exists():
                audio_path.unlink()
            failures.append({"source_item_id": item.source_item_id, "media_path": item.media_path, "error": str(exc)})
    return extracted, failures


def write_tier3_manifest(items: list[Tier3Item], output_dir: Path, failures: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [_item_to_row(item) for item in items]
    write_jsonl(output_dir / "manifest.jsonl", rows)
    metadata = {
        "suite": "COAST Tier3 LongForm",
        "count": len(rows),
        "source_datasets": {
            dataset: sum(1 for row in rows if row["source_dataset"] == dataset)
            for dataset in sorted({row["source_dataset"] for row in rows})
        },
        "scenario_groups": {
            group: sum(1 for row in rows if row["scenario_group"] == group)
            for group in sorted({row["scenario_group"] for row in rows})
        },
        "speaker_modes": {
            mode: sum(1 for row in rows if row["speaker_mode"] == mode)
            for mode in sorted({row["speaker_mode"] for row in rows})
        },
        "audio_extraction_failures": len(failures),
        "audio_extraction_failure_log": str(output_dir / "audio_extraction_failures.json"),
    }
    _write_json(output_dir / "metadata.json", metadata)
    _write_json(output_dir / "audio_extraction_failures.json", failures)
    return metadata


def load_tier3_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    return _read_jsonl(manifest_path)


TASK_SPECS: dict[str, dict[str, str]] = {
    "task_1_event_window_grounding": {
        "task_name": "event_window_grounding",
        "conditioning": "audio_text",
        "current_primary_metric": "accuracy",
        "recommended_metrics": "accuracy, retrieval_hit_rate",
        "comparison_scope": "within_tier",
        "metric_note": "Current release treats this as long-form candidate selection, not dense temporal IoU grounding.",
    },
    "task_2_coarse_event_order": {
        "task_name": "coarse_event_order",
        "conditioning": "audio_text",
        "current_primary_metric": "accuracy",
        "recommended_metrics": "accuracy, order_consistency",
        "comparison_scope": "within_tier",
        "metric_note": "Sequence order is evaluated with forced-choice correctness over rendered pair orderings.",
    },
    "task_3_narrative_continuation_plausibility": {
        "task_name": "narrative_continuation_plausibility",
        "conditioning": "audio_audio",
        "current_primary_metric": "accuracy",
        "recommended_metrics": "accuracy, choice_consistency",
        "comparison_scope": "within_tier",
        "metric_note": "Interpret as audio-continuation plausibility over long-form contexts rather than text-label matching.",
    },
    "task_4_long_context_event_retrieval": {
        "task_name": "long_context_event_retrieval",
        "conditioning": "audio_text",
        "current_primary_metric": "accuracy",
        "recommended_metrics": "recall@k, mAP, accuracy",
        "comparison_scope": "within_tier",
        "metric_note": "Current release uses top-1 correctness; retrieval metrics are recommended for fuller evaluation.",
    },
    "task_5_speaker_event_overlap": {
        "task_name": "speaker_event_overlap",
        "conditioning": "audio_text",
        "current_primary_metric": "accuracy",
        "recommended_metrics": "accuracy, macro_f1",
        "comparison_scope": "within_tier",
        "metric_note": "This task may be empty for sources without reliable speaker-overlap supervision.",
    },
    "task_6_sequential_causal_consistency": {
        "task_name": "sequential_causal_consistency",
        "conditioning": "audio_text",
        "current_primary_metric": "accuracy",
        "recommended_metrics": "accuracy, choice_consistency",
        "comparison_scope": "within_tier",
        "metric_note": "Tier 3 focuses on coarse causal consistency, not frame-exact synchronization.",
    },
}


def _shared_score(a: dict[str, Any], b: dict[str, Any]) -> float:
    score = 0.0
    if a["scenario_group"] == b["scenario_group"]:
        score += 1.0
    if a["speaker_mode"] == b["speaker_mode"]:
        score += 1.0
    score += len(set(_tokenize(a["primary_event"])) & set(_tokenize(b["primary_event"]))) * 0.5
    return score


def _sample_balanced_rows(rows: list[dict[str, Any]], *, max_source_items: int, rng: random.Random) -> list[dict[str, Any]]:
    if len(rows) <= max_source_items:
        return list(rows)
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_event[row["primary_event"]].append(row)
    for bucket in by_event.values():
        rng.shuffle(bucket)
    selected: list[dict[str, Any]] = []
    bucket_names = sorted(by_event)
    while len(selected) < max_source_items and bucket_names:
        next_bucket_names = []
        for bucket_name in bucket_names:
            bucket = by_event[bucket_name]
            if bucket:
                selected.append(bucket.pop())
                if len(selected) >= max_source_items:
                    break
            if bucket:
                next_bucket_names.append(bucket_name)
        bucket_names = next_bucket_names
    return selected


def _build_indices(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    by_primary_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_speaker_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_scenario_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_primary_event[row["primary_event"]].append(row)
        by_speaker_mode[row["speaker_mode"]].append(row)
        by_scenario_group[row["scenario_group"]].append(row)
    return {
        "by_primary_event": dict(by_primary_event),
        "by_speaker_mode": dict(by_speaker_mode),
        "by_scenario_group": dict(by_scenario_group),
    }


def _sample_from_pool(
    pool: list[dict[str, Any]],
    *,
    target_id: str,
    rng: random.Random,
    k: int,
) -> list[dict[str, Any]]:
    candidates = [row for row in pool if row["source_item_id"] != target_id]
    if not candidates:
        return []
    if len(candidates) <= k:
        return list(candidates)
    return rng.sample(candidates, k)


def _choose_similar(
    target: dict[str, Any],
    rows: list[dict[str, Any]],
    indices: dict[str, dict[str, list[dict[str, Any]]]],
    rng: random.Random,
    k: int = 1,
) -> list[dict[str, Any]]:
    primary_pool = indices["by_primary_event"].get(target["primary_event"], [])
    selected = _sample_from_pool(primary_pool, target_id=target["source_item_id"], rng=rng, k=k)
    if len(selected) >= k:
        return selected
    scenario_pool = indices["by_scenario_group"].get(target["scenario_group"], [])
    extras = _sample_from_pool(
        [row for row in scenario_pool if row["source_item_id"] not in {item["source_item_id"] for item in selected}],
        target_id=target["source_item_id"],
        rng=rng,
        k=k - len(selected),
    )
    return selected + extras


def _choose_dissimilar(
    target: dict[str, Any],
    rows: list[dict[str, Any]],
    indices: dict[str, dict[str, list[dict[str, Any]]]],
    rng: random.Random,
    k: int = 1,
) -> list[dict[str, Any]]:
    pool = [
        row
        for row in rows
        if row["source_item_id"] != target["source_item_id"] and row["primary_event"] != target["primary_event"]
    ]
    if not pool:
        pool = [row for row in rows if row["source_item_id"] != target["source_item_id"]]
    if not pool:
        return []
    if len(pool) <= k:
        return pool
    return rng.sample(pool, k)


def _render_order_pair(job: tuple[Path, Path, Path]) -> None:
    first_path, second_path, output_path = job
    if output_path.exists():
        return
    _concat_audio([first_path, second_path], output_path)


def build_tier3_tasks(
    *,
    manifest_path: Path,
    output_dir: Path,
    seed: int = 42,
    max_source_items: int = 4096,
    max_order_audio_tasks: int = 1024,
    render_workers: int | None = None,
) -> dict[str, Any]:
    all_rows = load_tier3_manifest(manifest_path)
    rng = random.Random(seed)
    rows = _sample_balanced_rows(all_rows, max_source_items=max_source_items, rng=rng)
    indices = _build_indices(rows)
    audio_dir = output_dir / "audio"
    task_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in TASK_SPECS}
    order_render_jobs: list[tuple[Path, Path, Path]] = []
    order_task_budget = max_order_audio_tasks

    for row in rows:
        dissimilar = _choose_dissimilar(row, rows, indices, rng, k=min(3, max(0, len(rows) - 1)))
        similar = _choose_similar(row, rows, indices, rng, k=min(2, max(0, len(rows) - 1)))
        candidates = [row, *dissimilar]
        rng.shuffle(candidates)
        task_rows["task_1_event_window_grounding"].append(
            {
                "id": f"t3-1-{row['source_item_id']}",
                "task_name": "event_window_grounding",
                "prompt": row["context_rich"],
                "candidate_audio_paths": [item["audio_path"] for item in candidates],
                "candidate_item_ids": [item["source_item_id"] for item in candidates],
                "gold_index": candidates.index(row),
            }
        )

        if similar and order_task_budget > 0:
            first = row
            second = similar[0]
            ab_path = audio_dir / "task2_coarse_order" / f"{first['source_item_id']}__then__{second['source_item_id']}.wav"
            ba_path = audio_dir / "task2_coarse_order" / f"{second['source_item_id']}__then__{first['source_item_id']}.wav"
            order_render_jobs.append((Path(first["audio_path"]), Path(second["audio_path"]), ab_path))
            order_render_jobs.append((Path(second["audio_path"]), Path(first["audio_path"]), ba_path))
            task_rows["task_2_coarse_event_order"].append(
                {
                    "id": f"t3-2-{row['source_item_id']}",
                    "task_name": "coarse_event_order",
                    "prompt": f"Which sequence better matches this long-form context: {row['event_sequence_text']}?",
                    "candidate_audio_paths": [str(ab_path), str(ba_path)],
                    "gold_index": 0,
                }
            )
            order_task_budget -= 1

        if similar:
            plausible = similar[0]
            continuation_candidates = [plausible, *dissimilar[:2]]
            continuation_candidates = continuation_candidates[: min(3, len(continuation_candidates))]
            rng.shuffle(continuation_candidates)
            gold_index = continuation_candidates.index(plausible)
            task_rows["task_3_narrative_continuation_plausibility"].append(
                {
                    "id": f"t3-3-{row['source_item_id']}",
                    "task_name": "narrative_continuation_plausibility",
                    "query_audio_path": row["audio_path"],
                    # Keep the prompt generic so the task cannot be solved by lexical overlap.
                    "prompt": "Choose the continuation clip that most plausibly comes next in the same activity sequence.",
                    "candidate_audio_paths": [item["audio_path"] for item in continuation_candidates],
                    "candidate_item_ids": [item["source_item_id"] for item in continuation_candidates],
                    "gold_index": gold_index,
                }
            )

        task_rows["task_4_long_context_event_retrieval"].append(
            {
                "id": f"t3-4-{row['source_item_id']}",
                "task_name": "long_context_event_retrieval",
                "query_text": row["primary_event"].replace("_", " "),
                "candidate_audio_paths": [item["audio_path"] for item in candidates],
                "candidate_item_ids": [item["source_item_id"] for item in candidates],
                "gold_index": candidates.index(row),
            }
        )

        if row["speaker_mode"] != "unknown":
            speaker_options = [row["speaker_mode"], "multi_speaker" if row["speaker_mode"] == "single_speaker" else "single_speaker"]
            task_rows["task_5_speaker_event_overlap"].append(
                {
                    "id": f"t3-5-{row['source_item_id']}",
                    "task_name": "speaker_event_overlap",
                    "query_audio_path": row["audio_path"],
                    "prompt": row["context_short"],
                    "text_options": speaker_options,
                    "gold_index": 0,
                }
            )

        if dissimilar:
            plausible = f"{row['primary_event'].replace('_', ' ')} followed by {row['secondary_event'].replace('_', ' ')}"
            implausible = f"{dissimilar[0]['primary_event'].replace('_', ' ')} followed by {row['primary_event'].replace('_', ' ')}"
            options = [plausible, implausible]
            indexed = list(enumerate(options))
            rng.shuffle(indexed)
            gold_index = next(index for index, (original_index, _) in enumerate(indexed) if original_index == 0)
            task_rows["task_6_sequential_causal_consistency"].append(
                {
                    "id": f"t3-6-{row['source_item_id']}",
                    "task_name": "sequential_causal_consistency",
                    "query_audio_path": row["audio_path"],
                    "prompt": row["context_rich"],
                    "text_options": [text for _, text in indexed],
                    "gold_index": gold_index,
                }
            )

    if order_render_jobs:
        workers = render_workers or min(16, max(1, os.cpu_count() or 1))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(_render_order_pair, order_render_jobs))

    counts = {}
    for task_dir_name, task_spec in TASK_SPECS.items():
        rows_for_task = task_rows[task_dir_name]
        task_dir = output_dir / task_dir_name
        write_jsonl(task_dir / "all.jsonl", rows_for_task)
        _write_json(
            task_dir / "metadata.json",
            {
                "task_name": task_spec["task_name"],
                "count": len(rows_for_task),
                "conditioning": task_spec["conditioning"],
            },
        )
        counts[task_dir_name] = len(rows_for_task)

    metadata = {
        "suite": "COAST Tier3 LongForm Tasks",
        "seed": seed,
        "manifest_path": str(manifest_path),
        "source_items_total": len(all_rows),
        "source_items": len(rows),
        "max_source_items": max_source_items,
        "max_order_audio_tasks": max_order_audio_tasks,
        "render_workers": render_workers or min(16, max(1, os.cpu_count() or 1)),
        "current_release_evaluation": {
            "default_primary_metric": "accuracy",
            "note": "Current Tier 3 release emphasizes choice accuracy for long-form retrieval, continuation, and ordering tasks. Dense temporal overlap metrics are not yet part of the packaged EPIC release.",
            "cross_tier_comparison": "Do not compare Tier 3 raw accuracy directly to Tier 1/2 without task-family normalization.",
        },
        "counts": counts,
        "tasks": [spec["task_name"] for spec in TASK_SPECS.values()],
    }
    _write_json(output_dir / "metadata.json", metadata)
    return metadata
