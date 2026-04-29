import json
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dynsalmon.coast import _ffmpeg_binary


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
YOUTUBE_ID_RE = re.compile(r"^(.+?)_(?:\d+x\d+)_full_video_")


def _infer_youtube_id(merge: dict, clip_name: str) -> str:
    value = str(merge.get("video_name") or "").strip()
    if value:
        return value
    match = YOUTUBE_ID_RE.match(clip_name)
    if match:
        return match.group(1)
    return clip_name.split("_")[0]


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except UnicodeDecodeError:
        # Some mirrored annotation files arrive with a few bad bytes on cluster sync;
        # keep the rebuild moving and prefer a lossy decode over dropping the whole task suite.
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_tokens(text: str) -> list[str]:
    cleaned = []
    for char in text.lower():
        cleaned.append(char if char.isalnum() else " ")
    return [token for token in "".join(cleaned).split() if token]


def _extract_string_leaves(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        strings: list[str] = []
        for nested in value.values():
            strings.extend(_extract_string_leaves(nested))
        return strings
    if isinstance(value, list):
        strings: list[str] = []
        for nested in value:
            strings.extend(_extract_string_leaves(nested))
        return strings
    return []


def _find_video_path(clips_root: Path, clip_name: str) -> Path | None:
    direct_matches = [clips_root / f"{clip_name}{suffix}" for suffix in VIDEO_SUFFIXES]
    for candidate in direct_matches:
        if candidate.exists():
            return candidate
    suffixed_matches = []
    for suffix in VIDEO_SUFFIXES:
        suffixed_matches.extend(sorted(clips_root.rglob(f"{clip_name}_*{suffix}")))
    if suffixed_matches:
        return suffixed_matches[0]
    for candidate in clips_root.rglob(f"{clip_name}.*"):
        if candidate.is_file() and candidate.suffix.lower() in VIDEO_SUFFIXES:
            return candidate
    return None


def _parse_list_entries(payload: Any) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if isinstance(payload, list):
        for entry in payload:
            if isinstance(entry, str):
                rows.append({"youtube_id": entry, "clip_name": entry})
                continue
            if not isinstance(entry, dict):
                continue
            youtube_id = (
                entry.get("youtube_id")
                or entry.get("video_id")
                or entry.get("yt_id")
                or entry.get("video_name")
                or ""
            )
            clip_name = (
                entry.get("clip_name")
                or entry.get("clip")
                or entry.get("name")
                or entry.get("video_name")
                or ""
            )
            if youtube_id and clip_name:
                rows.append({"youtube_id": str(youtube_id), "clip_name": str(clip_name)})
    return rows


def _find_auxiliary_annotation(directory: Path, clip_name: str, youtube_id: str) -> Path | None:
    exact = directory / f"{clip_name}.json"
    if exact.exists():
        return exact
    prefix_matches = sorted(directory.glob(f"{clip_name}*.json"))
    if prefix_matches:
        return prefix_matches[0]
    youtube_matches = sorted(directory.glob(f"{youtube_id}*.json"))
    if youtube_matches:
        return youtube_matches[0]
    return None


def _resolve_split_membership(annotation_root: Path) -> dict[str, set[str]]:
    memberships: dict[str, set[str]] = {"all": set(), "sft": set(), "benchmark": set()}
    all_list = annotation_root / "all_data_list.json"
    if all_list.exists():
        for row in _parse_list_entries(_read_json(all_list)):
            memberships["all"].add(row["clip_name"])
    sft_list = annotation_root / "SFT_set.json"
    if sft_list.exists():
        payload = _read_json(sft_list)
        if isinstance(payload, list):
            memberships["sft"].update(str(item) for item in payload)
    testset = annotation_root / "testset.json"
    if testset.exists():
        payload = _read_json(testset)
        if isinstance(payload, list):
            memberships["benchmark"].update(str(item) for item in payload)
    return memberships


def _parse_clip_index_from_name(name: str) -> int:
    numbers = re.findall(r"\d+", name)
    if not numbers:
        return 0
    return int(numbers[-1])


def _infer_delivery_label(text: str) -> str:
    lower = text.lower()
    if any(token in lower for token in ["laugh", "chuckle", "giggle"]):
        return "laughter"
    if any(token in lower for token in ["shout", "yell", "scream", "cheer"]):
        return "high_energy"
    if any(token in lower for token in ["whisper", "murmur", "soft", "quietly"]):
        return "soft"
    return "neutral_speech"


def _speaker_count_label(count: int) -> str:
    return "multiple_speakers" if count > 1 else "single_speaker"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


@dataclass(slots=True)
class SpeakerVidTier2Item:
    clip_name: str
    youtube_id: str
    clip_path: Path
    split: str
    is_talking: bool
    clip_speaker_num: int
    speakers: list[str]
    asr_text: str
    caption_text: str
    clear_score: float
    dover_score: float
    conf_score: float
    start_seconds: float
    duration_seconds: float
    rich_context_text: str
    short_context_text: str
    delivery_label: str
    same_video_group: str
    clip_index: int

    @property
    def source_item_id(self) -> str:
        return self.clip_name

    @property
    def speaker_count_label(self) -> str:
        return _speaker_count_label(self.clip_speaker_num)


TIER2_TASK_SPECS: dict[str, dict[str, str]] = {
    "task_1_phonic_target_grounding": {
        "task_name": "phonic_target_grounding",
        "current_primary_metric": "accuracy",
        "recommended_metrics": "accuracy, recall@k",
        "comparison_scope": "within_tier",
        "metric_note": "Use as a phonic/audio-text grounding task rather than a general retrieval benchmark.",
    },
    "task_2_speech_activity_grounding": {
        "task_name": "speech_activity_grounding",
        "current_primary_metric": "accuracy",
        "recommended_metrics": "accuracy, precision, recall, f1",
        "comparison_scope": "within_tier",
        "metric_note": "Treat as categorical speech presence grounding, not frame-level SAD.",
    },
    "task_3_speaker_count_grounding": {
        "task_name": "speaker_count_grounding",
        "current_primary_metric": "accuracy",
        "recommended_metrics": "accuracy, macro_f1",
        "comparison_scope": "within_tier",
        "metric_note": "Exact-count style categorical evaluation is appropriate for the current release.",
    },
    "task_4_dyadic_matching": {
        "task_name": "dyadic_matching",
        "current_primary_metric": "accuracy",
        "recommended_metrics": "accuracy, retrieval_hit_rate",
        "comparison_scope": "within_tier",
        "metric_note": "Use as a matching task; coverage depends on available multi-speaker clips.",
    },
    "task_5_delivery_style_grounding": {
        "task_name": "delivery_style_grounding",
        "current_primary_metric": "accuracy",
        "recommended_metrics": "accuracy, macro_f1",
        "comparison_scope": "within_tier",
        "metric_note": "Delivery labels are intentionally coarse and should be interpreted conservatively.",
    },
    "task_6_turn_holding_continuation": {
        "task_name": "turn_holding_continuation",
        "current_primary_metric": "accuracy",
        "recommended_metrics": "accuracy, choice_consistency",
        "comparison_scope": "within_tier",
        "metric_note": "Continuation tasks probe contextual acoustic grounding and are expected to be harder than simple classification.",
    },
    "task_7_speaker_count_aware_continuation": {
        "task_name": "speaker_count_aware_continuation",
        "current_primary_metric": "accuracy",
        "recommended_metrics": "accuracy, choice_consistency",
        "comparison_scope": "within_tier",
        "metric_note": "Evaluate within Tier 2 only; do not compare raw scores against non-continuation tasks.",
    },
    "task_8_delivery_conditioned_continuation": {
        "task_name": "delivery_conditioned_continuation",
        "current_primary_metric": "accuracy",
        "recommended_metrics": "accuracy, choice_consistency",
        "comparison_scope": "within_tier",
        "metric_note": "Continuation difficulty depends strongly on style-label sparsity and should be reported with class counts.",
    },
}


def load_speakervid_tier2_items(
    *,
    annotation_root: Path,
    clips_root: Path,
    min_clear_score: float = 0.45,
    min_dover_score: float = 0.45,
    require_asr: bool = True,
    max_items: int | None = None,
    split_preference: str = "all",
) -> list[SpeakerVidTier2Item]:
    merge_dir = annotation_root / "merge_anno"
    if not merge_dir.exists():
        merge_dir = annotation_root / "merged_anno"
    asr_dir = annotation_root / "asr"
    if not asr_dir.exists():
        asr_dir = annotation_root / "raw_labels" / "asr"
    caption_dir = annotation_root / "anno"
    if not caption_dir.exists():
        caption_dir = annotation_root / "raw_labels" / "anno"
    memberships = _resolve_split_membership(annotation_root)
    items: list[SpeakerVidTier2Item] = []

    if not merge_dir.exists():
        raise FileNotFoundError(f"merge_anno not found under {annotation_root}")

    for merge_path in sorted(merge_dir.glob("*.json")):
        if merge_path.name.startswith("._"):
            continue
        clip_name = merge_path.stem
        merge = _read_json(merge_path)
        youtube_id = _infer_youtube_id(merge, clip_name)
        clip_path = _find_video_path(clips_root, clip_name)
        if clip_path is None:
            continue

        clear_score = _safe_float(merge.get("clear"), 0.0)
        dover_score = _safe_float(merge.get("dover"), 0.0)
        conf_score = _safe_float(merge.get("conf"), 0.0)
        if clear_score < min_clear_score or dover_score < min_dover_score:
            continue

        asr_path = _find_auxiliary_annotation(asr_dir, clip_name, youtube_id)
        asr_payload = _read_json(asr_path) if asr_path and asr_path.exists() else {}
        asr_strings = _extract_string_leaves(asr_payload)
        asr_text = max(asr_strings, key=len) if asr_strings else ""
        if require_asr and not asr_text.strip():
            continue

        caption_path = _find_auxiliary_annotation(caption_dir, clip_name, youtube_id)
        caption_payload = _read_json(caption_path) if caption_path and caption_path.exists() else {}
        caption_strings = _extract_string_leaves(caption_payload)
        caption_text = " ".join(part.strip() for part in caption_strings if part.strip())

        split = "unassigned"
        if clip_name in memberships["benchmark"]:
            split = "benchmark"
        elif clip_name in memberships["sft"]:
            split = "sft"
        elif clip_name in memberships["all"]:
            split = split_preference

        is_talking = bool(_safe_int(merge.get("is_talking"), 0))
        clip_speaker_num = max(1, _safe_int(merge.get("clip_speaker_num"), 1))
        speakers = [str(value) for value in merge.get("speaker", []) if str(value)]
        start_seconds = _safe_float(
            merge.get("org_start_seconds", merge.get("start_seconds", merge.get("start", 0.0))),
            0.0,
        )
        duration_seconds = _safe_float(merge.get("duration"), 0.0)
        if duration_seconds <= 0:
            duration_seconds = max(0.0, _safe_float(merge.get("end_seconds"), 0.0) - start_seconds)
        if duration_seconds <= 0:
            continue

        combined_text = " ".join(part for part in [caption_text, asr_text] if part)
        delivery_label = _infer_delivery_label(combined_text)
        short_context = (
            "A speaking portrait clip."
            if is_talking
            else "A portrait clip with little or no visible speech."
        )
        rich_bits = [
            "A portrait-style video clip",
            "with a speaking subject" if is_talking else "with no strong visible vocalization",
            f"and {clip_speaker_num} visible speaker(s)",
        ]
        if caption_text:
            rich_bits.append(f"Caption cues: {caption_text}")
        rich_context = ". ".join(rich_bits) + "."

        items.append(
            SpeakerVidTier2Item(
                clip_name=clip_name,
                youtube_id=youtube_id,
                clip_path=clip_path,
                split=split,
                is_talking=is_talking,
                clip_speaker_num=clip_speaker_num,
                speakers=speakers,
                asr_text=asr_text.strip(),
                caption_text=caption_text.strip(),
                clear_score=clear_score,
                dover_score=dover_score,
                conf_score=conf_score,
                start_seconds=start_seconds,
                duration_seconds=duration_seconds,
                rich_context_text=rich_context,
                short_context_text=short_context,
                delivery_label=delivery_label,
                same_video_group=youtube_id,
                clip_index=_parse_clip_index_from_name(clip_name),
            )
        )
        if max_items is not None and len(items) >= max_items:
            break

    return items


def _run_ffmpeg(args: list[str]) -> None:
    subprocess.run([_ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def extract_audio_bank(
    items: list[SpeakerVidTier2Item],
    output_dir: Path,
    sample_rate: int = 16000,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    audio_map: dict[str, str] = {}
    extraction_failures: list[dict[str, Any]] = []
    for item in items:
        output_path = output_dir / f"{item.clip_name}.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not output_path.exists():
                _run_ffmpeg(
                    [
                        "-i",
                        str(item.clip_path),
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
            audio_map[item.clip_name] = str(output_path)
        except subprocess.CalledProcessError as exc:
            if output_path.exists():
                output_path.unlink()
            extraction_failures.append(
                {
                    "clip_name": item.clip_name,
                    "clip_path": str(item.clip_path),
                    "error": str(exc),
                }
            )
    return audio_map, extraction_failures


def write_tier2_manifest(
    *,
    items: list[SpeakerVidTier2Item],
    audio_map: dict[str, str],
    output_dir: Path,
    audio_extraction_failures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = []
    for item in items:
        audio_path = audio_map.get(item.clip_name)
        if not audio_path:
            continue
        rows.append(
            {
                "source_item_id": item.source_item_id,
                "clip_name": item.clip_name,
                "youtube_id": item.youtube_id,
                "clip_path": str(item.clip_path),
                "audio_path": audio_path,
                "split": item.split,
                "is_talking": item.is_talking,
                "clip_speaker_num": item.clip_speaker_num,
                "speakers": item.speakers,
                "asr_text": item.asr_text,
                "caption_text": item.caption_text,
                "clear_score": item.clear_score,
                "dover_score": item.dover_score,
                "conf_score": item.conf_score,
                "start_seconds": item.start_seconds,
                "duration_seconds": item.duration_seconds,
                "short_context_text": item.short_context_text,
                "rich_context_text": item.rich_context_text,
                "delivery_label": item.delivery_label,
                "same_video_group": item.same_video_group,
                "clip_index": item.clip_index,
            }
        )
    _write_jsonl(output_dir / "manifest.jsonl", rows)
    metadata = {
        "suite": "COAST Tier2 SpeakerVid",
        "count": len(rows),
        "source_dataset": "SpeakerVid-5M",
        "splits": {
            split: sum(1 for row in rows if row["split"] == split)
            for split in sorted({row["split"] for row in rows})
        },
        "talking_count": sum(1 for row in rows if row["is_talking"]),
        "multi_speaker_count": sum(1 for row in rows if row["clip_speaker_num"] > 1),
        "delivery_counts": {
            label: sum(1 for row in rows if row["delivery_label"] == label)
            for label in sorted({row["delivery_label"] for row in rows})
        },
        "audio_extraction_failures": len(audio_extraction_failures or []),
        "audio_extraction_failure_log": str(output_dir / "audio_extraction_failures.json"),
    }
    _write_json(output_dir / "metadata.json", metadata)
    _write_json(output_dir / "audio_extraction_failures.json", audio_extraction_failures or [])
    return metadata


def _candidate_pool(
    target: dict[str, Any],
    items: list[dict[str, Any]],
    *,
    exclude_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    exclude_ids = exclude_ids or set()
    return [
        item
        for item in items
        if item["source_item_id"] != target["source_item_id"] and item["source_item_id"] not in exclude_ids
    ]


def _shared_score(a: dict[str, Any], b: dict[str, Any]) -> float:
    score = 0.0
    if a["clip_speaker_num"] == b["clip_speaker_num"]:
        score += 1.0
    if a["is_talking"] == b["is_talking"]:
        score += 1.0
    if a["delivery_label"] == b["delivery_label"]:
        score += 1.0
    score += len(set(_normalize_tokens(a["caption_text"])) & set(_normalize_tokens(b["caption_text"]))) * 0.05
    return score


def _duration_bucket(row: dict[str, Any]) -> str:
    duration = float(row.get("duration_seconds", 0.0) or 0.0)
    if duration < 2.5:
        return "short"
    if duration < 5.0:
        return "medium"
    return "long"


def _controlled_signature(row: dict[str, Any]) -> str:
    return "|".join(
        [
            row.get("speaker_count_label", _speaker_count_label(int(row.get("clip_speaker_num", 1)))),
            row.get("delivery_label", "unknown"),
            "talking" if row.get("is_talking") else "not_talking",
            _duration_bucket(row),
        ]
    )


def _choose_similar(target: dict[str, Any], items: list[dict[str, Any]], rng: random.Random, k: int = 1) -> list[dict[str, Any]]:
    ranked = sorted(_candidate_pool(target, items), key=lambda item: (-_shared_score(target, item), item["source_item_id"]))
    top = ranked[: max(k, min(len(ranked), 8))]
    rng.shuffle(top)
    return top[:k]


def _choose_dissimilar(target: dict[str, Any], items: list[dict[str, Any]], rng: random.Random, k: int = 1) -> list[dict[str, Any]]:
    ranked = sorted(_candidate_pool(target, items), key=lambda item: (_shared_score(target, item), item["source_item_id"]))
    top = ranked[: max(k, min(len(ranked), 8))]
    rng.shuffle(top)
    return top[:k]


def _choose_filtered(
    target: dict[str, Any],
    items: list[dict[str, Any]],
    rng: random.Random,
    *,
    predicate,
    k: int = 1,
    reverse: bool = True,
    exclude_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    pool = [item for item in _candidate_pool(target, items, exclude_ids=exclude_ids) if predicate(item)]
    ranked = sorted(pool, key=lambda item: (_shared_score(target, item), item["source_item_id"]), reverse=reverse)
    top = ranked[: max(k, min(len(ranked), 8))]
    rng.shuffle(top)
    return top[:k]


def build_speakervid_tier2_tasks(
    *,
    manifest_path: Path,
    output_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    rows = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    rng = random.Random(seed)
    task_rows: dict[str, list[dict[str, Any]]] = {
        "task_1_phonic_target_grounding": [],
        "task_2_speech_activity_grounding": [],
        "task_3_speaker_count_grounding": [],
        "task_4_dyadic_matching": [],
        "task_5_delivery_style_grounding": [],
        "task_6_turn_holding_continuation": [],
        "task_7_speaker_count_aware_continuation": [],
        "task_8_delivery_conditioned_continuation": [],
    }

    by_video: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_video.setdefault(row["same_video_group"], []).append(row)
    for video_rows in by_video.values():
        video_rows.sort(key=lambda row: row.get("clip_index", 0))

    for row in rows:
        distractors = _choose_dissimilar(row, rows, rng, k=min(3, max(0, len(rows) - 1)))
        candidates = [row, *distractors]
        rng.shuffle(candidates)
        task_rows["task_1_phonic_target_grounding"].append(
            {
                "id": f"sv-t1-{row['source_item_id']}",
                "task_name": "phonic_target_grounding",
                "prompt": row["rich_context_text"],
                "source_item_ids": [item["source_item_id"] for item in candidates],
                "candidate_audio_paths": [item["audio_path"] for item in candidates],
                "candidate_labels": [item["clip_name"] for item in candidates],
                "gold_index": candidates.index(row),
            }
        )

        task_rows["task_2_speech_activity_grounding"].append(
            {
                "id": f"sv-t2-{row['source_item_id']}",
                "task_name": "speech_activity_grounding",
                "prompt": row["rich_context_text"],
                "source_item_ids": [row["source_item_id"]],
                "query_audio_path": row["audio_path"],
                "text_options": [
                    "The visible subject is actively speaking or vocalizing.",
                    "The visible subject is not strongly vocalizing.",
                ],
                "gold_index": 0 if row["is_talking"] else 1,
            }
        )

        task_rows["task_3_speaker_count_grounding"].append(
            {
                "id": f"sv-t3-{row['source_item_id']}",
                "task_name": "speaker_count_grounding",
                "prompt": row["short_context_text"],
                "source_item_ids": [row["source_item_id"]],
                "query_audio_path": row["audio_path"],
                "text_options": [
                    "A single dominant speaker is present.",
                    "Multiple speakers or a dyadic exchange are present.",
                ],
                "gold_index": 1 if row["clip_speaker_num"] > 1 else 0,
                "balance_bucket": f"{row['delivery_label']}|{_duration_bucket(row)}",
            }
        )

        task_rows["task_5_delivery_style_grounding"].append(
            {
                "id": f"sv-t5-{row['source_item_id']}",
                "task_name": "delivery_style_grounding",
                "prompt": row["short_context_text"],
                "source_item_ids": [row["source_item_id"]],
                "query_audio_path": row["audio_path"],
                "text_options": [
                    "Laughter or amused vocal delivery.",
                    "High-energy shouting or cheering.",
                    "Neutral or ordinary speech delivery.",
                    "Soft or whisper-like delivery.",
                ],
                "gold_index": {
                    "laughter": 0,
                    "high_energy": 1,
                    "neutral_speech": 2,
                    "soft": 3,
                }[row["delivery_label"]],
            }
        )

        video_neighbors = [item for item in by_video[row["same_video_group"]] if item["source_item_id"] != row["source_item_id"]]
        if video_neighbors:
            same_video = video_neighbors[0]
            controlled_negative = _choose_filtered(
                row,
                rows,
                rng,
                predicate=lambda item: (
                    item["same_video_group"] != row["same_video_group"]
                    and item["clip_speaker_num"] == row["clip_speaker_num"]
                    and item["delivery_label"] == row["delivery_label"]
                    and _duration_bucket(item) == _duration_bucket(row)
                    and item["is_talking"] == row["is_talking"]
                ),
                k=1,
                reverse=False,
            )
            wrong_match = controlled_negative[0] if controlled_negative else _choose_dissimilar(row, rows, rng, k=1)[0]
            candidates = [same_video, wrong_match]
            rng.shuffle(candidates)
            task_rows["task_4_dyadic_matching"].append(
                {
                    "id": f"sv-t4-{row['source_item_id']}",
                    "task_name": "dyadic_matching",
                    "prompt": (
                        "Which follow-up clip is more likely to come from the same people-centered scene "
                        "or conversation context?"
                    ),
                    "source_item_ids": [row["source_item_id"], same_video["source_item_id"], wrong_match["source_item_id"]],
                    "prompt_audio_path": row["audio_path"],
                    "candidate_audio_paths": [item["audio_path"] for item in candidates],
                    "candidate_labels": [item["clip_name"] for item in candidates],
                    "gold_index": candidates.index(same_video),
                    "balance_bucket": f"{row['delivery_label']}|{_duration_bucket(row)}",
                }
            )

        # Conservative continuation tasks: only build when we have real speech text
        # and enough support for a clean contextual contrast.
        row_token_count = len(_normalize_tokens(row.get("asr_text", "")))
        if row.get("asr_text") and row_token_count >= 4:
            same_video_with_text = [
                item
                for item in video_neighbors
                if item.get("asr_text") and len(_normalize_tokens(item["asr_text"])) >= 4
            ]
            if same_video_with_text:
                same_video_text = same_video_with_text[0]
                off_context = _choose_dissimilar(row, rows, rng, k=1)[0]
                if off_context.get("asr_text") and len(_normalize_tokens(off_context["asr_text"])) >= 4:
                    options = [same_video_text["asr_text"], off_context["asr_text"]]
                    indexed = list(enumerate(options))
                    rng.shuffle(indexed)
                    gold_index = next(index for index, (original_index, _) in enumerate(indexed) if original_index == 0)
                    task_rows["task_6_turn_holding_continuation"].append(
                        {
                            "id": f"sv-t6-{row['source_item_id']}",
                            "task_name": "turn_holding_continuation",
                            "prompt": (
                                "Given the current speech clip, choose the continuation text that best fits "
                                "the same ongoing speaking turn or immediate local context."
                            ),
                            "source_item_ids": [row["source_item_id"], same_video_text["source_item_id"], off_context["source_item_id"]],
                            "prompt_audio_path": row["audio_path"],
                            "text_options": [text for _, text in indexed],
                            "gold_index": gold_index,
                            "balance_bucket": f"{row['delivery_label']}|{_duration_bucket(row)}",
                        }
                    )

            same_count = _choose_filtered(
                row,
                rows,
                rng,
                predicate=lambda item: (
                    item.get("asr_text")
                    and len(_normalize_tokens(item["asr_text"])) >= 4
                    and item["clip_speaker_num"] == row["clip_speaker_num"]
                    and item["delivery_label"] == row["delivery_label"]
                    and _duration_bucket(item) == _duration_bucket(row)
                    and item["same_video_group"] != row["same_video_group"]
                ),
                k=1,
            )
            different_count = _choose_filtered(
                row,
                rows,
                rng,
                predicate=lambda item: (
                    item.get("asr_text")
                    and len(_normalize_tokens(item["asr_text"])) >= 4
                    and item["clip_speaker_num"] != row["clip_speaker_num"]
                    and item["delivery_label"] == row["delivery_label"]
                    and _duration_bucket(item) == _duration_bucket(row)
                    and item["same_video_group"] != row["same_video_group"]
                ),
                k=1,
                reverse=False,
            )
            if same_count and different_count:
                options = [same_count[0]["asr_text"], different_count[0]["asr_text"]]
                indexed = list(enumerate(options))
                rng.shuffle(indexed)
                gold_index = next(index for index, (original_index, _) in enumerate(indexed) if original_index == 0)
                task_rows["task_7_speaker_count_aware_continuation"].append(
                    {
                        "id": f"sv-t7-{row['source_item_id']}",
                        "task_name": "speaker_count_aware_continuation",
                        "prompt": (
                            "Choose the continuation text that best matches the current interaction scale "
                            "(single-speaker versus multi-speaker or dyadic context)."
                        ),
                        "source_item_ids": [row["source_item_id"], same_count[0]["source_item_id"], different_count[0]["source_item_id"]],
                        "prompt_audio_path": row["audio_path"],
                        "text_options": [text for _, text in indexed],
                        "gold_index": gold_index,
                        "balance_bucket": f"{row['delivery_label']}|{_duration_bucket(row)}",
                    }
                )

            if row["delivery_label"] in {"laughter", "high_energy", "neutral_speech", "soft"}:
                same_delivery = _choose_filtered(
                    row,
                    rows,
                    rng,
                    predicate=lambda item: (
                        item.get("asr_text")
                        and len(_normalize_tokens(item["asr_text"])) >= 4
                        and item["delivery_label"] == row["delivery_label"]
                        and item["clip_speaker_num"] == row["clip_speaker_num"]
                        and _duration_bucket(item) == _duration_bucket(row)
                        and item["same_video_group"] != row["same_video_group"]
                    ),
                    k=1,
                )
                different_delivery = _choose_filtered(
                    row,
                    rows,
                    rng,
                    predicate=lambda item: (
                        item.get("asr_text")
                        and len(_normalize_tokens(item["asr_text"])) >= 4
                        and item["delivery_label"] != row["delivery_label"]
                        and item["clip_speaker_num"] == row["clip_speaker_num"]
                        and _duration_bucket(item) == _duration_bucket(row)
                        and item["same_video_group"] != row["same_video_group"]
                    ),
                    k=1,
                    reverse=False,
                )
                if same_delivery and different_delivery:
                    options = [same_delivery[0]["asr_text"], different_delivery[0]["asr_text"]]
                    indexed = list(enumerate(options))
                    rng.shuffle(indexed)
                    gold_index = next(index for index, (original_index, _) in enumerate(indexed) if original_index == 0)
                    task_rows["task_8_delivery_conditioned_continuation"].append(
                        {
                            "id": f"sv-t8-{row['source_item_id']}",
                            "task_name": "delivery_conditioned_continuation",
                            "prompt": (
                                "Choose the continuation text that best fits the delivery style of the current "
                                "speech clip."
                            ),
                            "source_item_ids": [row["source_item_id"], same_delivery[0]["source_item_id"], different_delivery[0]["source_item_id"]],
                            "prompt_audio_path": row["audio_path"],
                            "text_options": [text for _, text in indexed],
                            "gold_index": gold_index,
                            "delivery_label": row["delivery_label"],
                            "balance_bucket": f"{_speaker_count_label(row['clip_speaker_num'])}|{_duration_bucket(row)}",
                        }
                    )

    counts = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for task_name, task_items in task_rows.items():
        counts[task_name] = len(task_items)
        _write_jsonl(output_dir / task_name / "all.jsonl", task_items)
        _write_json(
            output_dir / task_name / "metadata.json",
            {
                **TIER2_TASK_SPECS[task_name],
                "count": len(task_items),
                "path": str(output_dir / task_name / "all.jsonl"),
            },
        )

    metadata = {
        "suite": "COAST Tier2 SpeakerVid Tasks",
        "seed": seed,
        "manifest_path": str(manifest_path),
        "source_items": len(rows),
        "current_release_evaluation": {
            "default_primary_metric": "accuracy",
            "note": "Current Tier 2 packaged evaluation reports task-level choice accuracy. Cross-tier aggregation should be normalized by task family and chance level.",
            "tier_interpretation": "Tier 2 is best interpreted as speech-centered acoustic grounding for audio-only and audio+text models.",
        },
        "counts": counts,
        "task_index": {
            task_name: {
                **TIER2_TASK_SPECS[task_name],
                "path": str(output_dir / task_name / "all.jsonl"),
                "metadata_path": str(output_dir / task_name / "metadata.json"),
                "count": counts[task_name],
            }
            for task_name in task_rows
        },
        "tasks": [spec["task_name"] for spec in TIER2_TASK_SPECS.values()],
    }
    _write_json(output_dir / "metadata.json", metadata)
    return metadata
