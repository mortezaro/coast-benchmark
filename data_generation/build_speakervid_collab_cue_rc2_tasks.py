#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TASK_CONFIGS: dict[str, dict[str, Any]] = {
    "acknowledgment_cue_detection": {
        "labels": ["short_acknowledgment", "not_short_acknowledgment"],
        "prompt": "Listen to the speech clip. Is the speaker mainly giving a short acknowledgment such as okay, yeah, or mm-hmm?",
    },
    "clarification_question_detection": {
        "labels": ["clarification_question", "not_clarification_question"],
        "prompt": "Listen to the speech clip. Does the speaker ask a semantic or referential clarification question?",
    },
    "explicit_repair_cue_detection": {
        "labels": ["explicit_repair_cue", "no_explicit_repair_cue"],
        "prompt": "Listen to the speech clip. Does the speaker explicitly signal a hearing or understanding problem?",
    },
}

BAD_TEXT_RE = re.compile(r"\b(fuck|fucking|shit|bitch|cunt|dick|asshole)\b", re.I)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _youtube_video_id(row: dict[str, Any]) -> str:
    value = str(row.get("youtube_id") or row.get("source_item_id") or "")
    match = re.match(r"(.+?)_\d+x\d+_full_video", value)
    if match:
        return match.group(1)
    return value.split("_full_video", 1)[0]


def _resolve_download(stem: Path) -> Path | None:
    for path in sorted(stem.parent.glob(stem.name + ".*")):
        if path.is_file() and not path.name.endswith(".part"):
            return path
    return None


def _download_stem(row: dict[str, Any], raw_dir: Path, *, mode: str) -> Path:
    video_id = _youtube_video_id(row)
    if mode != "section":
        return raw_dir / video_id
    start = max(0.0, float(row.get("window_start_seconds") or row.get("start_seconds") or 0.0))
    duration = min(24.0, max(12.0, float(row.get("window_duration_seconds") or 12.0)))
    end = start + duration
    return raw_dir / f"{video_id}_{int(start * 1000):010d}_{int(end * 1000):010d}"


def _find_existing_source(video_id: str, source_roots: list[Path]) -> Path | None:
    for root in source_roots:
        if not root.exists():
            continue
        matches = sorted(path for path in root.glob(video_id + "*") if path.is_file() and not path.name.startswith("._"))
        if matches:
            return matches[0]
    return None


def _download_source(
    row: dict[str, Any],
    raw_dir: Path,
    cookies: Path | None,
    python_bin: str,
    *,
    mode: str,
    source_roots: list[Path],
) -> Path:
    video_id = _youtube_video_id(row)
    if mode != "section":
        existing_source = _find_existing_source(video_id, source_roots)
        if existing_source:
            return existing_source
    stem = _download_stem(row, raw_dir, mode=mode)
    existing = _resolve_download(stem)
    if existing:
        return existing

    command = [
        python_bin,
        "-m",
        "yt_dlp",
        "--quiet",
        "--no-warnings",
        "-f",
        "ba/best",
        "-o",
        str(stem) + ".%(ext)s",
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    if mode == "section":
        start = max(0.0, float(row.get("window_start_seconds") or row.get("start_seconds") or 0.0))
        duration = min(24.0, max(12.0, float(row.get("window_duration_seconds") or 12.0)))
        end = start + duration
        command[7:7] = ["--download-sections", f"*{start:.3f}-{end:.3f}"]
    if cookies and cookies.exists():
        command[3:3] = ["--cookies", str(cookies)]
    _run(command)
    resolved = _resolve_download(stem)
    if not resolved:
        raise FileNotFoundError(f"yt-dlp produced no file for {video_id}")
    return resolved


def _extract_wav(input_path: Path, output_path: Path, ffmpeg: str, *, start: float, duration: float) -> None:
    if output_path.exists():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output_path),
        ]
    )


def _safe_candidate(row: dict[str, Any]) -> bool:
    text = str(row.get("text_private") or "")
    return bool(text.strip()) and BAD_TEXT_RE.search(text) is None


def _priority(row: dict[str, Any]) -> tuple[float, float, float, str]:
    duration = float(row.get("window_duration_seconds") or 0.0)
    clear = float(row.get("clear_score") or 0.0)
    dover = float(row.get("dover_score") or 0.0)
    return (abs(duration - 18.0) * -1.0, clear, dover, str(row.get("source_item_id") or ""))


def _task_row(task_name: str, label: str, source: dict[str, Any], wav_path: Path, ordinal: int) -> dict[str, Any]:
    config = TASK_CONFIGS[task_name]
    return {
        "id": f"sv-cue-rc2-{task_name}-{label}-{ordinal:05d}",
        "task_name": task_name,
        "prompt": str(config["prompt"]),
        "label_space": list(config["labels"]),
        "gold_label": label,
        "query_audio_path": str(wav_path),
        "source_item_ids": [str(source.get("source_item_id") or "")],
        "balance_bucket": label,
        "source_dataset": "SpeakerVid-5M",
        "source_item_id": str(source.get("source_item_id") or ""),
        "youtube_id": str(source.get("youtube_id") or ""),
        "window_start_seconds": source.get("window_start_seconds"),
        "window_duration_seconds": source.get("window_duration_seconds"),
        "match_reason": source.get("match_reason"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full RC2 observable cue-detection task suite from mined SpeakerVid candidates.")
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--python-bin", default="python")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--cookies")
    parser.add_argument("--source-video-root", action="append")
    parser.add_argument("--max-per-label", type=int, default=500)
    parser.add_argument("--download-mode", choices=["source", "section"], default="source")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    candidate_root = Path(args.candidate_root)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_root = output_dir / "audio"
    raw_dir = output_dir / "_download_cache"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cookies = Path(args.cookies).expanduser() if args.cookies else None
    source_roots = [Path(value).expanduser() for value in args.source_video_root or []]

    suite_counts: dict[str, int] = {}
    failures: list[dict[str, Any]] = []

    for task_index, (task_name, config) in enumerate(TASK_CONFIGS.items(), start=1):
        by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in _read_jsonl(candidate_root / task_name / "candidates.jsonl"):
            label = str(row.get("label") or "")
            if label in config["labels"] and _safe_candidate(row):
                by_label[label].append(row)
        for label_rows in by_label.values():
            label_rows.sort(key=_priority, reverse=True)

        out_rows: list[dict[str, Any]] = []
        label_counts: Counter[str] = Counter()
        for label in config["labels"]:
            for source in by_label.get(label, []):
                if label_counts[label] >= args.max_per_label:
                    break
                source_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(source.get("source_item_id") or "unknown"))
                wav_path = audio_root / task_name / label / f"{source_id}.wav"
                try:
                    raw_path = _download_source(
                        source,
                        raw_dir,
                        cookies,
                        args.python_bin,
                        mode=args.download_mode,
                        source_roots=source_roots,
                    )
                    if args.download_mode == "section":
                        start = 0.0
                    else:
                        start = max(0.0, float(source.get("window_start_seconds") or source.get("start_seconds") or 0.0))
                    duration = min(24.0, max(12.0, float(source.get("window_duration_seconds") or 12.0)))
                    _extract_wav(raw_path, wav_path, args.ffmpeg, start=start, duration=duration)
                except Exception as exc:
                    failures.append(
                        {
                            "task_name": task_name,
                            "label": label,
                            "source_item_id": source.get("source_item_id", ""),
                            "youtube_id": source.get("youtube_id", ""),
                            "error": str(exc),
                        }
                    )
                    continue
                label_counts[label] += 1
                out_rows.append(_task_row(task_name, label, source, wav_path, label_counts[label]))

        task_dir = output_dir / f"task_{task_index:02d}_{task_name}"
        _write_jsonl(task_dir / "all.jsonl", out_rows)
        _write_json(
            task_dir / "metadata.json",
            {
                "task_name": task_name,
                "count": len(out_rows),
                "label_counts": dict(sorted(Counter(row["gold_label"] for row in out_rows).items())),
                "candidate_root": str(candidate_root),
                "max_per_label": args.max_per_label,
                "human_validity_note": "Transcript-seeded, audio-validated observable cue task. Public rows omit transcripts.",
            },
        )
        suite_counts[task_dir.name] = len(out_rows)

    _write_jsonl(output_dir / "download_failures.jsonl", failures)
    metadata = {
        "suite_type": "speakervid_collab_cue_rc2_tasks",
        "output_dir": str(output_dir),
        "candidate_root": str(candidate_root),
        "tasks": list(TASK_CONFIGS),
        "counts": suite_counts,
        "download_failures": len(failures),
        "standards": [
            "observable cue labels only",
            "transcripts used only for candidate seeding",
            "12-24 second audio context windows",
            "public rows omit transcripts",
            "balanced labels enforced later by standard-release validator",
        ],
    }
    _write_json(output_dir / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
