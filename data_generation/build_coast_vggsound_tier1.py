from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynsalmon.coast import _ffmpeg_binary


VGGSOUND_CSV_URL = "https://huggingface.co/datasets/Loie/VGGSound/resolve/main/vggsound.csv"
DEFAULT_KEYWORDS = [
    "hammer",
    "clap",
    "slap",
    "slam",
    "bang",
    "smash",
    "knock",
    "hit",
    "punch",
    "thud",
]
KEYWORD_PRESETS = {
    "baseline": DEFAULT_KEYWORDS,
    "people_impulse": [
        "clap",
        "slap",
        "smack",
        "hit",
        "punch",
        "kick",
        "knock",
        "tap",
        "thud",
        "bang",
        "cheer",
        "shout",
        "sneeze",
        "cough",
        "laugh",
    ],
    "varied_impulse": [
        "hammer",
        "clap",
        "slap",
        "slam",
        "bang",
        "smash",
        "knock",
        "hit",
        "punch",
        "thud",
        "gun",
        "shot",
        "explosion",
        "firework",
        "door",
        "engine",
    ],
}
PEOPLE_TOKENS = [
    "people",
    "person",
    "man",
    "woman",
    "boy",
    "girl",
    "human",
    "crowd",
    "audience",
    "child",
    "adult",
]


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "item"


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        handle.write(response.read())


def _read_vggsound_rows(csv_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for fields in reader:
            if not fields or len(fields) < 4:
                continue
            youtube_id, start_seconds, label, split = [field.strip() for field in fields[:4]]
            rows.append(
                {
                    "youtube_id": youtube_id,
                    "start_seconds": start_seconds,
                    "label": label,
                    "split": split,
                }
            )
    return rows


def _matches_keywords(label: str, keywords: list[str]) -> bool:
    lower = label.lower()
    return any(keyword in lower for keyword in keywords)


def _people_priority(label: str) -> int:
    lower = label.lower()
    return int(any(token in lower for token in PEOPLE_TOKENS))


def _select_rows(
    rows: list[dict[str, str]],
    *,
    limit: int,
    keywords: list[str],
    split: str | None,
    per_label_limit: int | None,
    shuffle_seed: int,
    prefer_people: bool,
) -> list[dict[str, str]]:
    filtered = []
    for row in rows:
        if split and row["split"] != split:
            continue
        if not _matches_keywords(row["label"], keywords):
            continue
        filtered.append(row)

    rng = random.Random(shuffle_seed)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in filtered:
        grouped[row["label"]].append(row)
    for label_rows in grouped.values():
        rng.shuffle(label_rows)

    labels = list(grouped)
    rng.shuffle(labels)
    labels.sort(key=lambda label: (-_people_priority(label) if prefer_people else 0, label))

    selected: list[dict[str, str]] = []
    label_counts: Counter[str] = Counter()
    made_progress = True
    while len(selected) < limit and made_progress:
        made_progress = False
        for label in labels:
            if len(selected) >= limit:
                break
            if per_label_limit is not None and label_counts[label] >= per_label_limit:
                continue
            if not grouped[label]:
                continue
            selected.append(grouped[label].pop())
            label_counts[label] += 1
            made_progress = True
    return selected


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _yt_dlp_binary() -> str:
    binary = shutil.which("yt-dlp")
    if binary:
        return binary
    raise RuntimeError("yt-dlp is required but was not found on PATH")


def _yt_dlp_base_command(output_path: Path) -> list[str]:
    format_selector = os.environ.get("YTDLP_FORMAT_SELECTOR", "").strip() or "bv*+ba/bestvideo*+bestaudio/best"
    command = [
        _yt_dlp_binary(),
        "-f",
        format_selector,
        "-o",
        str(output_path),
        "--no-playlist",
        "--quiet",
        "--no-warnings",
    ]
    cookies_path = os.environ.get("YTDLP_COOKIES", "").strip()
    if cookies_path:
        command.extend(["--cookies", cookies_path])
    sleep_interval = os.environ.get("YTDLP_SLEEP_INTERVAL", "").strip()
    if sleep_interval:
        command.extend(["--sleep-interval", sleep_interval])
    max_sleep_interval = os.environ.get("YTDLP_MAX_SLEEP_INTERVAL", "").strip()
    if max_sleep_interval:
        command.extend(["--max-sleep-interval", max_sleep_interval])
    js_runtimes = os.environ.get("YTDLP_JS_RUNTIMES", "").strip()
    if js_runtimes:
        command.extend(["--js-runtimes", js_runtimes])
    user_agent = os.environ.get("YTDLP_USER_AGENT", "").strip()
    if user_agent:
        command.extend(["--user-agent", user_agent])
    return command


def _resolve_download_output(output_path: Path) -> Path | None:
    if output_path.exists():
        return output_path
    matches = sorted(output_path.parent.glob(f"{output_path.name}*"))
    for match in matches:
        if match.is_file() and not match.name.endswith(".part"):
            return match
    return None


def _download_youtube_video(youtube_id: str, output_path: Path) -> Path:
    existing = _resolve_download_output(output_path)
    if existing is not None:
        return existing
    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    command = [*_yt_dlp_base_command(output_path), url]
    _run(command)
    post_sleep_seconds = float(os.environ.get("YTDLP_POST_SLEEP_SECONDS", "0").strip() or "0")
    if post_sleep_seconds > 0:
        time.sleep(post_sleep_seconds)
    resolved = _resolve_download_output(output_path)
    if resolved is None:
        raise FileNotFoundError(f"yt-dlp reported success but no output file was found for {youtube_id}")
    return resolved


def _trim_clip(raw_video_path: Path, clip_path: Path, start_seconds: float, duration_seconds: float) -> None:
    if clip_path.exists():
        return
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _ffmpeg_binary(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-t",
        f"{duration_seconds:.3f}",
        "-i",
        str(raw_video_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(clip_path),
    ]
    _run(command)


def build_vggsound_tier1(
    *,
    root_dir: Path,
    limit: int,
    keywords: list[str],
    split: str | None,
    duration_seconds: float,
    per_label_limit: int | None = None,
    shuffle_seed: int = 42,
    prefer_people: bool = False,
) -> dict[str, Any]:
    metadata_dir = root_dir / "metadata"
    raw_dir = root_dir / "raw_videos"
    manifests_dir = root_dir / "manifests"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    csv_path = metadata_dir / "vggsound.csv"
    if not csv_path.exists():
        _download_file(VGGSOUND_CSV_URL, csv_path)

    rows = _read_vggsound_rows(csv_path)
    selected = _select_rows(
        rows,
        limit=limit,
        keywords=keywords,
        split=split,
        per_label_limit=per_label_limit,
        shuffle_seed=shuffle_seed,
        prefer_people=prefer_people,
    )

    manifest_path = manifests_dir / "vggsound_tier1_manifest.jsonl"
    failures_path = manifests_dir / "vggsound_tier1_failures.jsonl"
    manifest_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for row in selected:
        youtube_id = row["youtube_id"]
        start_seconds = float(row["start_seconds"])
        label = row["label"]
        label_slug = _slugify(label)
        requested_raw_path = raw_dir / f"{youtube_id}.mp4"
        try:
            raw_path = _download_youtube_video(youtube_id, requested_raw_path)
            manifest_rows.append(
                {
                    "id": f"vggsound_{youtube_id}_{int(start_seconds):06d}",
                    "source_path": str(raw_path),
                    "source_kind": "video",
                    "source_dataset": "VGGSound",
                    "task_type": "A-Discrimination",
                    "event_category": label_slug,
                    "context_text": f"A video of {label.replace('_', ' ')}.",
                    "clip_start_sec": start_seconds,
                    "clip_duration_sec": duration_seconds,
                    "split": row["split"],
                    "vggsound": row,
                }
            )
        except Exception as exc:  # pragma: no cover - runtime data acquisition
            failures.append(
                {
                    "youtube_id": youtube_id,
                    "start_seconds": start_seconds,
                    "label": label,
                    "error": str(exc),
                }
            )

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with failures_path.open("w", encoding="utf-8") as handle:
        for row in failures:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "csv_path": str(csv_path),
        "manifest_path": str(manifest_path),
        "failures_path": str(failures_path),
        "selected_rows": len(selected),
        "downloaded_rows": len(manifest_rows),
        "failed_rows": len(failures),
        "keywords": keywords,
        "split": split,
        "duration_seconds": duration_seconds,
        "per_label_limit": per_label_limit,
        "shuffle_seed": shuffle_seed,
        "prefer_people": prefer_people,
        "label_histogram": dict(sorted(Counter(row["vggsound"]["label"] for row in manifest_rows).items())),
        "people_priority_rows": sum(_people_priority(row["vggsound"]["label"]) for row in manifest_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--keyword-preset", choices=tuple(KEYWORD_PRESETS), default=None)
    parser.add_argument("--keywords", default=",".join(DEFAULT_KEYWORDS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--per-label-limit", type=int, default=25)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--prefer-people", action="store_true")
    args = parser.parse_args()

    keywords = (
        KEYWORD_PRESETS[args.keyword_preset]
        if args.keyword_preset
        else [keyword.strip().lower() for keyword in args.keywords.split(",") if keyword.strip()]
    )

    metadata = build_vggsound_tier1(
        root_dir=Path(args.root_dir),
        limit=args.limit,
        keywords=keywords,
        split=args.split if args.split else None,
        duration_seconds=args.duration_seconds,
        per_label_limit=args.per_label_limit if args.per_label_limit and args.per_label_limit > 0 else None,
        shuffle_seed=args.shuffle_seed,
        prefer_people=args.prefer_people,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
