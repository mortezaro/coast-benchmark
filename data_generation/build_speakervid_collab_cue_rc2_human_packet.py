#!/usr/bin/env python3
"""Build a human packet for RC2 simple collaborative cue-detection tasks."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


TASKS = {
    "acknowledgment_cue_detection": {
        "title": "Acknowledgment Cue Detection",
        "labels": ["short_acknowledgment", "not_short_acknowledgment"],
        "per_label": 6,
        "prompt": "Listen to the clip. Is the speaker mainly giving a short acknowledgment such as okay, yeah, or mm-hmm?",
        "descriptions": {
            "short_acknowledgment": "The target utterance is mainly a short acknowledgment/backchannel.",
            "not_short_acknowledgment": "The target utterance is not mainly a short acknowledgment.",
        },
    },
    "clarification_question_detection": {
        "title": "Clarification Question Detection",
        "labels": ["clarification_question", "not_clarification_question"],
        "per_label": 6,
        "prompt": "Listen to the clip. Does the speaker ask a semantic or referential clarification question?",
        "descriptions": {
            "clarification_question": "The speaker asks what something means, which one, who/where/when exactly, or whether X is meant.",
            "not_clarification_question": "The speaker does not ask a clarification question.",
        },
    },
    "explicit_repair_cue_detection": {
        "title": "Explicit Repair Cue Detection",
        "labels": ["explicit_repair_cue", "no_explicit_repair_cue"],
        "per_label": 6,
        "prompt": "Listen to the clip. Does the speaker explicitly signal a hearing or understanding problem?",
        "descriptions": {
            "explicit_repair_cue": "The speaker says something like huh, pardon, sorry what, what was that, repeat that, or I didn't hear/catch that.",
            "no_explicit_repair_cue": "No explicit hearing/understanding repair cue is present.",
        },
    },
}

BAD_TEXT_RE = re.compile(r"\b(fuck|fucking|shit|bitch|cunt|dick|asshole)\b", re.I)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def youtube_video_id(row: dict[str, Any]) -> str:
    value = str(row.get("youtube_id") or row.get("source_item_id") or "")
    match = re.match(r"(.+?)_\d+x\d+_full_video", value)
    if match:
        return match.group(1)
    return value.split("_full_video", 1)[0]


def safe_text(row: dict[str, Any]) -> bool:
    return BAD_TEXT_RE.search(str(row.get("text_private") or "")) is None


def priority(task: str, label: str, row: dict[str, Any]) -> tuple[float, float, float]:
    text = str(row.get("text_private") or "")
    dur = float(row.get("window_duration_seconds") or 0)
    dover = float(row.get("dover_score") or 0)
    clear = float(row.get("clear_score") or 0)
    score = 0.0
    if 12 <= dur <= 22:
        score += 5
    if safe_text(row):
        score += 3
    if task == "explicit_repair_cue_detection" and label == "explicit_repair_cue":
        if re.search(r"^(what was that|what did you say|can you repeat|could you repeat|i didn.?t hear|i didn.?t catch|huh|pardon|sorry what)", text, re.I):
            score += 8
    if task == "clarification_question_detection" and label == "clarification_question":
        if re.search(r"^(what do you mean|which one|who do you mean|do you mean|are you saying|can you clarify|could you clarify)", text, re.I):
            score += 8
    if task == "acknowledgment_cue_detection" and label == "short_acknowledgment":
        if len(text.split()) <= 3:
            score += 6
    return (score, dover, clear)


def task_rows(candidate_root: Path, task: str) -> list[dict[str, Any]]:
    rows = read_jsonl(candidate_root / task / "candidates.jsonl")
    rows = [r for r in rows if safe_text(r)]
    rows.sort(key=lambda r: priority(task, str(r.get("label")), r), reverse=True)
    return rows


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def resolve_download(stem: Path) -> Path | None:
    for path in sorted(stem.parent.glob(stem.name + ".*")):
        if path.is_file() and not path.name.endswith(".part"):
            return path
    return None


def download_window(row: dict[str, Any], raw_dir: Path, cookies: Path | None, python_bin: str) -> Path:
    video_id = youtube_video_id(row)
    start = max(0.0, float(row.get("window_start_seconds") or row.get("start_seconds") or 0.0))
    duration = min(22.0, max(12.0, float(row.get("window_duration_seconds") or 12.0)))
    end = start + duration
    stem = raw_dir / f"{video_id}_{int(start * 1000):010d}_{int(end * 1000):010d}"
    existing = resolve_download(stem)
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
        "--download-sections",
        f"*{start:.3f}-{end:.3f}",
        "--force-keyframes-at-cuts",
        "-o",
        str(stem) + ".%(ext)s",
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    if cookies and cookies.exists():
        command[3:3] = ["--cookies", str(cookies)]
    run(command)
    resolved = resolve_download(stem)
    if not resolved:
        raise FileNotFoundError(f"yt-dlp produced no file for {video_id}")
    return resolved


def extract_wav(input_path: Path, output_path: Path, ffmpeg: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run([
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ])


def write_instruction(output_root: Path, task: str) -> None:
    config = TASKS[task]
    lines = [f"# {config['title']}", "", config["prompt"], "", "Choose one label:", ""]
    for label in config["labels"]:
        lines.append(f"- `{label}`: {config['descriptions'][label]}")
    lines.extend([
        "",
        "Also answer whether the judgment was possible from audio alone.",
        "If the clip is mostly monologue, speaker-overlap, unclear, or missing the cue, mark `answerable_from_audio_yes_no` as `no`.",
    ])
    (output_root / "task_instructions" / f"{task}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--cookies")
    args = parser.parse_args()

    candidate_root = Path(args.candidate_root)
    output_root = Path(args.output_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    (output_root / "items").mkdir(parents=True, exist_ok=True)
    (output_root / "task_instructions").mkdir(parents=True, exist_ok=True)
    raw_dir = output_root / "_download_cache"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cookies = Path(args.cookies).expanduser() if args.cookies else None

    manifest = []
    responses = []
    answers = []
    failures = []

    for task, config in TASKS.items():
        write_instruction(output_root, task)
        selected = defaultdict(int)
        for row in task_rows(candidate_root, task):
            label = str(row["label"])
            if label not in config["labels"] or selected[label] >= int(config["per_label"]):
                continue
            item_id = f"{task}_{label}_{selected[label] + 1:02d}"
            item_dir = output_root / "items" / task / item_id
            try:
                raw = download_window(row, raw_dir, cookies, args.python_bin)
                extract_wav(raw, item_dir / "query.wav", args.ffmpeg)
            except Exception as exc:
                failures.append({
                    "task_name": task,
                    "label": label,
                    "source_item_id": row.get("source_item_id", ""),
                    "youtube_id": row.get("youtube_id", ""),
                    "error": str(exc),
                })
                continue
            public = {
                "id": item_id,
                "task_name": task,
                "prompt": config["prompt"],
                "label_space": config["labels"],
                "query_audio_file": "query.wav",
                "answerability_question": "Could you make this judgment from audio alone?",
            }
            (item_dir / "item.json").write_text(json.dumps(public, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            rel_audio = f"items/{task}/{item_id}/query.wav"
            rel_json = f"items/{task}/{item_id}/item.json"
            manifest.append({"item_id": item_id, "task_name": task, "audio_file": rel_audio, "item_json": rel_json})
            responses.append({
                "item_id": item_id,
                "task_name": task,
                "chosen_answer": "",
                "answerable_from_audio_yes_no": "",
                "confidence_1_5": "",
                "difficulty_1_5": "",
                "comments": "",
            })
            answers.append({
                "item_id": item_id,
                "task_name": task,
                "gold_label": label,
                "transcript_private": row.get("text_private", ""),
                "source_item_id": row.get("source_item_id", ""),
                "youtube_id": row.get("youtube_id", ""),
                "window_start_seconds": row.get("window_start_seconds", ""),
                "window_duration_seconds": row.get("window_duration_seconds", ""),
                "match_reason": row.get("match_reason", ""),
            })
            selected[label] += 1
            if all(selected[label] >= int(config["per_label"]) for label in config["labels"]):
                break

    for name, rows in [
        ("items_manifest.csv", manifest),
        ("responses_template.csv", responses),
        ("answer_key_private.csv", answers),
        ("download_failures.csv", failures),
    ]:
        with (output_root / name).open("w", encoding="utf-8", newline="") as handle:
            fieldnames = sorted({k for row in rows for k in row}) if rows else ["empty"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    readme = [
        "# COAST RC2 Observable Cue Human Packet",
        "",
        "This packet replaces abstract collaborative-state labels with simple observable cue detection.",
        "",
        "Share with raters: `items/`, `task_instructions/`, and `responses_template.csv`.",
        "Keep `answer_key_private.csv` private.",
        "",
        "Raters should mark whether each judgment is possible from audio alone. Items marked not answerable should be removed before benchmark release.",
    ]
    (output_root / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    shutil.rmtree(raw_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
