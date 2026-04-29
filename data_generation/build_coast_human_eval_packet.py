import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


TASK_GUIDANCE = {
    "target_sound_grounding": "Listen to all candidate clips and choose the one that best matches the written event description.",
    "fine_grained_acoustic_contrast": "Listen to the candidate clips and choose the better acoustic match for the prompt.",
    "foreground_event_focus": "Listen to the query clip and decide whether the target event is foreground, background, or absent.",
    "acoustic_plausibility_ranking": "Rank the candidate clips from most plausible to least plausible given the prompt.",
    "periodicity_aware_grounding": "Listen to the query clip and choose whether it sounds more repetitive/rhythmic or more abrupt/impact-like.",
    "onset_sharpness_grounding": "Listen to the query clip and judge whether the sound onset is sharp or diffuse.",
    "audio_text_hard_negative_matching": "Listen to the query clip and choose the text description that best matches it.",
    "phonic_target_grounding": "Listen to all candidate clips and choose the one that best matches the people-centered speech prompt.",
    "speaker_count_grounding": "Listen to the query clip and decide whether it contains one dominant speaker or multiple speakers.",
    "dyadic_matching": "Listen to the prompt clip and the candidate clips, then choose the candidate most likely to come from the same conversational scene.",
    "delivery_style_grounding": "Listen to the query clip and pick the delivery style that best matches the speech.",
    "turn_holding_continuation": "Listen to the prompt clip and choose the continuation text that best fits the same ongoing turn.",
    "speaker_count_aware_continuation": "Listen to the prompt clip and choose the continuation text that best matches the single-speaker or multi-speaker interaction scale.",
    "delivery_conditioned_continuation": "Listen to the prompt clip and choose the continuation text that best matches the delivery style.",
    "event_window_grounding": "Listen to all candidate clips and choose the one that best matches the long-form event prompt.",
    "narrative_continuation_plausibility": "Listen to the query clip and choose the continuation text that best fits what plausibly happens next.",
    "long_context_event_retrieval": "Given the text query, listen to the candidate clips and choose the best audio match.",
    "sequential_causal_consistency": "Listen to the query clip and choose the text option that best matches the causal order implied by the audio.",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _copy_audio(src: str | None, dest: Path) -> None:
    if not src:
        return
    source_path = Path(src)
    if not source_path.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, dest)


def _public_item_payload(row: dict[str, Any], item_dir: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": row["id"],
        "task_name": row["task_name"],
        "prompt": row.get("prompt") or row.get("query_text") or row.get("context_text") or "",
    }
    if "text_options" in row:
        payload["text_options"] = row["text_options"]
    if "label_space" in row:
        payload["label_space"] = row["label_space"]
    if "candidate_audio_paths" in row:
        payload["candidate_audio_files"] = [f"candidate_{idx + 1}.wav" for idx in range(len(row["candidate_audio_paths"]))]
    if "query_audio_path" in row:
        payload["query_audio_file"] = "query.wav"
    if "prompt_audio_path" in row:
        payload["prompt_audio_file"] = "prompt.wav"
    return payload


def _answer_repr(row: dict[str, Any]) -> str:
    if "gold_label" in row:
        return str(row["gold_label"])
    if "gold_ranking" in row:
        return " > ".join(str(int(index) + 1) for index in row["gold_ranking"])
    return str(int(row["gold_index"]) + 1)


def _copy_row_assets(row: dict[str, Any], item_dir: Path) -> None:
    _copy_audio(row.get("query_audio_path"), item_dir / "query.wav")
    _copy_audio(row.get("prompt_audio_path"), item_dir / "prompt.wav")
    candidate_paths = row.get("candidate_audio_paths") or []
    for idx, candidate_path in enumerate(candidate_paths, start=1):
        _copy_audio(candidate_path, item_dir / f"candidate_{idx}.wav")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--examples-per-task", type=int, default=2)
    args = parser.parse_args()

    subset_root = Path(args.subset_root)
    output_root = Path(args.output_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    instructions_dir = output_root / "task_instructions"
    items_dir = output_root / "items"
    instructions_dir.mkdir(parents=True, exist_ok=True)
    items_dir.mkdir(parents=True, exist_ok=True)

    answer_rows: list[dict[str, str]] = []
    manifest_rows: list[dict[str, str]] = []

    retained_tasks: list[str] = []
    for tier in ("tier1", "tier2", "tier3"):
        meta = _read_json(subset_root / tier / "metadata.json")
        dropped = set(meta.get("dropped_tasks", []))
        for task_dir in sorted((subset_root / tier).glob("task_*")):
            if task_dir.name in dropped:
                continue
            rows = _read_jsonl(task_dir / "all.jsonl")
            if not rows:
                continue
            task_name = rows[0]["task_name"]
            retained_tasks.append(task_name)
            task_md = [
                f"# {task_name}",
                "",
                f"Tier: `{tier}`",
                "",
                TASK_GUIDANCE.get(task_name, "Listen carefully and choose the option that best matches the task prompt."),
                "",
                "Ask each rater to also record:",
                "",
                "- chosen answer",
                "- confidence from 1 to 5",
                "- difficulty from 1 to 5",
                "- optional free-text comments",
            ]
            (instructions_dir / f"{task_name}.md").write_text("\n".join(task_md) + "\n", encoding="utf-8")

            task_items_dir = items_dir / task_name
            task_items_dir.mkdir(parents=True, exist_ok=True)
            for example_index, row in enumerate(rows[: args.examples_per_task], start=1):
                item_dir = task_items_dir / f"{example_index:02d}_{row['id']}"
                item_dir.mkdir(parents=True, exist_ok=True)
                _copy_row_assets(row, item_dir)
                public_payload = _public_item_payload(row, item_dir)
                (item_dir / "item.json").write_text(json.dumps(public_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                answer_rows.append(
                    {
                        "task_name": task_name,
                        "item_id": row["id"],
                        "answer_key": _answer_repr(row),
                    }
                )
                manifest_rows.append(
                    {
                        "tier": tier,
                        "task_name": task_name,
                        "item_id": row["id"],
                        "item_dir": str(item_dir.relative_to(output_root)),
                    }
                )

    readme_lines = [
        "# COAST Human Evaluation Packet",
        "",
        "This packet is a small human-facing evaluation starter built from the validated COAST v2 subset.",
        "",
        "It is intended for two purposes:",
        "",
        "1. sanity-check whether the retained tasks make sense to human raters",
        "2. start collecting a lightweight human baseline and difficulty signal",
        "",
        "## What To Share",
        "",
        "- Share the `items/` and `task_instructions/` folders with raters.",
        "- Keep `answer_key_private.csv` private if you want blind evaluation.",
        "",
        "## Rater Instructions",
        "",
        "- Use headphones when possible.",
        "- Work task-by-task rather than mixing tasks.",
        "- For each item, record the chosen answer, confidence (1-5), difficulty (1-5), and any comments.",
        "- Do not look at the private answer key during blind evaluation.",
        "",
        "## Packet Contents",
        "",
        "- `task_instructions/`: one short instruction file per retained task",
        "- `items/`: example items and audio files",
        "- `responses_template.csv`: template for collecting human answers",
        "- `items_manifest.csv`: inventory of included examples",
        "- `answer_key_private.csv`: answer key for scoring after collection",
        "",
        "## Retained Tasks In This Packet",
        "",
    ]
    for task_name in sorted(retained_tasks):
        readme_lines.append(f"- `{task_name}`")
    (output_root / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    with (output_root / "items_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tier", "task_name", "item_id", "item_dir"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    with (output_root / "answer_key_private.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_name", "item_id", "answer_key"])
        writer.writeheader()
        writer.writerows(answer_rows)

    with (output_root / "responses_template.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rater_id", "task_name", "item_id", "chosen_answer", "confidence_1_to_5", "difficulty_1_to_5", "comments"],
        )
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(
                {
                    "rater_id": "",
                    "task_name": row["task_name"],
                    "item_id": row["item_id"],
                    "chosen_answer": "",
                    "confidence_1_to_5": "",
                    "difficulty_1_to_5": "",
                    "comments": "",
                }
            )


if __name__ == "__main__":
    main()
