import argparse
import csv
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

from dynsalmon.coast_grounding_eval import _read_jsonl, _score_prediction


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _iter_subset_tasks(subset_root: Path) -> list[tuple[str, Path]]:
    pairs: list[tuple[str, Path]] = []
    for tier_dir in sorted(path for path in subset_root.iterdir() if path.is_dir()):
        for task_dir in sorted(path for path in tier_dir.glob("task_*") if path.is_dir()):
            pairs.append((tier_dir.name, task_dir))
    return pairs


def _task_rows(task_dir: Path) -> list[dict[str, Any]]:
    return _read_jsonl(task_dir / "all.jsonl")


def _task_chance(row: dict[str, Any]) -> float:
    if row["task_name"] == "acoustic_plausibility_ranking":
        return 1.0 / float(max(1, len(row.get("candidate_audio_paths") or [])))
    if "candidate_audio_paths" in row:
        return 1.0 / float(max(1, len(row.get("candidate_audio_paths") or [])))
    return 1.0 / float(max(1, len(row.get("text_options") or row.get("label_space") or [])))


def _primary_mean(rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    total = 0.0
    for row, pred in zip(rows, predictions):
        total += float(_score_prediction(row["task_name"], row, pred.get("prediction", pred))["primary_metric"])
    return total / float(len(rows))


def _load_model_predictions(package_root: Path, model_name: str, tier_name: str, task_dir_name: str) -> list[dict[str, Any]]:
    return _read_jsonl(package_root / model_name / tier_name / task_dir_name / "predictions.jsonl")


def _shuffled_rows(rows: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    shuffled = [deepcopy(row) for row in rows]
    targets: list[Any] = []
    mode = None
    for row in rows:
        if "gold_label" in row:
            mode = "gold_label"
            targets.append(row["gold_label"])
        elif row["task_name"] == "acoustic_plausibility_ranking":
            mode = "gold_ranking"
            targets.append(list(row["gold_ranking"]))
        else:
            mode = "gold_index"
            targets.append(int(row["gold_index"]))
    rng = random.Random(seed)
    rng.shuffle(targets)
    for row, target in zip(shuffled, targets):
        if mode == "gold_label":
            row["gold_label"] = target
        elif mode == "gold_ranking":
            row["gold_ranking"] = target
        else:
            row["gold_index"] = target
    return shuffled


def _binary_flipped_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    flipped = [deepcopy(row) for row in rows]
    for row in flipped:
        if "gold_label" in row:
            labels = list(row.get("label_space") or [])
            if len(labels) != 2:
                return None
            current = row["gold_label"]
            other = labels[1] if str(labels[0]) == str(current) else labels[0]
            row["gold_label"] = other
        elif row["task_name"] == "acoustic_plausibility_ranking":
            candidates = list(row.get("candidate_audio_paths") or [])
            if len(candidates) != 2:
                return None
            top = int(row["gold_ranking"][0])
            row["gold_ranking"] = [1 - top, top]
        else:
            option_count = len(row.get("candidate_audio_paths") or row.get("text_options") or row.get("label_space") or [])
            if option_count != 2:
                return None
            row["gold_index"] = 1 - int(row["gold_index"])
    return flipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-root", required=True)
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    subset_root = Path(args.subset_root)
    package_root = Path(args.package_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    package_summary = _read_json(package_root / "matched_package_summary.json")
    model_names = list(package_summary["models"])
    retained_tasks: list[dict[str, Any]] = []

    label_shuffle_rows: list[dict[str, Any]] = []
    binary_flip_rows: list[dict[str, Any]] = []

    for tier_name, task_dir in _iter_subset_tasks(subset_root):
        rows = _task_rows(task_dir)
        if not rows:
            continue
        task_name = rows[0]["task_name"]
        if task_name == "speaker_event_overlap":
            continue
        chance = _task_chance(rows[0])
        retained_tasks.append(
            {
                "tier": tier_name,
                "task_dir": task_dir.name,
                "task_name": task_name,
                "chance": chance,
            }
        )
        shuffled_rows = _shuffled_rows(rows, seed=args.seed + len(retained_tasks))
        flipped_rows = _binary_flipped_rows(rows)

        for model_name in model_names:
            predictions = _load_model_predictions(package_root, model_name, tier_name, task_dir.name)
            label_shuffle_rows.append(
                {
                    "tier": tier_name,
                    "task_name": task_name,
                    "model_name": model_name,
                    "chance": chance,
                    "label_shuffle_mean": _primary_mean(shuffled_rows, predictions),
                }
            )
            if flipped_rows is not None:
                binary_flip_rows.append(
                    {
                        "tier": tier_name,
                        "task_name": task_name,
                        "model_name": model_name,
                        "chance": chance,
                        "binary_flip_mean": _primary_mean(flipped_rows, predictions),
                    }
                )

    label_shuffle_csv = output_root / "robustness_label_shuffle.csv"
    with label_shuffle_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["tier", "task_name", "model_name", "chance", "label_shuffle_mean"],
        )
        writer.writeheader()
        writer.writerows(label_shuffle_rows)

    binary_flip_csv = output_root / "robustness_binary_flip.csv"
    with binary_flip_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["tier", "task_name", "model_name", "chance", "binary_flip_mean"],
        )
        writer.writeheader()
        writer.writerows(binary_flip_rows)

    label_shuffle_summary: dict[str, float] = {}
    for model_name in model_names:
        values = [row["label_shuffle_mean"] for row in label_shuffle_rows if row["model_name"] == model_name]
        label_shuffle_summary[model_name] = sum(values) / float(len(values)) if values else 0.0

    binary_flip_summary: dict[str, float] = {}
    for model_name in model_names:
        values = [row["binary_flip_mean"] for row in binary_flip_rows if row["model_name"] == model_name]
        if values:
            binary_flip_summary[model_name] = sum(values) / float(len(values))

    md_lines = [
        "# COAST Robustness Probes",
        "",
        "Retained tasks were probed structurally without rerunning the models.",
        "",
        "Implemented probes:",
        "- `label_shuffle`: shuffle gold targets within a retained task and rescore the existing predictions",
        "- `binary_flip`: for binary retained tasks, flip the gold target and rescore the existing predictions",
        "",
        "Lower scores after these probes indicate that the original benchmark signal was tied to the intended labels rather than a trivial shortcut.",
        "",
        "## Average Label-Shuffle Score",
        "",
        "| Model | Mean |",
        "| --- | --- |",
    ]
    for model_name, value in sorted(label_shuffle_summary.items(), key=lambda item: item[1]):
        md_lines.append(f"| {model_name} | {value:.4f} |")

    if binary_flip_summary:
        md_lines.extend(
            [
                "",
                "## Average Binary-Flip Score",
                "",
                "| Model | Mean |",
                "| --- | --- |",
            ]
        )
        for model_name, value in sorted(binary_flip_summary.items(), key=lambda item: item[1]):
            md_lines.append(f"| {model_name} | {value:.4f} |")

    (output_root / "robustness_probe_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    _write_json(
        output_root / "robustness_probe_summary.json",
        {
            "subset_root": str(subset_root),
            "package_root": str(package_root),
            "retained_tasks": retained_tasks,
            "label_shuffle_summary": label_shuffle_summary,
            "binary_flip_summary": binary_flip_summary,
        },
    )


if __name__ == "__main__":
    main()
