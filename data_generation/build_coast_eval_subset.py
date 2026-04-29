import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dynsalmon.coast_grounding_eval import _score_prediction


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _task_chance_from_row(row: dict[str, Any]) -> float:
    if row["task_name"] == "acoustic_plausibility_ranking":
        return 1.0 / float(max(1, len(row.get("candidate_audio_paths") or [])))
    if "candidate_audio_paths" in row:
        return 1.0 / float(max(1, len(row.get("candidate_audio_paths") or [])))
    return 1.0 / float(max(1, len(row.get("text_options") or row.get("label_space") or [])))


def _gold_key(row: dict[str, Any]) -> str:
    if "gold_label" in row:
        base = f"label:{row['gold_label']}"
    elif row["task_name"] == "acoustic_plausibility_ranking":
        base = f"rank0:{int(row['gold_ranking'][0])}"
    else:
        base = f"index:{int(row['gold_index'])}"
    balance_bucket = row.get("balance_bucket")
    if balance_bucket:
        return f"{base}|bucket:{balance_bucket}"
    return base


def _round_robin_balanced_sample(rows: list[dict[str, Any]], *, max_per_task: int, rng: random.Random) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_gold_key(row)].append(row)
    for group_rows in groups.values():
        rng.shuffle(group_rows)

    ordered_group_keys = sorted(groups.keys(), key=lambda key: (len(groups[key]), key))
    subset: list[dict[str, Any]] = []
    while len(subset) < max_per_task:
        progressed = False
        for key in ordered_group_keys:
            if groups[key]:
                subset.append(groups[key].pop())
                progressed = True
                if len(subset) >= max_per_task:
                    break
        if not progressed:
            break
    rng.shuffle(subset)
    return subset


def _random_sample(rows: list[dict[str, Any]], *, max_per_task: int, rng: random.Random) -> list[dict[str, Any]]:
    cloned = list(rows)
    rng.shuffle(cloned)
    return cloned[: min(max_per_task, len(cloned))]


def _prediction_from_index(row: dict[str, Any], predicted_index: int) -> dict[str, Any]:
    prediction = {"predicted_index": predicted_index}
    if row["task_name"] == "acoustic_plausibility_ranking":
        candidate_count = len(row.get("candidate_audio_paths") or [])
        remainder = [idx for idx in range(candidate_count) if idx != predicted_index]
        prediction["predicted_ranking"] = [predicted_index, *remainder]
    return prediction


def _majority_predicted_index(rows: list[dict[str, Any]]) -> int | str:
    counts: Counter[int | str] = Counter()
    for row in rows:
        if "gold_label" in row:
            counts[str(row["gold_label"])] += 1
        elif row["task_name"] == "acoustic_plausibility_ranking":
            counts[int(row["gold_ranking"][0])] += 1
        else:
            counts[int(row["gold_index"])] += 1
    return sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))[0][0]


def _accuracy_for_predictions(rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    metrics = [_score_prediction(row["task_name"], row, pred)["primary_metric"] for row, pred in zip(rows, predictions)]
    return sum(float(metric) for metric in metrics) / float(len(metrics))


def _compute_validation(rows: list[dict[str, Any]], *, seed: int, random_trials: int) -> dict[str, Any]:
    if not rows:
        return {
            "chance": 0.0,
            "majority_accuracy": 0.0,
            "always_first_accuracy": 0.0,
            "random_accuracy": 0.0,
            "label_distribution": {},
        }

    chance = _task_chance_from_row(rows[0])
    majority_key = _majority_predicted_index(rows)
    label_distribution = Counter(_gold_key(row) for row in rows)

    majority_predictions: list[dict[str, Any]] = []
    always_first_predictions: list[dict[str, Any]] = []
    for row in rows:
        if "gold_label" in row:
            labels = [str(value) for value in row["label_space"]]
            try:
                majority_index = labels.index(str(majority_key))
            except ValueError:
                majority_index = 0
        else:
            majority_index = int(majority_key) if isinstance(majority_key, int) else 0
        majority_predictions.append(_prediction_from_index(row, majority_index))
        always_first_predictions.append(_prediction_from_index(row, 0))

    majority_accuracy = _accuracy_for_predictions(rows, majority_predictions)
    always_first_accuracy = _accuracy_for_predictions(rows, always_first_predictions)

    random_accuracy_total = 0.0
    for offset in range(random_trials):
        rng = random.Random(seed + 1000 + offset)
        predictions: list[dict[str, Any]] = []
        for row in rows:
            if row["task_name"] == "acoustic_plausibility_ranking":
                count = len(row.get("candidate_audio_paths") or [])
            elif "candidate_audio_paths" in row:
                count = len(row.get("candidate_audio_paths") or [])
            else:
                count = len(row.get("text_options") or row.get("label_space") or [])
            predictions.append(_prediction_from_index(row, rng.randrange(count) if count else 0))
        random_accuracy_total += _accuracy_for_predictions(rows, predictions)
    random_accuracy = random_accuracy_total / float(max(random_trials, 1))

    return {
        "chance": chance,
        "majority_accuracy": majority_accuracy,
        "always_first_accuracy": always_first_accuracy,
        "random_accuracy": random_accuracy,
        "label_distribution": dict(sorted(label_distribution.items())),
    }


def _validation_passes(
    diagnostics: dict[str, Any],
    *,
    reject_majority_margin: float,
    reject_always_first_margin: float,
    reject_random_margin: float,
) -> bool:
    chance = float(diagnostics["chance"])
    return (
        float(diagnostics["majority_accuracy"]) <= chance + reject_majority_margin
        and float(diagnostics["always_first_accuracy"]) <= chance + reject_always_first_margin
        and float(diagnostics["random_accuracy"]) <= chance + reject_random_margin
    )


def _validation_violation_score(
    diagnostics: dict[str, Any],
    *,
    reject_majority_margin: float,
    reject_always_first_margin: float,
    reject_random_margin: float,
) -> float:
    chance = float(diagnostics["chance"])
    violations = [
        max(0.0, float(diagnostics["majority_accuracy"]) - (chance + reject_majority_margin)),
        max(0.0, float(diagnostics["always_first_accuracy"]) - (chance + reject_always_first_margin)),
        max(0.0, float(diagnostics["random_accuracy"]) - (chance + reject_random_margin)),
    ]
    return max(violations)


def _sample_task_rows(
    rows: list[dict[str, Any]],
    *,
    strategy: str,
    max_per_task: int,
    seed: int,
    max_attempts: int,
    random_trials: int,
    reject_majority_margin: float,
    reject_always_first_margin: float,
    reject_random_margin: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(rows) <= max_per_task:
        diagnostics = _compute_validation(rows, seed=seed, random_trials=random_trials)
        diagnostics["attempts"] = 1
        diagnostics["strategy"] = strategy
        diagnostics["validation_passed"] = _validation_passes(
            diagnostics,
            reject_majority_margin=reject_majority_margin,
            reject_always_first_margin=reject_always_first_margin,
            reject_random_margin=reject_random_margin,
        )
        return rows, diagnostics

    best_subset: list[dict[str, Any]] | None = None
    best_diag: dict[str, Any] | None = None
    best_violation = float("inf")

    for attempt in range(max_attempts):
        rng = random.Random(seed + attempt * 9973)
        if strategy == "balanced":
            subset = _round_robin_balanced_sample(rows, max_per_task=max_per_task, rng=rng)
        else:
            subset = _random_sample(rows, max_per_task=max_per_task, rng=rng)
        diagnostics = _compute_validation(subset, seed=seed + attempt, random_trials=random_trials)
        diagnostics["attempts"] = attempt + 1
        diagnostics["strategy"] = strategy
        passes = _validation_passes(
            diagnostics,
            reject_majority_margin=reject_majority_margin,
            reject_always_first_margin=reject_always_first_margin,
            reject_random_margin=reject_random_margin,
        )
        diagnostics["validation_passed"] = passes
        violation = _validation_violation_score(
            diagnostics,
            reject_majority_margin=reject_majority_margin,
            reject_always_first_margin=reject_always_first_margin,
            reject_random_margin=reject_random_margin,
        )
        if violation < best_violation:
            best_violation = violation
            best_subset = subset
            best_diag = diagnostics
        if passes:
            return subset, diagnostics

    if best_subset is None or best_diag is None:
        raise RuntimeError("Failed to sample subset")
    return best_subset, best_diag


def _sample_task_rows_adaptive(
    rows: list[dict[str, Any]],
    *,
    strategy: str,
    max_per_task: int,
    min_per_task: int,
    seed: int,
    max_attempts: int,
    random_trials: int,
    reject_majority_margin: float,
    reject_always_first_margin: float,
    reject_random_margin: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    upper = min(max_per_task, len(rows))
    if upper <= min_per_task:
        return _sample_task_rows(
            rows,
            strategy=strategy,
            max_per_task=upper,
            seed=seed,
            max_attempts=max_attempts,
            random_trials=random_trials,
            reject_majority_margin=reject_majority_margin,
            reject_always_first_margin=reject_always_first_margin,
            reject_random_margin=reject_random_margin,
        )

    candidate_sizes = set(range(min_per_task, min(upper, 64) + 1))
    candidate_sizes.update(range(80, upper + 1, 16))
    candidate_sizes.update([upper, 512, 384, 256, 192, 160, 128, 96, 80, 64, 48, 40, 32, 24, 16, min_per_task])
    ordered_sizes = sorted(size for size in candidate_sizes if min_per_task <= size <= upper)

    best_subset: list[dict[str, Any]] | None = None
    best_diag: dict[str, Any] | None = None
    best_violation = float("inf")
    for size in reversed(ordered_sizes):
        subset, diagnostics = _sample_task_rows(
            rows,
            strategy=strategy,
            max_per_task=size,
            seed=seed + (upper - size) * 7919,
            max_attempts=max_attempts,
            random_trials=random_trials,
            reject_majority_margin=reject_majority_margin,
            reject_always_first_margin=reject_always_first_margin,
            reject_random_margin=reject_random_margin,
        )
        diagnostics["adaptive_target_size"] = size
        violation = _validation_violation_score(
            diagnostics,
            reject_majority_margin=reject_majority_margin,
            reject_always_first_margin=reject_always_first_margin,
            reject_random_margin=reject_random_margin,
        )
        if violation < best_violation or (violation == best_violation and best_subset is not None and len(subset) > len(best_subset)):
            best_subset = subset
            best_diag = diagnostics
            best_violation = violation
        if diagnostics["validation_passed"]:
            diagnostics["adaptive_downsampled"] = size < upper
            return subset, diagnostics

    if best_subset is None or best_diag is None:
        raise RuntimeError("Failed adaptive sampling")
    best_diag["adaptive_downsampled"] = len(best_subset) < upper
    return best_subset, best_diag


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-per-task", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", choices=["balanced", "random"], default="balanced")
    parser.add_argument("--max-attempts", type=int, default=128)
    parser.add_argument("--random-trials", type=int, default=64)
    parser.add_argument("--reject-majority-margin", type=float, default=0.10)
    parser.add_argument("--reject-always-first-margin", type=float, default=0.10)
    parser.add_argument("--reject-random-margin", type=float, default=0.05)
    parser.add_argument("--drop-invalid-tasks", action="store_true")
    parser.add_argument("--adaptive-downsample", action="store_true")
    parser.add_argument("--min-per-task", type=int, default=8)
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_dirs = sorted(path for path in suite_dir.glob("task_*") if path.is_dir())
    counts: dict[str, int] = {}
    sample_ids: dict[str, list[str]] = {}
    task_validations: dict[str, dict[str, Any]] = {}
    dropped_tasks: list[str] = []

    for task_dir in task_dirs:
        rows = _read_jsonl(task_dir / "all.jsonl")
        if not rows:
            dropped_tasks.append(task_dir.name)
            task_validations[task_dir.name] = {
                "chance": 0.0,
                "majority_accuracy": 0.0,
                "always_first_accuracy": 0.0,
                "random_accuracy": 0.0,
                "label_distribution": {},
                "attempts": 0,
                "strategy": args.strategy,
                "validation_passed": False,
                "task_name": task_dir.name,
                "dropped": True,
                "unscored": True,
                "reason": "empty_source_task",
            }
            continue
        sampler = _sample_task_rows_adaptive if args.adaptive_downsample else _sample_task_rows
        sampler_kwargs = {
            "rows": rows,
            "strategy": args.strategy,
            "max_per_task": min(args.max_per_task, len(rows)),
            "seed": args.seed + len(task_validations),
            "max_attempts": args.max_attempts,
            "random_trials": args.random_trials,
            "reject_majority_margin": args.reject_majority_margin,
            "reject_always_first_margin": args.reject_always_first_margin,
            "reject_random_margin": args.reject_random_margin,
        }
        if args.adaptive_downsample:
            sampler_kwargs["min_per_task"] = args.min_per_task
        subset, diagnostics = sampler(**sampler_kwargs)
        task_name = subset[0]["task_name"] if subset else task_dir.name
        if not diagnostics["validation_passed"] and args.drop_invalid_tasks:
            dropped_tasks.append(task_dir.name)
            task_validations[task_dir.name] = {
                **diagnostics,
                "task_name": task_name,
                "dropped": True,
            }
            continue

        counts[task_dir.name] = len(subset)
        sample_ids[task_dir.name] = [str(row["id"]) for row in subset]
        task_validations[task_dir.name] = {
            **diagnostics,
            "task_name": task_name,
            "dropped": False,
        }
        _write_jsonl(output_dir / task_dir.name / "all.jsonl", subset)

        metadata_path = task_dir / "metadata.json"
        if metadata_path.exists():
            metadata = _read_json(metadata_path)
        else:
            metadata = {"task_name": task_name}
        metadata.update(
            {
                "count": len(subset),
                "sample_seed": args.seed,
                "sampled_from": str(task_dir / "all.jsonl"),
                "sampling_strategy": args.strategy,
                "validation": task_validations[task_dir.name],
            }
        )
        _write_json(output_dir / task_dir.name / "metadata.json", metadata)

    root_metadata = {}
    if (suite_dir / "metadata.json").exists():
        root_metadata = _read_json(suite_dir / "metadata.json")
    root_metadata.update(
        {
            "sampled_from_suite_dir": str(suite_dir),
            "sample_seed": args.seed,
            "max_per_task": args.max_per_task,
            "sampling_strategy": args.strategy,
            "counts": counts,
            "sample_ids": sample_ids,
            "task_validations": task_validations,
            "dropped_tasks": dropped_tasks,
            "validation_thresholds": {
                "majority": args.reject_majority_margin,
                "always_first": args.reject_always_first_margin,
                "random": args.reject_random_margin,
            },
            "adaptive_downsample": args.adaptive_downsample,
            "min_per_task": args.min_per_task,
        }
    )
    _write_json(output_dir / "metadata.json", root_metadata)
    print(json.dumps(root_metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
