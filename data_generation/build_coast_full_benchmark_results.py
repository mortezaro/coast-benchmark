import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


BASELINE_COLUMNS = ["chance", "random", "majority", "always_first"]


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _task_metadata(validated_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for tier_dir in sorted(path for path in validated_root.glob("tier*") if path.is_dir()):
        tier = tier_dir.name
        for task_dir in sorted(path for path in tier_dir.glob("task_*") if path.is_dir()):
            metadata_path = task_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            metadata = _read_json(metadata_path)
            validation = metadata.get("validation", {})
            rows.append(
                {
                    "tier": tier,
                    "task_dir": task_dir.name,
                    "task_name": metadata.get("task_name", task_dir.name),
                    "count": metadata.get("count"),
                    "chance": validation.get("chance"),
                    "random": validation.get("random_accuracy"),
                    "majority": validation.get("majority_accuracy"),
                    "always_first": validation.get("always_first_accuracy"),
                }
            )
    return rows


def _model_summaries(benchmark_root: Path) -> Dict[str, Dict[Tuple[str, str], Dict[str, Any]]]:
    models: Dict[str, Dict[Tuple[str, str], Dict[str, Any]]] = {}
    for model_dir in sorted(path for path in benchmark_root.iterdir() if path.is_dir()):
        model_rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for tier_dir in sorted(path for path in model_dir.glob("tier*") if path.is_dir()):
            summary_path = tier_dir / "summary.json"
            if not summary_path.exists():
                continue
            summary = _read_json(summary_path)
            for task_dir, payload in summary.get("tasks", {}).items():
                model_rows[(tier_dir.name, task_dir)] = payload
        if model_rows:
            models[model_dir.name] = model_rows
    return models


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _write_csv(
    rows: List[Dict[str, Any]],
    models: Dict[str, Dict[Tuple[str, str], Dict[str, Any]]],
    output_path: Path,
) -> None:
    model_names = sorted(models)
    fieldnames = ["tier", "task_name", "count", *BASELINE_COLUMNS, *model_names, "best_model", "best_score"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            key = (row["tier"], row["task_dir"])
            scores = {
                model: models[model].get(key, {}).get("primary_metric_mean")
                for model in model_names
            }
            valid_scores = {model: score for model, score in scores.items() if score is not None}
            best_model = max(valid_scores, key=lambda name: float(valid_scores[name])) if valid_scores else ""
            output = {
                "tier": row["tier"],
                "task_name": row["task_name"],
                "count": row["count"],
                "best_model": best_model,
                "best_score": _fmt(valid_scores.get(best_model)) if best_model else "",
            }
            for column in BASELINE_COLUMNS:
                output[column] = _fmt(row[column])
            for model, score in scores.items():
                output[model] = _fmt(score)
            writer.writerow(output)


def _write_markdown(
    rows: List[Dict[str, Any]],
    models: Dict[str, Dict[Tuple[str, str], Dict[str, Any]]],
    output_path: Path,
) -> None:
    model_names = sorted(models)
    lines: List[str] = [
        "# COAST Compact V5 Full Benchmark Results",
        "",
        "Primary metric is task-appropriate choice/top-1 accuracy. Baselines come from the validated compact subset metadata.",
        "",
    ]
    for tier in sorted({row["tier"] for row in rows}):
        lines.append(f"## {tier.title()}")
        tier_rows = [row for row in rows if row["tier"] == tier]
        header = ["Subtask", "N", *BASELINE_COLUMNS, *model_names, "Best"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in tier_rows:
            key = (row["tier"], row["task_dir"])
            scores = {
                model: models[model].get(key, {}).get("primary_metric_mean")
                for model in model_names
            }
            valid_scores = {model: score for model, score in scores.items() if score is not None}
            best_model = max(valid_scores, key=lambda name: float(valid_scores[name])) if valid_scores else ""
            best = f"{best_model} ({_fmt(valid_scores[best_model])})" if best_model else ""
            cells = [
                str(row["task_name"]),
                str(row["count"] or ""),
                *[_fmt(row[column]) for column in BASELINE_COLUMNS],
                *[_fmt(scores[model]) for model in model_names],
                best,
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument("--validated-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _task_metadata(args.validated_root)
    models = _model_summaries(args.benchmark_root)
    _write_csv(rows, models, args.output_dir / "coast_compact_v5_full_results.csv")
    _write_markdown(rows, models, args.output_dir / "coast_compact_v5_full_results.md")
    status = {
        "benchmark_root": str(args.benchmark_root),
        "validated_root": str(args.validated_root),
        "model_count": len(models),
        "models": sorted(models),
        "task_count": len(rows),
        "outputs": [
            str(args.output_dir / "coast_compact_v5_full_results.csv"),
            str(args.output_dir / "coast_compact_v5_full_results.md"),
        ],
    }
    (args.output_dir / "coast_compact_v5_full_results_status.json").write_text(
        json.dumps(status, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
