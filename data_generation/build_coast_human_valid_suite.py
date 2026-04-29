from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_generation.build_coast_human_eval_packet_curated import (  # noqa: E402
    TASK_POLICIES,
    _duration_stats,
    _passes_policy,
)


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


def _task_name_from_rows(rows: list[dict[str, Any]], fallback: str) -> str:
    for row in rows:
        if row.get("task_name"):
            return str(row["task_name"])
    return fallback


def _passes_human_policy(task_name: str, row: dict[str, Any]) -> tuple[bool, str]:
    try:
        stats = _duration_stats(row)
    except Exception as exc:
        return False, f"duration_error:{type(exc).__name__}"
    return _passes_policy(task_name, row, stats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--include-task", action="append")
    parser.add_argument("--exclude-task", action="append")
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    include_tasks = set(args.include_task or [])
    exclude_tasks = set(args.exclude_task or [])
    task_counts: dict[str, int] = {}
    audit: dict[str, dict[str, int]] = {}

    for task_dir in sorted(path for path in suite_dir.glob("task_*") if path.is_dir()):
        all_path = task_dir / "all.jsonl"
        if not all_path.exists():
            continue
        rows = _read_jsonl(all_path)
        task_name = _task_name_from_rows(rows, task_dir.name)
        if task_name not in TASK_POLICIES or not TASK_POLICIES[task_name].get("include", False):
            continue
        if include_tasks and task_name not in include_tasks:
            continue
        if task_name in exclude_tasks:
            continue

        reason_counts: Counter[str] = Counter()
        kept: list[dict[str, Any]] = []
        for row in rows:
            ok, reason = _passes_human_policy(task_name, row)
            reason_counts[reason] += 1
            if ok:
                kept.append(row)

        audit[task_name] = dict(sorted(reason_counts.items()))
        if not kept:
            continue

        out_task_dir = output_dir / task_dir.name
        _write_jsonl(out_task_dir / "all.jsonl", kept)
        metadata = _read_json(task_dir / "metadata.json") if (task_dir / "metadata.json").exists() else {}
        metadata.update(
            {
                "task_name": task_name,
                "count": len(kept),
                "human_filtered_from": str(all_path),
                "human_filter_reasons": audit[task_name],
            }
        )
        _write_json(out_task_dir / "metadata.json", metadata)
        task_counts[task_dir.name] = len(kept)

    root_metadata = _read_json(suite_dir / "metadata.json") if (suite_dir / "metadata.json").exists() else {}
    root_metadata.update(
        {
            "human_filtered_from_suite_dir": str(suite_dir),
            "counts": task_counts,
            "human_filter_audit": audit,
            "tasks": sorted(audit),
        }
    )
    _write_json(output_dir / "metadata.json", root_metadata)
    print(json.dumps(root_metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
