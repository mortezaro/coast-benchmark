from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _read_first_task_name(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return str(json.loads(line)["task_name"])
    return path.parent.name


def _count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _task_dirs(suite_dir: Path) -> list[Path]:
    return sorted(path for path in suite_dir.glob("task_*") if path.is_dir())


def _find_task_dir(suite_dir: Path, task_name: str) -> Path:
    candidates = []
    for task_dir in _task_dirs(suite_dir):
        all_path = task_dir / "all.jsonl"
        if not all_path.exists():
            continue
        found_name = _read_first_task_name(all_path)
        if found_name == task_name or task_dir.name.endswith("_" + task_name):
            candidates.append(task_dir)
    if len(candidates) != 1:
        names = [path.name for path in _task_dirs(suite_dir)]
        raise ValueError(f"Expected one source for {task_name!r} in {suite_dir}, found {candidates}; available={names}")
    return candidates[0]


def _copy_task(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "all.jsonl", dst / "all.jsonl")
    if (src / "metadata.json").exists():
        metadata = _read_json(src / "metadata.json")
    else:
        metadata = {}
    count = _count_jsonl(src / "all.jsonl")
    metadata.update({"selected_from": str(src), "count": count})
    _write_json(dst / "metadata.json", metadata)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble a selected COAST task suite from one or more existing suites.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Source spec in the form /path/to/suite:task_a,task_b. Repeatable.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected: list[tuple[Path, str]] = []
    for source_spec in args.source:
        if ":" not in source_spec:
            raise ValueError(f"Invalid --source spec {source_spec!r}; expected /path:task_a,task_b")
        source_dir_text, task_names_text = source_spec.split(":", 1)
        source_dir = Path(source_dir_text)
        for task_name in [item.strip() for item in task_names_text.split(",") if item.strip()]:
            selected.append((source_dir, task_name))

    counts: dict[str, int] = {}
    source_dirs: dict[str, str] = {}
    tasks: list[str] = []
    for index, (source_dir, task_name) in enumerate(selected, start=1):
        src = _find_task_dir(source_dir, task_name)
        dst = output_dir / f"task_{index:02d}_{task_name}"
        counts[dst.name] = _copy_task(src, dst)
        source_dirs[dst.name] = str(src)
        tasks.append(task_name)

    metadata = {
        "suite_type": "selected_coast_task_suite",
        "output_dir": str(output_dir),
        "tasks": tasks,
        "counts": counts,
        "source_dirs": source_dirs,
    }
    _write_json(output_dir / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
