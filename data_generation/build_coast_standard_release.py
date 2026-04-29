from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=ROOT)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _subset_command(
    *,
    suite_dir: Path,
    output_dir: Path,
    max_per_task: int,
    seed: int,
    strategy: str,
    max_attempts: int,
    random_trials: int,
    reject_majority_margin: float,
    reject_always_first_margin: float,
    reject_random_margin: float,
    drop_invalid_tasks: bool,
    adaptive_downsample: bool,
    min_per_task: int,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "data_generation" / "build_coast_eval_subset.py"),
        "--suite-dir",
        str(suite_dir),
        "--output-dir",
        str(output_dir),
        "--max-per-task",
        str(max_per_task),
        "--seed",
        str(seed),
        "--strategy",
        strategy,
        "--max-attempts",
        str(max_attempts),
        "--random-trials",
        str(random_trials),
        "--reject-majority-margin",
        str(reject_majority_margin),
        "--reject-always-first-margin",
        str(reject_always_first_margin),
        "--reject-random-margin",
        str(reject_random_margin),
    ]
    if drop_invalid_tasks:
        command.append("--drop-invalid-tasks")
    if adaptive_downsample:
        command.extend(["--adaptive-downsample", "--min-per-task", str(min_per_task)])
    return command


def _human_packet_command(*, tier_roots: list[Path], output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "data_generation" / "build_coast_human_eval_packet_curated.py"),
    ]
    for tier_root in tier_roots:
        command.extend(["--task-root", str(tier_root)])
    command.extend(["--output-root", str(output_dir)])
    return command


def _build_release_readme(
    *,
    release_name: str,
    output_root: Path,
    tier_validations: dict[str, dict[str, Any]],
    subset_targets: dict[str, int],
    include_human_packet: bool,
) -> str:
    dropped_lines = []
    for tier_name in ("tier1", "tier2", "tier3"):
        dropped = tier_validations[tier_name].get("dropped_tasks", [])
        dropped_lines.append(f"- {tier_name}: `{dropped}`")

    human_lines = []
    if include_human_packet:
        human_lines = [
            "",
            "## Human Packet",
            "",
            "- `human_eval_packet/` is a starter packet for human sanity checks and human-baseline collection.",
            "- It should be treated as a review packet, not as proof of final human validity by itself.",
        ]

    return "\n".join(
        [
            f"# {release_name}",
            "",
            "This directory is a COAST standard-release scaffold built from the validated subset pipeline.",
            "",
            "It is intended to support a larger, cleaner benchmark release rather than a tiny exploratory slice.",
            "",
            "## Standards",
            "",
            "- task-first and tier-first reporting",
            "- trivial-baseline rejection during subset construction",
            "- explicit dropping of structurally invalid tasks",
            "- human-eval packet generation for sanity checking and future human baselines",
            "- versioned release metadata and build commands",
            "",
            "## Target Subset Sizes",
            "",
            f"- tier1: `{subset_targets['tier1']}` per task",
            f"- tier2: `{subset_targets['tier2']}` per task",
            f"- tier3: `{subset_targets['tier3']}` per task",
            "",
            "## Dropped Tasks In This Build",
            "",
            *dropped_lines,
            "",
            "## Key Outputs",
            "",
            "- `validated_subsets/`",
            "- `release_manifest.json`",
            "- `build_commands.json`",
            *human_lines,
            "",
            f"- output root: `{output_root}`",
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-name", default="COAST Standard Release Candidate")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tier1-suite-dir", required=True)
    parser.add_argument("--tier2-suite-dir", required=True)
    parser.add_argument("--tier3-suite-dir", required=True)
    parser.add_argument("--tier1-max-per-task", type=int, default=128)
    parser.add_argument("--tier2-max-per-task", type=int, default=128)
    parser.add_argument("--tier3-max-per-task", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", choices=["balanced", "random"], default="balanced")
    parser.add_argument("--max-attempts", type=int, default=256)
    parser.add_argument("--random-trials", type=int, default=128)
    parser.add_argument("--reject-majority-margin", type=float, default=0.10)
    parser.add_argument("--reject-always-first-margin", type=float, default=0.10)
    parser.add_argument("--reject-random-margin", type=float, default=0.05)
    parser.add_argument("--drop-invalid-tasks", action="store_true")
    parser.add_argument("--adaptive-downsample", action="store_true")
    parser.add_argument("--min-per-task", type=int, default=16)
    parser.add_argument("--build-human-packet", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    suite_dirs = {
        "tier1": Path(args.tier1_suite_dir),
        "tier2": Path(args.tier2_suite_dir),
        "tier3": Path(args.tier3_suite_dir),
    }
    subset_targets = {
        "tier1": args.tier1_max_per_task,
        "tier2": args.tier2_max_per_task,
        "tier3": args.tier3_max_per_task,
    }

    build_commands: dict[str, str] = {}
    validated_root = output_root / "validated_subsets"
    tier_validations: dict[str, dict[str, Any]] = {}

    for offset, tier_name in enumerate(("tier1", "tier2", "tier3")):
        command = _subset_command(
            suite_dir=suite_dirs[tier_name],
            output_dir=validated_root / tier_name,
            max_per_task=subset_targets[tier_name],
            seed=args.seed + offset * 1000,
            strategy=args.strategy,
            max_attempts=args.max_attempts,
            random_trials=args.random_trials,
            reject_majority_margin=args.reject_majority_margin,
            reject_always_first_margin=args.reject_always_first_margin,
            reject_random_margin=args.reject_random_margin,
            drop_invalid_tasks=args.drop_invalid_tasks,
            adaptive_downsample=args.adaptive_downsample,
            min_per_task=args.min_per_task,
        )
        build_commands[f"{tier_name}_validated_subset"] = shlex.join(command)
        _run(command)
        tier_validations[tier_name] = _read_json(validated_root / tier_name / "metadata.json")

    human_packet_dir: Path | None = None
    if args.build_human_packet:
        human_packet_dir = output_root / "human_eval_packet"
        command = _human_packet_command(
            tier_roots=[validated_root / "tier1", validated_root / "tier2", validated_root / "tier3"],
            output_dir=human_packet_dir,
        )
        build_commands["human_eval_packet"] = shlex.join(command)
        _run(command)

    release_manifest = {
        "release_name": args.release_name,
        "output_root": str(output_root),
        "suite_dirs": {key: str(value) for key, value in suite_dirs.items()},
        "validated_subset_dirs": {
            "tier1": str(validated_root / "tier1"),
            "tier2": str(validated_root / "tier2"),
            "tier3": str(validated_root / "tier3"),
        },
        "human_packet_dir": str(human_packet_dir) if human_packet_dir else None,
        "subset_targets": subset_targets,
        "validation_thresholds": {
            "majority": args.reject_majority_margin,
            "always_first": args.reject_always_first_margin,
            "random": args.reject_random_margin,
        },
        "drop_invalid_tasks": args.drop_invalid_tasks,
        "tier_validations": tier_validations,
        "build_commands_path": str(output_root / "build_commands.json"),
    }
    _write_json(output_root / "release_manifest.json", release_manifest)
    _write_json(output_root / "build_commands.json", build_commands)

    readme = _build_release_readme(
        release_name=args.release_name,
        output_root=output_root,
        tier_validations=tier_validations,
        subset_targets=subset_targets,
        include_human_packet=args.build_human_packet,
    )
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(release_manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
