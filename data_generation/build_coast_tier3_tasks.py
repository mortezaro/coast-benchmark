from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynsalmon.coast_tier3 import build_tier3_tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-source-items", type=int, default=4096)
    parser.add_argument("--max-order-audio-tasks", type=int, default=1024)
    parser.add_argument("--render-workers", type=int)
    args = parser.parse_args()

    metadata = build_tier3_tasks(
        manifest_path=Path(args.manifest_path),
        output_dir=Path(args.output_dir),
        seed=args.seed,
        max_source_items=args.max_source_items,
        max_order_audio_tasks=args.max_order_audio_tasks,
        render_workers=args.render_workers,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
