from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynsalmon.coast_tier3 import (
    extract_tier3_audio,
    load_ego4d_items,
    load_epic_items,
    load_lfav_items,
    write_tier3_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--lfav-annotation-root")
    parser.add_argument("--lfav-media-root")
    parser.add_argument("--epic-annotation-root")
    parser.add_argument("--epic-media-root")
    parser.add_argument("--ego4d-annotation-root")
    parser.add_argument("--ego4d-media-root")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    items = []

    if args.lfav_annotation_root and args.lfav_media_root:
        items.extend(load_lfav_items(Path(args.lfav_annotation_root), Path(args.lfav_media_root)))
    if args.epic_annotation_root and args.epic_media_root:
        items.extend(load_epic_items(Path(args.epic_annotation_root), Path(args.epic_media_root)))
    if args.ego4d_annotation_root and args.ego4d_media_root:
        items.extend(load_ego4d_items(Path(args.ego4d_annotation_root), Path(args.ego4d_media_root)))

    extracted_items, failures = extract_tier3_audio(items, output_dir / "audio", sample_rate=args.sample_rate)
    metadata = write_tier3_manifest(extracted_items, output_dir, failures)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
