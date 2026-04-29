from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynsalmon.coast_speakervid import extract_audio_bank, load_speakervid_tier2_items, write_tier2_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-root", required=True)
    parser.add_argument("--clips-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-clear-score", type=float, default=0.45)
    parser.add_argument("--min-dover-score", type=float, default=0.45)
    parser.add_argument("--require-asr", action="store_true")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--split-preference", default="all")
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    items = load_speakervid_tier2_items(
        annotation_root=Path(args.annotation_root),
        clips_root=Path(args.clips_root),
        min_clear_score=args.min_clear_score,
        min_dover_score=args.min_dover_score,
        require_asr=args.require_asr,
        max_items=args.max_items,
        split_preference=args.split_preference,
    )
    audio_map, audio_extraction_failures = extract_audio_bank(items, output_dir / "audio", sample_rate=args.sample_rate)
    metadata = write_tier2_manifest(
        items=items,
        audio_map=audio_map,
        output_dir=output_dir,
        audio_extraction_failures=audio_extraction_failures,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
