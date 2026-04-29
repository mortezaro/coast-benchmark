import os
import argparse
from pathlib import Path

from dynsalmon.coast_hf_benchmark import TierSuite, benchmark_three_tiers


def _configure_hf_cache(hf_home: Path) -> None:
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(hf_home / "transformers")
    os.environ["HUGGINGFACE_ASSETS_CACHE"] = str(hf_home / "assets")
    os.environ["XDG_CACHE_HOME"] = str(hf_home / ".xdg")
    for path in (
        hf_home,
        hf_home / "hub",
        hf_home / "transformers",
        hf_home / "assets",
        hf_home / ".xdg",
    ):
        path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        required=True,
        choices=[
            "clap",
            "qwen2_audio",
            "ultravox",
            "spiritlm",
            "phi4mm",
            "cast_s2s",
            "tinywave_spirit",
            "moshi",
            "openai_audio",
            "gemini_audio",
        ],
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--hf-home", default=os.environ.get("HF_HOME", ""))
    parser.add_argument("--tier1-suite-dir", required=True)
    parser.add_argument("--tier2-suite-dir", required=True)
    parser.add_argument("--tier3-suite-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if args.hf_home:
        _configure_hf_cache(Path(args.hf_home))
    benchmark_three_tiers(
        backend=args.backend,
        model_id=args.model_id,
        device=args.device,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        max_new_tokens=args.max_new_tokens,
        output_root=output_root,
        tier_suites=[
            TierSuite(
                tier_name="tier1",
                suite_dir=Path(args.tier1_suite_dir),
                output_dir=output_root / "tier1",
            ),
            TierSuite(
                tier_name="tier2",
                suite_dir=Path(args.tier2_suite_dir),
                output_dir=output_root / "tier2",
            ),
            TierSuite(
                tier_name="tier3",
                suite_dir=Path(args.tier3_suite_dir),
                output_dir=output_root / "tier3",
            ),
        ],
    )


if __name__ == "__main__":
    main()
