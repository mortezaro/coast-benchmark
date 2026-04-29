# COAST Compact V5 Evaluation Instructions

This note describes how to run and refresh model evaluation tables for the COAST Human-Valid 14-Task Compact V5 release.

The repository does not include raw audio or validated subset files. Replace the placeholder paths below with the locations used in your own environment.

## Required Paths

- Benchmark output root:
  `/path/to/coast_model_outputs`
- Validated subset root:
  `/path/to/coast_humanvalid_14task_compact_v5/validated_subsets`
- Local table output:
  `/path/to/results_tables`

The validated subset root should contain:

```text
validated_subsets/
  tier1/
  tier2/
  tier3/
```

Each tier should contain task directories with `all.jsonl` and `metadata.json`.

## Run One Local/HF Model

```bash
python data_generation/run_coast_hf_benchmark.py \
  --backend clap \
  --model-id laion/clap-htsat-unfused \
  --device cuda:0 \
  --output-root /path/to/coast_model_outputs/clap_htsat \
  --tier1-suite-dir /path/to/validated_subsets/tier1 \
  --tier2-suite-dir /path/to/validated_subsets/tier2 \
  --tier3-suite-dir /path/to/validated_subsets/tier3 \
  --bootstrap-samples 1000
```

## Run On Slurm

The Slurm launchers are templates. Set environment variables rather than editing private paths into the scripts:

```bash
BENCH_ROOT=/path/to/coast_model_outputs
SUBSET_ROOT=/path/to/validated_subsets

sbatch --export=ALL,\
BENCH_ROOT=$BENCH_ROOT,\
SUBSET_ROOT=$SUBSET_ROOT,\
MAX_PER_TASK=312,\
BOOTSTRAP_SAMPLES=1000,\
CONFIDENCE_LEVEL=0.95,\
SEED=42,\
BACKEND=clap,\
MODEL_ID=laion/clap-htsat-unfused,\
MODEL_SLUG=clap_htsat,\
MAX_NEW_TOKENS=1 \
  cluster/clariden/coast_run_single_hf_benchmark.slurm
```

For API audio models, set an environment file outside the repository:

```bash
API_ENV_FILE=/path/to/coast_api_keys.env
```

The env file can define `OPENAI_API_KEY`, `GEMINI_API_KEY`, or backend-specific keys. Do not commit this file.

## Refresh Result Tables

Run this after model jobs finish:

```bash
python data_generation/build_coast_full_benchmark_results.py \
  --benchmark-root /path/to/coast_model_outputs \
  --validated-root /path/to/validated_subsets \
  --output-dir /path/to/results_tables
```

Outputs:

- `coast_compact_v5_full_results.csv`
- `coast_compact_v5_full_results.md`
- `coast_compact_v5_full_results_status.json`

## Interpret Incomplete Runs

Incomplete or failed jobs should be documented separately.

Examples:

- gated model dependency unavailable
- API request-per-day limit reached
- model checkpoint download failure
- model wrote partial tier predictions but no root `benchmark_summary.json`

These are infrastructure/model-access outcomes, not benchmark scores.

## Safety Rule

Always evaluate models against a fixed validated subset. Do not let a scoring job rebuild or resample the benchmark.
