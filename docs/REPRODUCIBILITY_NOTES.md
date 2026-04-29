# Reproducibility Notes

This repository is intentionally source-code first. The raw audio/video data is not committed. The benchmark can be rebuilt when the user has access to the original sources and provides local paths to the builders.

## Expected Inputs

The builders expect local or cluster paths for:

- short acoustic-event clips
- speech-interaction clips and annotations
- long-context event clips and metadata

The code avoids hard-coding credentials. Dataset access should be handled outside the repository.

## Fixed Subset Evaluation

Model evaluation should use a fixed validated subset:

```bash
python data_generation/run_coast_hf_benchmark.py \
  --backend clap \
  --model-id laion/clap-htsat-unfused \
  --device cuda:0 \
  --output-root /path/to/model_output \
  --tier1-suite-dir /path/to/validated_subsets/tier1 \
  --tier2-suite-dir /path/to/validated_subsets/tier2 \
  --tier3-suite-dir /path/to/validated_subsets/tier3
```

Do not resample while evaluating models. Rebuild or resample tasks only during dataset construction.

## Table Generation

After model runs finish:

```bash
python data_generation/build_coast_full_benchmark_results.py \
  --benchmark-root /path/to/all_model_outputs \
  --validated-root /path/to/validated_subsets \
  --output-dir /path/to/results_tables
```

The table builder reads task metadata for baselines and model `summary.json` files for scores.

## Human Packets

Human packets should include:

- task instructions
- item manifests
- answer templates
- private answer keys
- confidence and difficulty fields
- an audio-alone answerability question where relevant

Human feedback should be treated as part of the benchmark design loop, not only as a post-hoc score.

## What Not To Commit

Do not commit:

- downloaded videos
- extracted clips
- raw audio
- cookies
- API keys
- cloud credentials
- local scratch paths
- model checkpoints
