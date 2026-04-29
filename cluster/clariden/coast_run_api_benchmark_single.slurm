#!/bin/bash
#SBATCH --job-name=coast_api_single
#SBATCH --partition=normal
#SBATCH --account=a0125
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

export WORKDIR=${WORKDIR:-${SLURM_SUBMIT_DIR:-$PWD}}
export COAST_ENV_ROOT=${COAST_ENV_ROOT:-$HOME/coast_env}
export COAST_MINIFORGE_ROOT=${COAST_MINIFORGE_ROOT:-$COAST_ENV_ROOT/miniforge3}
export COAST_DEP_STAMP=${COAST_DEP_STAMP:-$COAST_ENV_ROOT/deps/coast_api_benchmark_v1.stamp}
export BENCH_ROOT=${BENCH_ROOT:-$WORKDIR/coast_api_benchmark_run1}
export SUBSET_ROOT=${SUBSET_ROOT:-$BENCH_ROOT/subsets}
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$COAST_ENV_ROOT/.cache}
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HUGGINGFACE_ASSETS_CACHE=$HF_HOME/assets
export MAX_PER_TASK=${MAX_PER_TASK:-16}
export BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-1000}
export CONFIDENCE_LEVEL=${CONFIDENCE_LEVEL:-0.95}
export SEED=${SEED:-42}

export TIER1_SUITE=${TIER1_SUITE:-$WORKDIR/vggsound_tier1_people_impulse_run1/coast_grounding_release/grounding_tasks}
export TIER2_SUITE=${TIER2_SUITE:-$WORKDIR/coast_tier2_release/tasks}
export TIER3_SUITE=${TIER3_SUITE:-$WORKDIR/coast_tier3_release/tasks}

export BACKEND=${BACKEND:?Set BACKEND}
export MODEL_ID=${MODEL_ID:?Set MODEL_ID}
export MODEL_SLUG=${MODEL_SLUG:?Set MODEL_SLUG}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
export MODEL_OUTDIR=${MODEL_OUTDIR:-$BENCH_ROOT/$MODEL_SLUG}
export API_ENV_FILE=${API_ENV_FILE:-$COAST_ENV_ROOT/coast_api_keys.env}

srun bash -c '
    set -euo pipefail
    cd "$WORKDIR"

    mkdir -p "$COAST_ENV_ROOT" "$(dirname "$COAST_DEP_STAMP")" "$BENCH_ROOT" "$SUBSET_ROOT" "$MODEL_OUTDIR" "$HF_HOME"
    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$COAST_ENV_ROOT/.cache}"
    export HF_HOME="$HF_HOME"
    export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
    export HF_HUB_CACHE="$HF_HOME/hub"
    export TRANSFORMERS_CACHE="$HF_HOME/transformers"
    export HUGGINGFACE_ASSETS_CACHE="$HF_HOME/assets"
    mkdir -p "$XDG_CACHE_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HUGGINGFACE_ASSETS_CACHE"

    if [ ! -x "$COAST_MINIFORGE_ROOT/bin/python" ]; then
      echo "[COAST] Installing Miniforge into $COAST_MINIFORGE_ROOT"
      INSTALLER="$COAST_ENV_ROOT/Miniforge3-Linux-aarch64.sh"
      curl -L -o "$INSTALLER" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
      bash "$INSTALLER" -b -u -p "$COAST_MINIFORGE_ROOT"
    fi

    export PATH="$COAST_MINIFORGE_ROOT/bin:$PATH"
    export PYTHONPATH="$WORKDIR/src:${PYTHONPATH:-}"

    LOCK_DIR="$COAST_ENV_ROOT/locks"
    INSTALL_LOCK="$LOCK_DIR/coast_api_benchmark_install.lock"
    mkdir -p "$LOCK_DIR"
    if [ ! -f "$COAST_DEP_STAMP" ]; then
      while ! mkdir "$INSTALL_LOCK" 2>/dev/null; do
        sleep 5
      done
      trap "rmdir \"$INSTALL_LOCK\" 2>/dev/null || true" EXIT
      if [ ! -f "$COAST_DEP_STAMP" ]; then
        python -m pip install --upgrade pip wheel
        date -u +"%Y-%m-%dT%H:%M:%SZ" > "$COAST_DEP_STAMP"
      fi
      rmdir "$INSTALL_LOCK" 2>/dev/null || true
      trap - EXIT
    fi

    if [ ! -f "$API_ENV_FILE" ]; then
      echo "[COAST] Missing API env file: $API_ENV_FILE" >&2
      exit 1
    fi
    set -a
    source "$API_ENV_FILE"
    set +a
    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$COAST_ENV_ROOT/.cache}"
    export HF_HOME="$HF_HOME"
    export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
    export HF_HUB_CACHE="$HF_HOME/hub"
    export TRANSFORMERS_CACHE="$HF_HOME/transformers"
    export HUGGINGFACE_ASSETS_CACHE="$HF_HOME/assets"
    mkdir -p "$XDG_CACHE_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HUGGINGFACE_ASSETS_CACHE"

    if [ ! -f "$SUBSET_ROOT/tier1/metadata.json" ]; then
      python data_generation/build_coast_eval_subset.py \
        --suite-dir "$TIER1_SUITE" \
        --output-dir "$SUBSET_ROOT/tier1" \
        --max-per-task "$MAX_PER_TASK" \
        --seed "$SEED"
    fi
    if [ ! -f "$SUBSET_ROOT/tier2/metadata.json" ]; then
      python data_generation/build_coast_eval_subset.py \
        --suite-dir "$TIER2_SUITE" \
        --output-dir "$SUBSET_ROOT/tier2" \
        --max-per-task "$MAX_PER_TASK" \
        --seed "$SEED"
    fi
    if [ ! -f "$SUBSET_ROOT/tier3/metadata.json" ]; then
      python data_generation/build_coast_eval_subset.py \
        --suite-dir "$TIER3_SUITE" \
        --output-dir "$SUBSET_ROOT/tier3" \
        --max-per-task "$MAX_PER_TASK" \
        --seed "$SEED"
    fi

    python data_generation/run_coast_hf_benchmark.py \
      --backend "$BACKEND" \
      --model-id "$MODEL_ID" \
      --device "cpu" \
      --output-root "$MODEL_OUTDIR" \
      --hf-home "$HF_HOME" \
      --tier1-suite-dir "$SUBSET_ROOT/tier1" \
      --tier2-suite-dir "$SUBSET_ROOT/tier2" \
      --tier3-suite-dir "$SUBSET_ROOT/tier3" \
      --seed "$SEED" \
      --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
      --confidence-level "$CONFIDENCE_LEVEL" \
      --max-new-tokens "$MAX_NEW_TOKENS"
  '
