#!/bin/bash
#SBATCH --job-name=coast_hf_single
#SBATCH --partition=normal
#SBATCH --account=a0125
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

export WORKDIR=${WORKDIR:-${SLURM_SUBMIT_DIR:-$PWD}}
export COAST_ENV_ROOT=${COAST_ENV_ROOT:-$HOME/coast_env}
export COAST_MINIFORGE_ROOT=${COAST_MINIFORGE_ROOT:-$COAST_ENV_ROOT/miniforge3}
export COAST_DEP_STAMP=${COAST_DEP_STAMP:-$COAST_ENV_ROOT/deps/hf_benchmark_single_v3.stamp}
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$COAST_ENV_ROOT/.triton_cache}
export NO_TORCH_COMPILE=${NO_TORCH_COMPILE:-1}
export BENCH_ROOT=${BENCH_ROOT:-$WORKDIR/coast_model_benchmark_run2}
export SUBSET_ROOT=${SUBSET_ROOT:-$BENCH_ROOT/subsets}
export MAX_PER_TASK=${MAX_PER_TASK:-64}
export BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-1000}
export CONFIDENCE_LEVEL=${CONFIDENCE_LEVEL:-0.95}
export SEED=${SEED:-42}

export TIER1_SUITE=${TIER1_SUITE:-$WORKDIR/vggsound_tier1_people_impulse_run1/coast_grounding_release/grounding_tasks}
export TIER2_SUITE=${TIER2_SUITE:-$WORKDIR/coast_tier2_release/tasks}
export TIER3_SUITE=${TIER3_SUITE:-$WORKDIR/coast_tier3_release/tasks}

export BACKEND=${BACKEND:?Set BACKEND}
export MODEL_ID=${MODEL_ID:?Set MODEL_ID}
export MODEL_SLUG=${MODEL_SLUG:?Set MODEL_SLUG}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8}
export MODEL_OUTDIR=${MODEL_OUTDIR:-$BENCH_ROOT/$MODEL_SLUG}
export REFERENCE_DIR=${REFERENCE_DIR:-}
export SPIRITLM_CODE_ROOT=${SPIRITLM_CODE_ROOT:-$HOME/spiritlm_runtime}
export SPIRITLM_CHECKPOINTS_DIR=${SPIRITLM_CHECKPOINTS_DIR:-$SPIRITLM_CODE_ROOT/checkpoints}
export MOSHI_SOURCE_ROOT=${MOSHI_SOURCE_ROOT:-$HOME/moshi_src/moshi}
export WAVTOKENIZER_CODE_ROOT=${WAVTOKENIZER_CODE_ROOT:-$HOME/wavtokenizer_runtime/WavTokenizer}
export WAVTOKENIZER_CHECKPOINT=${WAVTOKENIZER_CHECKPOINT:-}
export WAVTOKENIZER_CONFIG=${WAVTOKENIZER_CONFIG:-}
export CAST_WAVTOKENIZER_MODEL_ID=${CAST_WAVTOKENIZER_MODEL_ID:-KrauthammerLab/cast-wavtokenizer-24k-40tps}
export TINYWAVE_TOKENIZER_VARIANT=${TINYWAVE_TOKENIZER_VARIANT:-}

srun bash -c '
    set -euo pipefail
    cd "$WORKDIR"

    mkdir -p "$COAST_ENV_ROOT" "$(dirname "$COAST_DEP_STAMP")" "$HF_HOME" "$BENCH_ROOT" "$SUBSET_ROOT" "$MODEL_OUTDIR"
    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$COAST_ENV_ROOT/.cache}"
    export HF_HOME="${HF_HOME}"
    export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
    export HF_HUB_CACHE="${HF_HOME}/hub"
    export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
    export HUGGINGFACE_ASSETS_CACHE="${HF_HOME}/assets"
    export CONDA_PKGS_DIRS=$COAST_ENV_ROOT/conda_pkgs
    export CONDARC=$COAST_ENV_ROOT/condarc
    mkdir -p "$XDG_CACHE_HOME" "$CONDA_PKGS_DIRS" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HUGGINGFACE_ASSETS_CACHE" "$TRITON_CACHE_DIR"

    if [ ! -x "$COAST_MINIFORGE_ROOT/bin/python" ]; then
      echo "[COAST] Installing Miniforge into $COAST_MINIFORGE_ROOT"
      INSTALLER="$COAST_ENV_ROOT/Miniforge3-Linux-aarch64.sh"
      curl -L -o "$INSTALLER" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
      bash "$INSTALLER" -b -u -p "$COAST_MINIFORGE_ROOT"
    fi

    export PATH="$COAST_MINIFORGE_ROOT/bin:$PATH"
    export PYTHONPATH="$WORKDIR/src:${PYTHONPATH:-}"
    export SPIRITLM_CODE_ROOT="$SPIRITLM_CODE_ROOT"
    export SPIRITLM_CHECKPOINTS_DIR="$SPIRITLM_CHECKPOINTS_DIR"
    export MOSHI_SOURCE_ROOT="$MOSHI_SOURCE_ROOT"
    export WAVTOKENIZER_CODE_ROOT="$WAVTOKENIZER_CODE_ROOT"
    export WAVTOKENIZER_CHECKPOINT="$WAVTOKENIZER_CHECKPOINT"
    export WAVTOKENIZER_CONFIG="$WAVTOKENIZER_CONFIG"
    export CAST_WAVTOKENIZER_MODEL_ID="$CAST_WAVTOKENIZER_MODEL_ID"
    export TINYWAVE_TOKENIZER_VARIANT="$TINYWAVE_TOKENIZER_VARIANT"
    export TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
    export NO_TORCH_COMPILE="$NO_TORCH_COMPILE"

    LOCK_DIR="$COAST_ENV_ROOT/locks"
    INSTALL_LOCK="$LOCK_DIR/hf_benchmark_single_install.lock"
    mkdir -p "$LOCK_DIR"
    if [ ! -f "$COAST_DEP_STAMP" ]; then
      while ! mkdir "$INSTALL_LOCK" 2>/dev/null; do
        sleep 5
      done
      trap "rmdir \"$INSTALL_LOCK\" 2>/dev/null || true" EXIT
      if [ ! -f "$COAST_DEP_STAMP" ]; then
        echo "[COAST] Installing persistent single-model benchmark deps into $COAST_MINIFORGE_ROOT"
        python -m pip install --upgrade pip wheel
        python -m pip install "setuptools<82" "transformers==4.51.3" accelerate peft librosa soundfile sentencepiece scipy backoff torchvision pillow torchaudio omegaconf local-attention encodec fairscale torchfcpe pyarrow pandas protobuf
        date -u +"%Y-%m-%dT%H:%M:%SZ" > "$COAST_DEP_STAMP"
      else
        echo "[COAST] Reusing persistent benchmark deps from $COAST_MINIFORGE_ROOT"
      fi
      rmdir "$INSTALL_LOCK" 2>/dev/null || true
      trap - EXIT
    else
      echo "[COAST] Reusing persistent benchmark deps from $COAST_MINIFORGE_ROOT"
    fi

    python -m pip install --quiet protobuf

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

    EXTRA_ARGS=()
    if [ -n "$REFERENCE_DIR" ]; then
      EXTRA_ARGS+=(--reference-predictions-dir "$REFERENCE_DIR")
    fi

    CUDA_VISIBLE_DEVICES=0 python data_generation/run_coast_hf_benchmark.py \
      --backend "$BACKEND" \
      --model-id "$MODEL_ID" \
      --device "cuda:0" \
      --output-root "$MODEL_OUTDIR" \
      --hf-home "$HF_HOME" \
      --tier1-suite-dir "$SUBSET_ROOT/tier1" \
      --tier2-suite-dir "$SUBSET_ROOT/tier2" \
      --tier3-suite-dir "$SUBSET_ROOT/tier3" \
      --seed "$SEED" \
      --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
      --confidence-level "$CONFIDENCE_LEVEL" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      "${EXTRA_ARGS[@]}"
  '
