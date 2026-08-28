#!/bin/bash
#SBATCH --job-name=nla-semantic-swap
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --chdir=.
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
ROOT="${NLA_ROOT:-$PWD}"
STUDY="$ROOT/pilots/wildcard-nla/semantic-swap"
SOURCE="$ROOT/pilots/wildcard-nla/context-baselines/results/latest"
PROMPTS="$ROOT/pilots/wildcard-nla/context-baselines/prompts.json"
PYTHON="$ROOT/.venv-nla/bin/python"
BASE="$ROOT/pilots/wildcard-nla/weights/base-qwen"
AR="$ROOT/pilots/wildcard-nla/weights/ar"
export HF_HOME="$ROOT/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

RUN_DIR="$STUDY/results/runs/job-${SLURM_JOB_ID:?}"
test ! -e "$RUN_DIR"
mkdir -p "$RUN_DIR/provenance" "$STUDY/logs/slurm"
cp "$STUDY/PROTOCOL.md" "$ROOT/code/nla/run_semantic_swap_study.py" "$ROOT/code/run_nla_semantic_swap.slurm" "$RUN_DIR/provenance/"
(
  cd "$RUN_DIR/provenance"
  sha256sum PROTOCOL.md run_semantic_swap_study.py run_nla_semantic_swap.slurm > input-sha256.txt
)

test -x "$PYTHON"
test -d "$BASE"
test -d "$AR"
test -f "$SOURCE/decision.json"
test -d "$SOURCE/extract"
test -d "$SOURCE/variants"
nvidia-smi
"$PYTHON" - <<'PY'
import sys, torch, transformers, numpy, safetensors
assert transformers.__version__ == "4.57.6", transformers.__version__
assert safetensors.__version__ == "0.7.0", safetensors.__version__
assert torch.cuda.is_available(), "CUDA unavailable"
assert "L40S" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
print({"python": sys.version, "torch": torch.__version__,
       "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0)})
PY
"$PYTHON" "$ROOT/code/nla/run_semantic_swap_study.py" --stage all \
  --study-dir "$STUDY" \
  --source-dir "$SOURCE" \
  --results-dir "$RUN_DIR" \
  --prompts "$PROMPTS" \
  --base-checkpoint "$BASE" \
  --ar-checkpoint "$AR" \
  --device cuda

ln -sfn "runs/job-$SLURM_JOB_ID" "$STUDY/results/latest.new"
mv -T "$STUDY/results/latest.new" "$STUDY/results/latest"
