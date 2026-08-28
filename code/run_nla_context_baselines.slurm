#!/bin/bash
#SBATCH --job-name=nla-context-baselines
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --chdir=.
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
ROOT="${NLA_ROOT:-$PWD}"
STUDY="$ROOT/pilots/wildcard-nla/context-baselines"
FROZEN="$ROOT/pilots/wildcard-nla/output-validity"
PYTHON="$ROOT/.venv-nla/bin/python"
BASE="$ROOT/pilots/wildcard-nla/weights/base-qwen"
AV="$ROOT/pilots/wildcard-nla/weights/av"
AR="$ROOT/pilots/wildcard-nla/weights/ar"
export HF_HOME="$ROOT/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

RUN_DIR="$STUDY/results/runs/job-${SLURM_JOB_ID:?}"
test ! -e "$RUN_DIR"
mkdir -p "$RUN_DIR/provenance" "$STUDY/logs/slurm"
cp "$STUDY/prompts.json" "$STUDY/PROTOCOL.md" "$ROOT/code/nla/run_context_baseline_study.py" "$ROOT/code/run_nla_context_baselines.slurm" "$RUN_DIR/provenance/"
(
  cd "$RUN_DIR/provenance"
  sha256sum prompts.json PROTOCOL.md run_context_baseline_study.py run_nla_context_baselines.slurm > input-sha256.txt
)

test -x "$PYTHON"
test -d "$BASE"; test -d "$AV"; test -d "$AR"
test -f "$STUDY/prompts.json"
test -d "$FROZEN/results/extract"
nvidia-smi
"$PYTHON" - <<'PY'
import sys, torch, transformers, numpy, safetensors, yaml
assert transformers.__version__ == "4.57.6", transformers.__version__
assert safetensors.__version__ == "0.7.0", safetensors.__version__
assert torch.cuda.is_available(), "CUDA unavailable"
assert "L40S" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
print({"python": sys.version, "torch": torch.__version__,
       "transformers": transformers.__version__, "numpy": numpy.__version__,
       "safetensors": safetensors.__version__, "gpu": torch.cuda.get_device_name(0)})
PY
"$PYTHON" "$ROOT/code/nla/run_context_baseline_study.py" --stage all \
  --study-dir "$STUDY" \
  --frozen-dir "$FROZEN" \
  --results-dir "$RUN_DIR" \
  --prompts "$STUDY/prompts.json" \
  --base-checkpoint "$BASE" \
  --av-checkpoint "$AV" \
  --ar-checkpoint "$AR" \
  --device cuda

ln -sfn "runs/job-$SLURM_JOB_ID" "$STUDY/results/latest.new" && mv -Tf "$STUDY/results/latest.new" "$STUDY/results/latest"
