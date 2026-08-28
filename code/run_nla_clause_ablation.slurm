#!/bin/bash
#SBATCH --job-name=nla-clause-ablation
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
STUDY="$ROOT/pilots/wildcard-nla/clause-ablation"
PYTHON="$ROOT/.venv-nla/bin/python"
BASE="$ROOT/pilots/wildcard-nla/weights/base-qwen"
AR="$ROOT/pilots/wildcard-nla/weights/ar"
export HF_HOME="$ROOT/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$STUDY/results" "$STUDY/logs/slurm"
test -x "$PYTHON"
test -d "$BASE"; test -d "$AR"
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
"$PYTHON" "$ROOT/code/nla/run_clause_ablation.py" --stage all \
  --study-dir "$STUDY" \
  --frozen-dir "$ROOT/pilots/wildcard-nla/output-validity" \
  --results-dir "$STUDY/results" \
  --base-checkpoint "$BASE" \
  --ar-checkpoint "$AR" \
  --device cuda
