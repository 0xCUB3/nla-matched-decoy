# NLA clause ablation (Experiment 2)

Clause-deletion and malformation variants on **content** activations only, reusing frozen Experiment 1 decoys and gold artifacts. See [PROTOCOL.md](./PROTOCOL.md) for predeclared rules and predictions.

## What is frozen vs new

| Frozen (Experiment 1) | New in Experiment 2 |
|----------------------|---------------------|
| `pilots/wildcard-nla/output-validity/` prompts, AV extracts, decoy MSE/JSD, gold/unrelated behavior controls | Variant texts under `results/variants/` |
| Own `full` scores used as gate reference | Per-variant AR + behavior own scores |
| — | `neutral_paraphrase` recomputed per variant for floors |
| — | `results/gate.json`, `decision.json`, `per-target/` |

Do not modify `output-validity/` when running this study.

## Prerequisites

- Experiment 1 completed through `results/decision.json` (or at least frozen `results/ar/`, `results/behavior/`, `results/av/` for content targets).
- Local weights under `pilots/wildcard-nla/weights/` (same layout as Experiment 1).
- Python env: repo `.venv-nla` on ORCD; locally use the same deps as `code/nla/run_clause_ablation.py`.

## Run locally (CPU-friendly stages)

From repo root:

```bash
STUDY=pilots/wildcard-nla/clause-ablation
FROZEN=pilots/wildcard-nla/output-validity
PYTHON=python3  # or .venv-nla/bin/python

# Variant JSON only (no GPU)
"$PYTHON" code/nla/run_clause_ablation.py --stage variants \
  --study-dir "$STUDY" --frozen-dir "$FROZEN" \
  --results-dir "$STUDY/results"

# GPU: AR then behavior (needs base + AR checkpoints)
"$PYTHON" code/nla/run_clause_ablation.py --stage ar \
  --study-dir "$STUDY" --frozen-dir "$FROZEN" \
  --results-dir "$STUDY/results" \
  --base-checkpoint pilots/wildcard-nla/weights/base-qwen \
  --ar-checkpoint pilots/wildcard-nla/weights/ar \
  --device cuda

"$PYTHON" code/nla/run_clause_ablation.py --stage behavior \
  --study-dir "$STUDY" --frozen-dir "$FROZEN" \
  --results-dir "$STUDY/results" \
  --base-checkpoint pilots/wildcard-nla/weights/base-qwen \
  --device cuda

# CPU: gate + decision
"$PYTHON" code/nla/run_clause_ablation.py --stage decide \
  --study-dir "$STUDY" --frozen-dir "$FROZEN" \
  --results-dir "$STUDY/results"
```

Or `--stage all` after variants exist (runs full pipeline including GPU stages).

## Run on ORCD (GPU)

From repo root on a login node:

```bash
sbatch code/run_nla_clause_ablation.slurm
```

Logs: `pilots/wildcard-nla/clause-ablation/logs/slurm/`. Results: `pilots/wildcard-nla/clause-ablation/results/`.

## Figure

After `decision.json` exists:

```bash
python3 code/nla/plot_clause_ablation.py
```

Writes `figures/variant-specificity.png`.

## Tests

```bash
python3 -m unittest tests.test_nla_clause_ablation -v
```
