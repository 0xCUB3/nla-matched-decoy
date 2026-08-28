# Predeclared NLA clause-ablation study (Experiment 2)

Frozen before any new scoring for this confirmatory extension on 2026-08-12. **Experiment 1** (`pilots/wildcard-nla/output-validity/`) stays frozen: no edits to its prompts, AV outputs, decoy matrices, gold extracts, or decision artifacts. Experiment 2 only adds clause-deletion variants and malformation injection on **content** activations already scored in Experiment 1.

## Scientific predictions (verbatim)

- final_only retains joint specificity for most targets (rate >= 0.5).
- generic_only collapses (rate <= 0.25; chance is 0.125).
- drop_final is the discriminating unknown: no point prediction.
- malformed is unknown: if joint specificity survives, malformation does not explain the frozen boundary failure; if it dies, the confound stands.

## Question and unit of analysis

**Question:** Which parts of a structurally valid three-clause NLA explanation carry joint AR + behavior specificity on early/late **content** positions?

The independent unit remains one prompt, clustered across its two content positions (`content_early`, `content_late`). Boundary (`boundary_after_user`) activations are **excluded** from Experiment 2 scoring (48 targets = 24 prompts × 2 strata).

## Frozen inputs from Experiment 1

Reuse without recomputation or mutation:

- `prompts.json` (same 24 prompts);
- frozen AV extraction records under Experiment 1 `results/av/` (must be `structural_valid` with exactly **3** primary clauses and a final clause starting with `Final token`);
- frozen decoy AR MSE and behavior JSD vectors per target (seven decoys + ownership from the 8-way group);
- frozen gold activation reinjection and unrelated control scores where the runner loads them from Experiment 1 `results/`.

## Explanation variants (predeclared)

For each eligible content target, build five texts from the frozen `score_text` and `primary_clauses`:

| Variant | Definition |
|---------|------------|
| `full` | Frozen `score_text` (must match Experiment 1 gate). |
| `drop_final` | `primary_clauses[:-1]` joined by blank lines (`\n\n`). |
| `final_only` | Last primary clause only. |
| `generic_only` | First primary clause only. |
| `malformed` | `"<explanation>\n" + score_text + "\n<|im_end|>"` |

Protocol failure (no fallback): joined clauses must equal `score_text`; exactly three clauses; final clause must start with `Final token`.

## New scoring (per variant)

For each variant text, re-run AR normalized MSE against the **same** true activation and behavior JSD under the **same** layer-20 hook protocol as Experiment 1. Decoy scores stay frozen; only the **own** candidate changes per variant.

**Neutral paraphrase floors:** Recompute `neutral_paraphrase(variant_text)` per variant for AR and behavior floors. Do **not** reuse Experiment 1 paraphrase scores for variant-specific `strong_*` rules.

## Full-recompute gate (mandatory)

Before writing `decision.json`, recompute metrics for variant `full` and compare to Experiment 1 frozen own scores:

- MSE: absolute delta ≤ `1e-4`;
- JSD: absolute delta ≤ `1e-5`.

Fail loud if any content target fails; never loosen tolerances. Write `results/gate.json`; abort decision on gate failure.

## Decision rule (unchanged from Experiment 1)

For each variant and target, form 8-vector AR MSE and behavior JSD (index 0 = own variant, indices 1–7 = frozen decoys). Apply `_strong_metrics`:

- uniquely rank-1 own candidate on AR and on behavior;
- margin above variant-specific paraphrase floor (`max(abs(paraphrase − own), 0.001)` for AR, `1e-5` minimum for behavior).

`joint_specific` = `strong_ar and strong_behavior`. Headline rates are counts over **48** content targets per variant.

## Statistics

- Seed `20260812`;
- `10000` prompt-cluster bootstrap resamples on joint-specific indicators (24 prompts);
- Report headline and breakdown rates in `decision.json`;
- **No** Experiment-1 frozen classification names (`LOCALIZED_AND_CAUSAL`, etc.) in Experiment 2 outputs.

## Pinned models and environment

Same pinned base / AV / AR revisions and local checkpoint layout as Experiment 1 (`../raw/checkpoint-manifest.json`). GPU stage uses the same stack versions asserted in the Slurm job (e.g. `transformers==4.57.6`, `safetensors==0.7.0`, L40S).

## Operational validity

Scientific interpretation requires gate pass, finite margins on all variant×target cells, and completion of AR + behavior stages for all 48 targets.
