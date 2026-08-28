# Clause ablation results

Slurm job `21345706` on `node4204` (L40S) finished in 4m21s with exit 0. Independent recount of the 48 `per-target` files matches `decision.json`.

## Gate

`results/gate.json` status `pass` on all 48 content targets. Max |MSE| delta vs frozen Experiment 1 own scores: `0.0`. Max |JSD| delta: `0.0`.

## Headline joint specificity (n = 48 content activations)

| Variant | Joint | AR strong | Behavior strong | Bootstrap joint interval |
|---|---:|---:|---:|---|
| full | 48/48 | 48/48 | 48/48 | [1.000, 1.000] |
| drop_final | 41/48 | 47/48 | 42/48 | [0.750, 0.958] |
| final_only | 48/48 | 48/48 | 48/48 | [1.000, 1.000] |
| generic_only | 5/48 | 12/48 | 6/48 | [0.021, 0.188] |
| malformed | 48/48 | 48/48 | 48/48 | [1.000, 1.000] |

Chance for unique 8-way top-1 is 0.125. Seed `20260812`, 10000 prompt-cluster bootstraps (24 prompts × 2 content positions). These comparisons are exploratory and asymmetric because each shortened own variant competed against frozen full-length decoys from Experiment 1 rather than matched same-variant decoys.

## Predeclared predictions

- final_only retains joint specificity for most targets (rate >= 0.5). **Observed** (48/48).
- generic_only collapses (rate <= 0.25; chance is 0.125). **Observed** (5/48).
- drop_final is the discriminating unknown: no point prediction. **Observed** 41/48.
- malformed is unknown: if joint specificity survives, malformation does not explain the frozen boundary failure; if it dies, the confound stands. **Survived** 48/48. Wrapper malformation alone is not sufficient to explain the boundary failure on content text, but this content-only test does not identify a cause or test actual boundary truncation.

## drop_final misses

Seven targets lost joint specificity, six on behavior and one on AR (`ov-planning-06` early). Four were compositional planning and three were safety. Social/character stayed 16/16. Strata were 20/24 early and 21/24 late.

## What this does not show

The malformed variant wraps a *valid content* explanation. It does not repair boundary generations, and it does not score boundary activations. Wrapper malformation alone is not sufficient to explain the boundary failure, but this does not identify a cause or test actual boundary truncation.

The sealed fresh-prompt same-variant context-baseline study now provides the matched comparison; see [`../context-baselines/RESULT.md`](../context-baselines/RESULT.md).

Figure: `figures/variant-specificity.png`.
