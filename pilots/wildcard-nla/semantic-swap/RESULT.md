# Experiment 4: semantic prefix swap

Authoritative run: Slurm job `21463217` on `node4103` (L40S), 51s, exit 0. Artifacts at `results/latest/` (`runs/job-21463217/`). Bridge gate passed on all six sealed full-prose targets.

## Headline

Classification: `TOKEN_CLAUSE_DOMINATES`.

Same-variant joint specificity of swapped-prefix text: **48/48**. Strong AR 48/48. Strong behavior 48/48.

Each swapped text keeps the own final token clause and replaces the first two clauses with the next prompt in the frozen 8-way group.

## Margin versus sealed full prose

Positive delta means sealed full prose had a larger tournament margin than the swapped text.

| Channel | Median delta | Monte Carlo |
|---|---:|---|
| AR | 0.116 | no sign flip of 10,000 matched; p ≤ 1/10,001 |
| JSD | 0.027 | 1/10,000 matched; p = 2/10,001 |

Full prose had a larger AR margin on 45/48 targets and a larger JSD margin on 37/48 targets.

## Interpretation

On this model, layer, and prompt set, the final token clause is sufficient for unique same-variant identification even when the scene-setting clauses come from a different prompt. The first two clauses are not required for the joint unique-rank test. They still add a small tournament-margin advantage relative to a wrong scene.

This does not show human-semantic faithfulness. It does show that beating a tokenizer-only line or a local window is compatible with the reconstructor mostly reading the final clause.
