# Predeclared NLA context-baseline study (Experiment 3)

Frozen before any Experiment 3 model output is generated. **Experiment 1** (`pilots/wildcard-nla/output-validity/`) and **Experiment 2** (`pilots/wildcard-nla/clause-ablation/`) stay frozen. This study uses a new 24-prompt inventory and a same-variant tournament; it does not reuse Experiment 1 AV text as decoys.

## Question and unit of analysis

**Question:** On fresh prompts, does full NLA prose reconstruct a content activation more specifically than token-name and local-window baselines, on both AR normalized MSE and the one-position JSD functional-reconstruction test?

The independent unit is one prompt, clustered across its two content positions (`content_early`, `content_late`). Boundary positions are excluded.

## Frozen sample

`prompts.json` is the Experiment 3 inventory:

- schema version 1, seed `20260813`, study `context-baselines`, experiment `3`;
- 24 prompts, exactly 8 each in `safety`, `compositional_planning`, and `social_character_ood`;
- SHA-256 `4fba0cbe1e4d99070788e15966c87f95a494bc3c046d3a83a3c49635daaacb6b`;
- string-disjoint from the locked Experiment 1 and multi-example-gate prompt files listed in `locked_prior_prompt_files`.

Each prompt contributes two layer-20 content activations (48 targets). Position selection reuses the Experiment 1 rendered-offset rule, but only the content strata are scored. Within each category × stratum group, both raw decoded target tokens and the whitespace-stripped display surface derived from decoded tokenizer token must be pairwise distinct.

## Same-variant texts

For every target, build five texts. Token-only and local-context baselines never consume AV `score_text`. Target token representations in baselines use the whitespace-stripped display surface derived from decoded tokenizer token; this remains a tokenizer-only baseline.

Pre-data amendment: `nla_scrubbed` changed before output generation from drop-final redaction to full-text redaction because drop-final removes the target clause.

Pre-data amendment (prompt inventory revision): The initial candidate prompt inventory failed the preflight distinct-token gate (diagnostic saved in `results/preflight-diagnostic.json`) due to function-word token collisions before any AV output was generated. The inventory was revised once from this diagnostic to vary syntactic phrasing and target distinct lexical content, updating the frozen prompt SHA-256. This is a pre-data engineering rejection and revision; the initial inventory is not confirmatory.

| Variant | Definition |
|---------|------------|
| `nla_full` | Frozen-rule `score_text` from the new AV generation |
| `token_only` | `Final token "{token}".` |
| `local_ctx` | `Final token "{token}". Local token window: "{window}".` |
| `nla_drop_final_sym` | `primary_clauses[:-1]` joined by blank lines |
| `nla_scrubbed` | `score_text` with every raw token and display surface occurrence replaced by `<TOKEN>` |

The local window is the decoded prompt-overlapping tokens within radius 5 of the target position. Protocol failure if neither raw nor display target token is present in score text, if raw or display target token remains after redaction, if a baseline equals AV text, or if any group has duplicate candidate texts.

Each category × stratum × variant inventory is an 8-way **same-variant** tournament: every candidate uses that variant, not a mix of full and shortened texts.

## Scoring

Reuse the Experiment 1 AR and one-position hook protocol on the new gold activations. For each variant, score all eight same-variant candidates plus a variant-specific `neutral_paraphrase` floor. The behavior metric remains **AR-mediated one-position functional reconstruction** (natural-log JSD after norm-matched layer-20 replacement at only the target position).

Ownership metadata never enters the AR payload. The only model-facing field is `explanation`.

## Bridge gate (mandatory)

Before interpreting Experiment 3, rescore the six frozen Experiment 1 content targets

- `ov-safety-01`, `ov-planning-01`, `ov-social-01` × `{content_early, content_late}`

using each target's frozen own `score_text` and gold activation. Compare recomputed own MSE and JSD to the frozen Experiment 1 values:

- MSE absolute delta ≤ `1e-4`;
- JSD absolute delta ≤ `1e-5`.

Write `results/gate.json`. Bridge failure makes the measurement `INVALID_MEASUREMENT`; do not loosen tolerances.

## Headline comparison and decision rule

Primary comparison: `nla_full` versus `local_ctx`, using paired tournament-margin deltas

- `ar_delta = ar_tournament_margin(nla_full) - ar_tournament_margin(local_ctx)`
- `jsd_delta = jsd_tournament_margin(nla_full) - jsd_tournament_margin(local_ctx)`

where each variant's tournament margin is `median(other 7 errors) - own error`, so a positive delta means the NLA prose margin is larger than the baseline margin.

Secondary / replication comparison: the same deltas versus `token_only`.

One-sided prompt-cluster sign-flip, seed `20260813`, `B = 10000`, statistic = median, `p = (exceedances + 1) / (B + 1)`. Clusters are the 24 prompts (two positions stay together).

`ar_sig` / `jsd_sig` require median delta `> 0` and `p ≤ 0.01`. `replicated` means the secondary comparison is significant on both metrics.

| Label | Rule |
|-------|------|
| `INVALID_MEASUREMENT` | Operational validity or bridge gate failed |
| `REPLICATION_FAILURE` | Operationally valid, but `nla_full` joint-specific rate ≤ 0.125 or ≤ 24 of 48 strong |
| `PROSE_EXCEEDS_CONTEXT` | Primary AR and JSD sign-flips both significant positive (median > 0, p ≤ 0.01), and secondary `token_only` comparison also both significant positive (median > 0, p ≤ 0.01) |
| `PROSE_REDUCES_TO_CONTEXT` | Reversed `local_ctx` vs `nla_full` sign-flips show positive local margins with p ≤ 0.01 on both AR and JSD |
| `PROSE_PARTIAL` | Otherwise |

`nla_drop_final_sym` and `nla_scrubbed` versus `local_ctx` are descriptive only. Same-variant joint-specificity rates are descriptive only. Experiment 1 classification names are not used.

## Operational validity

All of the following are required:

- preflight pass on the frozen prompt SHA-256 and 48 content extracts;
- structurally valid AV output on all 48 targets;
- gold reinjection JSD ≤ `1e-5` on every target;
- bridge gate pass;
- median random-direction JSD greater than median `nla_full` JSD;
- median unrelated-text AR MSE greater than median `nla_full` MSE.

## Pinned models and environment

Same pinned base / AV / AR revisions and `../raw/checkpoint-manifest.json` as Experiment 1. New AV generations are required because the prompt set is new.

## Execution provenance and status

Slurm job 21460276 completed with internal coherence, but was executed before run-level provenance sealing was added. Earlier failure diagnostics were overwritten pre-provenance during development and are not experimental evidence.

The upcoming sealed execution is a deterministic reproducibility rerun with identical inputs and frozen seeds under full provenance sealing. It is not a new preregistration or independent confirmation.

All future sealed executions save complete input manifests under `results/runs/job-<JOB_ID>/provenance/` and a post-decide `completion-manifest.json` inventory. The symlink `results/latest` points to the most recent successful sealed run.

## What this study is not

It is not a human semantic-faithfulness test, not an independent behavioral-validity test, and not a claim about boundary tokens. Experiment 2 remains an asymmetric full-decoy ablation and is not re-interpreted here.
