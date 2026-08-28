# Predeclared NLA semantic-swap study (Experiment 4)

Frozen before any Experiment 4 model output is generated. Experiments 1-3 stay frozen. This study does not generate new AV text. It reuses sealed Experiment 3 gold activations and three-clause splits.

## Question and unit of analysis

**Question:** If the scene-setting clauses are replaced with another prompt's clauses while the own final token clause is held fixed, does same-variant joint specificity collapse?

This is the inverse of Experiment 3's drop-final and scrubbed variants. Those keep semantics and remove or redact the token. This study keeps the token clause and replaces the semantics.

The independent unit is one prompt, clustered across its two content positions. Boundary positions remain excluded.

## Frozen inputs

- Sealed Experiment 3 run at `../context-baselines/results/latest/`, job `21461368`.
- Prompt inventory SHA-256 `4fba0cbe1e4d99070788e15966c87f95a494bc3c046d3a83a3c49635daaacb6b`.
- Seed `20260814`.
- Partner rule: inside each frozen 8-id category x stratum inventory, partner(i) = candidate_ids[(index + 1) mod 8]. This is a derangement.

## Swap construction

Each target has exactly three primary clauses. The swapped text is partner clause 1, then partner clause 2, then the own final clause, joined by blank lines.

Protocol failure if any record lacks three clauses, if the swap equals the original explanation, if the own final clause equals the partner final clause, or if any 8-way swapped inventory has duplicate texts.

The resulting inventory is an 8-way same-variant tournament: every candidate is a swapped-prefix text.

Ownership metadata never enters the AR payload. The only model-facing field is `explanation`.

## Scoring

Reuse the Experiment 3 AR and one-position hook protocol on the sealed gold activations. Score the eight swapped candidates plus a swap-specific neutral paraphrase floor. The behavior metric remains AR-mediated one-position functional reconstruction.

Experiment 3 full-prose tournament margins are taken from the sealed decision.json rows. They are not rescored.

## Bridge gate (mandatory)

Rescore the six Experiment 3 targets cb-safety-01, cb-planning-01, and cb-social-01 at both content positions using each target's sealed full-prose text and gold activation. Compare recomputed own MSE and JSD to the sealed Experiment 3 values:

- MSE absolute delta at most 1e-4
- JSD absolute delta at most 1e-5

Bridge failure makes the measurement INVALID_MEASUREMENT.

## Headline comparison and decision rule

Primary readout: same-variant joint-specific count of nla_swapped_prefix out of 48.

Secondary readout: paired tournament-margin deltas

- ar_delta = sealed full-prose AR tournament margin minus swapped AR tournament margin
- jsd_delta = sealed full-prose JSD tournament margin minus swapped JSD tournament margin

One-sided prompt-cluster sign-flip, seed 20260814, B = 10000, statistic = median. Clusters are the 24 prompts.

ar_sig / jsd_sig require median delta > 0 and p <= 0.01.

| Label | Rule |
|-------|------|
| INVALID_MEASUREMENT | Operational validity or bridge gate failed |
| SEMANTICS_CARRY_WEIGHT | Swapped joint-specific rate <= 0.25, and both margin deltas significant positive |
| TOKEN_CLAUSE_DOMINATES | Swapped joint-specific rate >= 0.75 |
| MIXED | Otherwise |

## What this does not claim

A collapse would show that the scene-setting clauses carry reconstruction-relevant information beyond the final token clause, on this model/layer and this prompt set. Survival would show that keeping the final token clause is enough for same-variant specificity even with a wrong scene. Neither result is human-semantic faithfulness, downstream behavior, or a boundary-position unconfound.
