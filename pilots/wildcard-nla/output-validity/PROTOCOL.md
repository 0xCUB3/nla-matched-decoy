# Predeclared NLA output-validity study

Frozen before any model output was generated for this confirmatory set on 2026-08-12. The prior 24-example lesion gate is a locked pilot and is excluded from confirmatory estimates. No threshold, candidate set, prompt, position, model, or decision rule below may change after AV generation begins.

## Question and unit of analysis

**Question:** Does a released NLA explanation identify its own activation more specifically than matched explanations from other natural prompts, and does that specificity survive a causal next-token intervention test?

The independent unit is one prompt, clustered across its three activation positions. Lesions or candidate explanations are never treated as independent observations.

## Frozen sample

`prompts.json` contains 24 new prompts, exactly 8 each in `safety`, `compositional_planning`, and `social_character_ood`. Each prompt contributes exactly three layer-20 activations, for 72 total:

1. `content_early`: the one-third quantile among alphanumeric-overlapping tokens inside the literal user prompt;
2. `content_late`: the two-thirds quantile among those tokens;
3. `boundary_after_user`: the first special token strictly after the user prompt span and before the final rendered-chat token.

Render the one-user chat with `apply_chat_template(..., tokenize=False, add_generation_prompt=True)`, locate the literal prompt's unique character span, then tokenize the rendered string with offsets. Tokens overlapping the prompt span and whose decoded text contains a Unicode alphanumeric character are content candidates. Quantile indices are `floor((m-1)/3)` and `floor(2*(m-1)/3)`. Boundary candidates have IDs in `tokenizer.all_special_ids` or decoded text matching `<|...|>`. Any missing, non-unique, or non-distinct position is protocol failure; no fallback is allowed.

## Pinned components

Use the exact checkpoint files and SHA-256 manifest already frozen at `../raw/checkpoint-manifest.json`:

- base: `Qwen/Qwen2.5-7B-Instruct` revision `a09a35458c702b33eeacc393d103063234e8bc28`;
- AV: `kitft/nla-qwen2.5-7b-L20-av` revision `b88469162777ae6553bc14208eb0cb579336f8f4`;
- AR: `kitft/nla-qwen2.5-7b-L20-ar` revision `e2c9e57eac213d37a31612087f645ab6332c1bb6`.

Load locally only. The true activation is base `hidden_states[21][0, position]`; behavior replaces only that position at `model.model.layers[20]`. AV uses the released sidecar template, sidecar norm scale, `inputs_embeds`, greedy generation, and 180 new-token maximum. AR uses the released sidecar and value head with no text cleanup beyond the frozen rule below.

## Frozen output-validity rules

For raw AV text, record opening/closing explanation-tag counts, order, generated-token count, EOS termination, and literal special-token strings. `wrapper_valid` requires exactly one opening and one closing tag in that order. If wrapper-valid, `score_text` is the text inside the tags; otherwise it is the entire raw generation, unchanged. Primary clauses are nonempty line units when there are at least two, otherwise sentence-boundary units, capped at eight by deterministic final merge.

`structural_valid` requires all of:

- `wrapper_valid`;
- 2 or 3 primary clauses;
- no literal `<|...|>` token inside `score_text`;
- generation did not hit the 180-token cap without EOS.

Malformed output is an observed interface outcome, not repaired or silently excluded.

## Matched-decoy specificity assay

Within each of the 9 category × position-stratum groups are 8 activations. For every target activation, the candidate set is exactly all 8 AV `score_text` outputs in its group: one own explanation and seven matched decoys. Score every candidate through AR against the target true activation and save the reconstructed vector. Also score a deterministic neutral paraphrase of the own text and the fixed unrelated sentence `A purple bicycle is parked beside a weekday calendar in a quiet room.`

For each target:

- `own_ar_rank`: ascending normalized AR MSE among 8 candidates;
- `ar_margin`: median(decoy MSE) minus own MSE;
- `ar_floor`: `max(abs(paraphrase MSE - own MSE), 0.001)`;
- `strong_ar`: own rank is uniquely 1 and `ar_margin > ar_floor`.

Reload the base model. Norm-match each reconstructed vector to the true activation, replace only the frozen target position, and compute natural-log next-token JSD from the unmodified baseline. Also run exact gold reinjection and a seeded same-norm random direction. Define:

- `own_behavior_rank`: ascending JSD among the same 8 candidates;
- `behavior_margin`: median(decoy JSD) minus own JSD;
- `behavior_floor`: `max(abs(paraphrase JSD - own JSD), 1e-5)`;
- `strong_behavior`: own rank is uniquely 1 and `behavior_margin > behavior_floor`;
- `joint_specific`: `strong_ar and strong_behavior`.

Candidate explanations remain raw/frozen; no semantic judge or generated label enters the confirmatory decision.

## Statistics

Report overall and by category, position stratum, and structural validity:

- AR, behavior, and joint top-1 rates;
- median ranks and margins;
- prompt-cluster bootstrap 95% percentile intervals using seed `20260812`, 10,000 resamples of 24 prompts;
- Spearman correlation between AR and behavior ranks/margins;
- structural-valid minus invalid joint-specific rate if both strata contain at least 12 activations (otherwise descriptive only).

For AR and behavior separately, run 10,000 fixed-seed matched-assignment permutations. Within each category × stratum group, randomly permute the one-to-one ownership mapping between its 8 targets and 8 candidate explanations; recompute the top-1 count from the already frozen score matrix. The one-sided p-value is `(1 + permutations with count >= observed)/(10001)`. No lesion-level binomial test is permitted.

## Operational validity

A scientific decision requires:

- exact prompts, 72 distinct positions, 8 targets in every category × stratum group;
- exact checkpoint-manifest verification and local-only loading;
- all extraction, AV, validity, AR, vector, and behavior records present and finite;
- gold-reinjection JSD `<=1e-5` for every activation;
- median random-direction JSD strictly greater than median own-explanation JSD;
- median unrelated AR MSE strictly greater than median own AR MSE.

Otherwise report `INVALID_MEASUREMENT`, preserving the reason.

## Frozen interpretation

- `LOCALIZED_AND_CAUSAL`: operationally valid; AR and behavior permutation p-values both `<=0.01`; the prompt-cluster-bootstrap lower 95% bound for joint-specific rate is greater than chance `0.125`; and joint-specific rate is at least `0.25` in at least two position strata.
- `RECONSTRUCTION_NOT_CAUSAL`: operationally valid; AR p `<=0.01`, but behavior p `>0.05` or the joint-specific lower bound is not above `0.125`.
- `NOT_SEMANTICALLY_LOCALIZED`: operationally valid and AR p `>0.05`.
- `MIXED_OR_UNDERPOWERED`: every other operationally valid outcome.

Structural-validity comparisons are secondary and cannot change this classification. Raw examples and boundary-token strata must remain visible in the report.

## Runtime and stopping

Use one L40S, the pinned `.venv-nla` runtime (Transformers 4.57.6, safetensors 0.7.0), 128G host memory, local-only model loading, and separate stage processes. A technical failure may receive a minimal implementation repair only if all scientific choices remain unchanged; archive every failed attempt. Do not add prompts, seeds, raters, models, thresholds, or post-hoc exclusions.
