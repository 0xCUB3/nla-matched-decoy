# NLA output-validity study result

## Frozen outcome

`LOCALIZED_AND_CAUSAL` with **operational validity passed**.

The predeclared confirmatory study used 24 new prompts, 3 deterministic layer-20 positions per prompt (72 activations), and exact released Qwen2.5-7B base/AV/AR checkpoints. Within each category × position group, each activation's explanation competed against seven matched decoys. The own explanation was assessed by both AR reconstruction and causal next-token JSD after norm-matched reinjection.

## Headline evidence

- AR unique/strong top-1: **57/72 = 79.2%**.
- Behavior unique top-1: **51/72 = 70.8%**.
- Strong behavior: **48/72 = 66.7%**.
- Joint-specific: **48/72 = 66.7%**; prompt-cluster-bootstrap 95% CI **[66.7%, 66.7%]**.
- Matched-assignment permutation p-values (10,000, seed `20260812`): AR **0.00009999**, behavior **0.00009999**.
- Median own AR rank `1`, behavior rank `1`; median AR margin `0.7661`, behavior margin `0.5379`.
- AR/behavior Spearman: ranks `0.5078`, margins `0.7400`.

The degenerate joint CI is real: every prompt had exactly two joint-positive content positions and one joint-negative boundary position.

## Position is the decisive boundary

| Position stratum | AR top-1 | Behavior top-1 | Joint-specific | Structural validity |
|---|---:|---:|---:|---:|
| content early | 24/24 | 24/24 | 24/24 | 24/24 |
| content late | 24/24 | 24/24 | 24/24 | 24/24 |
| after-user boundary | 9/24 | 3/24 | 0/24 | 0/24 |

Every category reproduced 8/8 joint specificity at both content positions and 0/8 at the boundary. Category-aggregated joint specificity was therefore 16/24 in safety, planning, and social/OOD alike.

All 48 content AV outputs had exactly one opening and closing wrapper and three clauses. All 24 boundary outputs had one opening tag, no closing tag, a literal `<|im_end|>` in scored text, and no token-cap failure. Structural validity and position are therefore perfectly confounded. The valid-minus-invalid difference of `+1.0` cannot identify an independent effect of wrapper quality.

## Operational controls and provenance

- Gold-reinjection maximum JSD: `0.0`.
- Median random-direction JSD `0.68985` > own-explanation JSD `0.08601`.
- Median unrelated AR MSE `1.08213` > own AR MSE `0.18388`.
- 9 exact 8×8 candidate matrices; candidate IDs/order and one-own ownership verified.
- 72 records each for extraction, AV, validity, AR, and behavior; 720 AR vectors plus 72 activations, all finite shape `(3584,)`.
- All 41 checkpoint files matched frozen size and SHA-256.
- Job `20248106`: `COMPLETED`, exit `0:0`, elapsed `00:07:23`, peak batch RSS `56,722,972K`, one requested L40S, Transformers `4.57.6`, safetensors `0.7.0`, offline staged processes.
- Independent raw-artifact recomputation reproduced every headline statistic and classification. Static audit found no ownership/category metadata entering AR or behavior scoring.

## Interpretation

> For this released Qwen2.5 NLA configuration, explanations generated from ordinary content-token activations identify their own activation among seven category- and position-matched natural decoys, and the reconstructed activation preserves a uniquely closer next-token effect. The interface fails at the after-user boundary.

This supports activation-specific localization and causal relevance for these content positions. It does **not** establish that the prose is human-semantic, globally faithful, sufficient for longer-horizon behavior, or valid at special-token boundaries. The matched-decoy assay rules out category and position alone, not all shared lexical or prompt-family cues.

## Important caveats

- Structural validity cannot be separated from position in this design.
- Boundary behavior JSDs are near saturation/ties; their rank differences are not scientifically informative.
- The causal measurement is one-step next-token JSD, not generated behavior or task success.
- Three activations from each prompt are clustered; all uncertainty and decisions use prompt clusters.
- This is one model family, one layer, and released checkpoints trained for this layer.
- `pilots/wildcard-nla/raw/environment.json` belongs to the earlier MPS pilot. This study's runtime is `environment-freeze.txt` and `gpu-environment.txt` here.

## Outcome taxonomy

- Protocol failure: not observed.
- Measurement failure: not observed; all frozen operational predicates passed.
- Positive boundary result: content-position localization/causal relevance passed exactly as predeclared.
- Negative boundary result: after-user special-token explanations were malformed and never jointly specific.
- General semantic faithfulness: not claimed.

## Artifacts

- Frozen protocol: `PROTOCOL.md`
- Inputs: `prompts.json`
- Final record: `results/decision.json`
- Raw stages: `results/{extract,av,validity,ar,ar-vectors,behavior}/`
- Runtime/provenance: `environment-freeze.txt`, `gpu-environment.txt`, `slurm-accounting.txt`, `execution-artifact-hashes.txt`
