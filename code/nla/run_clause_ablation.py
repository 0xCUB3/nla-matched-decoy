#!/usr/bin/env python3
"""Clause-deletion ablation + malformation injection (Experiment 2).

Reuses frozen NLA output-validity artifacts for gold activations, decoy scores,
and the full-explanation gate. Does not modify Experiment 1.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import run_output_validity_study as ov

StudyError = ov.StudyError
SEED = ov.SEED
LAYER_INDEX = ov.LAYER_INDEX
CATEGORIES = ov.CATEGORIES
AR_REPO = ov.AR_REPO
AR_REVISION = ov.AR_REVISION
BASE_REPO = ov.BASE_REPO
BASE_REVISION = ov.BASE_REVISION
N_PERMUTATIONS = ov.N_PERMUTATIONS
CHANCE_TOP1 = ov.CHANCE_TOP1
UNRELATED_TEXT = ov.UNRELATED_TEXT
json_dump = ov.json_dump
set_determinism = ov.set_determinism
load_prompt_spec = ov.load_prompt_spec
finite = ov.finite
neutral_paraphrase = ov.neutral_paraphrase
_target_stem = ov._target_stem
_load_extraction_record = ov._load_extraction_record
_load_tokenizer_model = ov._load_tokenizer_model
_load_ar_head = ov._load_ar_head
_load_yaml = ov._load_yaml
_score_ar_text = ov._score_ar_text
normalize_vector = ov.normalize_vector
js_divergence = ov.js_divergence
hooked_logits = ov.hooked_logits
_base_logits = ov._base_logits
_strong_metrics = ov._strong_metrics
prompt_cluster_bootstrap = ov.prompt_cluster_bootstrap
_require_json = ov._require_json

DEFAULT_STUDY_DIR = Path(__file__).resolve().parents[2] / "pilots/wildcard-nla/clause-ablation"
DEFAULT_FROZEN_DIR = Path(__file__).resolve().parents[2] / "pilots/wildcard-nla/output-validity"

CONTENT_STRATA = ("content_early", "content_late")
VARIANT_NAMES = ("full", "drop_final", "final_only", "generic_only", "malformed")
N_CONTENT_TARGETS = 24 * len(CONTENT_STRATA)
GATE_MSE_TOL = 1e-4
GATE_JSD_TOL = 1e-5
MALFORMED_SUFFIX = "\n<|im_end|>"


def build_variant_texts(primary_clauses: Sequence[str], score_text: str) -> dict[str, str]:
    """Build the five predeclared explanation variants from frozen clauses."""
    joined = "\n\n".join(primary_clauses)
    if joined != score_text:
        raise StudyError("primary_clauses join does not match frozen score_text")
    if len(primary_clauses) != 3:
        raise StudyError(f"expected 3 primary clauses, found {len(primary_clauses)}")
    if not primary_clauses[-1].startswith("Final token"):
        raise StudyError("final clause must start with 'Final token'")
    return {
        "full": score_text,
        "drop_final": "\n\n".join(primary_clauses[:-1]),
        "final_only": primary_clauses[-1],
        "generic_only": primary_clauses[0],
        "malformed": f"<explanation>\n{score_text}{MALFORMED_SUFFIX}",
    }


def validate_content_validity(record: Mapping[str, Any], stem: str) -> None:
    if record.get("structural_valid") is not True:
        raise StudyError(f"content target not structural_valid: {stem}")
    if record.get("primary_clause_count") != 3:
        raise StudyError(f"content target primary_clause_count != 3: {stem}")
    clauses = record.get("primary_clauses")
    if not isinstance(clauses, list) or len(clauses) != 3:
        raise StudyError(f"content target missing 3 primary_clauses: {stem}")
    if not str(clauses[-1]).startswith("Final token"):
        raise StudyError(f"final clause must start with 'Final token': {stem}")


def compare_gate_values(
    recomputed_mse: float,
    frozen_mse: float,
    recomputed_jsd: float,
    frozen_jsd: float,
    *,
    mse_tol: float = GATE_MSE_TOL,
    jsd_tol: float = GATE_JSD_TOL,
) -> dict[str, Any]:
    mse_delta = abs(float(recomputed_mse) - float(frozen_mse))
    jsd_delta = abs(float(recomputed_jsd) - float(frozen_jsd))
    passed = mse_delta <= mse_tol and jsd_delta <= jsd_tol
    return {
        "passed": passed,
        "recomputed_mse": float(recomputed_mse),
        "frozen_mse": float(frozen_mse),
        "mse_delta": mse_delta,
        "mse_tol": mse_tol,
        "recomputed_jsd": float(recomputed_jsd),
        "frozen_jsd": float(frozen_jsd),
        "jsd_delta": jsd_delta,
        "jsd_tol": jsd_tol,
    }


def _iter_content_targets(prompts: Sequence[Mapping[str, Any]]):
    for prompt in prompts:
        for stratum in CONTENT_STRATA:
            yield prompt, stratum


def _frozen_results(args: argparse.Namespace) -> Path:
    return args.frozen_dir / "results"


def _load_frozen_validity(frozen_results: Path, prompt_id: str, stratum: str) -> dict[str, Any]:
    path = frozen_results / "validity" / f"{_target_stem(prompt_id, stratum)}.json"
    return _require_json(path)


def _load_frozen_ar(frozen_results: Path, prompt_id: str, stratum: str) -> dict[str, Any]:
    return _require_json(frozen_results / "ar" / f"{_target_stem(prompt_id, stratum)}.json")


def _load_frozen_behavior(frozen_results: Path, prompt_id: str, stratum: str) -> dict[str, Any]:
    return _require_json(frozen_results / "behavior" / f"{_target_stem(prompt_id, stratum)}.json")


def _frozen_decoys(ar_record: Mapping[str, Any]) -> list[dict[str, Any]]:
    decoys = [row for row in ar_record["candidates"] if not row.get("own")]
    if len(decoys) != 7:
        raise StudyError("expected exactly 7 non-own decoys in frozen AR record")
    return decoys


def _decoy_mse_list(decoys: Sequence[Mapping[str, Any]]) -> list[float]:
    return [float(row["mse_nrm"]) for row in decoys]


def _decoy_jsd_list(frozen_behavior: Mapping[str, Any], decoy_ids: Sequence[str]) -> list[float]:
    by_id = {row["candidate_id"]: float(row["jsd"]) for row in frozen_behavior["candidates"]}
    return [by_id[cid] for cid in decoy_ids]


def _frozen_own_scores(ar_record: Mapping[str, Any], behavior_record: Mapping[str, Any]) -> tuple[float, float]:
    own_ar = next(row for row in ar_record["candidates"] if row.get("own"))
    own_bh = next(row for row in behavior_record["candidates"] if row.get("own"))
    return float(own_ar["mse_nrm"]), float(own_bh["jsd"])


def stage_variants(args: argparse.Namespace) -> None:
    prompts = load_prompt_spec(args.prompts)
    frozen_results = _frozen_results(args)
    out = args.results_dir / "variants"
    out.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for prompt, stratum in _iter_content_targets(prompts):
        stem = _target_stem(prompt["id"], stratum)
        validity = _load_frozen_validity(frozen_results, prompt["id"], stratum)
        validate_content_validity(validity, stem)
        clauses = validity["primary_clauses"]
        score_text = validity["score_text"]
        texts = build_variant_texts(clauses, score_text)
        record = {
            "status": "pass",
            "activation_id": validity["activation_id"],
            "prompt_id": prompt["id"],
            "prompt_index": int(prompt["index"]),
            "category": prompt["category"],
            "position_stratum": stratum,
            "position": validity.get("position"),
            "layer_index": LAYER_INDEX,
            "seed": SEED,
            "primary_clauses": list(clauses),
            "score_text": score_text,
            "variant_texts": texts,
            "frozen_validity_path": str(frozen_results / "validity" / f"{stem}.json"),
            "record_path": str(out / f"{stem}.json"),
        }
        json_dump(out / f"{stem}.json", record)
        manifest_rows.append({"stem": stem, "activation_id": record["activation_id"], "record_path": record["record_path"]})
    if len(manifest_rows) != N_CONTENT_TARGETS:
        raise StudyError(f"expected {N_CONTENT_TARGETS} content variant records, found {len(manifest_rows)}")
    json_dump(out / "manifest.json", {"status": "pass", "seed": SEED, "count": len(manifest_rows), "records": manifest_rows})


def stage_ar(args: argparse.Namespace) -> None:
    import torch

    set_determinism()
    prompts = load_prompt_spec(args.prompts)
    frozen_results = _frozen_results(args)
    tokenizer, model, device, dtype = _load_tokenizer_model(args.ar_checkpoint, args.device, AR_REVISION)
    head = _load_ar_head(args.ar_checkpoint, model, dtype, device)
    sidecar = _load_yaml(args.ar_checkpoint / "nla_meta.yaml")
    template, mse_scale = str(sidecar["prompt_templates"]["ar"]), float(sidecar["extraction"]["mse_scale"])
    out = args.results_dir / "ar"
    vectors = args.results_dir / "ar-vectors"
    out.mkdir(parents=True, exist_ok=True)
    vectors.mkdir(parents=True, exist_ok=True)
    for prompt, stratum in _iter_content_targets(prompts):
        stem = _target_stem(prompt["id"], stratum)
        variant_spec = _require_json(args.results_dir / "variants" / f"{stem}.json")
        target_meta, gold = _load_extraction_record(frozen_results, prompt["id"], stratum)
        frozen_ar = _load_frozen_ar(frozen_results, prompt["id"], stratum)
        decoys = _frozen_decoys(frozen_ar)
        gold_n = normalize_vector(gold, mse_scale)
        variant_rows: dict[str, Any] = {}
        for name in VARIANT_NAMES:
            text = variant_spec["variant_texts"][name]
            paraphrase = neutral_paraphrase(text)
            own_vector, own_tokens = _score_ar_text(tokenizer, model, head, device, template, text)
            own_n = normalize_vector(own_vector, mse_scale)
            own_mse = float(((own_n - gold_n) ** 2).mean().item())
            own_path = vectors / f"{stem}__{name}__own.pt"
            torch.save(own_vector, own_path)
            para_vector, para_tokens = _score_ar_text(tokenizer, model, head, device, template, paraphrase)
            para_n = normalize_vector(para_vector, mse_scale)
            para_mse = float(((para_n - gold_n) ** 2).mean().item())
            para_path = vectors / f"{stem}__{name}__paraphrase.pt"
            torch.save(para_vector, para_path)
            variant_rows[name] = {
                "text": text,
                "paraphrase_text": paraphrase,
                "own_mse_nrm": own_mse,
                "paraphrase_mse_nrm": para_mse,
                "own_token_count": own_tokens,
                "paraphrase_token_count": para_tokens,
                "own_vector_path": str(own_path),
                "paraphrase_vector_path": str(para_path),
                "finite": finite(own_mse) and finite(para_mse),
            }
        frozen_unrelated = next(
            (row for row in frozen_ar.get("controls", []) if row.get("candidate_source") == "unrelated"),
            None,
        )
        json_dump(
            out / f"{stem}.json",
            {
                "status": "pass",
                "activation_id": target_meta["activation_id"],
                "prompt_id": prompt["id"],
                "prompt_index": int(prompt["index"]),
                "category": prompt["category"],
                "position_stratum": stratum,
                "position": target_meta["position"],
                "layer_index": LAYER_INDEX,
                "seed": SEED,
                "checkpoint": {"repo": AR_REPO, "revision": AR_REVISION},
                "mse_scale": mse_scale,
                "decoys": [
                    {"candidate_id": row["candidate_id"], "mse_nrm": float(row["mse_nrm"])} for row in decoys
                ],
                "variants": variant_rows,
                "frozen_unrelated_mse_nrm": (
                    float(frozen_unrelated["mse_nrm"]) if isinstance(frozen_unrelated, Mapping) else None
                ),
                "finite": all(row["finite"] for row in variant_rows.values()),
                "record_path": str(out / f"{stem}.json"),
            },
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def stage_behavior(args: argparse.Namespace) -> None:
    import torch

    set_determinism()
    prompts = load_prompt_spec(args.prompts)
    frozen_results = _frozen_results(args)
    tokenizer, model, device, dtype = _load_tokenizer_model(args.base_checkpoint, args.device, BASE_REVISION)
    out = args.results_dir / "behavior"
    out.mkdir(parents=True, exist_ok=True)
    for prompt, stratum in _iter_content_targets(prompts):
        stem = _target_stem(prompt["id"], stratum)
        ar_record = _require_json(args.results_dir / "ar" / f"{stem}.json")
        frozen_behavior = _load_frozen_behavior(frozen_results, prompt["id"], stratum)
        target_meta, gold = _load_extraction_record(frozen_results, prompt["id"], stratum)
        alignment = target_meta.get("alignment")
        if not isinstance(alignment, Mapping):
            raise StudyError(f"missing rendered alignment for {target_meta['activation_id']}")
        ids = torch.tensor([alignment["token_ids"]], device=device)
        mask = torch.ones_like(ids)
        position = int(target_meta["position"])
        baseline = _base_logits(model, ids, mask)
        norm = float(gold.norm().item())
        decoy_ids = [row["candidate_id"] for row in ar_record["decoys"]]
        decoy_jsds = _decoy_jsd_list(frozen_behavior, decoy_ids)
        variant_rows: dict[str, Any] = {}
        for name in VARIANT_NAMES:
            spec = ar_record["variants"][name]
            own_vector = torch.load(spec["own_vector_path"], map_location="cpu", weights_only=True).float()
            para_vector = torch.load(spec["paraphrase_vector_path"], map_location="cpu", weights_only=True).float()
            own_replacement = own_vector / own_vector.norm().clamp_min(1e-12) * norm
            own_logits = hooked_logits(model, ids, mask, position, own_replacement.to(device))
            own_jsd = js_divergence(baseline[position], own_logits[position])
            para_replacement = para_vector / para_vector.norm().clamp_min(1e-12) * norm
            para_logits = hooked_logits(model, ids, mask, position, para_replacement.to(device))
            para_jsd = js_divergence(baseline[position], para_logits[position])
            variant_rows[name] = {
                "own_jsd": own_jsd,
                "paraphrase_jsd": para_jsd,
                "finite": finite(own_jsd) and finite(para_jsd),
            }
        json_dump(
            out / f"{stem}.json",
            {
                "status": "pass",
                "activation_id": target_meta["activation_id"],
                "prompt_id": prompt["id"],
                "prompt_index": int(prompt["index"]),
                "category": prompt["category"],
                "position_stratum": stratum,
                "position": position,
                "layer_index": LAYER_INDEX,
                "seed": SEED,
                "checkpoint": {"repo": BASE_REPO, "revision": BASE_REVISION},
                "decoys": [{"candidate_id": cid, "jsd": jsd} for cid, jsd in zip(decoy_ids, decoy_jsds)],
                "variants": variant_rows,
                "finite": all(row["finite"] for row in variant_rows.values()),
                "record_path": str(out / f"{stem}.json"),
            },
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _variant_metrics(
    own_mse: float,
    own_jsd: float,
    decoy_mse: Sequence[float],
    decoy_jsd: Sequence[float],
    paraphrase_mse: float,
    paraphrase_jsd: float,
) -> dict[str, Any]:
    ar_vector = [float(own_mse)] + [float(x) for x in decoy_mse]
    behavior_vector = [float(own_jsd)] + [float(x) for x in decoy_jsd]
    return _strong_metrics(ar_vector, behavior_vector, 0, float(paraphrase_mse), float(paraphrase_jsd))


def _rate(records: Sequence[Mapping[str, Any]], key: str = "joint_specific") -> float:
    if not records:
        return float("nan")
    return sum(1 for row in records if row.get(key)) / len(records)


def _run_full_gate(args: argparse.Namespace, prompts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    frozen_results = _frozen_results(args)
    rows = []
    all_pass = True
    for prompt, stratum in _iter_content_targets(prompts):
        stem = _target_stem(prompt["id"], stratum)
        ar_record = _require_json(args.results_dir / "ar" / f"{stem}.json")
        behavior_record = _require_json(args.results_dir / "behavior" / f"{stem}.json")
        frozen_ar = _load_frozen_ar(frozen_results, prompt["id"], stratum)
        frozen_behavior = _load_frozen_behavior(frozen_results, prompt["id"], stratum)
        frozen_mse, frozen_jsd = _frozen_own_scores(frozen_ar, frozen_behavior)
        full_ar = ar_record["variants"]["full"]
        full_bh = behavior_record["variants"]["full"]
        comparison = compare_gate_values(
            full_ar["own_mse_nrm"],
            frozen_mse,
            full_bh["own_jsd"],
            frozen_jsd,
        )
        if not comparison["passed"]:
            all_pass = False
        rows.append(
            {
                "stem": stem,
                "activation_id": ar_record["activation_id"],
                **comparison,
            }
        )
    return {
        "status": "pass" if all_pass else "fail",
        "target_count": len(rows),
        "mse_tol": GATE_MSE_TOL,
        "jsd_tol": GATE_JSD_TOL,
        "targets": rows,
    }


def stage_decide(args: argparse.Namespace) -> dict[str, Any] | None:
    prompts = load_prompt_spec(args.prompts)
    frozen_results = _frozen_results(args)
    per_target_dir = args.results_dir / "per-target"
    per_target_dir.mkdir(parents=True, exist_ok=True)

    gate = _run_full_gate(args, prompts)
    json_dump(args.results_dir / "gate.json", gate)
    if gate["status"] != "pass":
        raise StudyError("full-recompute gate failed; decision.json not written")

    flat_rows: list[dict[str, Any]] = []
    for prompt, stratum in _iter_content_targets(prompts):
        stem = _target_stem(prompt["id"], stratum)
        ar_record = _require_json(args.results_dir / "ar" / f"{stem}.json")
        behavior_record = _require_json(args.results_dir / "behavior" / f"{stem}.json")
        decoy_mse = [float(row["mse_nrm"]) for row in ar_record["decoys"]]
        decoy_jsd = [float(row["jsd"]) for row in behavior_record["decoys"]]
        variant_metrics: dict[str, Any] = {}
        for name in VARIANT_NAMES:
            ar_v = ar_record["variants"][name]
            bh_v = behavior_record["variants"][name]
            metrics = _variant_metrics(
                ar_v["own_mse_nrm"],
                bh_v["own_jsd"],
                decoy_mse,
                decoy_jsd,
                ar_v["paraphrase_mse_nrm"],
                bh_v["paraphrase_jsd"],
            )
            variant_metrics[name] = metrics
            flat_rows.append(
                {
                    "stem": stem,
                    "activation_id": ar_record["activation_id"],
                    "prompt_id": prompt["id"],
                    "prompt_index": int(prompt["index"]),
                    "category": prompt["category"],
                    "position_stratum": stratum,
                    "variant": name,
                    "joint_specific": bool(metrics["joint_specific"]),
                    "strong_ar": bool(metrics["strong_ar"]),
                    "strong_behavior": bool(metrics["strong_behavior"]),
                    "own_ar_rank": metrics["own_ar_rank"],
                    "own_behavior_rank": metrics["own_behavior_rank"],
                    "ar_margin": metrics["ar_margin"],
                    "behavior_margin": metrics["behavior_margin"],
                    "ar_floor": metrics["ar_floor"],
                    "behavior_floor": metrics["behavior_floor"],
                }
            )
        target_record = {
            "status": "pass",
            "stem": stem,
            "activation_id": ar_record["activation_id"],
            "prompt_id": prompt["id"],
            "category": prompt["category"],
            "position_stratum": stratum,
            "variants": variant_metrics,
            "frozen_unrelated_mse_nrm": ar_record.get("frozen_unrelated_mse_nrm"),
            "record_path": str(per_target_dir / f"{stem}.json"),
        }
        json_dump(per_target_dir / f"{stem}.json", target_record)

    def rows_for_variant(name: str) -> list[dict[str, Any]]:
        return [row for row in flat_rows if row["variant"] == name]

    def bootstrap_joint(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return prompt_cluster_bootstrap(
            [{"prompt_id": row["prompt_id"], "joint_specific": float(row["joint_specific"])} for row in records],
            "joint_specific",
            seed=SEED,
            n_resamples=N_PERMUTATIONS,
        )

    breakdown: dict[str, Any] = {}
    headline: dict[str, Any] = {}
    for name in VARIANT_NAMES:
        subset = rows_for_variant(name)
        headline[name] = {
            "n": len(subset),
            "joint_specific_count": sum(1 for row in subset if row["joint_specific"]),
            "joint_specific_rate": _rate(subset),
            "bootstrap_joint_specific": bootstrap_joint(subset),
        }
        breakdown[f"variant::{name}"] = headline[name]
        for category in CATEGORIES:
            cat_rows = [row for row in subset if row["category"] == category]
            breakdown[f"variant::{name}::category::{category}"] = {
                "n": len(cat_rows),
                "joint_specific_count": sum(1 for row in cat_rows if row["joint_specific"]),
                "joint_specific_rate": _rate(cat_rows),
            }
        for stratum in CONTENT_STRATA:
            stratum_rows = [row for row in subset if row["position_stratum"] == stratum]
            breakdown[f"variant::{name}::position_stratum::{stratum}"] = {
                "n": len(stratum_rows),
                "joint_specific_count": sum(1 for row in stratum_rows if row["joint_specific"]),
                "joint_specific_rate": _rate(stratum_rows),
            }

    predictions = {
        "final_only_retains": "final_only retains joint specificity for most targets (rate >= 0.5).",
        "generic_only_collapses": "generic_only collapses (rate <= 0.25; chance is 0.125).",
        "drop_final_unknown": "drop_final is the discriminating unknown: no point prediction.",
        "malformed_unknown": (
            "malformed is unknown: if joint specificity survives, malformation does not explain the "
            "frozen boundary failure; if it dies, the confound stands."
        ),
    }

    result = {
        "status": "pass",
        "study": "clause-ablation",
        "experiment": 2,
        "seed": SEED,
        "target_count": N_CONTENT_TARGETS,
        "variants": list(VARIANT_NAMES),
        "gate_path": str(args.results_dir / "gate.json"),
        "gate_status": gate["status"],
        "operational_validity": {
            "gate_pass": gate["status"] == "pass",
            "content_targets": N_CONTENT_TARGETS,
            "finite_scores": all(finite(row["ar_margin"]) and finite(row["behavior_margin"]) for row in flat_rows),
            "gold_extract_from_frozen": True,
        },
        "predictions": predictions,
        "headline": headline,
        "breakdown": breakdown,
        "per_target_dir": str(per_target_dir),
        "chance_joint_top1": CHANCE_TOP1,
        "frozen_study_dir": str(args.frozen_dir),
        "config": {
            "layer_index": LAYER_INDEX,
            "bootstrap_seed": SEED,
            "bootstrap_resamples": N_PERMUTATIONS,
            "prompt_clusters": 24,
            "positions_per_prompt": len(CONTENT_STRATA),
        },
    }
    json_dump(args.results_dir / "decision.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="NLA clause-deletion ablation (Experiment 2)")
    ap.add_argument("--stage", choices=("variants", "ar", "behavior", "decide", "all"), required=True)
    ap.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    ap.add_argument("--frozen-dir", type=Path, default=DEFAULT_FROZEN_DIR)
    ap.add_argument("--results-dir", type=Path)
    ap.add_argument("--prompts", type=Path)
    ap.add_argument("--base-checkpoint", type=Path)
    ap.add_argument("--ar-checkpoint", type=Path)
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = ap.parse_args(argv)
    args.study_dir = args.study_dir.resolve()
    args.frozen_dir = args.frozen_dir.resolve()
    args.results_dir = (args.results_dir or args.study_dir / "results").resolve()
    args.prompts = (args.prompts or args.frozen_dir / "prompts.json").resolve()
    if not args.prompts.is_file():
        raise StudyError(f"missing frozen prompts: {args.prompts}")
    frozen_validity = args.frozen_dir / "results" / "validity" / "manifest.json"
    if not frozen_validity.is_file():
        raise StudyError(f"missing frozen validity manifest: {frozen_validity}")
    weights = args.study_dir.parent / "weights"
    args.base_checkpoint = (args.base_checkpoint or weights / "base-qwen").resolve()
    args.ar_checkpoint = (args.ar_checkpoint or weights / "ar").resolve()
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    stages = {
        "variants": stage_variants,
        "ar": stage_ar,
        "behavior": stage_behavior,
        "decide": stage_decide,
    }
    if args.stage == "all":
        for stage in ("variants", "ar", "behavior", "decide"):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--stage",
                stage,
                "--study-dir",
                str(args.study_dir),
                "--frozen-dir",
                str(args.frozen_dir),
                "--results-dir",
                str(args.results_dir),
                "--prompts",
                str(args.prompts),
                "--base-checkpoint",
                str(args.base_checkpoint),
                "--ar-checkpoint",
                str(args.ar_checkpoint),
                "--device",
                args.device,
            ]
            subprocess.run(command, check=True)
        return
    stages[args.stage](args)


if __name__ == "__main__":
    main()
