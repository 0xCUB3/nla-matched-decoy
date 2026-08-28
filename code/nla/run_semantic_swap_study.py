#!/usr/bin/env python3
"""Semantic-swap follow-up (Experiment 4) on sealed Experiment 3 artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import run_context_baseline_study as cb
import run_output_validity_study as ov

StudyError = cb.StudyError
SEED = 20260814
VARIANT = "nla_swapped_prefix"
N_PERMUTATIONS = cb.N_PERMUTATIONS
GATE_MSE_TOL = cb.GATE_MSE_TOL
GATE_JSD_TOL = cb.GATE_JSD_TOL
GOLD_JSD_MAX = cb.GOLD_JSD_MAX
FROZEN_PROMPT_SHA256 = cb.FROZEN_PROMPT_SHA256
BEHAVIOR_METRIC_NAME = cb.BEHAVIOR_METRIC_NAME
DEFAULT_STUDY_DIR = Path(__file__).resolve().parents[2] / "pilots/wildcard-nla/semantic-swap"
DEFAULT_SOURCE_DIR = Path(__file__).resolve().parents[2] / "pilots/wildcard-nla/context-baselines/results/latest"
DEFAULT_PROMPT_PATH = Path(__file__).resolve().parents[2] / "pilots/wildcard-nla/context-baselines/prompts.json"
BRIDGE_TARGETS = (
    ("cb-safety-01", "content_early"),
    ("cb-safety-01", "content_late"),
    ("cb-planning-01", "content_early"),
    ("cb-planning-01", "content_late"),
    ("cb-social-01", "content_early"),
    ("cb-social-01", "content_late"),
)
DECISION_LABELS = (
    "SEMANTICS_CARRY_WEIGHT",
    "TOKEN_CLAUSE_DOMINATES",
    "MIXED",
    "INVALID_MEASUREMENT",
)
json_dump = ov.json_dump
finite = ov.finite


def swapped_prefix_text(own_clauses: Sequence[str], partner_clauses: Sequence[str]) -> str:
    own = [str(item) for item in own_clauses]
    partner = [str(item) for item in partner_clauses]
    if len(own) != 3 or len(partner) != 3:
        raise StudyError("swap requires exactly three clauses on both records")
    if any(not item.strip() for item in own + partner):
        raise StudyError("swap received an empty clause")
    if own[2] == partner[2]:
        raise StudyError("partner final clause equals own final clause")
    text = "\n\n".join([partner[0], partner[1], own[2]])
    if text == "\n\n".join(own):
        raise StudyError("semantic swap produced the original explanation")
    return text


def cycle_partners(candidate_ids: Sequence[str]) -> dict[str, str]:
    ids = [str(item) for item in candidate_ids]
    if len(ids) != 8 or len(set(ids)) != 8:
        raise StudyError("partner cycle requires 8 distinct ids")
    return {ids[index]: ids[(index + 1) % 8] for index in range(8)}


def classify_swap(*, operational: bool, joint_rate: float, ar_sig: bool, jsd_sig: bool) -> str:
    if not operational:
        return "INVALID_MEASUREMENT"
    if joint_rate <= 0.25 and ar_sig and jsd_sig:
        return "SEMANTICS_CARRY_WEIGHT"
    if joint_rate >= 0.75:
        return "TOKEN_CLAUSE_DOMINATES"
    return "MIXED"


def _source_variant(source_dir: Path, prompt_id: str, stratum: str) -> dict[str, Any]:
    path = source_dir / "variants" / f"{ov._target_stem(prompt_id, stratum)}.json"
    return ov._require_json(path)


def _source_decision_rows(source_dir: Path) -> dict[str, dict[str, Any]]:
    decision = ov._require_json(source_dir / "decision.json")
    if decision.get("classification") != "PROSE_EXCEEDS_CONTEXT":
        raise StudyError("sealed Experiment 3 classification is not PROSE_EXCEEDS_CONTEXT")
    rows = decision.get("rows")
    if not isinstance(rows, list) or len(rows) != 48:
        raise StudyError("sealed Experiment 3 decision lacks 48 rows")
    by_id = {}
    for row in rows:
        activation_id = row.get("activation_id")
        if not isinstance(activation_id, str):
            raise StudyError("sealed decision row lacks activation_id")
        by_id[activation_id] = row
    if len(by_id) != 48:
        raise StudyError("sealed decision rows are not unique")
    return by_id


def stage_variants(args: argparse.Namespace) -> None:
    prompts = cb.load_context_prompt_spec(args.prompts)
    digest = hashlib.sha256(args.prompts.read_bytes()).hexdigest()
    if digest != FROZEN_PROMPT_SHA256:
        raise StudyError("Experiment 4 prompt hash is not the frozen Experiment 3 inventory")
    out = args.results_dir / "variants"
    inventories = args.results_dir / "inventories"
    out.mkdir(parents=True, exist_ok=True)
    inventories.mkdir(parents=True, exist_ok=True)
    records_by_group: dict[str, list[dict[str, Any]]] = {}
    for prompt, stratum in cb._iter_content_targets(prompts):
        source = _source_variant(args.source_dir, prompt["id"], stratum)
        clauses = source.get("primary_clauses")
        if not isinstance(clauses, list) or len(clauses) != 3:
            raise StudyError(f"sealed variant lacks three clauses: {source.get('activation_id')}")
        if not source.get("structural_valid"):
            raise StudyError(f"sealed variant is not structurally valid: {source.get('activation_id')}")
        record = {
            "status": "pass",
            "activation_id": source["activation_id"],
            "prompt_id": prompt["id"],
            "prompt_index": int(prompt["index"]),
            "category": prompt["category"],
            "position_stratum": stratum,
            "primary_clauses": list(clauses),
            "score_text": source["score_text"],
            "token_text": source["token_text"],
            "structural_valid": True,
            "source_record_path": source.get("record_path"),
        }
        records_by_group.setdefault(cb._group_id(prompt["category"], stratum), []).append(record)
    manifest_rows = []
    for category in cb.CATEGORIES:
        for stratum in cb.CONTENT_STRATA:
            group = cb._group_id(category, stratum)
            rows = records_by_group[group]
            expected = cb._expected_candidate_ids(prompts, category, stratum)
            actual = [row["activation_id"] for row in rows]
            if actual != expected:
                raise StudyError(f"candidate inventory/order mismatch for {group}")
            partners = cycle_partners(expected)
            by_id = {row["activation_id"]: row for row in rows}
            texts = []
            for row in rows:
                partner = by_id[partners[row["activation_id"]]]
                text = swapped_prefix_text(row["primary_clauses"], partner["primary_clauses"])
                row["partner_id"] = partner["activation_id"]
                row["variant_texts"] = {VARIANT: text}
                texts.append(text)
                json_dump(out / f"{ov._target_stem(row['prompt_id'], stratum)}.json", row)
                manifest_rows.append(row["activation_id"])
            if len(set(texts)) != 8:
                raise StudyError(f"duplicate swapped texts for {group}")
            payload = {
                "status": "pass",
                "group_id": group,
                "category": category,
                "position_stratum": stratum,
                "variant": VARIANT,
                "seed": SEED,
                "candidate_count": 8,
                "candidate_ids": expected,
                "partner_ids": [partners[item] for item in expected],
                "candidate_texts": texts,
            }
            json_dump(inventories / f"{category}__{stratum}__{VARIANT}.json", payload)
    if len(manifest_rows) != 48:
        raise StudyError(f"expected 48 swap records, found {len(manifest_rows)}")
    json_dump(out / "manifest.json", {"status": "pass", "seed": SEED, "count": len(manifest_rows)})
    json_dump(inventories / "manifest.json", {"status": "pass", "seed": SEED, "variant": VARIANT})


def stage_ar(args: argparse.Namespace) -> None:
    import torch

    cb.set_determinism(SEED)
    prompts = cb.load_context_prompt_spec(args.prompts)
    tokenizer, model, device, dtype = cb._load_tokenizer_model(args.ar_checkpoint, args.device, cb.AR_REVISION)
    head = cb._load_ar_head(args.ar_checkpoint, model, dtype, device)
    sidecar = cb._load_yaml(args.ar_checkpoint / "nla_meta.yaml")
    template, mse_scale = str(sidecar["prompt_templates"]["ar"]), float(sidecar["extraction"]["mse_scale"])
    out = args.results_dir / "ar"
    vectors = args.results_dir / "ar-vectors"
    out.mkdir(parents=True, exist_ok=True)
    vectors.mkdir(parents=True, exist_ok=True)
    for prompt, stratum in cb._iter_content_targets(prompts):
        stem = ov._target_stem(prompt["id"], stratum)
        target_meta, gold = ov._load_extraction_record(args.source_dir, prompt["id"], stratum)
        gold_n = ov.normalize_vector(gold, mse_scale)
        inventory = ov._require_json(args.results_dir / "inventories" / f"{prompt['category']}__{stratum}__{VARIANT}.json")
        own_id = target_meta["activation_id"]
        candidates = []
        for candidate_index, (candidate_id, text) in enumerate(zip(inventory["candidate_ids"], inventory["candidate_texts"])):
            vector, token_count = cb._score_payload_text(tokenizer, model, head, device, template, text)
            vector_n = ov.normalize_vector(vector, mse_scale)
            mse = float(((vector_n - gold_n) ** 2).mean().item())
            vector_path = vectors / f"{stem}__{VARIANT}__candidate-{candidate_index:02d}.pt"
            torch.save(vector, vector_path)
            candidates.append({
                "candidate_id": candidate_id,
                "variant_text": text,
                "own": candidate_id == own_id,
                "token_count": token_count,
                "mse_nrm": mse,
                "vector_path": str(vector_path),
                "finite": finite(mse) and bool(torch.isfinite(vector).all()),
            })
        own = next(row for row in candidates if row["own"])
        paraphrase = ov.neutral_paraphrase(own["variant_text"])
        para_vector, para_tokens = cb._score_payload_text(tokenizer, model, head, device, template, paraphrase)
        para_n = ov.normalize_vector(para_vector, mse_scale)
        para_mse = float(((para_n - gold_n) ** 2).mean().item())
        para_path = vectors / f"{stem}__{VARIANT}__paraphrase.pt"
        torch.save(para_vector, para_path)
        json_dump(out / f"{stem}.json", {
            "status": "pass",
            "activation_id": own_id,
            "prompt_id": prompt["id"],
            "category": prompt["category"],
            "position_stratum": stratum,
            "layer_index": cb.LAYER_INDEX,
            "seed": SEED,
            "unrelated_mse_nrm": None,
            "variants": {
                VARIANT: {
                    "candidate_ids": list(inventory["candidate_ids"]),
                    "candidate_texts": list(inventory["candidate_texts"]),
                    "candidates": candidates,
                    "paraphrase_text": paraphrase,
                    "paraphrase_mse_nrm": para_mse,
                    "paraphrase_token_count": para_tokens,
                    "paraphrase_vector_path": str(para_path),
                    "finite": all(row["finite"] for row in candidates) and finite(para_mse),
                }
            },
        })
    _run_bridge_ar(args, tokenizer, model, head, device, template, mse_scale)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _run_bridge_ar(args: argparse.Namespace, tokenizer: Any, model: Any, head: Any, device: Any, template: str, mse_scale: float) -> None:
    import torch

    sealed_rows = _source_decision_rows(args.source_dir)
    out = args.results_dir / "bridge"
    vectors = out / "ar-vectors"
    out.mkdir(parents=True, exist_ok=True)
    vectors.mkdir(parents=True, exist_ok=True)
    rows = []
    for prompt_id, stratum in BRIDGE_TARGETS:
        stem = ov._target_stem(prompt_id, stratum)
        target_meta, gold = ov._load_extraction_record(args.source_dir, prompt_id, stratum)
        gold_n = ov.normalize_vector(gold, mse_scale)
        source = _source_variant(args.source_dir, prompt_id, stratum)
        text = source["variant_texts"]["nla_full"]
        vector, token_count = cb._score_payload_text(tokenizer, model, head, device, template, text)
        vector_n = ov.normalize_vector(vector, mse_scale)
        mse = float(((vector_n - gold_n) ** 2).mean().item())
        path = vectors / f"{stem}.pt"
        torch.save(vector, path)
        sealed = sealed_rows[target_meta["activation_id"]]["variants"]["nla_full"]
        rows.append({
            "stem": stem,
            "activation_id": target_meta["activation_id"],
            "recomputed_mse": mse,
            "frozen_mse": float(sealed["own_mse"]),
            "token_count": token_count,
            "vector_path": str(path),
        })
    json_dump(out / "ar.json", {"status": "pass", "target_count": len(rows), "targets": rows})


def stage_behavior(args: argparse.Namespace) -> None:
    import torch

    cb.set_determinism(SEED)
    prompts = cb.load_context_prompt_spec(args.prompts)
    tokenizer, model, device, dtype = cb._load_tokenizer_model(args.base_checkpoint, args.device, cb.BASE_REVISION)
    out = args.results_dir / "behavior"
    out.mkdir(parents=True, exist_ok=True)
    for prompt, stratum in cb._iter_content_targets(prompts):
        stem = ov._target_stem(prompt["id"], stratum)
        ar_record = ov._require_json(args.results_dir / "ar" / f"{stem}.json")
        target_meta, gold = ov._load_extraction_record(args.source_dir, prompt["id"], stratum)
        alignment = target_meta.get("alignment")
        if not isinstance(alignment, Mapping):
            raise StudyError(f"missing rendered alignment for {target_meta['activation_id']}")
        ids = torch.tensor([alignment["token_ids"]], device=device)
        mask = torch.ones_like(ids)
        position = int(target_meta["position"])
        baseline = ov._base_logits(model, ids, mask)
        gold_logits = ov.hooked_logits(model, ids, mask, position, gold.to(device))
        gold_jsd = ov.js_divergence(baseline[position], gold_logits[position])
        block = ar_record["variants"][VARIANT]
        candidates = []
        for row in block["candidates"]:
            vector = torch.load(row["vector_path"], map_location="cpu", weights_only=True).float()
            replacement = vector / vector.norm().clamp_min(1e-12) * float(gold.norm().item())
            logits = ov.hooked_logits(model, ids, mask, position, replacement.to(device))
            jsd = ov.js_divergence(baseline[position], logits[position])
            candidates.append({
                "candidate_id": row["candidate_id"],
                "own": row["own"],
                "jsd": jsd,
                "vector_path": row["vector_path"],
                "finite": finite(jsd),
                "metric": BEHAVIOR_METRIC_NAME,
            })
        para_vector = torch.load(block["paraphrase_vector_path"], map_location="cpu", weights_only=True).float()
        para_replacement = para_vector / para_vector.norm().clamp_min(1e-12) * float(gold.norm().item())
        para_logits = ov.hooked_logits(model, ids, mask, position, para_replacement.to(device))
        para_jsd = ov.js_divergence(baseline[position], para_logits[position])
        json_dump(out / f"{stem}.json", {
            "status": "pass",
            "activation_id": target_meta["activation_id"],
            "prompt_id": prompt["id"],
            "position_stratum": stratum,
            "position": position,
            "layer_index": cb.LAYER_INDEX,
            "seed": SEED,
            "behavior_metric": BEHAVIOR_METRIC_NAME,
            "gold_reinjection_jsd": gold_jsd,
            "variants": {
                VARIANT: {
                    "candidate_ids": list(block["candidate_ids"]),
                    "candidates": candidates,
                    "paraphrase_jsd": para_jsd,
                    "finite": all(row["finite"] for row in candidates) and finite(para_jsd),
                }
            },
            "finite": finite(gold_jsd),
        })
    _run_bridge_behavior(args, model, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _run_bridge_behavior(args: argparse.Namespace, model: Any, device: Any) -> None:
    import torch

    sealed_rows = _source_decision_rows(args.source_dir)
    rows = []
    for prompt_id, stratum in BRIDGE_TARGETS:
        stem = ov._target_stem(prompt_id, stratum)
        target_meta, gold = ov._load_extraction_record(args.source_dir, prompt_id, stratum)
        alignment = target_meta.get("alignment")
        if not isinstance(alignment, Mapping):
            raise StudyError(f"missing frozen alignment for bridge {stem}")
        ids = torch.tensor([alignment["token_ids"]], device=device)
        mask = torch.ones_like(ids)
        position = int(target_meta["position"])
        baseline = ov._base_logits(model, ids, mask)
        vector = torch.load(args.results_dir / "bridge" / "ar-vectors" / f"{stem}.pt", map_location="cpu", weights_only=True).float()
        replacement = vector / vector.norm().clamp_min(1e-12) * float(gold.norm().item())
        logits = ov.hooked_logits(model, ids, mask, position, replacement.to(device))
        jsd = ov.js_divergence(baseline[position], logits[position])
        sealed = sealed_rows[target_meta["activation_id"]]["variants"]["nla_full"]
        rows.append({
            "stem": stem,
            "activation_id": target_meta["activation_id"],
            "recomputed_jsd": jsd,
            "frozen_jsd": float(sealed["own_jsd"]),
        })
    json_dump(args.results_dir / "bridge" / "behavior.json", {"status": "pass", "target_count": len(rows), "targets": rows})


def _bridge_gate(args: argparse.Namespace) -> dict[str, Any]:
    ar = ov._require_json(args.results_dir / "bridge" / "ar.json")
    behavior = ov._require_json(args.results_dir / "bridge" / "behavior.json")
    if ar.get("target_count") != 6 or behavior.get("target_count") != 6:
        raise StudyError("bridge target count is not 6")
    by_stem = {row["stem"]: row for row in behavior["targets"]}
    comparisons = []
    passed = True
    for row in ar["targets"]:
        other = by_stem[row["stem"]]
        comparison = cb.compare_bridge_values(
            row["recomputed_mse"], row["frozen_mse"],
            other["recomputed_jsd"], other["frozen_jsd"],
        )
        comparison["stem"] = row["stem"]
        comparisons.append(comparison)
        passed = passed and bool(comparison["passed"])
    return {"status": "pass" if passed else "fail", "passed": passed, "target_count": len(comparisons), "targets": comparisons}


def stage_decide(args: argparse.Namespace) -> dict[str, Any]:
    prompts = cb.load_context_prompt_spec(args.prompts)
    sealed_rows = _source_decision_rows(args.source_dir)
    gate = _bridge_gate(args)
    json_dump(args.results_dir / "gate.json", gate)
    rows: list[dict[str, Any]] = []
    reasons: list[str] = []
    gold_jsds: list[float] = []
    for prompt, stratum in cb._iter_content_targets(prompts):
        stem = ov._target_stem(prompt["id"], stratum)
        ar_record = ov._require_json(args.results_dir / "ar" / f"{stem}.json")
        behavior = ov._require_json(args.results_dir / "behavior" / f"{stem}.json")
        variants = ov._require_json(args.results_dir / "variants" / f"{stem}.json")
        activation_id = cb._activation_id(prompt["id"], stratum)
        swapped = cb._metrics_for_variant(
            ar_record["variants"][VARIANT],
            behavior["variants"][VARIANT],
            activation_id,
        )
        sealed = sealed_rows[activation_id]["variants"]["nla_full"]
        gold_jsd = float(behavior["gold_reinjection_jsd"])
        gold_jsds.append(gold_jsd)
        if not finite(gold_jsd) or gold_jsd > GOLD_JSD_MAX:
            reasons.append(f"gold reinjection failed for {stem}")
        rows.append({
            "activation_id": activation_id,
            "prompt_id": prompt["id"],
            "prompt_index": int(prompt["index"]),
            "category": prompt["category"],
            "position_stratum": stratum,
            "partner_id": variants["partner_id"],
            "swapped_text": variants["variant_texts"][VARIANT],
            "gold_reinjection_jsd": gold_jsd,
            "swapped": swapped,
            "sealed_full": {
                "ar_tournament_margin": float(sealed["ar_tournament_margin"]),
                "jsd_tournament_margin": float(sealed["jsd_tournament_margin"]),
                "joint_specific": bool(sealed["joint_specific"]),
            },
            "ar_delta": cb.pair_delta(float(sealed["ar_tournament_margin"]), swapped["ar_tournament_margin"]),
            "jsd_delta": cb.pair_delta(float(sealed["jsd_tournament_margin"]), swapped["jsd_tournament_margin"]),
        })
    joint_count = sum(1 for row in rows if row["swapped"]["joint_specific"])
    joint_rate = joint_count / 48
    comparison = {
        "ar": cb.prompt_cluster_sign_flip(rows, "ar_delta", seed=SEED),
        "jsd": cb.prompt_cluster_sign_flip(rows, "jsd_delta", seed=SEED),
    }
    comparison["ar_median"] = float(comparison["ar"]["observed"])
    comparison["jsd_median"] = float(comparison["jsd"]["observed"])
    comparison["ar_significant"] = comparison["ar_median"] > 0 and float(comparison["ar"]["p_value"]) <= 0.01
    comparison["jsd_significant"] = comparison["jsd_median"] > 0 and float(comparison["jsd"]["p_value"]) <= 0.01
    operational = gate["passed"] and not reasons and max(gold_jsds) <= GOLD_JSD_MAX
    classification = classify_swap(
        operational=operational,
        joint_rate=joint_rate,
        ar_sig=comparison["ar_significant"],
        jsd_sig=comparison["jsd_significant"],
    )
    result = {
        "status": "pass" if operational else "fail",
        "study": "semantic-swap",
        "experiment": 4,
        "classification": classification,
        "operational_validity": operational,
        "invalid_reasons": reasons,
        "seed": SEED,
        "prompt_sha256": FROZEN_PROMPT_SHA256,
        "target_count": 48,
        "behavior_metric": BEHAVIOR_METRIC_NAME,
        "swapped_joint_specific_count": joint_count,
        "swapped_joint_specific_rate": joint_rate,
        "comparison": comparison,
        "gate_status": gate["status"],
        "rows": rows,
        "source_dir": str(args.source_dir),
    }
    if classification not in DECISION_LABELS:
        raise StudyError(f"unrecognized classification {classification}")
    json_dump(args.results_dir / "decision.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic-swap follow-up on sealed Experiment 3 artifacts")
    parser.add_argument("--stage", choices=("variants", "ar", "behavior", "decide", "all"), required=True)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--ar-checkpoint", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args(argv)
    args.study_dir = args.study_dir.resolve()
    args.source_dir = args.source_dir.resolve()
    args.results_dir = (args.results_dir or args.study_dir / "results").resolve()
    args.prompts = args.prompts.resolve()
    weights = args.study_dir.parent / "weights"
    args.base_checkpoint = (args.base_checkpoint or weights / "base-qwen").resolve()
    args.ar_checkpoint = (args.ar_checkpoint or weights / "ar").resolve()
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in ("variants", "all"):
        stage_variants(args)
    if args.stage in ("ar", "all"):
        stage_ar(args)
    if args.stage in ("behavior", "all"):
        stage_behavior(args)
    if args.stage in ("decide", "all"):
        stage_decide(args)


if __name__ == "__main__":
    main()
