#!/usr/bin/env python3
"""Fresh-prompt same-variant context-baseline study (Experiment 3).

Stages stay separate processes. Dependency-light transforms, inventories,
payloads, and the prompt-cluster sign-flip live here so they can be tested
without loading the 7B checkpoints. GPU imports stay lazy and match the
frozen Experiment 1 runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import run_output_validity_study as ov

StudyError = ov.StudyError
LAYER_INDEX = ov.LAYER_INDEX
HF_HIDDEN_STATES_INDEX = ov.HF_HIDDEN_STATES_INDEX
MAX_NEW_TOKENS = ov.MAX_NEW_TOKENS
CATEGORIES = ov.CATEGORIES
AR_REPO = ov.AR_REPO
AR_REVISION = ov.AR_REVISION
AV_REPO = ov.AV_REPO
AV_REVISION = ov.AV_REVISION
BASE_REPO = ov.BASE_REPO
BASE_REVISION = ov.BASE_REVISION
N_PERMUTATIONS = ov.N_PERMUTATIONS
CHANCE_TOP1 = ov.CHANCE_TOP1
UNRELATED_TEXT = ov.UNRELATED_TEXT
json_dump = ov.json_dump
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
unique_top1 = ov.unique_top1
rank_ascending = ov.rank_ascending
_require_json = ov._require_json
validate_av_output = ov.validate_av_output
rendered_chat_positions = ov.rendered_chat_positions
_decode_token = ov._decode_token
_input_ids = ov._input_ids
_tokenize_rendered = ov._tokenize_rendered

SEED = 20260813
CONTENT_STRATA = ("content_early", "content_late")
VARIANT_NAMES = (
    "nla_full",
    "token_only",
    "local_ctx",
    "nla_drop_final_sym",
    "nla_scrubbed",
)
BASELINE_VARIANTS = ("token_only", "local_ctx")


def sha256_file(path: Path | str) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_file_inventory(results_dir: Path | str) -> list[dict[str, Any]]:
    root = Path(results_dir).resolve()
    inventory: list[dict[str, Any]] = []
    if not root.is_dir():
        return inventory
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if path.name == "completion-manifest.json":
            continue
        rel = path.relative_to(root).as_posix()
        inventory.append({
            "path": rel,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        })
    inventory.sort(key=lambda item: item["path"])
    return inventory


def build_completion_manifest(args: argparse.Namespace) -> dict[str, Any]:
    results_dir = Path(args.results_dir).resolve()
    decision_path = results_dir / "decision.json"
    if not decision_path.is_file():
        raise StudyError(f"cannot build completion manifest: missing {decision_path}")

    runner_path = Path(__file__).resolve()
    runner_sha256 = sha256_file(runner_path)

    prompt_path = Path(args.prompts).resolve() if hasattr(args, "prompts") and args.prompts else None
    prompt_sha256 = sha256_file(prompt_path) if (prompt_path and prompt_path.is_file()) else FROZEN_PROMPT_SHA256

    study_dir = getattr(args, "study_dir", None)
    protocol_path = (Path(study_dir).resolve() / "PROTOCOL.md") if study_dir else (results_dir.parent / "PROTOCOL.md")
    protocol_sha256 = sha256_file(protocol_path) if (protocol_path and protocol_path.is_file()) else None

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    utc_timestamp = datetime.now(timezone.utc).isoformat()
    decision_sha256 = sha256_file(decision_path)

    inventory = build_file_inventory(results_dir)

    return {
        "schema_version": 1,
        "runner_sha256": runner_sha256,
        "prompt_sha256": prompt_sha256,
        "protocol_sha256": protocol_sha256,
        "slurm_job_id": slurm_job_id,
        "utc_timestamp": utc_timestamp,
        "result_dir": str(results_dir),
        "decision_sha256": decision_sha256,
        "stage_statuses": {
            "preflight": "pass",
            "av": "pass",
            "variants": "pass",
            "ar": "pass",
            "behavior": "pass",
            "decide": "pass",
        },
        "file_inventory": inventory,
    }


def write_completion_manifest(args: argparse.Namespace) -> Path:
    results_dir = Path(args.results_dir).resolve()
    manifest = build_completion_manifest(args)
    manifest_path = results_dir / "completion-manifest.json"
    json_dump(manifest_path, manifest)
    return manifest_path
PRIMARY_VARIANTS = ("nla_full", "local_ctx")
SECONDARY_VARIANTS = ("nla_full", "token_only")
N_PROMPTS = 24
N_POSITIONS_PER_PROMPT = 2
N_CONTENT_TARGETS = N_PROMPTS * len(CONTENT_STRATA)
WINDOW_RADIUS = 5
TOKEN_PLACEHOLDER = "<TOKEN>"
GATE_MSE_TOL = 1e-4
GATE_JSD_TOL = 1e-5
GOLD_JSD_MAX = 1e-5
FROZEN_PROMPT_SHA256 = "4fba0cbe1e4d99070788e15966c87f95a494bc3c046d3a83a3c49635daaacb6b"
BEHAVIOR_METRIC_NAME = "AR-mediated one-position functional reconstruction"
DEFAULT_STUDY_DIR = Path(__file__).resolve().parents[2] / "pilots/wildcard-nla/context-baselines"
DEFAULT_FROZEN_DIR = Path(__file__).resolve().parents[2] / "pilots/wildcard-nla/output-validity"
BRIDGE_TARGETS = (
    ("ov-safety-01", "content_early"),
    ("ov-safety-01", "content_late"),
    ("ov-planning-01", "content_early"),
    ("ov-planning-01", "content_late"),
    ("ov-social-01", "content_early"),
    ("ov-social-01", "content_late"),
)
DECISION_LABELS = (
    "PROSE_EXCEEDS_CONTEXT",
    "PROSE_REDUCES_TO_CONTEXT",
    "PROSE_PARTIAL",
    "REPLICATION_FAILURE",
    "INVALID_MEASUREMENT",
)
OWNERSHIP_METADATA_KEYS = frozenset({
    "own", "candidate_id", "source_activation_id", "source_prompt_id",
    "source_category", "source_position_stratum", "activation_id",
    "prompt_id", "prompt_index", "group_id", "candidate_source",
    "category", "position_stratum",
})


def set_determinism(seed: int = SEED) -> None:
    ov.set_determinism(seed)


def _iter_content_targets(prompts: Sequence[Mapping[str, Any]]):
    for prompt in prompts:
        for stratum in CONTENT_STRATA:
            yield prompt, stratum


def _group_id(category: str, stratum: str) -> str:
    return f"{category}::{stratum}"


def _activation_id(prompt_id: str, stratum: str) -> str:
    return f"{prompt_id}::{stratum}"


def _locked_prompt_paths(spec: Mapping[str, Any], prompt_path: Path) -> list[Path]:
    raw = spec.get("locked_prior_prompt_files", [])
    if not isinstance(raw, list):
        raise StudyError("locked_prior_prompt_files must be a list")
    paths = []
    for relative in raw:
        if not isinstance(relative, str) or not relative:
            raise StudyError("locked prior prompt path is empty")
        paths.append((prompt_path.parent / relative).resolve())
    return paths


def _prompt_texts(path: Path) -> set[str]:
    if not path.is_file():
        raise StudyError(f"missing locked prompt file: {path}")
    raw = json.loads(path.read_text())
    rows = raw.get("prompts")
    if not isinstance(rows, list):
        raise StudyError(f"locked prompt file lacks prompts: {path}")
    texts = set()
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise StudyError(f"malformed locked prompt row in {path}")
        texts.add(row["prompt"])
    return texts


def load_context_prompt_spec(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text())
    prompts = raw.get("prompts")
    if raw.get("schema_version") != 1 or raw.get("seed") != SEED or not isinstance(prompts, list):
        raise StudyError("prompt file does not contain the frozen Experiment 3 schema/seed")
    if raw.get("study") != "context-baselines" or raw.get("experiment") != 3:
        raise StudyError("prompt file is not the frozen context-baselines Experiment 3 inventory")
    if len(prompts) != N_PROMPTS:
        raise StudyError(f"expected {N_PROMPTS} prompts, found {len(prompts)}")
    ids: set[str] = set()
    texts: set[str] = set()
    counts = {c: 0 for c in CATEGORIES}
    for expected, row in enumerate(prompts):
        if not isinstance(row, dict) or row.get("index") != expected:
            raise StudyError("prompt indices must be consecutive 0..23")
        identifier, category, text = row.get("id"), row.get("category"), row.get("prompt")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise StudyError("prompt ids must be unique non-empty strings")
        if not isinstance(text, str) or not text.strip() or text in texts:
            raise StudyError("prompt texts must be unique, non-empty strings")
        if category not in CATEGORIES:
            raise StudyError(f"unknown prompt category: {category}")
        ids.add(identifier)
        texts.add(text)
        counts[category] += 1
    if counts != {c: 8 for c in CATEGORIES}:
        raise StudyError(f"prompt category counts are not 8/8/8: {counts}")
    return prompts


def validate_fresh_prompt_inventory(path: Path) -> dict[str, Any]:
    """CPU-testable inventory: 24 category-balanced prompts, disjoint from Experiment 1."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != FROZEN_PROMPT_SHA256:
        raise StudyError("prompts.json does not match the frozen Experiment 3 file")
    raw = json.loads(path.read_text())
    prompts = load_context_prompt_spec(path)
    locked_paths = _locked_prompt_paths(raw, path)
    locked_texts: set[str] = set()
    for locked in locked_paths:
        locked_texts.update(_prompt_texts(locked))
    overlap = {row["prompt"] for row in prompts} & locked_texts
    if overlap:
        raise StudyError(f"fresh prompts are not string-disjoint from locked inventories: {sorted(overlap)[:3]}")
    return {
        "status": "pass",
        "seed": SEED,
        "prompt_count": len(prompts),
        "category_counts": {c: sum(p["category"] == c for p in prompts) for c in CATEGORIES},
        "prompt_sha256": digest,
        "locked_prior_prompt_files": [str(p) for p in locked_paths],
        "string_disjoint_from_locked": True,
        "prompts": prompts,
    }


def display_token(token: str) -> str:
    if not isinstance(token, str):
        raise StudyError("display_token requires a string token")
    surface = token.strip()
    if not surface:
        raise StudyError("display_token stripped to an empty string")
    return surface


def token_only_text(token: str) -> str:
    surface = display_token(token)
    return f'Final token "{surface}".'


def local_ctx_text(token: str, window: str) -> str:
    surface = display_token(token)
    if not isinstance(window, str) or window == "":
        raise StudyError("local_ctx requires a decoded local token window")
    return f'Final token "{surface}". Local token window: "{window}".'


def drop_final_sym_text(primary_clauses: Sequence[str]) -> str:
    if not isinstance(primary_clauses, Sequence) or isinstance(primary_clauses, (str, bytes)):
        raise StudyError("nla_drop_final_sym requires a clause sequence")
    clauses = [str(x) for x in primary_clauses]
    if len(clauses) < 2:
        raise StudyError("nla_drop_final_sym requires at least two primary clauses")
    return "\n\n".join(clauses[:-1])


def scrub_target_token(text: str, token: str, placeholder: str = TOKEN_PLACEHOLDER) -> str:
    if not isinstance(text, str) or not text:
        raise StudyError("nla_scrubbed requires non-empty score_text")
    if not isinstance(token, str) or not token:
        raise StudyError("nla_scrubbed requires a non-empty decoded target token")
    if not isinstance(placeholder, str) or not placeholder:
        raise StudyError("nla_scrubbed requires a non-empty placeholder")
    surface = display_token(token)
    raw = token
    scrubbed = text.replace(raw, placeholder)
    if surface != raw:
        scrubbed = scrubbed.replace(surface, placeholder)
    if scrubbed == text:
        raise StudyError("decoded target token and display surface are absent from score_text; redaction failed")
    if raw in scrubbed or surface in scrubbed:
        raise StudyError("decoded target token or display surface remained in score_text after redaction")
    return scrubbed


def prompt_overlapping_indices(alignment: Mapping[str, Any]) -> list[int]:
    span = alignment.get("prompt_span")
    offsets = alignment.get("token_offsets")
    if not isinstance(span, Sequence) or len(span) != 2 or not isinstance(offsets, Sequence):
        raise StudyError("alignment lacks prompt span or token offsets")
    start, end = int(span[0]), int(span[1])
    indices = []
    for index, pair in enumerate(offsets):
        if not isinstance(pair, Sequence) or len(pair) != 2:
            raise StudyError("token offset is not a pair")
        left, right = int(pair[0]), int(pair[1])
        if left < end and right > start:
            indices.append(index)
    if not indices:
        raise StudyError("no prompt-overlapping tokens")
    return indices


def local_token_window(alignment: Mapping[str, Any], position: int, radius: int = WINDOW_RADIUS) -> str:
    """Deterministic ±5 decoded prompt tokens around the target position."""
    texts = alignment.get("token_texts")
    if not isinstance(texts, Sequence) or isinstance(texts, (str, bytes)):
        raise StudyError("alignment lacks decoded token texts")
    if not (0 <= int(position) < len(texts)):
        raise StudyError("target position is outside alignment")
    prompt_idx = prompt_overlapping_indices(alignment)
    if int(position) not in prompt_idx:
        raise StudyError("target position is not a decoded prompt token")
    center = prompt_idx.index(int(position))
    lo = max(0, center - int(radius))
    hi = min(len(prompt_idx), center + int(radius) + 1)
    window = "".join(str(texts[prompt_idx[i]]) for i in range(lo, hi))
    if not window:
        raise StudyError("local token window is empty")
    return window


def build_variant_texts(
    *,
    score_text: str,
    primary_clauses: Sequence[str],
    token: str,
    window: str,
) -> dict[str, str]:
    """Build the five frozen variants. Baselines never consume AV score text."""
    if not isinstance(score_text, str) or not score_text:
        raise StudyError("nla_full requires AV score_text")
    drop = drop_final_sym_text(primary_clauses)
    texts = {
        "nla_full": score_text,
        "token_only": token_only_text(token),
        "local_ctx": local_ctx_text(token, window),
        "nla_drop_final_sym": drop,
        "nla_scrubbed": scrub_target_token(score_text, token),
    }
    if texts["token_only"] in (score_text, drop) or texts["local_ctx"] in (score_text, drop):
        raise StudyError("baseline variants must not reuse AV score text")
    return texts


def build_symmetric_candidate_texts(records: Sequence[Mapping[str, Any]], variant: str) -> list[str]:
    if variant not in VARIANT_NAMES:
        raise StudyError(f"unknown variant: {variant}")
    if len(records) != 8:
        raise StudyError(f"symmetric tournament requires 8 records, found {len(records)}")
    texts = []
    for row in records:
        if not isinstance(row, Mapping):
            raise StudyError("candidate record is not an object")
        variant_texts = row.get("variant_texts")
        if not isinstance(variant_texts, Mapping) or variant not in variant_texts:
            raise StudyError(f"candidate record lacks variant text {variant}")
        text = variant_texts[variant]
        if not isinstance(text, str) or not text:
            raise StudyError(f"empty candidate text for {variant}")
        texts.append(text)
    if len(set(texts)) != 8:
        raise StudyError(f"duplicate candidate texts for variant {variant}")
    return texts


def diagnose_token_duplicates(tokens: Sequence[str], *, group_id: str = "") -> dict[str, Any]:
    """Pure CPU diagnostic helper for pairwise distinctness of raw and surface tokens."""
    raw_tokens = [str(t) for t in tokens]
    raw_counts: dict[str, int] = {}
    raw_indices: dict[str, list[int]] = {}
    for idx, tok in enumerate(raw_tokens):
        raw_counts[tok] = raw_counts.get(tok, 0) + 1
        raw_indices.setdefault(tok, []).append(idx)

    raw_duplicates = [
        {"token": tok, "count": count, "indices": raw_indices[tok]}
        for tok, count in raw_counts.items()
        if count > 1
    ]

    surface_tokens: list[str] = []
    surface_errors: list[dict[str, Any]] = []
    for idx, tok in enumerate(raw_tokens):
        try:
            surface_tokens.append(display_token(tok))
        except Exception as exc:
            surface_tokens.append("")
            surface_errors.append({"index": idx, "token": tok, "error": str(exc)})

    surface_counts: dict[str, int] = {}
    surface_indices: dict[str, list[int]] = {}
    for idx, surf in enumerate(surface_tokens):
        if surf:
            surface_counts[surf] = surface_counts.get(surf, 0) + 1
            surface_indices.setdefault(surf, []).append(idx)

    surface_duplicates = [
        {"surface_token": surf, "count": count, "indices": surface_indices[surf]}
        for surf, count in surface_counts.items()
        if count > 1
    ]

    empty_tokens = [idx for idx, tok in enumerate(raw_tokens) if not isinstance(tok, str) or tok == ""]
    has_violations = bool(raw_duplicates or surface_duplicates or surface_errors or empty_tokens or len(raw_tokens) != 8)

    return {
        "group_id": group_id,
        "token_count": len(raw_tokens),
        "raw_tokens": raw_tokens,
        "surface_tokens": surface_tokens,
        "raw_duplicates": raw_duplicates,
        "surface_duplicates": surface_duplicates,
        "surface_errors": surface_errors,
        "empty_tokens": empty_tokens,
        "has_violations": has_violations,
        "is_valid": not has_violations,
    }


def validate_pairwise_distinct_tokens(tokens: Sequence[str], *, group_id: str) -> None:
    diag = diagnose_token_duplicates(tokens, group_id=group_id)
    if diag["token_count"] != 8:
        raise StudyError(f"{group_id} does not have 8 decoded target tokens")
    if diag["empty_tokens"] or diag["surface_errors"]:
        raise StudyError(f"{group_id} has an empty decoded target token")
    if diag["raw_duplicates"]:
        raise StudyError(f"{group_id} decoded target tokens are not pairwise distinct")
    if diag["surface_duplicates"]:
        raise StudyError(f"{group_id} display/surface target tokens are not pairwise distinct")


def build_preflight_diagnostic(
    *,
    prompt_sha: str,
    selected_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Construct the preflight-diagnostic JSON payload. Contains no AV outputs."""
    if len(selected_rows) != N_CONTENT_TARGETS:
        raise StudyError(f"expected {N_CONTENT_TARGETS} selected rows, found {len(selected_rows)}")

    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in selected_rows:
        cat = row.get("category")
        strat = row.get("stratum") or row.get("position_stratum")
        if not cat or not strat:
            raise StudyError("selected row missing category or stratum")
        gid = _group_id(str(cat), str(strat))
        by_group.setdefault(gid, []).append(dict(row))

    inventories: dict[str, Any] = {}
    violations: list[dict[str, Any]] = []
    group_diagnostics: dict[str, Any] = {}

    for cat in CATEGORIES:
        for strat in CONTENT_STRATA:
            gid = _group_id(cat, strat)
            rows = by_group.get(gid, [])
            raw_tokens = [str(r.get("raw_token", r.get("raw_decoded_token", ""))) for r in rows]
            diag = diagnose_token_duplicates(raw_tokens, group_id=gid)
            group_diagnostics[gid] = diag

            inventory_records = []
            for idx, r in enumerate(rows):
                rec = {
                    "candidate_index": idx,
                    "prompt_id": r.get("prompt_id"),
                    "category": cat,
                    "stratum": strat,
                    "position": r.get("position"),
                    "raw_token": r.get("raw_token"),
                    "raw_decoded_token": r.get("raw_decoded_token", r.get("raw_token")),
                    "display_token": r.get("display_token"),
                    "display_surface_token": r.get("display_surface_token", r.get("display_token")),
                    "local_window": r.get("local_window"),
                }
                inventory_records.append(rec)

            inventories[gid] = {
                "group_id": gid,
                "category": cat,
                "stratum": strat,
                "count": len(rows),
                "candidate_order_prompt_ids": [r.get("prompt_id") for r in rows],
                "candidate_order_raw_tokens": raw_tokens,
                "candidate_order_surface_tokens": [r.get("display_token") for r in rows],
                "items": inventory_records,
            }

            if diag["has_violations"]:
                raw_dups_with_prompts = []
                for dup in diag["raw_duplicates"]:
                    raw_dups_with_prompts.append({
                        "token": dup["token"],
                        "count": dup["count"],
                        "indices": dup["indices"],
                        "prompt_ids": [rows[i]["prompt_id"] for i in dup["indices"] if i < len(rows)],
                    })
                surf_dups_with_prompts = []
                for dup in diag["surface_duplicates"]:
                    surf_dups_with_prompts.append({
                        "surface_token": dup["surface_token"],
                        "count": dup["count"],
                        "indices": dup["indices"],
                        "prompt_ids": [rows[i]["prompt_id"] for i in dup["indices"] if i < len(rows)],
                        "raw_tokens": [raw_tokens[i] for i in dup["indices"] if i < len(raw_tokens)],
                    })
                violations.append({
                    "group_id": gid,
                    "category": cat,
                    "stratum": strat,
                    "raw_duplicates": raw_dups_with_prompts,
                    "surface_duplicates": surf_dups_with_prompts,
                    "surface_errors": diag["surface_errors"],
                    "empty_tokens": diag["empty_tokens"],
                })

    has_violations = len(violations) > 0
    return {
        "status": "fail" if has_violations else "pass",
        "has_violations": has_violations,
        "prompt_sha256": prompt_sha,
        "prompt_sha": prompt_sha,
        "row_count": len(selected_rows),
        "selected_rows": list(selected_rows),
        "inventories": inventories,
        "violations": violations,
        "group_diagnostics": group_diagnostics,
    }


def model_text_payload(record: Mapping[str, Any]) -> dict[str, str]:
    """The only mapping that may enter AR scoring. Ownership metadata is excluded."""
    text = record.get("variant_text")
    if not isinstance(text, str) or not text:
        raise StudyError("model payload lacks explanation text")
    payload = {"explanation": text}
    assert_payload_excludes_ownership(payload)
    return payload


def assert_payload_excludes_ownership(payload: Mapping[str, Any]) -> None:
    leaked = sorted(OWNERSHIP_METADATA_KEYS.intersection(payload))
    if leaked:
        raise StudyError(f"ownership metadata in model payload: {leaked}")


def compare_bridge_values(
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
    return {
        "passed": mse_delta <= mse_tol and jsd_delta <= jsd_tol,
        "recomputed_mse": float(recomputed_mse),
        "frozen_mse": float(frozen_mse),
        "mse_delta": mse_delta,
        "mse_tol": mse_tol,
        "recomputed_jsd": float(recomputed_jsd),
        "frozen_jsd": float(frozen_jsd),
        "jsd_delta": jsd_delta,
        "jsd_tol": jsd_tol,
    }


def _cluster_statistic(values: Sequence[float], statistic: str) -> float:
    if not values:
        raise StudyError("cannot summarize an empty prompt cluster")
    if statistic == "mean":
        return sum(values) / len(values)
    if statistic == "median":
        result = ov._median(values)
        if result is None:
            raise StudyError("cannot summarize an empty prompt cluster")
        return result
    raise StudyError(f"unsupported prompt-cluster statistic: {statistic}")


def prompt_cluster_sign_flip(
    records: Sequence[Mapping[str, Any]],
    value_key: str,
    *,
    seed: int = SEED,
    n_draws: int = N_PERMUTATIONS,
    statistic: str = "median",
) -> dict[str, Any]:
    """One-sided prompt-cluster sign-flip. p = (exceedances + 1) / (B + 1)."""
    if int(n_draws) < 1:
        raise StudyError("sign-flip requires at least one draw")
    by_prompt: dict[str, list[float]] = {}
    for row in records:
        prompt_id = row.get("prompt_id")
        value = row.get(value_key)
        if not isinstance(prompt_id, str) or not prompt_id or not finite(value):
            raise StudyError("sign-flip record lacks finite prompt/value")
        by_prompt.setdefault(prompt_id, []).append(float(value))
    if not by_prompt:
        raise StudyError("sign-flip requires at least one prompt cluster")
    ids = sorted(by_prompt)
    observed_values = [value for prompt_id in ids for value in by_prompt[prompt_id]]
    observed = _cluster_statistic(observed_values, statistic)
    rng = random.Random(seed)
    nulls: list[float] = []
    exceedances = 0
    for _ in range(int(n_draws)):
        values: list[float] = []
        for prompt_id in ids:
            sign = -1.0 if rng.random() < 0.5 else 1.0
            values.extend(sign * value for value in by_prompt[prompt_id])
        stat = _cluster_statistic(values, statistic)
        nulls.append(stat)
        if stat >= observed:
            exceedances += 1
    return {
        "seed": seed,
        "draws": int(n_draws),
        "statistic": statistic,
        "observed": observed,
        "exceedances": exceedances,
        "p_value": (exceedances + 1) / (int(n_draws) + 1),
        "cluster_count": len(ids),
        "cluster_sizes": [len(by_prompt[prompt_id]) for prompt_id in ids],
        "one_sided": "greater",
    }


def classify_context_decision(
    operational: bool,
    nla_full_joint_rate: float,
    nla_full_joint_count: int,
    primary_ar_median: float,
    primary_ar_p: float,
    primary_jsd_median: float,
    primary_jsd_p: float,
    secondary_ar_median: float,
    secondary_ar_p: float,
    secondary_jsd_median: float,
    secondary_jsd_p: float,
    reversed_ar_median: float,
    reversed_ar_p: float,
    reversed_jsd_median: float,
    reversed_jsd_p: float,
) -> str:
    if not operational:
        return "INVALID_MEASUREMENT"
    if nla_full_joint_rate <= 0.125 or nla_full_joint_count <= 24:
        return "REPLICATION_FAILURE"
    primary_sig = primary_ar_median > 0 and primary_ar_p <= 0.01 and primary_jsd_median > 0 and primary_jsd_p <= 0.01
    secondary_sig = secondary_ar_median > 0 and secondary_ar_p <= 0.01 and secondary_jsd_median > 0 and secondary_jsd_p <= 0.01
    if primary_sig and secondary_sig:
        return "PROSE_EXCEEDS_CONTEXT"
    reversed_sig = reversed_ar_median > 0 and reversed_ar_p <= 0.01 and reversed_jsd_median > 0 and reversed_jsd_p <= 0.01
    if reversed_sig:
        return "PROSE_REDUCES_TO_CONTEXT"
    return "PROSE_PARTIAL"


def _group_prompts(prompts: Sequence[Mapping[str, Any]], category: str) -> list[Mapping[str, Any]]:
    rows = [row for row in prompts if row["category"] == category]
    if len(rows) != 8:
        raise StudyError(f"category {category} does not have 8 prompts")
    return rows


def _expected_candidate_ids(prompts: Sequence[Mapping[str, Any]], category: str, stratum: str) -> list[str]:
    return [_activation_id(row["id"], stratum) for row in _group_prompts(prompts, category)]


def _score_payload_text(tokenizer: Any, model: Any, head: Any, device: Any, template: str, text: str):
    payload = model_text_payload({"variant_text": text})
    return _score_ar_text(tokenizer, model, head, device, template, payload["explanation"])


def _frozen_own_scores(ar_record: Mapping[str, Any], behavior_record: Mapping[str, Any]) -> tuple[float, float]:
    own_ar = next(row for row in ar_record["candidates"] if row.get("own"))
    own_bh = next(row for row in behavior_record["candidates"] if row.get("own"))
    return float(own_ar["mse_nrm"]), float(own_bh["jsd"])


def stage_preflight(args: argparse.Namespace) -> None:
    import torch

    inventory = validate_fresh_prompt_inventory(args.prompts)
    prompts = inventory["prompts"]
    for checkpoint, repo in ((args.base_checkpoint, BASE_REPO), (args.av_checkpoint, AV_REPO), (args.ar_checkpoint, AR_REPO)):
        ov._require_checkpoint(checkpoint, repo)
    manifest = ov.verify_checkpoint_manifest(args)
    set_determinism()
    tokenizer, model, device, dtype = _load_tokenizer_model(args.base_checkpoint, args.device, BASE_REVISION)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / "extract"
    out.mkdir(parents=True, exist_ok=True)

    alignments: dict[str, dict[str, Any]] = {}
    selected_rows: list[dict[str, Any]] = []
    for prompt in prompts:
        alignment = rendered_chat_positions(tokenizer, prompt["prompt"])
        alignments[prompt["id"]] = alignment
        for stratum in CONTENT_STRATA:
            position_info = alignment["positions"][stratum]
            position = int(position_info["position"])
            raw_token = str(position_info["token_text"])
            try:
                surface_token = display_token(raw_token)
            except Exception:
                surface_token = ""
            try:
                window = local_token_window(alignment, position)
            except Exception:
                window = ""
            selected_rows.append({
                "prompt_id": prompt["id"],
                "prompt_index": int(prompt["index"]),
                "category": prompt["category"],
                "stratum": stratum,
                "position_stratum": stratum,
                "position": position,
                "token_id": position_info["token_id"],
                "raw_token": raw_token,
                "raw_decoded_token": raw_token,
                "display_token": surface_token,
                "display_surface_token": surface_token,
                "local_window": window,
            })

    diagnostic = build_preflight_diagnostic(
        prompt_sha=inventory["prompt_sha256"],
        selected_rows=selected_rows,
    )
    json_dump(args.results_dir / "preflight-diagnostic.json", diagnostic)

    for category in CATEGORIES:
        for stratum in CONTENT_STRATA:
            gid = _group_id(category, stratum)
            group_tokens = [r["raw_token"] for r in selected_rows if r["category"] == category and r["stratum"] == stratum]
            validate_pairwise_distinct_tokens(group_tokens, group_id=gid)
    manifest_rows = []
    for prompt in prompts:
        alignment = alignments[prompt["id"]]
        encoded = _tokenize_rendered(tokenizer, alignment["rendered_text"])
        ids = torch.tensor([_input_ids(encoded)], device=device)
        mask = torch.ones_like(ids)
        with torch.inference_mode():
            result = model(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
        hidden = result.hidden_states[HF_HIDDEN_STATES_INDEX]
        for stratum in CONTENT_STRATA:
            position_info = alignment["positions"][stratum]
            position = int(position_info["position"])
            activation = hidden[0, position].detach().float().cpu()
            activation_id = _activation_id(prompt["id"], stratum)
            stem = _target_stem(prompt["id"], stratum)
            metadata = {
                "status": "pass",
                "activation_id": activation_id,
                "prompt_id": prompt["id"],
                "prompt_index": int(prompt["index"]),
                "category": prompt["category"],
                "prompt": prompt["prompt"],
                "position_stratum": stratum,
                "position": position,
                "token_id": position_info["token_id"],
                "token_text": position_info["token_text"],
                "layer_index": LAYER_INDEX,
                "hf_hidden_states_index": HF_HIDDEN_STATES_INDEX,
                "seed": SEED,
                "base_model": BASE_REPO,
                "base_revision": BASE_REVISION,
                "device": str(device),
                "dtype": str(dtype),
                "finite": bool(torch.isfinite(activation).all()),
                "activation_shape": list(activation.shape),
                "activation_norm_fp32": float(activation.norm().item()),
                "rendered_prompt_path": str(args.prompts),
                "record_path": str(out / f"{stem}.json"),
                "activation_path": str(out / f"{stem}.pt"),
                "alignment": alignment,
            }
            torch.save(activation, out / f"{stem}.pt")
            json_dump(out / f"{stem}.json", metadata)
            manifest_rows.append({
                "activation_id": activation_id,
                "prompt_id": prompt["id"],
                "position_stratum": stratum,
                "token_text": position_info["token_text"],
                "record_path": str(out / f"{stem}.json"),
                "activation_path": str(out / f"{stem}.pt"),
            })
    if len(manifest_rows) != N_CONTENT_TARGETS:
        raise StudyError(f"expected {N_CONTENT_TARGETS} extract records, found {len(manifest_rows)}")
    json_dump(out / "manifest.json", {"status": "pass", "seed": SEED, "count": len(manifest_rows), "records": manifest_rows})
    json_dump(args.results_dir / "preflight.json", {
        "status": "pass",
        "seed": SEED,
        "prompt_file": str(args.prompts),
        "prompt_sha256": inventory["prompt_sha256"],
        "prompt_count": len(prompts),
        "category_counts": inventory["category_counts"],
        "string_disjoint_from_locked": True,
        "locked_prior_prompt_files": inventory["locked_prior_prompt_files"],
        "layer_index": LAYER_INDEX,
        "hf_hidden_states_index": HF_HIDDEN_STATES_INDEX,
        "position_strata": list(CONTENT_STRATA),
        "activation_count": N_CONTENT_TARGETS,
        "pairwise_distinct_target_tokens": {
            group: True for group in (_group_id(c, s) for c in CATEGORIES for s in CONTENT_STRATA)
        },
        "causal_hook": "model.model.layers[20] output, only target position",
        "behavior_metric": BEHAVIOR_METRIC_NAME,
        "max_new_tokens": MAX_NEW_TOKENS,
        "candidate_group_size": 8,
        "statistics": {"seed": SEED, "sign_flip_draws": N_PERMUTATIONS, "chance": CHANCE_TOP1},
        "checkpoints": {
            "base": {"repo": BASE_REPO, "revision": BASE_REVISION, "path": str(args.base_checkpoint)},
            "av": {"repo": AV_REPO, "revision": AV_REVISION, "path": str(args.av_checkpoint)},
            "ar": {"repo": AR_REPO, "revision": AR_REVISION, "path": str(args.ar_checkpoint)},
        },
        "manifest_verification": manifest,
        "bridge_targets": [f"{pid}::{stratum}" for pid, stratum in BRIDGE_TARGETS],
    })
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def stage_av(args: argparse.Namespace) -> None:
    import torch

    set_determinism()
    prompts = load_context_prompt_spec(args.prompts)
    tokenizer, model, device, dtype = _load_tokenizer_model(args.av_checkpoint, args.device, AV_REVISION)
    meta = _load_yaml(args.av_checkpoint / "nla_meta.yaml")
    template = str(meta["prompt_templates"]["av"])
    marker = str(meta["tokens"]["injection_char"])
    marker_id = int(meta["tokens"]["injection_token_id"])
    scale = float(meta["extraction"]["injection_scale"])
    out = args.results_dir / "av"
    out.mkdir(parents=True, exist_ok=True)
    for prompt, stratum in _iter_content_targets(prompts):
        extract_meta, activation = _load_extraction_record(args.results_dir, prompt["id"], stratum)
        content = template.format(injection_char=marker)
        payload = model_text_payload({"variant_text": content})
        ids = torch.tensor([_input_ids(tokenizer.apply_chat_template(
            [{"role": "user", "content": payload["explanation"]}], tokenize=True, add_generation_prompt=True, return_tensors="pt"))], device=device)
        matches = (ids[0] == marker_id).nonzero(as_tuple=False).flatten().tolist()
        if len(matches) != 1:
            raise StudyError(f"AV injection marker count for {prompt['id']}::{stratum}: {matches}")
        marker_position = matches[0]
        if int(ids[0, marker_position - 1]) != int(meta["tokens"]["injection_left_neighbor_id"]) or int(ids[0, marker_position + 1]) != int(meta["tokens"]["injection_right_neighbor_id"]):
            raise StudyError(f"AV injection neighbors do not match sidecar for {prompt['id']}::{stratum}")
        scaled = activation * (scale / max(float(activation.norm()), 1e-12))
        with torch.inference_mode():
            embeds = model.get_input_embeddings()(ids).clone()
            embeds[0, marker_position] = scaled.to(device=device, dtype=embeds.dtype)
            generated = model.generate(
                inputs_embeds=embeds, attention_mask=torch.ones_like(ids),
                max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids = generated[0].detach().cpu().tolist()
        raw_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        special_strings = [
            _decode_token(tokenizer, token_id) for token_id in generated_ids
            if token_id in set(getattr(tokenizer, "all_special_ids", ()))
            or __import__("re").fullmatch(r"<\|[^|]+\|>", _decode_token(tokenizer, token_id))
        ]
        eos_terminated = tokenizer.eos_token_id is not None and int(tokenizer.eos_token_id) in generated_ids
        stem = _target_stem(prompt["id"], stratum)
        json_dump(out / f"{stem}.json", {
            "status": "pass",
            "activation_id": extract_meta["activation_id"],
            "prompt_id": prompt["id"],
            "prompt_index": int(prompt["index"]),
            "category": prompt["category"],
            "position_stratum": stratum,
            "position": extract_meta["position"],
            "layer_index": LAYER_INDEX,
            "seed": SEED,
            "checkpoint": {"repo": AV_REPO, "revision": AV_REVISION},
            "device": str(device),
            "dtype": str(dtype),
            "injection_position": int(marker_position),
            "injection_scale": scale,
            "generated_ids": generated_ids,
            "generated_token_count": len(generated_ids),
            "eos_terminated": bool(eos_terminated),
            "literal_special_token_strings": special_strings,
            "raw_text": raw_text,
            "record_path": str(out / f"{stem}.json"),
        })
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _load_extract_meta(results: Path, prompt_id: str, stratum: str) -> dict[str, Any]:
    path = results / "extract" / f"{_target_stem(prompt_id, stratum)}.json"
    return _require_json(path)


def stage_variants(args: argparse.Namespace) -> None:
    prompts = load_context_prompt_spec(args.prompts)
    out = args.results_dir / "variants"
    inventories = args.results_dir / "inventories"
    out.mkdir(parents=True, exist_ok=True)
    inventories.mkdir(parents=True, exist_ok=True)
    records_by_group: dict[str, list[dict[str, Any]]] = {}
    manifest_rows = []
    for prompt, stratum in _iter_content_targets(prompts):
        stem = _target_stem(prompt["id"], stratum)
        extract = _load_extract_meta(args.results_dir, prompt["id"], stratum)
        av = _require_json(args.results_dir / "av" / f"{stem}.json")
        if av.get("activation_id") != _activation_id(prompt["id"], stratum):
            raise StudyError(f"AV identity mismatch for {stem}")
        validity = validate_av_output(
            av["raw_text"], av["generated_token_count"], av["eos_terminated"],
            av.get("literal_special_token_strings", ()),
        )
        token = str(extract["token_text"])
        window = local_token_window(extract["alignment"], int(extract["position"]))
        texts = build_variant_texts(
            score_text=validity["score_text"],
            primary_clauses=validity["primary_clauses"],
            token=token,
            window=window,
        )
        record = {
            "status": "pass",
            "activation_id": extract["activation_id"],
            "prompt_id": prompt["id"],
            "prompt_index": int(prompt["index"]),
            "category": prompt["category"],
            "position_stratum": stratum,
            "position": extract["position"],
            "token_text": token,
            "local_window": window,
            "layer_index": LAYER_INDEX,
            "seed": SEED,
            "primary_clauses": list(validity["primary_clauses"]),
            "score_text": validity["score_text"],
            "structural_valid": validity["structural_valid"],
            "variant_texts": texts,
            "baselines_used_av_score_text": False,
            "record_path": str(out / f"{stem}.json"),
        }
        json_dump(out / f"{stem}.json", record)
        records_by_group.setdefault(_group_id(prompt["category"], stratum), []).append(record)
        manifest_rows.append({"stem": stem, "activation_id": record["activation_id"], "record_path": record["record_path"]})
    if len(manifest_rows) != N_CONTENT_TARGETS:
        raise StudyError(f"expected {N_CONTENT_TARGETS} variant records, found {len(manifest_rows)}")
    inventory_manifest = []
    for category in CATEGORIES:
        for stratum in CONTENT_STRATA:
            group = _group_id(category, stratum)
            rows = records_by_group[group]
            expected = _expected_candidate_ids(prompts, category, stratum)
            actual = [row["activation_id"] for row in rows]
            if actual != expected:
                raise StudyError(f"candidate inventory/order mismatch for {group}")
            for variant in VARIANT_NAMES:
                texts = build_symmetric_candidate_texts(rows, variant)
                payload = {
                    "status": "pass",
                    "group_id": group,
                    "category": category,
                    "position_stratum": stratum,
                    "variant": variant,
                    "seed": SEED,
                    "candidate_count": 8,
                    "candidate_ids": expected,
                    "candidate_texts": texts,
                    "record_path": str(inventories / f"{category}__{stratum}__{variant}.json"),
                }
                json_dump(inventories / f"{category}__{stratum}__{variant}.json", payload)
                inventory_manifest.append(payload["record_path"])
    json_dump(out / "manifest.json", {"status": "pass", "seed": SEED, "count": len(manifest_rows), "records": manifest_rows})
    json_dump(inventories / "manifest.json", {"status": "pass", "seed": SEED, "count": len(inventory_manifest), "records": inventory_manifest})


def _load_group_inventory(results: Path, category: str, stratum: str, variant: str) -> dict[str, Any]:
    path = results / "inventories" / f"{category}__{stratum}__{variant}.json"
    record = _require_json(path)
    if record.get("candidate_count") != 8 or len(record.get("candidate_ids", ())) != 8:
        raise StudyError(f"inventory is not 8-way: {path}")
    if len(record.get("candidate_texts", ())) != 8:
        raise StudyError(f"inventory texts are not 8-way: {path}")
    if len(set(record["candidate_texts"])) != 8:
        raise StudyError(f"inventory has duplicate texts: {path}")
    return record


def _write_matrix(path: Path, payload: Mapping[str, Any]) -> None:
    json_dump(path, payload)


def stage_ar(args: argparse.Namespace) -> None:
    import torch

    set_determinism()
    prompts = load_context_prompt_spec(args.prompts)
    tokenizer, model, device, dtype = _load_tokenizer_model(args.ar_checkpoint, args.device, AR_REVISION)
    head = _load_ar_head(args.ar_checkpoint, model, dtype, device)
    sidecar = _load_yaml(args.ar_checkpoint / "nla_meta.yaml")
    template, mse_scale = str(sidecar["prompt_templates"]["ar"]), float(sidecar["extraction"]["mse_scale"])
    out = args.results_dir / "ar"
    vectors = args.results_dir / "ar-vectors"
    matrices = args.results_dir / "matrices"
    out.mkdir(parents=True, exist_ok=True)
    vectors.mkdir(parents=True, exist_ok=True)
    matrices.mkdir(parents=True, exist_ok=True)
    for prompt, stratum in _iter_content_targets(prompts):
        stem = _target_stem(prompt["id"], stratum)
        target_meta, gold = _load_extraction_record(args.results_dir, prompt["id"], stratum)
        gold_n = normalize_vector(gold, mse_scale)
        variant_rows: dict[str, Any] = {}
        for variant in VARIANT_NAMES:
            inventory = _load_group_inventory(args.results_dir, prompt["category"], stratum, variant)
            if inventory["candidate_ids"] != _expected_candidate_ids(prompts, prompt["category"], stratum):
                raise StudyError(f"candidate inventory/order mismatch for {stem} {variant}")
            own_id = target_meta["activation_id"]
            if own_id not in inventory["candidate_ids"]:
                raise StudyError(f"group inventory lacks own candidate {own_id}")
            candidates = []
            for candidate_index, (candidate_id, text) in enumerate(zip(inventory["candidate_ids"], inventory["candidate_texts"])):
                vector, token_count = _score_payload_text(tokenizer, model, head, device, template, text)
                vector_n = normalize_vector(vector, mse_scale)
                mse = float(((vector_n - gold_n) ** 2).mean().item())
                vector_path = vectors / f"{stem}__{variant}__candidate-{candidate_index:02d}.pt"
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
            paraphrase = neutral_paraphrase(own["variant_text"])
            para_vector, para_tokens = _score_payload_text(tokenizer, model, head, device, template, paraphrase)
            para_n = normalize_vector(para_vector, mse_scale)
            para_mse = float(((para_n - gold_n) ** 2).mean().item())
            para_path = vectors / f"{stem}__{variant}__paraphrase.pt"
            torch.save(para_vector, para_path)
            variant_rows[variant] = {
                "candidate_ids": list(inventory["candidate_ids"]),
                "candidate_texts": list(inventory["candidate_texts"]),
                "candidates": candidates,
                "paraphrase_text": paraphrase,
                "paraphrase_mse_nrm": para_mse,
                "paraphrase_token_count": para_tokens,
                "paraphrase_vector_path": str(para_path),
                "finite": all(row["finite"] for row in candidates) and finite(para_mse),
            }
        unrelated_vector, unrelated_tokens = _score_payload_text(tokenizer, model, head, device, template, UNRELATED_TEXT)
        unrelated_n = normalize_vector(unrelated_vector, mse_scale)
        unrelated_mse = float(((unrelated_n - gold_n) ** 2).mean().item())
        unrelated_path = vectors / f"{stem}__unrelated.pt"
        torch.save(unrelated_vector, unrelated_path)
        json_dump(out / f"{stem}.json", {
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
            "variants": variant_rows,
            "unrelated_mse_nrm": unrelated_mse,
            "unrelated_token_count": unrelated_tokens,
            "unrelated_vector_path": str(unrelated_path),
            "finite": all(row["finite"] for row in variant_rows.values()) and finite(unrelated_mse),
            "record_path": str(out / f"{stem}.json"),
        })
    for category in CATEGORIES:
        for stratum in CONTENT_STRATA:
            ids = _expected_candidate_ids(prompts, category, stratum)
            for variant in VARIANT_NAMES:
                matrix = []
                texts = None
                for target_id in ids:
                    prompt_id, target_stratum = target_id.split("::", 1)
                    ar_record = _require_json(args.results_dir / "ar" / f"{_target_stem(prompt_id, target_stratum)}.json")
                    block = ar_record["variants"][variant]
                    if block["candidate_ids"] != ids:
                        raise StudyError(f"AR candidate order mismatch for {target_id} {variant}")
                    matrix.append([float(row["mse_nrm"]) for row in block["candidates"]])
                    texts = list(block["candidate_texts"])
                _write_matrix(matrices / f"{category}__{stratum}__{variant}.json", {
                    "status": "pass",
                    "group_id": _group_id(category, stratum),
                    "variant": variant,
                    "seed": SEED,
                    "target_ids": ids,
                    "candidate_ids": ids,
                    "candidate_texts": texts,
                    "ar_mse": matrix,
                    "behavior_jsd": None,
                    "behavior_metric": BEHAVIOR_METRIC_NAME,
                    "record_path": str(matrices / f"{category}__{stratum}__{variant}.json"),
                })
    _run_bridge_ar(args, tokenizer, model, head, device, template, mse_scale)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _run_bridge_ar(args: argparse.Namespace, tokenizer: Any, model: Any, head: Any, device: Any, template: str, mse_scale: float) -> None:
    import torch

    frozen_results = args.frozen_dir / "results"
    rows = []
    for prompt_id, stratum in BRIDGE_TARGETS:
        stem = _target_stem(prompt_id, stratum)
        target_meta, gold = _load_extraction_record(frozen_results, prompt_id, stratum)
        frozen_ar = _require_json(frozen_results / "ar" / f"{stem}.json")
        own_text = next(row["score_text"] for row in frozen_ar["candidates"] if row.get("own"))
        gold_n = normalize_vector(gold, mse_scale)
        vector, token_count = _score_payload_text(tokenizer, model, head, device, template, own_text)
        vector_n = normalize_vector(vector, mse_scale)
        mse = float(((vector_n - gold_n) ** 2).mean().item())
        frozen_mse = float(next(row["mse_nrm"] for row in frozen_ar["candidates"] if row.get("own")))
        path = args.results_dir / "bridge" / "ar-vectors" / f"{stem}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(vector, path)
        rows.append({
            "stem": stem,
            "activation_id": target_meta["activation_id"],
            "recomputed_mse": mse,
            "frozen_mse": frozen_mse,
            "token_count": token_count,
            "vector_path": str(path),
            "finite": finite(mse) and bool(torch.isfinite(vector).all()),
        })
    json_dump(args.results_dir / "bridge" / "ar.json", {
        "status": "pass",
        "seed": SEED,
        "target_count": len(rows),
        "targets": rows,
    })


def stage_behavior(args: argparse.Namespace) -> None:
    import torch

    set_determinism()
    prompts = load_context_prompt_spec(args.prompts)
    tokenizer, model, device, dtype = _load_tokenizer_model(args.base_checkpoint, args.device, BASE_REVISION)
    out = args.results_dir / "behavior"
    out.mkdir(parents=True, exist_ok=True)
    for prompt, stratum in _iter_content_targets(prompts):
        stem = _target_stem(prompt["id"], stratum)
        ar_record = _require_json(args.results_dir / "ar" / f"{stem}.json")
        target_meta, gold = _load_extraction_record(args.results_dir, prompt["id"], stratum)
        alignment = target_meta.get("alignment")
        if not isinstance(alignment, Mapping):
            raise StudyError(f"missing rendered alignment for {target_meta['activation_id']}")
        ids = torch.tensor([alignment["token_ids"]], device=device)
        mask = torch.ones_like(ids)
        position = int(target_meta["position"])
        baseline = _base_logits(model, ids, mask)
        gold_logits = hooked_logits(model, ids, mask, position, gold.to(device))
        gold_jsd = js_divergence(baseline[position], gold_logits[position])
        norm = float(gold.norm().item())
        random_seed = SEED + int(prompt["index"]) * N_POSITIONS_PER_PROMPT + CONTENT_STRATA.index(stratum) + 1000
        generator = torch.Generator(device="cpu").manual_seed(random_seed)
        random_vector = torch.randn(gold.shape, generator=generator)
        random_vector = random_vector / random_vector.norm().clamp_min(1e-12) * norm
        random_logits = hooked_logits(model, ids, mask, position, random_vector.to(device))
        variant_rows: dict[str, Any] = {}
        for variant in VARIANT_NAMES:
            block = ar_record["variants"][variant]
            candidates = []
            for row in block["candidates"]:
                vector = torch.load(row["vector_path"], map_location="cpu", weights_only=True).float()
                replacement = vector / vector.norm().clamp_min(1e-12) * norm
                logits = hooked_logits(model, ids, mask, position, replacement.to(device))
                jsd = js_divergence(baseline[position], logits[position])
                candidates.append({
                    "candidate_id": row["candidate_id"],
                    "own": row["own"],
                    "jsd": jsd,
                    "vector_path": row["vector_path"],
                    "finite": finite(jsd),
                    "metric": BEHAVIOR_METRIC_NAME,
                })
            para_vector = torch.load(block["paraphrase_vector_path"], map_location="cpu", weights_only=True).float()
            para_replacement = para_vector / para_vector.norm().clamp_min(1e-12) * norm
            para_logits = hooked_logits(model, ids, mask, position, para_replacement.to(device))
            para_jsd = js_divergence(baseline[position], para_logits[position])
            variant_rows[variant] = {
                "candidate_ids": list(block["candidate_ids"]),
                "candidates": candidates,
                "paraphrase_jsd": para_jsd,
                "finite": all(row["finite"] for row in candidates) and finite(para_jsd),
            }
        json_dump(out / f"{stem}.json", {
            "status": "pass",
            "activation_id": target_meta["activation_id"],
            "prompt_id": prompt["id"],
            "prompt_index": int(prompt["index"]),
            "category": prompt["category"],
            "position_stratum": stratum,
            "position": position,
            "layer_index": LAYER_INDEX,
            "seed": SEED,
            "random_direction_seed": random_seed,
            "checkpoint": {"repo": BASE_REPO, "revision": BASE_REVISION},
            "behavior_metric": BEHAVIOR_METRIC_NAME,
            "gold_reinjection_jsd": gold_jsd,
            "random_direction_jsd": js_divergence(baseline[position], random_logits[position]),
            "variants": variant_rows,
            "finite": finite(gold_jsd) and all(row["finite"] for row in variant_rows.values()),
            "record_path": str(out / f"{stem}.json"),
        })
    for category in CATEGORIES:
        for stratum in CONTENT_STRATA:
            ids = _expected_candidate_ids(prompts, category, stratum)
            for variant in VARIANT_NAMES:
                path = args.results_dir / "matrices" / f"{category}__{stratum}__{variant}.json"
                matrix_record = _require_json(path)
                jsd_matrix = []
                for target_id in ids:
                    prompt_id, target_stratum = target_id.split("::", 1)
                    behavior = _require_json(args.results_dir / "behavior" / f"{_target_stem(prompt_id, target_stratum)}.json")
                    block = behavior["variants"][variant]
                    if block["candidate_ids"] != ids:
                        raise StudyError(f"behavior candidate order mismatch for {target_id} {variant}")
                    jsd_matrix.append([float(row["jsd"]) for row in block["candidates"]])
                matrix_record["behavior_jsd"] = jsd_matrix
                matrix_record["behavior_metric"] = BEHAVIOR_METRIC_NAME
                _write_matrix(path, matrix_record)
    _run_bridge_behavior(args, tokenizer, model, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _run_bridge_behavior(args: argparse.Namespace, tokenizer: Any, model: Any, device: Any) -> None:
    import torch

    frozen_results = args.frozen_dir / "results"
    rows = []
    for prompt_id, stratum in BRIDGE_TARGETS:
        stem = _target_stem(prompt_id, stratum)
        target_meta, gold = _load_extraction_record(frozen_results, prompt_id, stratum)
        frozen_behavior = _require_json(frozen_results / "behavior" / f"{stem}.json")
        alignment = target_meta.get("alignment")
        if not isinstance(alignment, Mapping):
            raise StudyError(f"missing frozen alignment for bridge {stem}")
        ids = torch.tensor([alignment["token_ids"]], device=device)
        mask = torch.ones_like(ids)
        position = int(target_meta["position"])
        baseline = _base_logits(model, ids, mask)
        vector = torch.load(args.results_dir / "bridge" / "ar-vectors" / f"{stem}.pt", map_location="cpu", weights_only=True).float()
        replacement = vector / vector.norm().clamp_min(1e-12) * float(gold.norm().item())
        logits = hooked_logits(model, ids, mask, position, replacement.to(device))
        jsd = js_divergence(baseline[position], logits[position])
        _, frozen_jsd = _frozen_own_scores(
            _require_json(frozen_results / "ar" / f"{stem}.json"),
            frozen_behavior,
        )
        rows.append({
            "stem": stem,
            "activation_id": target_meta["activation_id"],
            "recomputed_jsd": jsd,
            "frozen_jsd": frozen_jsd,
            "finite": finite(jsd),
        })
    json_dump(args.results_dir / "bridge" / "behavior.json", {
        "status": "pass",
        "seed": SEED,
        "target_count": len(rows),
        "targets": rows,
    })


def tournament_margin(values: Sequence[float], own_index: int) -> float:
    """Calculate tournament margin: median(other seven) - own.

    Metric values are errors, so larger output margin is better.
    """
    if len(values) != 8:
        raise StudyError(f"tournament margin requires exactly 8 values, got {len(values)}")
    if not (0 <= int(own_index) < 8):
        raise StudyError(f"own_index {own_index} out of bounds for 8 values")
    other_values = [float(values[i]) for i in range(8) if i != int(own_index)]
    if not all(finite(v) for v in other_values) or not finite(values[int(own_index)]):
        raise StudyError("non-finite value in tournament margin calculation")
    other_median = ov._median(other_values)
    if other_median is None:
        raise StudyError("failed to compute median of other candidate values")
    return float(other_median) - float(values[int(own_index)])


def pair_delta(target_margin: float, baseline_margin: float) -> float:
    """Positive when the target tournament margin is larger than baseline."""
    if not finite(target_margin) or not finite(baseline_margin):
        raise StudyError("pair delta requires finite tournament margins")
    return float(target_margin) - float(baseline_margin)


def _own_index(candidates: Sequence[Mapping[str, Any]], activation_id: str) -> int:
    indices = [i for i, row in enumerate(candidates) if row.get("own") is True]
    if len(indices) != 1:
        raise StudyError(f"candidate ownership is not unique for {activation_id}")
    if candidates[indices[0]].get("candidate_id") != activation_id:
        raise StudyError(f"own candidate id mismatch for {activation_id}")
    return indices[0]


def _metrics_for_variant(
    ar_block: Mapping[str, Any],
    behavior_block: Mapping[str, Any],
    activation_id: str,
) -> dict[str, Any]:
    if list(ar_block.get("candidate_ids", ())) != list(behavior_block.get("candidate_ids", ())):
        raise StudyError(f"AR/behavior candidate order mismatch for {activation_id}")
    ar_own = _own_index(ar_block["candidates"], activation_id)
    bh_own = _own_index(behavior_block["candidates"], activation_id)
    if ar_own != bh_own:
        raise StudyError(f"AR/behavior own index mismatch for {activation_id}")
    ar_values = [float(row["mse_nrm"]) for row in ar_block["candidates"]]
    jsd_values = [float(row["jsd"]) for row in behavior_block["candidates"]]
    if not all(finite(value) for value in ar_values + jsd_values):
        raise StudyError(f"non-finite same-variant scores for {activation_id}")
    ar_margin = tournament_margin(ar_values, ar_own)
    jsd_margin = tournament_margin(jsd_values, bh_own)
    metrics = _strong_metrics(
        ar_values,
        jsd_values,
        ar_own,
        float(ar_block["paraphrase_mse_nrm"]),
        float(behavior_block["paraphrase_jsd"]),
    )
    return {
        **metrics,
        "own_mse": ar_values[ar_own],
        "own_jsd": jsd_values[ar_own],
        "ar_tournament_margin": ar_margin,
        "jsd_tournament_margin": jsd_margin,
        "paraphrase_mse": float(ar_block["paraphrase_mse_nrm"]),
        "paraphrase_jsd": float(behavior_block["paraphrase_jsd"]),
        "ar_values": ar_values,
        "jsd_values": jsd_values,
        "candidate_ids": list(ar_block["candidate_ids"]),
    }


def _comparison_bundle(records: Sequence[Mapping[str, Any]], ar_key: str, jsd_key: str) -> dict[str, Any]:
    ar_flip = prompt_cluster_sign_flip(records, ar_key)
    jsd_flip = prompt_cluster_sign_flip(records, jsd_key)
    ar_median = float(ar_flip["observed"])
    jsd_median = float(jsd_flip["observed"])
    return {
        "ar": ar_flip,
        "jsd": jsd_flip,
        "ar_median": ar_median,
        "jsd_median": jsd_median,
        "ar_significant": ar_median > 0 and float(ar_flip["p_value"]) <= 0.01,
        "jsd_significant": jsd_median > 0 and float(jsd_flip["p_value"]) <= 0.01,
        "n": len(records),
    }


def _bridge_gate(args: argparse.Namespace) -> dict[str, Any]:
    ar = _require_json(args.results_dir / "bridge" / "ar.json")
    behavior = _require_json(args.results_dir / "bridge" / "behavior.json")
    if ar.get("target_count") != len(BRIDGE_TARGETS) or behavior.get("target_count") != len(BRIDGE_TARGETS):
        raise StudyError("bridge target count is not 6")
    if len(ar.get("targets", ())) != len(BRIDGE_TARGETS) or len(behavior.get("targets", ())) != len(BRIDGE_TARGETS):
        raise StudyError("bridge target rows are incomplete")
    by_stem = {row["stem"]: row for row in behavior["targets"]}
    comparisons = []
    passed = True
    for row in ar["targets"]:
        other = by_stem.get(row["stem"])
        if not isinstance(other, Mapping):
            raise StudyError(f"missing bridge behavior row for {row['stem']}")
        comparison = compare_bridge_values(
            row["recomputed_mse"], row["frozen_mse"],
            other["recomputed_jsd"], other["frozen_jsd"],
        )
        comparison["stem"] = row["stem"]
        comparison["activation_id"] = row.get("activation_id")
        comparisons.append(comparison)
        passed = passed and bool(comparison["passed"])
    return {
        "status": "pass" if passed else "fail",
        "passed": passed,
        "mse_tol": GATE_MSE_TOL,
        "jsd_tol": GATE_JSD_TOL,
        "target_count": len(comparisons),
        "targets": comparisons,
    }


def _require_preflight(args: argparse.Namespace) -> dict[str, Any]:
    preflight = _require_json(args.results_dir / "preflight.json")
    digest = hashlib.sha256(args.prompts.read_bytes()).hexdigest()
    if preflight.get("status") != "pass" or preflight.get("seed") != SEED:
        raise StudyError("preflight is not a passing Experiment 3 record")
    if preflight.get("prompt_sha256") != digest or digest != FROZEN_PROMPT_SHA256:
        raise StudyError("preflight prompt hash is not the frozen Experiment 3 inventory")
    if preflight.get("activation_count") != N_CONTENT_TARGETS:
        raise StudyError("preflight activation count is not 48")
    return preflight


def _subset_delta_stats(rows: Sequence[Mapping[str, Any]], comparison: str) -> dict[str, Any]:
    ar_values = [float(row["comparisons"][comparison]["ar_delta"]) for row in rows]
    jsd_values = [float(row["comparisons"][comparison]["jsd_delta"]) for row in rows]
    return {
        "n": len(rows),
        "median_ar_delta": ov._median(ar_values),
        "median_jsd_delta": ov._median(jsd_values),
        "prose_better_ar_count": sum(value > 0 for value in ar_values),
        "prose_better_jsd_count": sum(value > 0 for value in jsd_values),
    }


def stage_decide(args: argparse.Namespace) -> dict[str, Any]:
    prompts = load_context_prompt_spec(args.prompts)
    preflight = _require_preflight(args)
    gate = _bridge_gate(args)
    json_dump(args.results_dir / "gate.json", gate)

    rows: list[dict[str, Any]] = []
    reasons: list[str] = []
    gold_jsds: list[float] = []
    random_jsds: list[float] = []
    own_nla_mse: list[float] = []
    own_nla_jsd: list[float] = []
    unrelated_mse: list[float] = []
    for prompt, stratum in _iter_content_targets(prompts):
        stem = _target_stem(prompt["id"], stratum)
        extract = _require_json(args.results_dir / "extract" / f"{stem}.json")
        variants = _require_json(args.results_dir / "variants" / f"{stem}.json")
        ar_record = _require_json(args.results_dir / "ar" / f"{stem}.json")
        behavior = _require_json(args.results_dir / "behavior" / f"{stem}.json")
        activation_id = _activation_id(prompt["id"], stratum)
        if extract.get("activation_id") != activation_id or ar_record.get("activation_id") != activation_id:
            raise StudyError(f"identity mismatch for {stem}")
        if variants.get("activation_id") != activation_id or behavior.get("activation_id") != activation_id:
            raise StudyError(f"variant/behavior identity mismatch for {stem}")
        if extract.get("layer_index") != LAYER_INDEX or ar_record.get("layer_index") != LAYER_INDEX:
            reasons.append(f"layer provenance mismatch for {stem}")
        if not variants.get("structural_valid"):
            reasons.append(f"AV output is not structurally valid for {stem}")
        gold_jsd = float(behavior["gold_reinjection_jsd"])
        gold_jsds.append(gold_jsd)
        if not finite(gold_jsd) or gold_jsd > GOLD_JSD_MAX:
            reasons.append(f"gold reinjection failed for {stem}")
        variant_metrics = {}
        for variant in VARIANT_NAMES:
            variant_metrics[variant] = _metrics_for_variant(
                ar_record["variants"][variant],
                behavior["variants"][variant],
                activation_id,
            )
        prose = variant_metrics["nla_full"]
        comparisons = {
            "nla_full_vs_local_ctx": {
                "ar_delta": pair_delta(prose["ar_tournament_margin"], variant_metrics["local_ctx"]["ar_tournament_margin"]),
                "jsd_delta": pair_delta(prose["jsd_tournament_margin"], variant_metrics["local_ctx"]["jsd_tournament_margin"]),
            },
            "nla_full_vs_token_only": {
                "ar_delta": pair_delta(prose["ar_tournament_margin"], variant_metrics["token_only"]["ar_tournament_margin"]),
                "jsd_delta": pair_delta(prose["jsd_tournament_margin"], variant_metrics["token_only"]["jsd_tournament_margin"]),
            },
            "nla_drop_final_sym_vs_local_ctx": {
                "ar_delta": pair_delta(variant_metrics["nla_drop_final_sym"]["ar_tournament_margin"], variant_metrics["local_ctx"]["ar_tournament_margin"]),
                "jsd_delta": pair_delta(variant_metrics["nla_drop_final_sym"]["jsd_tournament_margin"], variant_metrics["local_ctx"]["jsd_tournament_margin"]),
            },
            "nla_scrubbed_vs_local_ctx": {
                "ar_delta": pair_delta(variant_metrics["nla_scrubbed"]["ar_tournament_margin"], variant_metrics["local_ctx"]["ar_tournament_margin"]),
                "jsd_delta": pair_delta(variant_metrics["nla_scrubbed"]["jsd_tournament_margin"], variant_metrics["local_ctx"]["jsd_tournament_margin"]),
            },
        }
        random_jsds.append(float(behavior["random_direction_jsd"]))
        own_nla_mse.append(float(prose["own_mse"]))
        own_nla_jsd.append(float(prose["own_jsd"]))
        unrelated_mse.append(float(ar_record["unrelated_mse_nrm"]))
        rows.append({
            "activation_id": activation_id,
            "prompt_id": prompt["id"],
            "prompt_index": int(prompt["index"]),
            "category": prompt["category"],
            "position_stratum": stratum,
            "position": extract["position"],
            "token_text": variants["token_text"],
            "variant_texts": variants["variant_texts"],
            "structural_valid": bool(variants["structural_valid"]),
            "gold_reinjection_jsd": gold_jsd,
            "random_direction_jsd": float(behavior["random_direction_jsd"]),
            "unrelated_mse_nrm": float(ar_record["unrelated_mse_nrm"]),
            "variants": variant_metrics,
            "comparisons": comparisons,
        })
    if len(rows) != N_CONTENT_TARGETS:
        reasons.append(f"expected {N_CONTENT_TARGETS} targets, found {len(rows)}")
    if not gate["passed"]:
        reasons.append("frozen Experiment 1 bridge exceeded tolerances")
    if ov._median(random_jsds) is None or not (ov._median(random_jsds) > ov._median(own_nla_jsd)):
        reasons.append("median random-direction JSD is not greater than median nla_full JSD")
    if ov._median(unrelated_mse) is None or not (ov._median(unrelated_mse) > ov._median(own_nla_mse)):
        reasons.append("median unrelated AR MSE is not greater than median nla_full MSE")

    for category in CATEGORIES:
        for stratum in CONTENT_STRATA:
            group = _group_id(category, stratum)
            group_rows = [row for row in rows if row["category"] == category and row["position_stratum"] == stratum]
            if len(group_rows) != 8:
                reasons.append(f"group {group} does not have 8 targets, found {len(group_rows)}")
                continue
            for variant in VARIANT_NAMES:
                candidate_texts = [row["variant_texts"][variant] for row in group_rows]
                if len(candidate_texts) != 8 or len(set(candidate_texts)) != 8 or any(not isinstance(t, str) or not t for t in candidate_texts):
                    reasons.append(f"duplicate or invalid candidate texts for group {group} variant {variant}")
                for row in group_rows:
                    v_metrics = row["variants"][variant]
                    if len(v_metrics["candidate_ids"]) != 8:
                        reasons.append(f"candidate count is not 8 for {row['activation_id']} variant {variant}")
                    if len(v_metrics["ar_values"]) != 8 or not all(finite(v) for v in v_metrics["ar_values"]):
                        reasons.append(f"non-finite AR candidate scores for {row['activation_id']} variant {variant}")
                    if len(v_metrics["jsd_values"]) != 8 or not all(finite(v) for v in v_metrics["jsd_values"]):
                        reasons.append(f"non-finite behavior candidate scores for {row['activation_id']} variant {variant}")
                    if not finite(v_metrics["paraphrase_mse"]) or not finite(v_metrics["paraphrase_jsd"]):
                        reasons.append(f"non-finite paraphrase scores for {row['activation_id']} variant {variant}")

    primary_records = [
        {
            "prompt_id": row["prompt_id"],
            "ar_delta": row["comparisons"]["nla_full_vs_local_ctx"]["ar_delta"],
            "jsd_delta": row["comparisons"]["nla_full_vs_local_ctx"]["jsd_delta"],
        }
        for row in rows
    ]
    secondary_records = [
        {
            "prompt_id": row["prompt_id"],
            "ar_delta": row["comparisons"]["nla_full_vs_token_only"]["ar_delta"],
            "jsd_delta": row["comparisons"]["nla_full_vs_token_only"]["jsd_delta"],
        }
        for row in rows
    ]
    reversed_primary_records = [
        {
            "prompt_id": row["prompt_id"],
            "ar_delta": pair_delta(
                row["variants"]["local_ctx"]["ar_tournament_margin"],
                row["variants"]["nla_full"]["ar_tournament_margin"],
            ),
            "jsd_delta": pair_delta(
                row["variants"]["local_ctx"]["jsd_tournament_margin"],
                row["variants"]["nla_full"]["jsd_tournament_margin"],
            ),
        }
        for row in rows
    ]
    primary = _comparison_bundle(primary_records, "ar_delta", "jsd_delta")
    secondary = _comparison_bundle(secondary_records, "ar_delta", "jsd_delta")
    reversed_primary = _comparison_bundle(reversed_primary_records, "ar_delta", "jsd_delta")

    replicated = bool(secondary["ar_significant"] and secondary["jsd_significant"])

    nla_full_joint_count = sum(bool(row["variants"]["nla_full"]["joint_specific"]) for row in rows)
    nla_full_joint_rate = (nla_full_joint_count / len(rows)) if rows else 0.0
    operational = not reasons
    classification = classify_context_decision(
        operational,
        nla_full_joint_rate,
        nla_full_joint_count,
        primary["ar_median"],
        float(primary["ar"]["p_value"]),
        primary["jsd_median"],
        float(primary["jsd"]["p_value"]),
        secondary["ar_median"],
        float(secondary["ar"]["p_value"]),
        secondary["jsd_median"],
        float(secondary["jsd"]["p_value"]),
        reversed_primary["ar_median"],
        float(reversed_primary["ar"]["p_value"]),
        reversed_primary["jsd_median"],
        float(reversed_primary["jsd"]["p_value"]),
    )
    if classification not in DECISION_LABELS:
        raise StudyError(f"unknown decision label: {classification}")

    def variant_rate(name: str) -> dict[str, Any]:
        subset = [row["variants"][name] for row in rows]
        return {
            "n": len(subset),
            "joint_specific_count": sum(bool(row["joint_specific"]) for row in subset),
            "joint_specific_rate": (sum(bool(row["joint_specific"]) for row in subset) / len(subset)) if subset else None,
            "strong_ar_rate": (sum(bool(row["strong_ar"]) for row in subset) / len(subset)) if subset else None,
            "strong_behavior_rate": (sum(bool(row["strong_behavior"]) for row in subset) / len(subset)) if subset else None,
        }

    breakdown = {
        "overall_primary": _subset_delta_stats(rows, "nla_full_vs_local_ctx"),
        "overall_secondary": _subset_delta_stats(rows, "nla_full_vs_token_only"),
    }
    for category in CATEGORIES:
        cat_rows = [row for row in rows if row["category"] == category]
        breakdown[f"category::{category}::primary"] = _subset_delta_stats(cat_rows, "nla_full_vs_local_ctx")
    for stratum in CONTENT_STRATA:
        stratum_rows = [row for row in rows if row["position_stratum"] == stratum]
        breakdown[f"position_stratum::{stratum}::primary"] = _subset_delta_stats(stratum_rows, "nla_full_vs_local_ctx")

    result = {
        "status": "pass" if operational else "invalid",
        "study": "context-baselines",
        "experiment": 3,
        "classification": classification,
        "operational_validity": operational,
        "invalid_reasons": sorted(set(reasons)),
        "seed": SEED,
        "prompt_sha256": preflight["prompt_sha256"],
        "target_count": len(rows),
        "behavior_metric": BEHAVIOR_METRIC_NAME,
        "primary_comparison": "nla_full_vs_local_ctx",
        "secondary_comparison": "nla_full_vs_token_only",
        "replicated_on_token_only": replicated,
        "primary": primary,
        "secondary": secondary,
        "reversed_primary": reversed_primary,
        "descriptive": {
            "nla_drop_final_sym_vs_local_ctx": _subset_delta_stats(rows, "nla_drop_final_sym_vs_local_ctx"),
            "nla_scrubbed_vs_local_ctx": _subset_delta_stats(rows, "nla_scrubbed_vs_local_ctx"),
            "same_variant_specificity": {name: variant_rate(name) for name in VARIANT_NAMES},
        },
        "breakdown": breakdown,
        "gate_path": str(args.results_dir / "gate.json"),
        "gate_status": gate["status"],
        "controls": {
            "gold_reinjection_max_jsd": max(gold_jsds) if gold_jsds else None,
            "gold_jsd_max": GOLD_JSD_MAX,
            "median_random_direction_jsd": ov._median(random_jsds),
            "median_nla_full_jsd": ov._median(own_nla_jsd),
            "median_unrelated_ar_mse": ov._median(unrelated_mse),
            "median_nla_full_mse": ov._median(own_nla_mse),
        },
        "rows": rows,
        "config": {
            "base": [BASE_REPO, BASE_REVISION],
            "av": [AV_REPO, AV_REVISION],
            "ar": [AR_REPO, AR_REVISION],
            "layer_index": LAYER_INDEX,
            "hf_hidden_states_index": HF_HIDDEN_STATES_INDEX,
            "position_strata": list(CONTENT_STRATA),
            "variants": list(VARIANT_NAMES),
            "sign_flip_draws": N_PERMUTATIONS,
            "chance_top1": CHANCE_TOP1,
            "window_radius": WINDOW_RADIUS,
        },
        "frozen_study_dir": str(args.frozen_dir),
    }
    json_dump(args.results_dir / "decision.json", result)
    if result.get("status") == "pass":
        write_completion_manifest(args)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fresh-prompt same-variant context-baseline study (Experiment 3)")
    ap.add_argument("--stage", choices=("preflight", "av", "variants", "ar", "behavior", "decide", "all"), required=True)
    ap.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    ap.add_argument("--frozen-dir", type=Path, default=DEFAULT_FROZEN_DIR)
    ap.add_argument("--results-dir", type=Path)
    ap.add_argument("--prompts", type=Path)
    ap.add_argument("--base-checkpoint", type=Path)
    ap.add_argument("--av-checkpoint", type=Path)
    ap.add_argument("--ar-checkpoint", type=Path)
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = ap.parse_args(argv)
    args.study_dir = args.study_dir.resolve()
    args.frozen_dir = args.frozen_dir.resolve()
    args.results_dir = (args.results_dir or args.study_dir / "results").resolve()
    args.prompts = (args.prompts or args.study_dir / "prompts.json").resolve()
    weights = args.study_dir.parent / "weights"
    args.base_checkpoint = (args.base_checkpoint or weights / "base-qwen").resolve()
    args.av_checkpoint = (args.av_checkpoint or weights / "av").resolve()
    args.ar_checkpoint = (args.ar_checkpoint or weights / "ar").resolve()
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    stages = {
        "preflight": stage_preflight,
        "av": stage_av,
        "variants": stage_variants,
        "ar": stage_ar,
        "behavior": stage_behavior,
        "decide": stage_decide,
    }
    if args.stage == "all":
        for stage in ("preflight", "av", "variants", "ar", "behavior", "decide"):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--stage", stage,
                "--study-dir", str(args.study_dir),
                "--frozen-dir", str(args.frozen_dir),
                "--results-dir", str(args.results_dir),
                "--prompts", str(args.prompts),
                "--base-checkpoint", str(args.base_checkpoint),
                "--av-checkpoint", str(args.av_checkpoint),
                "--ar-checkpoint", str(args.ar_checkpoint),
                "--device", args.device,
            ]
            subprocess.run(command, check=True)
        return
    stages[args.stage](args)


if __name__ == "__main__":
    main()
       