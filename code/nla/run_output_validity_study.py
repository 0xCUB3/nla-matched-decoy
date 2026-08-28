#!/usr/bin/env python3
"""Frozen NLA output-validity study.

The implementation is deliberately staged: model stages are small, separate
processes and every later stage requires the complete raw records produced by
its predecessors.  The dependency-light parsers and statistics are kept here
so they can be tested without loading the 7B checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

SEED = 20260812
LAYER_INDEX = 20
HF_HIDDEN_STATES_INDEX = 21
MAX_NEW_TOKENS = 180
MAX_PRIMARY_CLAUSES = 8
N_PROMPTS = 24
N_POSITIONS_PER_PROMPT = 3
N_ACTIVATIONS = 72
N_PERMUTATIONS = 10_000
CHANCE_TOP1 = 0.125
CATEGORIES = ("safety", "compositional_planning", "social_character_ood")
POSITION_STRATA = ("content_early", "content_late", "boundary_after_user")
UNRELATED_TEXT = "A purple bicycle is parked beside a weekday calendar in a quiet room."
BASE_REPO = "Qwen/Qwen2.5-7B-Instruct"
BASE_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
AV_REPO = "kitft/nla-qwen2.5-7b-L20-av"
AV_REVISION = "b88469162777ae6553bc14208eb0cb579336f8f4"
AR_REPO = "kitft/nla-qwen2.5-7b-L20-ar"
AR_REVISION = "e2c9e57eac213d37a31612087f645ab6332c1bb6"
FROZEN_PROMPT_SHA256 = "8b3349189fbfa38fa07b2a5e098d1275d8f708c48646909147790f87f5df0300"
DEFAULT_STUDY_DIR = Path(__file__).resolve().parents[2] / "pilots/wildcard-nla/output-validity"


class StudyError(RuntimeError):
    """A missing or malformed protocol artifact is a loud technical failure."""


GateError = StudyError  # convenient compatibility name for tests and callers


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(v) for v in value)
    return True


def set_determinism(seed: int = SEED) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_prompt_spec(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text())
    prompts = raw.get("prompts")
    if raw.get("schema_version") != 1 or raw.get("seed") != SEED or not isinstance(prompts, list):
        raise StudyError("prompt file does not contain the frozen schema/seed")
    if len(prompts) != N_PROMPTS:
        raise StudyError(f"expected {N_PROMPTS} prompts, found {len(prompts)}")
    ids: set[str] = set(); texts: set[str] = set(); counts = {c: 0 for c in CATEGORIES}
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
        ids.add(identifier); texts.add(text); counts[category] += 1
    if counts != {c: 8 for c in CATEGORIES}:
        raise StudyError(f"prompt category counts are not 8/8/8: {counts}")
    return prompts


def _input_ids(tokenized: Any) -> list[int]:
    if isinstance(tokenized, Mapping):
        tokenized = tokenized["input_ids"]
    if hasattr(tokenized, "detach"):
        tokenized = tokenized.detach().cpu().tolist()
    if tokenized and isinstance(tokenized[0], (list, tuple)):
        tokenized = tokenized[0]
    return [int(x) for x in tokenized]


def _offsets(tokenized: Any) -> list[tuple[int, int]]:
    if isinstance(tokenized, Mapping):
        tokenized = tokenized["offset_mapping"]
    if hasattr(tokenized, "detach"):
        tokenized = tokenized.detach().cpu().tolist()
    if tokenized and isinstance(tokenized[0], (list, tuple)) and tokenized[0] and isinstance(tokenized[0][0], (list, tuple)):
        tokenized = tokenized[0]
    return [(int(x[0]), int(x[1])) for x in tokenized]


def _tokenize_rendered(tokenizer: Any, rendered: str, *, offsets: bool = False) -> Any:
    kwargs = {"add_special_tokens": False}
    if offsets:
        kwargs["return_offsets_mapping"] = True
    return tokenizer(rendered, **kwargs)


def _decode_token(tokenizer: Any, token_id: int) -> str:
    value = tokenizer.decode([int(token_id)], skip_special_tokens=False) if hasattr(tokenizer, "decode") else str(token_id)
    return str(value)


def rendered_chat_positions(tokenizer: Any, prompt: str) -> dict[str, Any]:
    """Select the two content quantiles and first post-user boundary token.

    The rendered character span, not a separately tokenized prompt, is the
    alignment authority.  Missing, repeated, or non-distinct candidates fail.
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    if not isinstance(rendered, str):
        raise StudyError("chat template did not return rendered text")
    start = rendered.find(prompt)
    end = rendered.rfind(prompt)
    if not prompt or start < 0 or start != end:
        raise StudyError("literal user prompt span is missing or non-unique")
    prompt_end = start + len(prompt)
    tokenized = _tokenize_rendered(tokenizer, rendered, offsets=True)
    ids = _input_ids(tokenized)
    offsets = _offsets(tokenized)
    if len(ids) != len(offsets) or len(ids) < 2:
        raise StudyError("rendered tokenization lacks aligned offsets")
    specials = set(int(x) for x in getattr(tokenizer, "all_special_ids", ()))
    decoded = [_decode_token(tokenizer, token_id) for token_id in ids]
    content = []
    for position, ((left, right), text) in enumerate(zip(offsets, decoded)):
        overlaps = left < prompt_end and right > start
        if overlaps and any(char.isalnum() for char in text):
            content.append(position)
    if not content:
        raise StudyError("no alphanumeric prompt-overlapping content token")
    early = content[(len(content) - 1) // 3]
    late = content[(2 * (len(content) - 1)) // 3]
    boundary = []
    for position, ((left, _right), text) in enumerate(zip(offsets, decoded)):
        if position >= len(ids) - 1:
            continue
        is_special = ids[position] in specials or bool(re.fullmatch(r"<\|[^|]+\|>", text))
        if left >= prompt_end and is_special:
            boundary.append(position)
    if not boundary:
        raise StudyError("no special boundary token after user span")
    positions = {"content_early": early, "content_late": late, "boundary_after_user": boundary[0]}
    if len(set(positions.values())) != 3:
        raise StudyError(f"position strata are not distinct: {positions}")
    return {
        "rendered_text": rendered, "prompt_span": [start, prompt_end],
        "token_ids": ids, "token_offsets": [list(x) for x in offsets],
        "token_texts": decoded, "positions": {
            name: {"position": int(pos), "token_id": int(ids[pos]), "token_text": decoded[pos]}
            for name, pos in positions.items()
        },
    }


# Compatibility aliases make the frozen position rule easy to find in audits.
select_positions = rendered_chat_positions
select_target_positions = rendered_chat_positions


def _clean_clause(value: str) -> str:
    return re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", value).strip()


def parse_primary_clauses(text: str, cap: int = MAX_PRIMARY_CLAUSES) -> list[str]:
    if cap < 1:
        raise StudyError("parser cap must be positive")
    lines = [_clean_clause(x) for x in text.splitlines() if _clean_clause(x)]
    clauses = lines if len(lines) >= 2 else [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]
    clauses = [x for x in clauses if x]
    if len(clauses) > cap:
        clauses = clauses[: cap - 1] + [" ".join(clauses[cap - 1:])]
    return clauses


def parse_secondary_clauses(text: str, cap: int = MAX_PRIMARY_CLAUSES) -> list[str]:
    if cap < 1:
        raise StudyError("parser cap must be positive")
    text = re.sub(r"(?:^|\n)\s*(?:[-*•]|\d+[.)])\s*", " ", text)
    clauses = [x.strip(" \t\r\n;,") for x in re.split(r"(?:;|(?<=[.!?])\s+)", text) if x.strip(" \t\r\n;,")]
    if len(clauses) > cap:
        clauses = clauses[: cap - 1] + [" ".join(clauses[cap - 1:])]
    return [x for x in clauses if x]


def count_wrapper_tags(raw_text: str) -> dict[str, Any]:
    opening = list(re.finditer(r"<explanation>", raw_text, flags=re.I))
    closing = list(re.finditer(r"</explanation>", raw_text, flags=re.I))
    wrapper_valid = len(opening) == 1 and len(closing) == 1 and opening[0].start() < closing[0].start()
    score_text = (raw_text[opening[0].end():closing[0].start()].strip() if wrapper_valid else raw_text)
    return {
        "opening_tag_count": len(opening), "closing_tag_count": len(closing),
        "opening_positions": [m.start() for m in opening], "closing_positions": [m.start() for m in closing],
        "wrapper_valid": wrapper_valid, "score_text": score_text,
    }


def score_text(raw_text: str) -> str:
    """Apply the frozen wrapper rule and return the exact text to score."""
    return count_wrapper_tags(raw_text)["score_text"]


def wrapper_validity(raw_text: str) -> dict[str, Any]:
    return count_wrapper_tags(raw_text)


def validate_av_output(raw_text: str, generated_token_count: int, eos_terminated: bool,
                       special_token_strings: Sequence[str] = ()) -> dict[str, Any]:
    wrapper = count_wrapper_tags(raw_text)
    clauses = parse_primary_clauses(wrapper["score_text"])
    literal_special = bool(re.search(r"<\|[^|]+\|>", wrapper["score_text"]))
    structural = bool(
        wrapper["wrapper_valid"] and len(clauses) in (2, 3) and not literal_special
        and not (int(generated_token_count) >= MAX_NEW_TOKENS and not eos_terminated)
    )
    return {
        **wrapper, "generated_token_count": int(generated_token_count),
        "eos_terminated": bool(eos_terminated), "hit_token_cap_without_eos": bool(
            int(generated_token_count) >= MAX_NEW_TOKENS and not eos_terminated),
        "literal_special_token_strings": list(special_token_strings),
        "primary_clauses": clauses, "primary_clause_count": len(clauses),
        "literal_special_token_in_score_text": literal_special, "structural_valid": structural,
    }


def neutral_paraphrase(text: str) -> str:
    replacements = (("The ", "This "), ("the ", "that "), (" likely ", " probably "),
                    (" suggests ", " indicates "), (" should ", " ought to "))
    result = text
    for old, new in replacements:
        result = result.replace(old, new)
    if result == text:
        result = text.rstrip(" .") + " in neutral terms."
    return result.strip()


def _target_stem(prompt_id: str, stratum: str) -> str:
    return f"{prompt_id}__{stratum}"


def _load_yaml(path: Path) -> Mapping[str, Any]:
    import yaml
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, Mapping):
        raise StudyError(f"malformed YAML sidecar: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest_model(path: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    files = spec.get("downloaded_files")
    if not path.is_dir() or not isinstance(files, list) or not files:
        raise StudyError(f"missing checkpoint or manifest entries: {path}")
    verified = []
    for record in files:
        if not isinstance(record, Mapping):
            raise StudyError(f"malformed manifest entry for {path}")
        relative, expected_size, expected_hash = record.get("path"), record.get("size_bytes"), record.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_size, int) or not isinstance(expected_hash, str):
            raise StudyError(f"malformed checkpoint manifest entry for {path}: {record!r}")
        target = path / relative
        if not target.is_file():
            raise StudyError(f"manifest file missing: {target}")
        actual_size = target.stat().st_size
        if actual_size != expected_size:
            raise StudyError(f"manifest size mismatch for {target}: {actual_size} != {expected_size}")
        actual_hash = _sha256_file(target)
        if actual_hash != expected_hash:
            raise StudyError(f"manifest SHA-256 mismatch for {target}")
        verified.append(relative)
    return {"status": "pass", "file_count": len(verified), "files": verified}


def _manifest_path(study_dir: Path) -> Path:
    return study_dir.parent / "raw" / "checkpoint-manifest.json"


def verify_checkpoint_manifest(args: argparse.Namespace) -> dict[str, Any]:
    path = _manifest_path(args.study_dir)
    if not path.is_file():
        raise StudyError(f"missing frozen checkpoint manifest: {path}")
    manifest = json.loads(path.read_text())
    models = manifest.get("models", {})
    roles = {
        "base": ("base_qwen", args.base_checkpoint, BASE_REPO, BASE_REVISION),
        "av": ("av", args.av_checkpoint, AV_REPO, AV_REVISION),
        "ar": ("ar", args.ar_checkpoint, AR_REPO, AR_REVISION),
    }
    result: dict[str, Any] = {"status": "pass", "manifest_path": str(path),
                              "manifest_sha256": _sha256_file(path), "models": {}}
    for role, (key, checkpoint, repo, revision) in roles.items():
        spec = models.get(key)
        if not isinstance(spec, Mapping) or spec.get("repo") != repo or spec.get("revision") != revision:
            raise StudyError(f"manifest provenance mismatch for {role}")
        result["models"][role] = {"repo": repo, "revision": revision, "path": str(checkpoint),
                                  **verify_manifest_model(checkpoint, spec)}
    return result


def _require_checkpoint(path: Path, repo: str) -> None:
    if not path.is_dir():
        raise StudyError(f"missing local checkpoint: {path}")
    if repo != BASE_REPO and not (path / "nla_meta.yaml").is_file():
        raise StudyError(f"missing NLA sidecar: {path / 'nla_meta.yaml'}")


def _load_args_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Path]:
    prompts = load_prompt_spec(args.prompts)
    return prompts, args.results_dir


def stage_validate(args: argparse.Namespace) -> None:
    prompts = load_prompt_spec(args.prompts)
    prompt_hash = hashlib.sha256(args.prompts.read_bytes()).hexdigest()
    if prompt_hash != FROZEN_PROMPT_SHA256:
        raise StudyError("prompts.json does not match the frozen confirmatory file")
    for checkpoint, repo in ((args.base_checkpoint, BASE_REPO), (args.av_checkpoint, AV_REPO), (args.ar_checkpoint, AR_REPO)):
        _require_checkpoint(checkpoint, repo)
    manifest = verify_checkpoint_manifest(args)
    json_dump(args.results_dir / "validation.json", {
        "status": "pass", "seed": SEED, "prompt_file": str(args.prompts), "prompt_sha256": prompt_hash,
        "prompt_count": len(prompts), "category_counts": {c: sum(p["category"] == c for p in prompts) for c in CATEGORIES},
        "layer_index": LAYER_INDEX, "hf_hidden_states_index": HF_HIDDEN_STATES_INDEX,
        "position_strata": list(POSITION_STRATA), "activation_count": N_ACTIVATIONS,
        "causal_hook": "model.model.layers[20] output, only target position", "local_only": True,
        "max_new_tokens": MAX_NEW_TOKENS, "candidate_group_size": 8,
        "statistics": {"seed": SEED, "permutations": N_PERMUTATIONS, "chance": CHANCE_TOP1},
        "checkpoints": {"base": {"repo": BASE_REPO, "revision": BASE_REVISION, "path": str(args.base_checkpoint)},
                        "av": {"repo": AV_REPO, "revision": AV_REVISION, "path": str(args.av_checkpoint)},
                        "ar": {"repo": AR_REPO, "revision": AR_REVISION, "path": str(args.ar_checkpoint)}},
        "manifest_verification": manifest,
    })


def _load_tokenizer_model(path: Path, device_name: str, revision: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if device_name == "cuda" or (device_name == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(path, revision=revision, local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(path, revision=revision, local_files_only=True,
        torch_dtype=dtype, low_cpu_mem_usage=True, attn_implementation="sdpa", trust_remote_code=False).to(device).eval()
    return tokenizer, model, device, dtype


def _load_extraction_record(results: Path, prompt_id: str, stratum: str) -> tuple[dict[str, Any], Any]:
    import torch
    stem = _target_stem(prompt_id, stratum)
    meta_path, tensor_path = results / "extract" / f"{stem}.json", results / "extract" / f"{stem}.pt"
    if not meta_path.is_file() or not tensor_path.is_file():
        raise StudyError(f"missing extraction record or tensor for {stem}")
    meta = json.loads(meta_path.read_text())
    activation = torch.load(tensor_path, map_location="cpu", weights_only=True).float()
    if activation.ndim != 1 or not bool(torch.isfinite(activation).all()):
        raise StudyError(f"invalid activation for {stem}")
    return meta, activation


def _all_target_keys(prompts: Sequence[Mapping[str, Any]]) -> Iterable[tuple[Mapping[str, Any], str]]:
    for prompt in prompts:
        for stratum in POSITION_STRATA:
            yield prompt, stratum


def stage_extract(args: argparse.Namespace) -> None:
    import torch
    set_determinism()
    prompts = load_prompt_spec(args.prompts)
    tokenizer, model, device, dtype = _load_tokenizer_model(args.base_checkpoint, args.device, BASE_REVISION)
    out = args.results_dir / "extract"; out.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for prompt in prompts:
        alignment = rendered_chat_positions(tokenizer, prompt["prompt"])
        encoded = _tokenize_rendered(tokenizer, alignment["rendered_text"])
        ids = torch.tensor([_input_ids(encoded)], device=device)
        mask = torch.ones_like(ids)
        with torch.inference_mode():
            result = model(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
        hidden = result.hidden_states[HF_HIDDEN_STATES_INDEX]
        for stratum in POSITION_STRATA:
            position_info = alignment["positions"][stratum]
            position = int(position_info["position"])
            activation = hidden[0, position].detach().float().cpu()
            activation_id = f"{prompt['id']}::{stratum}"
            stem = _target_stem(prompt["id"], stratum)
            metadata = {
                "status": "pass", "activation_id": activation_id, "prompt_id": prompt["id"],
                "prompt_index": int(prompt["index"]), "category": prompt["category"], "prompt": prompt["prompt"],
                "position_stratum": stratum, "position": position, "token_id": position_info["token_id"],
                "token_text": position_info["token_text"], "layer_index": LAYER_INDEX,
                "hf_hidden_states_index": HF_HIDDEN_STATES_INDEX, "seed": SEED,
                "base_model": BASE_REPO, "base_revision": BASE_REVISION, "device": str(device), "dtype": str(dtype),
                "finite": bool(torch.isfinite(activation).all()), "activation_shape": list(activation.shape),
                "activation_norm_fp32": float(activation.norm().item()), "rendered_prompt_path": str(args.prompts),
                "record_path": str(out / f"{stem}.json"), "activation_path": str(out / f"{stem}.pt"),
                "alignment": alignment,
            }
            torch.save(activation, out / f"{stem}.pt")
            json_dump(out / f"{stem}.json", metadata)
            manifest_rows.append({"activation_id": activation_id, "prompt_id": prompt["id"],
                                  "position_stratum": stratum, "record_path": str(out / f"{stem}.json"),
                                  "activation_path": str(out / f"{stem}.pt")})
    json_dump(out / "manifest.json", {"status": "pass", "seed": SEED, "count": len(manifest_rows), "records": manifest_rows})
    del model
    if device.type == "cuda": torch.cuda.empty_cache()


def _load_av_record(results: Path, prompt_id: str, stratum: str) -> dict[str, Any]:
    path = results / "av" / f"{_target_stem(prompt_id, stratum)}.json"
    if not path.is_file():
        raise StudyError(f"missing AV record: {path}")
    return json.loads(path.read_text())


def stage_av(args: argparse.Namespace) -> None:
    import torch
    set_determinism()
    prompts = load_prompt_spec(args.prompts)
    tokenizer, model, device, dtype = _load_tokenizer_model(args.av_checkpoint, args.device, AV_REVISION)
    meta = _load_yaml(args.av_checkpoint / "nla_meta.yaml")
    template = str(meta["prompt_templates"]["av"]); marker = str(meta["tokens"]["injection_char"])
    marker_id = int(meta["tokens"]["injection_token_id"]); scale = float(meta["extraction"]["injection_scale"])
    out = args.results_dir / "av"; out.mkdir(parents=True, exist_ok=True)
    for prompt, stratum in _all_target_keys(prompts):
        extract_meta, activation = _load_extraction_record(args.results_dir, prompt["id"], stratum)
        content = template.format(injection_char=marker)
        ids = torch.tensor([_input_ids(tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True, return_tensors="pt"))], device=device)
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
            generated = model.generate(inputs_embeds=embeds, attention_mask=torch.ones_like(ids),
                max_new_tokens=MAX_NEW_TOKENS, do_sample=False, eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id)
        generated_ids = generated[0].detach().cpu().tolist()
        raw_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        special_strings = [_decode_token(tokenizer, token_id) for token_id in generated_ids
                           if token_id in set(getattr(tokenizer, "all_special_ids", ())) or re.fullmatch(r"<\|[^|]+\|>", _decode_token(tokenizer, token_id))]
        eos_terminated = tokenizer.eos_token_id is not None and int(tokenizer.eos_token_id) in generated_ids
        stem = _target_stem(prompt["id"], stratum)
        json_dump(out / f"{stem}.json", {
            "status": "pass", "activation_id": extract_meta["activation_id"], "prompt_id": prompt["id"],
            "prompt_index": int(prompt["index"]), "category": prompt["category"], "position_stratum": stratum,
            "position": extract_meta["position"], "layer_index": LAYER_INDEX, "seed": SEED,
            "checkpoint": {"repo": AV_REPO, "revision": AV_REVISION}, "device": str(device), "dtype": str(dtype),
            "injection_position": int(marker_position), "injection_scale": scale,
            "generated_ids": generated_ids, "generated_token_count": len(generated_ids),
            "eos_terminated": bool(eos_terminated), "literal_special_token_strings": special_strings,
            "raw_text": raw_text, "record_path": str(out / f"{stem}.json"),
        })
    del model
    if device.type == "cuda": torch.cuda.empty_cache()


def stage_validity(args: argparse.Namespace) -> None:
    prompts = load_prompt_spec(args.prompts)
    out = args.results_dir / "validity"; out.mkdir(parents=True, exist_ok=True)
    candidates_by_group: dict[str, list[dict[str, Any]]] = {}
    group_order: list[str] = []
    for prompt, stratum in _all_target_keys(prompts):
        av = _load_av_record(args.results_dir, prompt["id"], stratum)
        if av.get("activation_id") != f"{prompt['id']}::{stratum}" or av.get("checkpoint", {}).get("repo") != AV_REPO or av.get("checkpoint", {}).get("revision") != AV_REVISION:
            raise StudyError(f"AV provenance mismatch for {prompt['id']}::{stratum}")
        validity = validate_av_output(av["raw_text"], av["generated_token_count"], av["eos_terminated"], av.get("literal_special_token_strings", ()))
        activation_id = f"{prompt['id']}::{stratum}"
        row = {
            "status": "pass", "activation_id": activation_id, "prompt_id": prompt["id"], "prompt_index": int(prompt["index"]),
            "category": prompt["category"], "position_stratum": stratum, "position": av.get("position"),
            "layer_index": LAYER_INDEX, "seed": SEED, "checkpoint": av["checkpoint"],
            "candidate_source": "own_av_output", "own": True, "av_record_path": str(args.results_dir / "av" / f"{_target_stem(prompt['id'], stratum)}.json"),
            "record_path": str(out / f"{_target_stem(prompt['id'], stratum)}.json"), **validity,
        }
        json_dump(out / f"{_target_stem(prompt['id'], stratum)}.json", row)
        group_key = f"{prompt['category']}::{stratum}"
        if group_key not in candidates_by_group:
            candidates_by_group[group_key] = []; group_order.append(group_key)
        candidates_by_group[group_key].append({
            "candidate_id": activation_id, "source_activation_id": activation_id, "source_prompt_id": prompt["id"],
            "source_category": prompt["category"], "source_position_stratum": stratum, "candidate_source": "own_av_output",
            "own": True, "seed": SEED, "layer_index": LAYER_INDEX, "position": av.get("position"),
            "checkpoint": {"repo": AV_REPO, "revision": AV_REVISION},
            "score_text": validity["score_text"], "structural_valid": validity["structural_valid"],
            "validity_record_path": str(out / f"{_target_stem(prompt['id'], stratum)}.json"),
        })
    groups_dir = out / "groups"; groups_dir.mkdir(parents=True, exist_ok=True)
    group_records = []
    for group_key in group_order:
        candidates = candidates_by_group[group_key]
        if len(candidates) != 8:
            raise StudyError(f"candidate group {group_key} has {len(candidates)} records, not 8")
        category, stratum = group_key.split("::", 1)
        record = {"status": "pass", "group_id": group_key, "category": category, "position_stratum": stratum,
                  "seed": SEED, "candidate_count": len(candidates), "candidate_ids": [x["candidate_id"] for x in candidates],
                  "candidates": candidates, "record_path": str(groups_dir / f"{category}__{stratum}.json")}
        json_dump(groups_dir / f"{category}__{stratum}.json", record); group_records.append(record)
    json_dump(out / "manifest.json", {"status": "pass", "seed": SEED, "count": N_ACTIVATIONS,
                                       "groups": group_records})


def _load_ar_head(model_path: Path, model: Any, dtype: Any, device: Any):
    import torch
    from safetensors.torch import load_file
    model.model.norm = torch.nn.Identity(); model.lm_head = torch.nn.Identity()
    weights = load_file(str(model_path / "value_head.safetensors"))
    key = "weight" if "weight" in weights else next(iter(weights))
    head = torch.nn.Linear(model.config.hidden_size, model.config.hidden_size, bias=False, dtype=dtype, device=device)
    head.load_state_dict({"weight": weights[key].to(device=device, dtype=dtype)}); return head.eval()


def _score_ar_text(tokenizer: Any, model: Any, head: Any, device: Any, template: str, text: str):
    import torch
    encoded = tokenizer(template.format(explanation=text), add_special_tokens=True, return_tensors="pt")
    ids, mask = encoded["input_ids"].to(device), encoded["attention_mask"].to(device)
    with torch.inference_mode():
        output = model(input_ids=ids, attention_mask=mask, use_cache=False)
        last = int(mask[0].sum().item()) - 1
        hidden = output.logits[0, last].to(dtype=head.weight.dtype)
        vector = head(hidden).float().cpu()
    return vector, int(mask[0].sum().item())


def normalize_vector(vector: Any, scale: float):
    import torch
    value = vector.float()
    return value / value.norm().clamp_min(1e-12) * float(scale)


def stage_ar(args: argparse.Namespace) -> None:
    import torch
    set_determinism()
    prompts = load_prompt_spec(args.prompts)
    groups_manifest = args.results_dir / "validity" / "manifest.json"
    if not groups_manifest.is_file():
        raise StudyError("missing validity manifest; AR cannot invent candidate groups")
    validity_manifest = json.loads(groups_manifest.read_text())
    if validity_manifest.get("count") != N_ACTIVATIONS:
        raise StudyError("validity manifest is incomplete")
    tokenizer, model, device, dtype = _load_tokenizer_model(args.ar_checkpoint, args.device, AR_REVISION)
    head = _load_ar_head(args.ar_checkpoint, model, dtype, device)
    sidecar = _load_yaml(args.ar_checkpoint / "nla_meta.yaml")
    template, mse_scale = str(sidecar["prompt_templates"]["ar"]), float(sidecar["extraction"]["mse_scale"])
    out, vectors = args.results_dir / "ar", args.results_dir / "ar-vectors"; out.mkdir(parents=True, exist_ok=True); vectors.mkdir(parents=True, exist_ok=True)
    group_cache = {g["group_id"]: g for g in validity_manifest["groups"]}
    for prompt, stratum in _all_target_keys(prompts):
        target_meta, gold = _load_extraction_record(args.results_dir, prompt["id"], stratum)
        group = group_cache.get(f"{prompt['category']}::{stratum}")
        if not isinstance(group, Mapping) or len(group.get("candidates", ())) != 8:
            raise StudyError(f"invalid 8-way candidate group for {prompt['id']}::{stratum}")
        gold_n = normalize_vector(gold, mse_scale)
        rows = []
        for candidate_index, candidate in enumerate(group["candidates"]):
            vector, token_count = _score_ar_text(tokenizer, model, head, device, template, candidate["score_text"])
            vector_n = normalize_vector(vector, mse_scale)
            mse = float(((vector_n - gold_n) ** 2).mean().item())
            vector_path = vectors / f"{_target_stem(prompt['id'], stratum)}__candidate-{candidate_index:02d}.pt"
            torch.save(vector, vector_path)
            rows.append({
                "candidate_id": candidate["candidate_id"], "source_activation_id": candidate["source_activation_id"],
                "source_prompt_id": candidate["source_prompt_id"], "source_category": candidate["source_category"],
                "source_position_stratum": candidate["source_position_stratum"], "candidate_source": candidate["candidate_source"],
                "own": candidate["candidate_id"] == target_meta["activation_id"], "seed": SEED,
                "layer_index": LAYER_INDEX, "position": target_meta["position"],
                "checkpoint": {"repo": AR_REPO, "revision": AR_REVISION}, "score_text": candidate["score_text"],
                "structural_valid": candidate["structural_valid"], "token_count": token_count, "mse_nrm": mse,
                "vector_path": str(vector_path), "record_path": str(out / f"{_target_stem(prompt['id'], stratum)}.json"),
                "finite": finite(mse) and bool(torch.isfinite(vector).all()),
            })
        own = next((r for r in rows if r["own"]), None)
        if own is None: raise StudyError(f"candidate group lacks own explanation for {target_meta['activation_id']}")
        own_text = own["score_text"]
        controls = []
        for name, text in (("own_paraphrase", neutral_paraphrase(own_text)), ("unrelated", UNRELATED_TEXT)):
            vector, token_count = _score_ar_text(tokenizer, model, head, device, template, text)
            vector_n = normalize_vector(vector, mse_scale)
            mse = float(((vector_n - gold_n) ** 2).mean().item())
            vector_path = vectors / f"{_target_stem(prompt['id'], stratum)}__{name}.pt"
            torch.save(vector, vector_path)
            controls.append({"candidate_id": f"{target_meta['activation_id']}::{name}", "source_activation_id": target_meta["activation_id"],
                             "source_prompt_id": prompt["id"], "source_category": prompt["category"], "source_position_stratum": stratum,
                             "candidate_source": name, "own": name == "own_paraphrase", "seed": SEED,
                             "layer_index": LAYER_INDEX, "position": target_meta["position"],
                             "checkpoint": {"repo": AR_REPO, "revision": AR_REVISION}, "score_text": text,
                             "token_count": token_count, "mse_nrm": mse, "vector_path": str(vector_path),
                             "record_path": str(out / f"{_target_stem(prompt['id'], stratum)}.json"),
                             "finite": finite(mse) and bool(torch.isfinite(vector).all())})
        stem = _target_stem(prompt["id"], stratum)
        json_dump(out / f"{stem}.json", {
            "status": "pass", "activation_id": target_meta["activation_id"], "prompt_id": prompt["id"], "prompt_index": int(prompt["index"]),
            "category": prompt["category"], "position_stratum": stratum, "position": target_meta["position"], "layer_index": LAYER_INDEX,
            "seed": SEED, "checkpoint": {"repo": AR_REPO, "revision": AR_REVISION}, "mse_scale": mse_scale,
            "candidate_group_id": group["group_id"], "candidate_count": len(rows), "candidates": rows, "controls": controls,
            "finite": all(r["finite"] for r in rows + controls), "record_path": str(out / f"{stem}.json"),
        })
    del model
    if device.type == "cuda": torch.cuda.empty_cache()


def js_divergence(logits_a: Any, logits_b: Any) -> float:
    import torch
    a, b = logits_a.float(), logits_b.float()
    pa, pb = torch.softmax(a, dim=-1), torch.softmax(b, dim=-1)
    m = (pa + pb) / 2
    value = 0.5 * (torch.sum(pa * (torch.log(pa.clamp_min(1e-12)) - torch.log(m.clamp_min(1e-12)))) +
                   torch.sum(pb * (torch.log(pb.clamp_min(1e-12)) - torch.log(m.clamp_min(1e-12)))))
    return float(value.item())


def replace_one_position_hook(position: int, vector: Any):
    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        import torch
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        changed = tensor.clone()
        replacement = vector.to(device=tensor.device, dtype=tensor.dtype)
        if replacement.ndim == 1: replacement = replacement.unsqueeze(0)
        changed[:, int(position), :] = replacement
        if isinstance(output, tuple): return (changed,) + output[1:]
        if isinstance(output, list): return [changed] + output[1:]
        return changed
    return hook


def _base_logits(model: Any, ids: Any, mask: Any) -> Any:
    import torch
    with torch.inference_mode():
        return model(input_ids=ids, attention_mask=mask, use_cache=False).logits[0]


def hooked_logits(model: Any, ids: Any, mask: Any, position: int, vector: Any) -> Any:
    handle = model.model.layers[LAYER_INDEX].register_forward_hook(replace_one_position_hook(position, vector))
    try: return _base_logits(model, ids, mask)
    finally: handle.remove()


def _load_ar_json(results: Path, prompt_id: str, stratum: str) -> dict[str, Any]:
    path = results / "ar" / f"{_target_stem(prompt_id, stratum)}.json"
    if not path.is_file(): raise StudyError(f"missing AR record: {path}")
    return json.loads(path.read_text())


def stage_behavior(args: argparse.Namespace) -> None:
    import torch
    set_determinism()
    prompts = load_prompt_spec(args.prompts)
    tokenizer, model, device, dtype = _load_tokenizer_model(args.base_checkpoint, args.device, BASE_REVISION)
    out = args.results_dir / "behavior"; out.mkdir(parents=True, exist_ok=True)
    for prompt, stratum in _all_target_keys(prompts):
        target_meta, gold = _load_extraction_record(args.results_dir, prompt["id"], stratum)
        ar = _load_ar_json(args.results_dir, prompt["id"], stratum)
        if ar.get("checkpoint", {}).get("repo") != AR_REPO or ar.get("checkpoint", {}).get("revision") != AR_REVISION or ar.get("candidate_count") != 8:
            raise StudyError(f"AR provenance/candidate count mismatch for {target_meta['activation_id']}")
        alignment = target_meta.get("alignment")
        if not isinstance(alignment, Mapping): raise StudyError(f"missing rendered alignment for {target_meta['activation_id']}")
        ids = torch.tensor([alignment["token_ids"]], device=device); mask = torch.ones_like(ids)
        position = int(target_meta["position"]); baseline = _base_logits(model, ids, mask)
        gold_logits = hooked_logits(model, ids, mask, position, gold.to(device))
        gold_jsd = js_divergence(baseline[position], gold_logits[position])
        norm = float(gold.norm().item())
        random_seed = SEED + int(prompt["index"]) * N_POSITIONS_PER_PROMPT + POSITION_STRATA.index(stratum) + 1000
        generator = torch.Generator(device="cpu").manual_seed(random_seed)
        random_vector = torch.randn(gold.shape, generator=generator); random_vector = random_vector / random_vector.norm().clamp_min(1e-12) * norm
        random_logits = hooked_logits(model, ids, mask, position, random_vector.to(device))
        variants = []
        for row in ar["candidates"]:
            vector_path = Path(row["vector_path"])
            if not vector_path.is_file(): raise StudyError(f"missing AR vector: {vector_path}")
            vector = torch.load(vector_path, map_location="cpu", weights_only=True).float()
            replacement = vector / vector.norm().clamp_min(1e-12) * norm
            logits = hooked_logits(model, ids, mask, position, replacement.to(device))
            jsd = js_divergence(baseline[position], logits[position])
            variants.append({"candidate_id": row["candidate_id"], "source_activation_id": row["source_activation_id"],
                             "source_prompt_id": row["source_prompt_id"], "source_category": row["source_category"],
                             "source_position_stratum": row["source_position_stratum"], "candidate_source": row["candidate_source"],
                             "own": row["own"], "seed": SEED, "layer_index": LAYER_INDEX, "position": position,
                             "checkpoint": {"repo": BASE_REPO, "revision": BASE_REVISION}, "vector_path": row["vector_path"], "jsd": jsd,
                             "replacement_norm": float(replacement.norm().item()), "target_activation_norm": norm,
                             "rescaled_to_original_norm": True, "finite": finite(jsd)})
        controls = []
        for row in ar["controls"]:
            vector = torch.load(row["vector_path"], map_location="cpu", weights_only=True).float()
            replacement = vector / vector.norm().clamp_min(1e-12) * norm
            logits = hooked_logits(model, ids, mask, position, replacement.to(device))
            jsd = js_divergence(baseline[position], logits[position])
            controls.append({"candidate_id": row["candidate_id"], "source_activation_id": row["source_activation_id"],
                             "candidate_source": row["candidate_source"], "own": row["own"], "seed": SEED,
                             "layer_index": LAYER_INDEX, "position": position,
                             "checkpoint": {"repo": BASE_REPO, "revision": BASE_REVISION}, "vector_path": row["vector_path"], "jsd": jsd,
                             "replacement_norm": float(replacement.norm().item()), "target_activation_norm": norm,
                             "rescaled_to_original_norm": True, "finite": finite(jsd)})
        json_dump(out / f"{_target_stem(prompt['id'], stratum)}.json", {
            "status": "pass", "activation_id": target_meta["activation_id"], "prompt_id": prompt["id"], "prompt_index": int(prompt["index"]),
            "category": prompt["category"], "position_stratum": stratum, "position": position, "layer_index": LAYER_INDEX,
            "seed": SEED, "random_direction_seed": random_seed, "checkpoint": {"repo": BASE_REPO, "revision": BASE_REVISION},
            "gold_reinjection_jsd": gold_jsd, "random_direction_jsd": js_divergence(baseline[position], random_logits[position]),
            "candidates": variants, "controls": controls, "finite": finite(gold_jsd) and all(x["finite"] for x in variants + controls),
            "record_path": str(out / f"{_target_stem(prompt['id'], stratum)}.json"),
        })
    del model
    if device.type == "cuda": torch.cuda.empty_cache()


def rank_ascending(values: Sequence[float], index: int) -> int:
    """One-based ascending rank; exact ties share a rank."""
    if not values or not (0 <= index < len(values)) or not all(finite(x) for x in values):
        raise StudyError("cannot rank missing/non-finite values")
    return 1 + sum(float(x) < float(values[index]) for x in values)


def unique_top1(values: Sequence[float], index: int) -> bool:
    if not values or not all(finite(x) for x in values): return False
    minimum = min(values)
    return values[index] == minimum and sum(x == minimum for x in values) == 1


def _median(values: Sequence[float]) -> float | None:
    if not values: return None
    ordered = sorted(float(x) for x in values); n = len(ordered)
    return ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i]); result = [0.0] * len(values); i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]: j += 1
        rank = (i + 1 + j) / 2.0
        for k in order[i:j]: result[k] = rank
        i = j
    return result


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2 or not all(finite(x) and finite(y) for x, y in zip(xs, ys)):
        return None
    rx, ry = _average_ranks(xs), _average_ranks(ys); mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    den = math.sqrt(sum((x - mx) ** 2 for x in rx) * sum((y - my) ** 2 for y in ry))
    return None if den == 0 else sum((x - mx) * (y - my) for x, y in zip(rx, ry)) / den


def matched_assignment_permutation(matrices: Mapping[str, Sequence[Sequence[float]]], observed_count: int | None = None,
                                   *, seed: int = SEED, n_permutations: int = N_PERMUTATIONS) -> dict[str, Any]:
    """Permute one-to-one ownership within each 8-target group."""
    groups = list(matrices.values())
    if not groups or any(len(group) != 8 or any(len(row) != 8 for row in group) for group in groups):
        raise StudyError("matched-assignment matrices must be 8x8")
    observed = sum(unique_top1(row, i) for group in groups for i, row in enumerate(group)) if observed_count is None else int(observed_count)
    rng = random.Random(seed); counts = []
    for _ in range(int(n_permutations)):
        count = 0
        for group in groups:
            assignment = list(range(8)); rng.shuffle(assignment)
            count += sum(unique_top1(row, assignment[i]) for i, row in enumerate(group))
        counts.append(count)
    exceed = sum(x >= observed for x in counts)
    return {"observed_count": observed, "permutations": int(n_permutations), "seed": seed,
            "null_counts": counts, "p_value": (1 + exceed) / (int(n_permutations) + 1)}


def percentile(values: Sequence[float], q: float) -> float:
    if not values: raise StudyError("cannot percentile empty values")
    if not 0 <= q <= 1: raise StudyError("percentile q outside [0,1]")
    ordered = sorted(float(x) for x in values); position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] if low == high else ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _cluster_values(records: Sequence[Mapping[str, Any]], value_key: str) -> dict[str, list[float]]:
    by_prompt: dict[str, list[float]] = {}
    for row in records:
        prompt_id = row.get("prompt_id")
        value = row.get(value_key)
        if not isinstance(prompt_id, str) or not prompt_id or not finite(value):
            raise StudyError("bootstrap record lacks finite prompt/value")
        by_prompt.setdefault(prompt_id, []).append(float(value))
    if not by_prompt:
        raise StudyError("prompt-cluster bootstrap requires at least one prompt cluster")
    return by_prompt


def _cluster_statistic(values: Sequence[float], statistic: str) -> float:
    if not values:
        raise StudyError("cannot summarize an empty prompt cluster draw")
    if statistic == "mean":
        return sum(values) / len(values)
    if statistic == "median":
        result = _median(values)
        if result is None:
            raise StudyError("cannot summarize an empty prompt cluster draw")
        return result
    raise StudyError(f"unsupported prompt-cluster statistic: {statistic}")


def prompt_cluster_bootstrap(records: Sequence[Mapping[str, Any]], value_key: str = "joint_specific",
                             *, seed: int = SEED, n_resamples: int = N_PERMUTATIONS,
                             statistic: str = "mean") -> dict[str, Any]:
    """Bootstrap prompt clusters, retaining every row in each drawn cluster.

    The frozen overall estimate has 24 clusters and three positions per cluster,
    while subset reports can have 8 or 24 clusters and one to three rows per
    cluster.  No row-level resampling is permitted.
    """
    if int(n_resamples) < 1:
        raise StudyError("prompt-cluster bootstrap requires at least one resample")
    by_prompt = _cluster_values(records, value_key)
    ids = sorted(by_prompt)
    rng = random.Random(seed)
    estimates = []
    for _ in range(int(n_resamples)):
        drawn = [rng.choice(ids) for _ in ids]
        values = [value for prompt_id in drawn for value in by_prompt[prompt_id]]
        estimates.append(_cluster_statistic(values, statistic))
    observed_values = [value for prompt_id in ids for value in by_prompt[prompt_id]]
    return {"seed": seed, "resamples": int(n_resamples), "cluster_count": len(ids),
            "cluster_sizes": [len(by_prompt[prompt_id]) for prompt_id in ids], "statistic": statistic,
            "observed": _cluster_statistic(observed_values, statistic), "lower": percentile(estimates, .025),
            "upper": percentile(estimates, .975)}


def prompt_cluster_bootstrap_difference(records_a: Sequence[Mapping[str, Any]],
                                        records_b: Sequence[Mapping[str, Any]],
                                        value_key: str = "joint_specific", *,
                                        seed: int = SEED, n_resamples: int = N_PERMUTATIONS,
                                        statistic: str = "mean") -> dict[str, Any]:
    """Bootstrap the difference of two prompt-cluster estimates."""
    if int(n_resamples) < 1:
        raise StudyError("prompt-cluster bootstrap requires at least one resample")
    by_a, by_b = _cluster_values(records_a, value_key), _cluster_values(records_b, value_key)
    ids_a, ids_b = sorted(by_a), sorted(by_b)
    observed_a = [value for prompt_id in ids_a for value in by_a[prompt_id]]
    observed_b = [value for prompt_id in ids_b for value in by_b[prompt_id]]
    observed = _cluster_statistic(observed_a, statistic) - _cluster_statistic(observed_b, statistic)
    rng = random.Random(seed)
    estimates = []
    for _ in range(int(n_resamples)):
        draw_a = [rng.choice(ids_a) for _ in ids_a]
        draw_b = [rng.choice(ids_b) for _ in ids_b]
        values_a = [value for prompt_id in draw_a for value in by_a[prompt_id]]
        values_b = [value for prompt_id in draw_b for value in by_b[prompt_id]]
        estimates.append(_cluster_statistic(values_a, statistic) - _cluster_statistic(values_b, statistic))
    return {"seed": seed, "resamples": int(n_resamples), "statistic": statistic,
            "cluster_count_a": len(ids_a), "cluster_count_b": len(ids_b), "observed": observed,
            "lower": percentile(estimates, .025), "upper": percentile(estimates, .975)}


def _strong_metrics(ar_mse: Sequence[float], behavior_jsd: Sequence[float], own_index: int,
                    paraphrase_mse: float, paraphrase_jsd: float) -> dict[str, Any]:
    own_mse, own_jsd = ar_mse[own_index], behavior_jsd[own_index]
    ar_margin = _median([ar_mse[i] for i in range(len(ar_mse)) if i != own_index]) - own_mse
    behavior_margin = _median([behavior_jsd[i] for i in range(len(behavior_jsd)) if i != own_index]) - own_jsd
    ar_floor = max(abs(paraphrase_mse - own_mse), 0.001)
    behavior_floor = max(abs(paraphrase_jsd - own_jsd), 1e-5)
    ar_rank, behavior_rank = rank_ascending(ar_mse, own_index), rank_ascending(behavior_jsd, own_index)
    strong_ar = unique_top1(ar_mse, own_index) and ar_margin > ar_floor
    strong_behavior = unique_top1(behavior_jsd, own_index) and behavior_margin > behavior_floor
    return {"own_index": own_index, "own_ar_rank": ar_rank, "ar_margin": ar_margin, "ar_floor": ar_floor, "strong_ar": strong_ar,
            "own_behavior_rank": behavior_rank, "behavior_margin": behavior_margin, "behavior_floor": behavior_floor,
            "strong_behavior": strong_behavior, "joint_specific": strong_ar and strong_behavior}


def frozen_classification(operational: bool, ar_p: float, behavior_p: float, joint_lower: float,
                          joint_rates_by_stratum: Mapping[str, float]) -> str:
    if not operational: return "INVALID_MEASUREMENT"
    if ar_p <= 0.01 and behavior_p <= 0.01 and joint_lower > CHANCE_TOP1 and sum(v >= .25 for v in joint_rates_by_stratum.values()) >= 2:
        return "LOCALIZED_AND_CAUSAL"
    if ar_p <= 0.01 and (behavior_p > 0.05 or joint_lower <= CHANCE_TOP1):
        return "RECONSTRUCTION_NOT_CAUSAL"
    if ar_p > 0.05: return "NOT_SEMANTICALLY_LOCALIZED"
    return "MIXED_OR_UNDERPOWERED"


# Public names used by lightweight audit scripts.
compute_target_metrics = _strong_metrics
permutation_test = matched_assignment_permutation
bootstrap_prompt_clusters = prompt_cluster_bootstrap
classify_decision = frozen_classification


def _require_json(path: Path) -> dict[str, Any]:
    if not path.is_file(): raise StudyError(f"missing required record: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict): raise StudyError(f"record is not an object: {path}")
    return value


def _expected_group_candidates(prompts: Sequence[Mapping[str, Any]], category: str,
                               stratum: str) -> list[str]:
    return [f"{prompt['id']}::{stratum}" for prompt in prompts if prompt.get("category") == category]


def _validate_group_record(group: Mapping[str, Any], expected_group_id: str,
                           expected_ids: Sequence[str], source: str) -> None:
    if group.get("status") != "pass" or group.get("group_id") != expected_group_id:
        raise StudyError(f"invalid {source} validity group: {expected_group_id}")
    category, stratum = expected_group_id.split("::", 1)
    if group.get("category") != category or group.get("position_stratum") != stratum or group.get("seed") != SEED:
        raise StudyError(f"{source} validity group provenance mismatch: {expected_group_id}")
    candidates = group.get("candidates")
    if group.get("candidate_count") != 8 or not isinstance(candidates, list) or len(candidates) != 8:
        raise StudyError(f"{source} validity group is not exactly 8-way: {expected_group_id}")
    candidate_ids = group.get("candidate_ids")
    actual_ids = [row.get("candidate_id") if isinstance(row, Mapping) else None for row in candidates]
    if candidate_ids != list(expected_ids) or actual_ids != list(expected_ids):
        raise StudyError(f"{source} validity candidate identity/order mismatch: {expected_group_id}")
    for candidate, candidate_id in zip(candidates, expected_ids):
        if not isinstance(candidate, Mapping) or candidate.get("candidate_id") != candidate_id:
            raise StudyError(f"malformed {source} validity candidate: {expected_group_id}")
        if (candidate.get("source_activation_id") != candidate_id
                or candidate.get("source_prompt_id") != candidate_id.split("::", 1)[0]
                or candidate.get("source_category") != category
                or candidate.get("source_position_stratum") != stratum
                or candidate.get("candidate_source") != "own_av_output"
                or candidate.get("own") is not True
                or candidate.get("seed") != SEED
                or candidate.get("layer_index") != LAYER_INDEX
                or candidate.get("checkpoint", {}).get("repo") != AV_REPO
                or candidate.get("checkpoint", {}).get("revision") != AV_REVISION):
            raise StudyError(f"{source} validity candidate provenance mismatch: {expected_group_id}")


def _load_validity_groups(results: Path, prompts: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Load and cross-check both levels of the frozen validity manifest."""
    manifest_path = results / "validity" / "manifest.json"
    manifest = _require_json(manifest_path)
    expected_ids = [f"{category}::{stratum}" for category in CATEGORIES for stratum in POSITION_STRATA]
    if (manifest.get("status") != "pass" or manifest.get("seed") != SEED
            or manifest.get("count") != N_ACTIVATIONS or manifest.get("groups") is None):
        raise StudyError("invalid validity manifest provenance/cardinality")
    groups = manifest["groups"]
    if not isinstance(groups, list) or len(groups) != len(expected_ids):
        raise StudyError("validity manifest does not contain exactly 9 groups")
    result: dict[str, dict[str, Any]] = {}
    for nested, group_id in zip(groups, expected_ids):
        category, stratum = group_id.split("::", 1)
        ids = _expected_group_candidates(prompts, category, stratum)
        if len(ids) != 8:
            raise StudyError(f"prompt set does not define exactly 8 candidates for {group_id}")
        if not isinstance(nested, Mapping):
            raise StudyError(f"malformed nested validity group: {group_id}")
        _validate_group_record(nested, group_id, ids, "nested manifest")
        disk_path = results / "validity" / "groups" / f"{category}__{stratum}.json"
        disk = _require_json(disk_path)
        _validate_group_record(disk, group_id, ids, "group file")
        if (disk.get("record_path") != str(disk_path) or nested.get("record_path") != disk.get("record_path")
                or disk.get("candidate_ids") != nested.get("candidate_ids") or disk.get("candidates") != nested.get("candidates")):
            raise StudyError(f"nested validity manifest disagrees with group file: {group_id}")
        for candidate in disk["candidates"]:
            validity_path = results / "validity" / f"{_target_stem(candidate['source_prompt_id'], stratum)}.json"
            if candidate.get("validity_record_path") != str(validity_path):
                raise StudyError(f"validity candidate path mismatch: {candidate['candidate_id']}")
            validity = _require_json(validity_path)
            if (validity.get("status") != "pass" or validity.get("activation_id") != candidate["candidate_id"]
                    or validity.get("prompt_id") != candidate["source_prompt_id"]
                    or validity.get("category") != category or validity.get("position_stratum") != stratum
                    or validity.get("seed") != SEED or validity.get("checkpoint", {}).get("repo") != AV_REPO
                    or validity.get("checkpoint", {}).get("revision") != AV_REVISION
                    or validity.get("score_text") != candidate.get("score_text")
                    or validity.get("structural_valid") != candidate.get("structural_valid")):
                raise StudyError(f"validity record disagrees with group manifest: {candidate['candidate_id']}")
        result[group_id] = dict(disk)
    if [group.get("group_id") for group in groups] != expected_ids:
        raise StudyError("validity manifest group identity/order mismatch")
    return result


def _row_top1(row: Mapping[str, Any], values_key: str) -> bool:
    own_index = row.get("own_index")
    if not isinstance(own_index, int) or isinstance(own_index, bool):
        raise StudyError("row lacks explicit own candidate index")
    candidate_ids = row.get("candidate_ids")
    own_candidate_id = row.get("own_candidate_id")
    if candidate_ids is not None:
        if not isinstance(candidate_ids, list) or len(candidate_ids) != 8 or not 0 <= own_index < len(candidate_ids):
            raise StudyError("row candidate index/cardinality mismatch")
        if own_candidate_id != candidate_ids[own_index]:
            raise StudyError("row own candidate ID/index mismatch")
    values = row.get(values_key)
    if not isinstance(values, list):
        raise StudyError(f"row lacks candidate values: {values_key}")
    return unique_top1(values, own_index)


def _bootstrap_subset_report(rows: Sequence[Mapping[str, Any]], *, seed: int = SEED,
                             n_resamples: int = N_PERMUTATIONS) -> dict[str, Any]:
    metric_specs = (
        ("ar_top1_rate", "_ar_top1", "mean"),
        ("behavior_top1_rate", "_behavior_top1", "mean"),
        ("joint_specific_rate", "joint_specific", "mean"),
        ("median_ar_rank", "own_ar_rank", "median"),
        ("median_behavior_rank", "own_behavior_rank", "median"),
        ("median_ar_margin", "ar_margin", "median"),
        ("median_behavior_margin", "behavior_margin", "median"),
    )
    report: dict[str, Any] = {"n": len(rows), "cluster_count": len({r.get("prompt_id") for r in rows}),
                              "seed": seed, "resamples": int(n_resamples), "metrics": {}}
    if not rows:
        report["cluster_count"] = 0
        for name, _key, _statistic in metric_specs:
            report["metrics"][name] = None
            report[name] = None
        return report
    metric_rows = []
    for row in rows:
        metric_rows.append({"prompt_id": row.get("prompt_id"), "_ar_top1": float(_row_top1(row, "ar_mse")),
                            "_behavior_top1": float(_row_top1(row, "behavior_jsd")),
                            "joint_specific": float(bool(row.get("joint_specific"))),
                            "own_ar_rank": row.get("own_ar_rank"), "own_behavior_rank": row.get("own_behavior_rank"),
                            "ar_margin": row.get("ar_margin"), "behavior_margin": row.get("behavior_margin")})
    for name, key, statistic in metric_specs:
        interval = prompt_cluster_bootstrap(metric_rows, key, seed=seed,
                                            n_resamples=n_resamples, statistic=statistic)
        report["metrics"][name] = interval
        report[name] = interval
    return report


def stage_decide(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    prompts = load_prompt_spec(args.prompts)
    validation = _require_json(args.results_dir / "validation.json")
    exact_provenance = (
        validation.get("status") == "pass" and validation.get("seed") == SEED and validation.get("prompt_sha256") == hashlib.sha256(args.prompts.read_bytes()).hexdigest()
        and validation.get("prompt_sha256") == FROZEN_PROMPT_SHA256 and validation.get("activation_count") == N_ACTIVATIONS
        and validation.get("local_only") is True and validation.get("layer_index") == LAYER_INDEX
        and validation.get("hf_hidden_states_index") == HF_HIDDEN_STATES_INDEX
        and validation.get("manifest_verification", {}).get("status") == "pass"
        and validation.get("checkpoints", {}).get("base", {}).get("repo") == BASE_REPO
        and validation.get("checkpoints", {}).get("base", {}).get("revision") == BASE_REVISION
        and validation.get("checkpoints", {}).get("av", {}).get("repo") == AV_REPO
        and validation.get("checkpoints", {}).get("av", {}).get("revision") == AV_REVISION
        and validation.get("checkpoints", {}).get("ar", {}).get("repo") == AR_REPO
        and validation.get("checkpoints", {}).get("ar", {}).get("revision") == AR_REVISION
    )
    validity_groups = _load_validity_groups(args.results_dir, prompts)
    rows = []; ar_matrices: dict[str, list[list[float]]] = {}; behavior_matrices: dict[str, list[list[float]]] = {}
    reasons = []
    all_random, all_own_jsd, all_own_mse, all_unrelated_mse = [], [], [], []
    for prompt, stratum in _all_target_keys(prompts):
        stem = _target_stem(prompt["id"], stratum)
        extract = _require_json(args.results_dir / "extract" / f"{stem}.json")
        av = _require_json(args.results_dir / "av" / f"{stem}.json")
        validity = _require_json(args.results_dir / "validity" / f"{stem}.json")
        ar = _require_json(args.results_dir / "ar" / f"{stem}.json")
        behavior = _require_json(args.results_dir / "behavior" / f"{stem}.json")
        if extract.get("status") != "pass" or av.get("status") != "pass" or validity.get("status") != "pass" or ar.get("status") != "pass" or behavior.get("status") != "pass":
            reasons.append(f"missing pass status in extraction/AV/validity/AR/behavior for {stem}")
        if av.get("activation_id") != f"{prompt['id']}::{stratum}" or av.get("seed") != SEED or av.get("position") != extract.get("position"):
            reasons.append(f"AV identity/seed/position mismatch for {stem}")
        if av.get("checkpoint", {}).get("repo") != AV_REPO or av.get("checkpoint", {}).get("revision") != AV_REVISION:
            reasons.append(f"AV provenance mismatch for {stem}")
        if validity.get("activation_id") != f"{prompt['id']}::{stratum}" or validity.get("seed") != SEED:
            reasons.append(f"validity identity/seed mismatch for {stem}")
        if validity.get("checkpoint", {}).get("repo") != AV_REPO or validity.get("checkpoint", {}).get("revision") != AV_REVISION:
            reasons.append(f"validity provenance mismatch for {stem}")
        tensor_path = args.results_dir / "extract" / f"{stem}.pt"
        if not tensor_path.is_file(): raise StudyError(f"missing activation tensor: {tensor_path}")
        activation = torch.load(tensor_path, map_location="cpu", weights_only=True)
        if extract.get("activation_id") != f"{prompt['id']}::{stratum}" or extract.get("position_stratum") != stratum:
            reasons.append(f"extraction identity mismatch for {stem}")
        if extract.get("base_model") != BASE_REPO or extract.get("base_revision") != BASE_REVISION or extract.get("layer_index") != LAYER_INDEX or extract.get("hf_hidden_states_index") != HF_HIDDEN_STATES_INDEX:
            reasons.append(f"extraction provenance mismatch for {stem}")
        if activation.ndim != 1 or not bool(torch.isfinite(activation).all()) or extract.get("finite") is not True:
            reasons.append(f"non-finite extraction for {stem}")
        if ar.get("checkpoint", {}).get("repo") != AR_REPO or ar.get("checkpoint", {}).get("revision") != AR_REVISION or ar.get("candidate_count") != 8 or ar.get("layer_index") != LAYER_INDEX or ar.get("position") != extract.get("position"):
            reasons.append(f"AR provenance/count mismatch for {stem}")
        if behavior.get("checkpoint", {}).get("repo") != BASE_REPO or behavior.get("checkpoint", {}).get("revision") != BASE_REVISION or behavior.get("layer_index") != LAYER_INDEX or behavior.get("position") != extract.get("position") or behavior.get("activation_id") != extract.get("activation_id") or behavior.get("seed") != SEED:
            reasons.append(f"behavior provenance/identity mismatch for {stem}")
        candidates, behavior_candidates = ar.get("candidates"), behavior.get("candidates")
        if (not isinstance(candidates, list) or len(candidates) != 8
                or not isinstance(behavior_candidates, list) or len(behavior_candidates) != 8
                or any(not isinstance(row, Mapping) for row in candidates + behavior_candidates)):
            raise StudyError(f"missing 8-way candidate records for {stem}")
        group_id = f"{prompt['category']}::{stratum}"
        group = validity_groups[group_id]
        expected_ids = group["candidate_ids"]
        ar_ids = [row.get("candidate_id") if isinstance(row, Mapping) else None for row in candidates]
        behavior_ids = [row.get("candidate_id") if isinstance(row, Mapping) else None for row in behavior_candidates]
        if ar_ids != expected_ids or behavior_ids != expected_ids:
            raise StudyError(f"candidate identity/order mismatch for {stem}")
        if len(set(ar_ids)) != 8 or len(set(behavior_ids)) != 8:
            raise StudyError(f"candidate IDs are not unique for {stem}")
        if (any(not isinstance(row.get("own"), bool) for row in candidates)
                or any(not isinstance(row.get("own"), bool) for row in behavior_candidates)):
            raise StudyError(f"candidate ownership flags are not boolean for {stem}")
        by_id = {x.get("candidate_id"): x for x in behavior_candidates}
        for index, (ar_candidate, behavior_candidate, group_candidate) in enumerate(zip(candidates, behavior_candidates, group["candidates"])):
            for field in ("candidate_id", "source_activation_id", "source_prompt_id", "source_category", "source_position_stratum", "candidate_source"):
                if ar_candidate.get(field) != group_candidate.get(field) or behavior_candidate.get(field) != group_candidate.get(field):
                    raise StudyError(f"candidate provenance/order mismatch at index {index} for {stem}")
        ar_own_indices = [i for i, row in enumerate(candidates) if row.get("own") is True]
        behavior_own_indices = [i for i, row in enumerate(behavior_candidates) if row.get("own") is True]
        if len(ar_own_indices) != 1 or len(behavior_own_indices) != 1 or ar_own_indices != behavior_own_indices:
            raise StudyError(f"candidate ownership is not exactly one aligned own candidate for {stem}")
        own_index = ar_own_indices[0]
        target_activation_id = f"{prompt['id']}::{stratum}"
        if expected_ids[own_index] != target_activation_id or candidates[own_index].get("candidate_id") != target_activation_id:
            raise StudyError(f"own candidate ID mismatch for {stem}")
        ar_values = [x.get("mse_nrm") for x in candidates]; behavior_values = [by_id[x.get("candidate_id")].get("jsd") for x in candidates]
        if not all(finite(x) for x in ar_values + behavior_values): reasons.append(f"non-finite candidate metric for {stem}")
        if any(not Path(x.get("vector_path", "")).is_file() or x.get("finite") is not True for x in candidates):
            reasons.append(f"missing/non-finite AR vector for {stem}")
        if any(x.get("finite") is not True for x in behavior_candidates):
            reasons.append(f"non-finite behavior candidate for {stem}")
        controls_ar = {x.get("candidate_source"): x for x in ar.get("controls", [])}; controls_behavior = {x.get("candidate_source"): x for x in behavior.get("controls", [])}
        if "own_paraphrase" not in controls_ar or "unrelated" not in controls_ar or "own_paraphrase" not in controls_behavior:
            raise StudyError(f"missing controls for {stem}")
        metrics = _strong_metrics(ar_values, behavior_values, own_index, controls_ar["own_paraphrase"]["mse_nrm"], controls_behavior["own_paraphrase"]["jsd"])
        row = {"activation_id": extract["activation_id"], "prompt_id": prompt["id"], "prompt_index": int(prompt["index"]),
               "category": prompt["category"], "position_stratum": stratum, "position": extract["position"], "layer_index": LAYER_INDEX,
               "candidate_source": "matched_category_position_group", "candidate_ids": list(expected_ids), "own_index": own_index,
               "own_candidate_id": candidates[own_index]["candidate_id"],
               "structural_valid": bool(candidates[own_index].get("structural_valid")), "ar_mse": ar_values,
               "behavior_jsd": behavior_values, "own_paraphrase_mse": controls_ar["own_paraphrase"]["mse_nrm"],
               "unrelated_mse": controls_ar["unrelated"]["mse_nrm"], "random_direction_jsd": behavior.get("random_direction_jsd"),
               **metrics}
        rows.append(row); group = f"{prompt['category']}::{stratum}"; ar_matrices.setdefault(group, []).append(ar_values); behavior_matrices.setdefault(group, []).append(behavior_values)
        all_random.append(behavior.get("random_direction_jsd")); all_own_jsd.append(behavior_values[own_index]); all_own_mse.append(ar_values[own_index]); all_unrelated_mse.append(controls_ar["unrelated"]["mse_nrm"])
        if not finite(behavior.get("gold_reinjection_jsd")) or behavior["gold_reinjection_jsd"] > 1e-5: reasons.append(f"gold reinjection failed for {stem}")
        if not behavior.get("finite") or not ar.get("finite"): reasons.append(f"stage finite flag failed for {stem}")
    if len(rows) != N_ACTIVATIONS: reasons.append(f"expected {N_ACTIVATIONS} activations, found {len(rows)}")
    if any(len(v) != 8 for v in ar_matrices.values()) or len(ar_matrices) != 9: reasons.append("candidate group cardinality failure")
    if len({(r["prompt_id"], r["position"]) for r in rows}) != N_ACTIVATIONS:
        reasons.append("prompt-position pairs are not 72 distinct activations")
    operational = exact_provenance and not reasons and _median(all_random) > _median(all_own_jsd) and _median(all_unrelated_mse) > _median(all_own_mse)
    if not exact_provenance: reasons.append("exact provenance/configuration failed")
    if not (_median(all_random) > _median(all_own_jsd)): reasons.append("median random-direction JSD is not greater than median own JSD")
    if not (_median(all_unrelated_mse) > _median(all_own_mse)): reasons.append("median unrelated AR MSE is not greater than median own AR MSE")
    overall = {"ar_top1_rate": sum(_row_top1(r, "ar_mse") for r in rows) / len(rows),
               "behavior_top1_rate": sum(_row_top1(r, "behavior_jsd") for r in rows) / len(rows),
               "joint_specific_rate": sum(r["joint_specific"] for r in rows) / len(rows),
               "median_ar_rank": _median([r["own_ar_rank"] for r in rows]), "median_behavior_rank": _median([r["own_behavior_rank"] for r in rows]),
               "median_ar_margin": _median([r["ar_margin"] for r in rows]), "median_behavior_margin": _median([r["behavior_margin"] for r in rows])}
    def subset_stats(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {"n": len(subset), "ar_top1_rate": sum(_row_top1(r, "ar_mse") for r in subset) / len(subset) if subset else None,
                "behavior_top1_rate": sum(_row_top1(r, "behavior_jsd") for r in subset) / len(subset) if subset else None,
                "joint_specific_rate": sum(r["joint_specific"] for r in subset) / len(subset) if subset else None,
                "median_ar_rank": _median([r["own_ar_rank"] for r in subset]), "median_behavior_rank": _median([r["own_behavior_rank"] for r in subset]),
                "median_ar_margin": _median([r["ar_margin"] for r in subset]), "median_behavior_margin": _median([r["behavior_margin"] for r in subset])}
    breakdown = {"overall": overall}
    for category in CATEGORIES: breakdown[f"category::{category}"] = subset_stats([r for r in rows if r["category"] == category])
    for stratum in POSITION_STRATA: breakdown[f"position_stratum::{stratum}"] = subset_stats([r for r in rows if r["position_stratum"] == stratum])
    for valid in (True, False): breakdown[f"structural_valid::{str(valid).lower()}"] = subset_stats([r for r in rows if r["structural_valid"] is valid])
    valid_rows = [r for r in rows if r["structural_valid"]]
    invalid_rows = [r for r in rows if not r["structural_valid"]]
    enough_structural_rows = len(valid_rows) >= 12 and len(invalid_rows) >= 12
    structural_comparison = {
        "valid_n": len(valid_rows), "invalid_n": len(invalid_rows),
        "joint_specific_rate_valid": sum(r["joint_specific"] for r in valid_rows) / len(valid_rows) if valid_rows else None,
        "joint_specific_rate_invalid": sum(r["joint_specific"] for r in invalid_rows) / len(invalid_rows) if invalid_rows else None,
        "difference_valid_minus_invalid": (sum(r["joint_specific"] for r in valid_rows) / len(valid_rows) - sum(r["joint_specific"] for r in invalid_rows) / len(invalid_rows)) if enough_structural_rows else None,
        "bootstrap_95_valid_minus_invalid": prompt_cluster_bootstrap_difference(
            valid_rows, invalid_rows, seed=SEED, n_resamples=N_PERMUTATIONS) if enough_structural_rows else None,
        "descriptive_only": not enough_structural_rows,
    }
    bootstrap = prompt_cluster_bootstrap(rows, seed=SEED, n_resamples=N_PERMUTATIONS)
    overall_bootstrap_report = _bootstrap_subset_report(rows, seed=SEED, n_resamples=N_PERMUTATIONS)
    bootstrap["metrics"] = overall_bootstrap_report["metrics"]
    bootstrap["overall"] = overall_bootstrap_report
    bootstrap["by_subset"] = {"overall": overall_bootstrap_report}
    for category in CATEGORIES:
        bootstrap["by_subset"][f"category::{category}"] = _bootstrap_subset_report(
            [r for r in rows if r["category"] == category])
    for stratum in POSITION_STRATA:
        bootstrap["by_subset"][f"position_stratum::{stratum}"] = _bootstrap_subset_report(
            [r for r in rows if r["position_stratum"] == stratum])
    for valid in (True, False):
        bootstrap["by_subset"][f"structural_valid::{str(valid).lower()}"] = _bootstrap_subset_report(
            [r for r in rows if r["structural_valid"] is valid])
    ar_perm = matched_assignment_permutation(ar_matrices); behavior_perm = matched_assignment_permutation(behavior_matrices)
    strata_joint = {s: breakdown[f"position_stratum::{s}"]["joint_specific_rate"] for s in POSITION_STRATA}
    classification = frozen_classification(operational, ar_perm["p_value"], behavior_perm["p_value"], bootstrap["lower"], strata_joint)
    result = {"status": "pass" if operational else "invalid", "classification": classification, "operational_validity": operational,
              "invalid_reasons": sorted(set(reasons)), "seed": SEED, "layer_index": LAYER_INDEX, "rows": rows,
              "breakdown": breakdown, "structural_comparison": structural_comparison, "bootstrap": bootstrap,
              "permutation": {"ar": ar_perm, "behavior": behavior_perm},
              "spearman": {"ar_vs_behavior_ranks": spearman([r["own_ar_rank"] for r in rows], [r["own_behavior_rank"] for r in rows]),
                           "ar_vs_behavior_margins": spearman([r["ar_margin"] for r in rows], [r["behavior_margin"] for r in rows])},
              "controls": {"gold_reinjection_max_jsd": max((float(_require_json(args.results_dir / "behavior" / f"{_target_stem(p['id'], s)}.json")["gold_reinjection_jsd"]) for p, s in _all_target_keys(prompts)), default=float("nan")),
                           "median_random_direction_jsd": _median(all_random), "median_own_behavior_jsd": _median(all_own_jsd),
                           "median_unrelated_ar_mse": _median(all_unrelated_mse), "median_own_ar_mse": _median(all_own_mse)},
              "config": {"base": [BASE_REPO, BASE_REVISION], "av": [AV_REPO, AV_REVISION], "ar": [AR_REPO, AR_REVISION],
                         "layer_index": LAYER_INDEX, "hf_hidden_states_index": HF_HIDDEN_STATES_INDEX, "position_strata": list(POSITION_STRATA),
                         "permutations": N_PERMUTATIONS, "bootstrap_resamples": N_PERMUTATIONS}}
    json_dump(args.results_dir / "decision.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("validate", "extract", "av", "validity", "ar", "behavior", "decide", "all"), required=True)
    ap.add_argument("--study-dir", "--gate-dir", dest="study_dir", type=Path, default=DEFAULT_STUDY_DIR)
    ap.add_argument("--results-dir", type=Path)
    ap.add_argument("--prompts", type=Path)
    ap.add_argument("--base-checkpoint", type=Path)
    ap.add_argument("--av-checkpoint", type=Path)
    ap.add_argument("--ar-checkpoint", type=Path)
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = ap.parse_args(argv)
    if args.max_new_tokens != MAX_NEW_TOKENS:
        ap.error(f"--max-new-tokens is frozen at {MAX_NEW_TOKENS}")
    args.study_dir = args.study_dir.resolve()
    args.results_dir = (args.results_dir or args.study_dir / "results").resolve()
    args.prompts = (args.prompts or args.study_dir / "prompts.json").resolve()
    weights = args.study_dir.parent / "weights"
    args.base_checkpoint = (args.base_checkpoint or weights / "base-qwen").resolve()
    args.av_checkpoint = (args.av_checkpoint or weights / "av").resolve()
    args.ar_checkpoint = (args.ar_checkpoint or weights / "ar").resolve()
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv); args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "all":
        for stage in ("validate", "extract", "av", "validity", "ar", "behavior", "decide"):
            command = [sys.executable, str(Path(__file__).resolve()), "--stage", stage, "--study-dir", str(args.study_dir),
                       "--results-dir", str(args.results_dir), "--prompts", str(args.prompts),
                       "--base-checkpoint", str(args.base_checkpoint), "--av-checkpoint", str(args.av_checkpoint),
                       "--ar-checkpoint", str(args.ar_checkpoint), "--device", args.device]
            subprocess.run(command, check=True)
        return
    {"validate": stage_validate, "extract": stage_extract, "av": stage_av, "validity": stage_validity,
     "ar": stage_ar, "behavior": stage_behavior, "decide": stage_decide}[args.stage](args)


if __name__ == "__main__":
    main()
