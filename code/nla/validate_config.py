#!/usr/bin/env python3
"""Validate released NLA sidecars against the downloaded tokenizers."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from transformers import AutoTokenizer
import yaml


def validate_sidecar(checkpoint: Path, tokenizer, injection_scale_override=None):
    meta = yaml.safe_load((checkpoint / "nla_meta.yaml").read_text())
    d_model = meta["d_model"]
    extraction = meta.get("extraction", {})
    injection_scale = extraction.get("injection_scale")
    if injection_scale is None:
        injection_scale = injection_scale_override
    assert injection_scale is not None
    tokens = meta["tokens"]
    marker_ids = tokenizer.encode(tokens["injection_char"], add_special_tokens=False)
    assert marker_ids == [tokens["injection_token_id"]], (marker_ids, tokens)
    content = meta["prompt_templates"]["av"].format(injection_char=tokens["injection_char"])
    ids = tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True)
    matches = [i for i, token in enumerate(ids) if token == tokens["injection_token_id"]]
    assert len(matches) == 1, matches
    p = matches[0]
    assert ids[p - 1] == tokens["injection_left_neighbor_id"], (ids[p-1], tokens)
    assert ids[p + 1] == tokens["injection_right_neighbor_id"], (ids[p+1], tokens)
    return {"d_model": d_model, "role": meta.get("role"), "injection_char": tokens["injection_char"],
            "injection_token_id": tokens["injection_token_id"], "injection_scale": injection_scale,
            "prompt_token_count": len(ids), "injection_position": p, "status": "pass"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nla-source", type=Path, required=True)
    ap.add_argument("--av", type=Path, required=True)
    ap.add_argument("--ar", type=Path, required=True)
    ap.add_argument("--base-tokenizer", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = {"checks": [], "status": "pass"}
    av_tok = AutoTokenizer.from_pretrained(args.av, local_files_only=True)
    ar_tok = AutoTokenizer.from_pretrained(args.ar, local_files_only=True)
    base_tok = AutoTokenizer.from_pretrained(args.base_tokenizer, local_files_only=True)
    result["checks"].append({"checkpoint": "av", **validate_sidecar(args.av, av_tok)})
    result["checks"].append({"checkpoint": "ar", **validate_sidecar(args.ar, ar_tok, injection_scale_override=150.0)})
    result["base_tokenizer_marker_ids"] = {
        "marker": "㈎",
        "ids": base_tok.encode("㈎", add_special_tokens=False),
    }
    assert result["base_tokenizer_marker_ids"]["ids"] == [149705]
    result["checks"].append({"checkpoint": "base-tokenizer", "marker": "㈎",
                              "ids": result["base_tokenizer_marker_ids"]["ids"],
                              "status": "pass"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
