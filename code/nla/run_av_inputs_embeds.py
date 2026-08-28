#!/usr/bin/env python3
"""Pure-transformers AV decode using a true activation and inputs_embeds."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 20260811


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--activation", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    ap.add_argument("--max-new-tokens", type=int, default=180)
    args = ap.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    elif args.device == "mps" or (args.device == "auto" and torch.backends.mps.is_available()):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=dtype,
        low_cpu_mem_usage=True, attn_implementation="sdpa",
    ).to(device).eval()
    meta = __import__("yaml").safe_load((args.model / "nla_meta.yaml").read_text())
    template = meta["prompt_templates"]["av"]
    marker = meta["tokens"]["injection_char"]
    content = template.format(injection_char=marker)
    ids_cpu = tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt",
    )
    ids = ids_cpu.to(device)
    marker_id = int(meta["tokens"]["injection_token_id"])
    matches = (ids[0] == marker_id).nonzero(as_tuple=False).flatten().tolist()
    assert len(matches) == 1, matches
    p = matches[0]
    assert int(ids[0, p - 1]) == meta["tokens"]["injection_left_neighbor_id"]
    assert int(ids[0, p + 1]) == meta["tokens"]["injection_right_neighbor_id"]
    activation = torch.load(args.activation, map_location="cpu", weights_only=True).float()
    injection_scale = float(meta["extraction"]["injection_scale"])
    scaled = activation * (injection_scale / max(float(activation.norm().item()), 1e-12))
    with torch.inference_mode():
        embeds = model.get_input_embeddings()(ids)
        embeds = embeds.clone()
        embeds[0, p] = scaled.to(device=device, dtype=embeds.dtype)
        mask = torch.ones_like(ids, device=device)
        generated = model.generate(
            inputs_embeds=embeds, attention_mask=mask,
            max_new_tokens=args.max_new_tokens, do_sample=False,
            eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id,
        )
    # With inputs_embeds, Transformers returns only generated IDs on current versions.
    generated_ids = generated[0].detach().cpu().tolist()
    text = tok.decode(generated_ids, skip_special_tokens=False)
    record = {
        "seed": SEED, "device": str(device), "dtype": str(dtype),
        "checkpoint": str(args.model), "activation": str(args.activation),
        "input_token_count": int(ids.shape[1]), "injection_position": int(p),
        "injection_scale": injection_scale, "activation_norm": float(activation.norm().item()),
        "generated_token_count": len(generated_ids), "generated_ids": generated_ids,
        "raw_text": text,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
