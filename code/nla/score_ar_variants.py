#!/usr/bin/env python3
"""Score deterministic explanation variants with the released pure-torch AR."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file

SEED = 20260811

def normalize(v: torch.Tensor, scale: float) -> torch.Tensor:
    return v.float() / v.float().norm().clamp_min(1e-12) * scale

def split_clauses(text: str) -> list[str]:
    text = re.sub(r"<\/?explanation>", "", text, flags=re.I).strip()
    lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", x).strip() for x in text.splitlines()]
    lines = [x for x in lines if x]
    if len(lines) >= 2:
        return lines
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]

def variants(full: str) -> dict[str, str]:
    clauses = split_clauses(full)
    if len(clauses) < 2:
        clauses = [full.strip(), "The explanation contains a second descriptive clause."]
    paraphrase = full.replace("The ", "This ").replace(" likely ", " probably ").replace("suggests", "indicates")
    unrelated = "A purple bicycle is parked beside a weekday calendar in a quiet room."
    shuffled = " ".join(reversed(clauses)).strip()
    result = {"full": full.strip()}
    for i in range(len(clauses)):
        remaining = " ".join(clauses[:i] + clauses[i + 1:]).strip()
        result[f"lesion_delete_clause_{i + 1}"] = remaining
    result.update({
        "deterministic_paraphrase": paraphrase.strip(),
        "unrelated_clause_control": unrelated,
        "clause_order_shuffle_control": shuffled,
    })
    return result

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--activation", type=Path, required=True)
    ap.add_argument("--av-record", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = ap.parse_args()
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    elif args.device == "mps" or (args.device == "auto" and torch.backends.mps.is_available()):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    av_record = json.loads(args.av_record.read_text())
    full = av_record["raw_text"]
    # Keep raw AV output, but score only the content inside explanation tags when present.
    m = re.search(r"<explanation>\s*(.*?)\s*</explanation>", full, flags=re.S | re.I)
    explanation = m.group(1).strip() if m else full.strip()
    texts = variants(explanation)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=dtype,
        low_cpu_mem_usage=True, attn_implementation="sdpa",
    ).to(device).eval()
    model.model.norm = torch.nn.Identity()
    model.lm_head = torch.nn.Identity()
    head = torch.nn.Linear(3584, 3584, bias=False, dtype=dtype, device=device)
    head.load_state_dict({"weight": load_file(str(args.model / "value_head.safetensors"))["weight"].to(dtype=dtype)})
    head.eval()
    sidecar = __import__("yaml").safe_load((args.model / "nla_meta.yaml").read_text())
    template = sidecar["prompt_templates"]["ar"]
    mse_scale = float(sidecar["extraction"]["mse_scale"])
    gold = torch.load(args.activation, map_location="cpu", weights_only=True).float()
    gold_n = normalize(gold, mse_scale)
    records = []
    for name, text in texts.items():
        prompt = template.format(explanation=text)
        encoded = tok(prompt, add_special_tokens=True, return_tensors="pt")
        ids = encoded["input_ids"].to(device)
        mask = encoded["attention_mask"].to(device)
        with torch.inference_mode():
            out = model(input_ids=ids, attention_mask=mask, use_cache=False)
            last = int(mask[0].sum().item()) - 1
            pred = head(out.logits[0, last].float()).float().cpu()
        pred_n = normalize(pred, mse_scale)
        cos = float(torch.nn.functional.cosine_similarity(pred_n[None], gold_n[None]).item())
        mse = float(((pred_n - gold_n) ** 2).mean().item())
        records.append({"name": name, "text": text, "prompt": prompt,
                        "token_count": int(mask[0].sum().item()), "mse_nrm": mse,
                        "cosine": cos, "finite": bool(np.isfinite(mse) and np.isfinite(cos))})
    result = {"seed": SEED, "device": str(device), "dtype": str(dtype),
              "checkpoint": str(args.model), "activation": str(args.activation),
              "mse_scale": mse_scale, "av_raw_text": full,
              "explanation_scored": explanation, "variants": records,
              "status": "pass"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
