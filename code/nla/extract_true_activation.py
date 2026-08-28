#!/usr/bin/env python3
"""Extract one deterministic, true Qwen2.5 layer-20 activation on MPS/CPU."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 20260811
PROMPT = (
    "A safety engineer reviews a proposed change to a chemical reactor controller. "
    "The change would disable an interlock during maintenance, and the engineer must "
    "decide whether the assistant should explain the bypass or recommend a safe test."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
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
    enc = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt",
    )
    input_ids = enc.to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    eligible = list(range(max(5, input_ids.shape[1] // 3), input_ids.shape[1] - 1))
    pos = random.Random(SEED).choice(eligible)
    with torch.inference_mode():
        result = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True, use_cache=False)
    # HF hidden_states[0] is embeddings; layer 20 output is hidden_states[21].
    activation = result.hidden_states[21][0, pos].detach().float().cpu()
    token_id = int(input_ids[0, pos].item())
    left = max(0, pos - 8); right = min(input_ids.shape[1], pos + 9)
    metadata = {
        "seed": SEED, "device": str(device), "dtype": str(dtype),
        "base_model": str(args.model), "layer_index": 20,
        "hf_hidden_states_index": 21, "prompt": PROMPT,
        "chat_token_count": int(input_ids.shape[1]), "position": pos,
        "token_id": token_id, "token_text": tok.decode([token_id]),
        "local_token_context": tok.decode(input_ids[0, left:right].cpu().tolist()),
        "activation_norm_fp32": float(activation.norm().item()),
        "activation_shape": list(activation.shape),
    }
    torch.save(activation, out / "true_activation.pt")
    (out / "activation_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
