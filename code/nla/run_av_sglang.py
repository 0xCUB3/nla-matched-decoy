#!/usr/bin/env python3
"""Cloud/SGLang AV decode for a saved activation."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nla-repo", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--activation", type=Path, required=True)
    ap.add_argument("--sglang-url", default="http://127.0.0.1:30000")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.nla_repo))
    from nla_inference import NLAClient
    activation = torch.load(args.activation, map_location="cpu", weights_only=True).float()
    client = NLAClient(str(args.checkpoint), sglang_url=args.sglang_url)
    raw = client.generate(activation, extract_explanation=False,
                          temperature=0.0, max_new_tokens=180,
                          skip_special_tokens=False)
    result = {"checkpoint": str(args.checkpoint), "activation": str(args.activation),
              "sglang_url": args.sglang_url, "temperature": 0.0,
              "max_new_tokens": 180, "raw_text": raw}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
