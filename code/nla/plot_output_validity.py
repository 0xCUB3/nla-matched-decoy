#!/usr/bin/env python3
"""Create the final position-stratified summary figure from decision.json."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DECISION = ROOT / "pilots/wildcard-nla/output-validity/results/decision.json"
OUT = ROOT / "pilots/wildcard-nla/output-validity/figures/position-specificity.png"

def main() -> None:
    decision = json.loads(DECISION.read_text())
    keys = ["content_early", "content_late", "boundary_after_user"]
    labels = ["Early content", "Late content", "After-user boundary"]
    fields = [("ar_top1_rate", "AR top-1"), ("behavior_top1_rate", "Behavior top-1"), ("joint_specific_rate", "Joint-specific")]
    values = [[decision["breakdown"][f"position_stratum::{key}"][field] for key in keys] for field, _ in fields]
    x = list(range(3)); width = 0.24
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colors = ["#4472C4", "#ED7D31", "#70AD47"]
    for j, ((_, name), row) in enumerate(zip(fields, values)):
        bars = ax.bar([i + (j-1)*width for i in x], row, width, label=name, color=colors[j])
        for bar, value in zip(bars, row):
            ax.text(bar.get_x()+bar.get_width()/2, value+0.025, f"{round(value*24):.0f}/24", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.125, color="#666666", linestyle="--", linewidth=1, label="8-way chance")
    ax.set_ylim(0, 1.12); ax.set_ylabel("Rate"); ax.set_xticks(x, labels)
    ax.set_title("NLA activation specificity is position-dependent")
    ax.legend(frameon=False, ncol=2, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220); print(OUT)

if __name__ == "__main__": main()
