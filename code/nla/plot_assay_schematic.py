#!/usr/bin/env python3
"""Render the matched-decoy causal-assay schematic used in the write-up."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parents[2] / "pilots/wildcard-nla/output-validity/figures/assay-schematic.png"


def box(ax, x, y, w, h, title, body, color):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.5,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=12, weight="bold", color=color)
    ax.text(x + w / 2, y + h * 0.34, body, ha="center", va="center", fontsize=9.5, color="#263238", linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, label=""):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, linewidth=1.5, color="#546e7a"))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.03, label, ha="center", va="bottom", fontsize=8.5, color="#455a64")


def main():
    fig, ax = plt.subplots(figsize=(13.5, 4.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, 0.02, 0.30, 0.16, 0.42, "1. Target", "Layer-20 activation\nat one frozen position", "#1565c0")
    box(ax, 0.225, 0.30, 0.16, 0.42, "2. AV text", "Generate the target's\nnatural-language explanation", "#6a1b9a")
    box(ax, 0.43, 0.30, 0.18, 0.42, "3. Matched decoys", "Own explanation + 7 from\nthe same category × position", "#ad1457")
    box(ax, 0.655, 0.55, 0.16, 0.30, "4a. AR test", "Unique lowest\nnormalized MSE", "#2e7d32")
    box(ax, 0.655, 0.15, 0.16, 0.30, "4b. Causal test", "One-position reinjection;\nunique lowest next-token JSD", "#ef6c00")
    box(ax, 0.86, 0.30, 0.12, 0.42, "5. Joint pass", "Both tests pass\nbeyond frozen\nparaphrase floors", "#37474f")

    arrow(ax, 0.18, 0.51, 0.225, 0.51)
    arrow(ax, 0.385, 0.51, 0.43, 0.51)
    arrow(ax, 0.61, 0.56, 0.655, 0.69)
    arrow(ax, 0.61, 0.46, 0.655, 0.31)
    arrow(ax, 0.815, 0.69, 0.86, 0.57)
    arrow(ax, 0.815, 0.31, 0.86, 0.43)

    ax.text(0.5, 0.94, "Matched-decoy causal assay", ha="center", va="center", fontsize=17, weight="bold", color="#102027")
    ax.text(0.5, 0.06, "Frozen before generation: prompts, positions, checkpoints, thresholds, controls, permutations, and decision rule", ha="center", va="center", fontsize=9.5, color="#455a64")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
