#!/usr/bin/env python3
"""Bar chart of joint-specific rates by clause-ablation variant."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DECISION = ROOT / "pilots/wildcard-nla/clause-ablation/results/decision.json"
OUT = ROOT / "pilots/wildcard-nla/clause-ablation/figures/variant-specificity.png"

VARIANT_ORDER = ("full", "drop_final", "final_only", "generic_only", "malformed")
VARIANT_LABELS = {
    "full": "Full",
    "drop_final": "Drop final",
    "final_only": "Final only",
    "generic_only": "Generic only",
    "malformed": "Malformed",
}
CHANCE = 0.125
N_TARGETS = 48


def load_headline_rates(decision_path: Path = DECISION) -> tuple[list[str], list[float], list[int]]:
    """Return labels, joint_specific_rate values, and joint_specific counts per variant."""
    decision = json.loads(decision_path.read_text())
    headline = decision["headline"]
    labels: list[str] = []
    rates: list[float] = []
    counts: list[int] = []
    for name in VARIANT_ORDER:
        row = headline[name]
        labels.append(VARIANT_LABELS[name])
        rates.append(float(row["joint_specific_rate"]))
        counts.append(int(row["joint_specific_count"]))
    return labels, rates, counts


def plot_variant_specificity(
    labels: list[str],
    rates: list[float],
    counts: list[int],
    out_path: Path = OUT,
) -> Path:
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    color = "#4472C4"
    bars = ax.bar(x, rates, width=0.55, color=color)
    for bar, rate, count in zip(bars, rates, counts):
        inside = rate >= 0.8
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate - 0.045 if inside else rate + 0.025,
            f"{count}/{N_TARGETS}",
            ha="center",
            va="top" if inside else "bottom",
            fontsize=9,
            color="white" if inside else "black",
        )
    ax.axhline(CHANCE, color="#666666", linestyle="--", linewidth=1)
    ax.text(2.0, CHANCE - 0.04, "8-way chance", ha="center", va="top", color="#666666", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Rate")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_title("NLA clause ablation: joint specificity by explanation variant")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main() -> None:
    labels, rates, counts = load_headline_rates()
    path = plot_variant_specificity(labels, rates, counts)
    print(path)


if __name__ == "__main__":
    main()
