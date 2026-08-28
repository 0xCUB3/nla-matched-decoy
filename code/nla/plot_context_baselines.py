#!/usr/bin/env python3
"""Plot sealed Experiment 3 paired tournament-margin deltas and joint-specificity rates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DECISION = ROOT / "pilots/wildcard-nla/context-baselines/results/latest/decision.json"
DEFAULT_MARGIN_OUT = ROOT / "pilots/wildcard-nla/context-baselines/figures/context-margin-deltas.png"
DEFAULT_JOINT_OUT = ROOT / "pilots/wildcard-nla/context-baselines/figures/context-joint-specificity.png"

COMPARISONS = (
    ("nla_full_vs_local_ctx", "Full − local context"),
    ("nla_full_vs_token_only", "Full − token only"),
)

JOINT_ORDER = (
    ("nla_full", "Full NLA prose"),
    ("nla_drop_final_sym", "Without final clause"),
    ("nla_scrubbed", "Token surface redacted"),
    ("local_ctx", "Local context"),
    ("token_only", "Token only"),
)


def load_margin_deltas(decision_path: Path = DEFAULT_DECISION) -> dict[str, Any]:
    """Load the two preregistered paired tournament-margin comparisons."""
    decision = json.loads(Path(decision_path).read_text())
    if decision.get("classification") not in {
        "PROSE_EXCEEDS_CONTEXT", "PROSE_REDUCES_TO_CONTEXT", "PROSE_PARTIAL",
        "REPLICATION_FAILURE", "INVALID_MEASUREMENT",
    }:
        raise ValueError("unrecognized Experiment 3 classification")
    rows = decision.get("rows")
    if not isinstance(rows, list) or len(rows) != 48:
        raise ValueError("expected 48 Experiment 3 decision rows")
    loaded: dict[str, Any] = {"classification": decision["classification"], "channels": {}}
    for channel in ("ar", "jsd"):
        series = []
        for key, label in COMPARISONS:
            field = f"{channel}_delta"
            values = [float(row["comparisons"][key][field]) for row in rows]
            if len(values) != 48:
                raise ValueError("comparison lacks 48 values")
            series.append({"key": key, "label": label, "values": values})
        loaded["channels"][channel] = series
    return loaded


def load_joint_rates(decision_path: Path = DEFAULT_DECISION) -> list[dict[str, Any]]:
    """Load same-variant joint-specificity counts in display order."""
    decision = json.loads(Path(decision_path).read_text())
    spec = decision.get("descriptive", {}).get("same_variant_specificity")
    if not isinstance(spec, dict):
        raise ValueError("missing same-variant joint-specificity block")
    rates: list[dict[str, Any]] = []
    for key, label in JOINT_ORDER:
        block = spec.get(key)
        if not isinstance(block, dict):
            raise ValueError(f"missing joint-specificity for {key}")
        count = int(block["joint_specific_count"])
        n = int(block["n"])
        if n <= 0:
            raise ValueError(f"non-positive n for {key}")
        rates.append({"key": key, "label": label, "count": count, "n": n, "rate": count / n})
    return rates


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return (ordered[23] + ordered[24]) / 2


def plot_margin_deltas(data: dict[str, Any], out_path: Path = DEFAULT_MARGIN_OUT) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.7), sharey=False)
    for axis, (channel, title) in zip(axes, (("ar", "AR normalized-MSE margin"), ("jsd", "One-position JSD margin"))):
        series = data["channels"][channel]
        for index, item in enumerate(series):
            values = item["values"]
            jitter = [index + (((row * 37) % 17) - 8) / 95 for row in range(len(values))]
            axis.scatter(jitter, values, color="#4472C4", alpha=0.72, s=22, edgecolors="none")
            median = _median(values)
            axis.hlines(median, index - 0.27, index + 0.27, color="#B22222", linewidth=2.2)
            axis.text(index, median, f" median {median:.3f}", ha="center", va="bottom" if median >= 0 else "top", fontsize=8)
        axis.axhline(0, color="#555555", linestyle="--", linewidth=1)
        axis.set_xticks([0, 1], [item["label"] for item in series], rotation=12, ha="right")
        axis.set_title(f"{title}\nMonte Carlo p ≤ 1/10,001 for both comparisons", fontsize=12)
        axis.set_ylabel("Full NLA tournament margin − baseline margin")
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Fresh symmetric context baselines", y=1.02, fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_joint_rates(rates: list[dict[str, Any]], out_path: Path = DEFAULT_JOINT_OUT) -> Path:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8.6, 4.4))
    labels = [item["label"] for item in rates]
    counts = [item["count"] for item in rates]
    colors = ["#1f4e79", "#5b8ab8", "#8fa8c8", "#c47c15", "#9c9c9c"]
    bars = axis.bar(range(len(rates)), counts, color=colors, width=0.72)
    axis.set_xticks(range(len(rates)), labels, rotation=18, ha="right")
    axis.set_ylabel("Joint-specific targets out of 48")
    axis.set_ylim(0, 52)
    axis.set_title("Same-variant joint specificity on 24 fresh prompts")
    for bar, item in zip(bars, rates):
        axis.text(bar.get_x() + bar.get_width() / 2, item["count"] + 1.1, f"{item['count']}/48", ha="center", va="bottom", fontsize=10)
    axis.axhline(6, color="#777777", linestyle="--", linewidth=1)  # unique 8-way top-1 chance is 0.125 → 6/48
    axis.text(4.38, 8.4, "unique top-1 chance 6/48", ha="right", va="bottom", fontsize=8, color="#555555")
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision", type=Path, default=DEFAULT_DECISION)
    parser.add_argument("--margin-out", type=Path, default=DEFAULT_MARGIN_OUT)
    parser.add_argument("--joint-out", type=Path, default=DEFAULT_JOINT_OUT)
    args = parser.parse_args()
    print(plot_margin_deltas(load_margin_deltas(args.decision), args.margin_out))
    print(plot_joint_rates(load_joint_rates(args.decision), args.joint_out))


if __name__ == "__main__":
    main()
