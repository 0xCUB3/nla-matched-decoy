from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "nla"))
import plot_context_baselines as plot


class PlotContextBaselinesTests(unittest.TestCase):
    def test_load_margin_deltas_reads_48_rows(self) -> None:
        rows = []
        for index in range(48):
            rows.append({"comparisons": {
                "nla_full_vs_local_ctx": {"ar_delta": 0.1 + index, "jsd_delta": 0.2 + index},
                "nla_full_vs_token_only": {"ar_delta": 0.3 + index, "jsd_delta": 0.4 + index},
            }})
        payload = {"classification": "PROSE_EXCEEDS_CONTEXT", "rows": rows}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "decision.json"
            path.write_text(json.dumps(payload))
            data = plot.load_margin_deltas(path)
        self.assertEqual(data["classification"], "PROSE_EXCEEDS_CONTEXT")
        self.assertEqual(len(data["channels"]["ar"][0]["values"]), 48)
        self.assertEqual(data["channels"]["ar"][0]["values"][0], 0.1)
        self.assertEqual(data["channels"]["jsd"][1]["values"][-1], 47.4)

    def test_load_joint_rates_preserves_display_order(self) -> None:
        spec = {
            "nla_full": {"n": 48, "joint_specific_count": 47},
            "token_only": {"n": 48, "joint_specific_count": 0},
            "local_ctx": {"n": 48, "joint_specific_count": 7},
            "nla_drop_final_sym": {"n": 48, "joint_specific_count": 36},
            "nla_scrubbed": {"n": 48, "joint_specific_count": 29},
        }
        payload = {"classification": "PROSE_EXCEEDS_CONTEXT", "descriptive": {"same_variant_specificity": spec}, "rows": [{"comparisons": {
            "nla_full_vs_local_ctx": {"ar_delta": 0.1, "jsd_delta": 0.2},
            "nla_full_vs_token_only": {"ar_delta": 0.3, "jsd_delta": 0.4},
        }}] * 48}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "decision.json"
            path.write_text(json.dumps(payload))
            rates = plot.load_joint_rates(path)
        self.assertEqual([item["key"] for item in rates], [
            "nla_full", "nla_drop_final_sym", "nla_scrubbed", "local_ctx", "token_only",
        ])
        self.assertEqual([item["count"] for item in rates], [47, 36, 29, 7, 0])


if __name__ == "__main__":
    unittest.main()
