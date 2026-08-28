import sys
import unittest
from pathlib import Path

CODE = Path(__file__).parents[1] / "code" / "nla"
sys.path.insert(0, str(CODE))
import run_clause_ablation as ca


class ClauseAblationTests(unittest.TestCase):
    def setUp(self):
        self.clauses = [
            "Generic opening about the task.",
            "Middle detail about constraints.",
            "Final token choice follows from the above.",
        ]
        self.score_text = "\n\n".join(self.clauses)

    def test_build_variant_texts_three_clause_fixture(self):
        variants = ca.build_variant_texts(self.clauses, self.score_text)
        self.assertEqual(variants["full"], self.score_text)
        self.assertEqual(variants["drop_final"], "\n\n".join(self.clauses[:-1]))
        self.assertEqual(variants["final_only"], self.clauses[-1])
        self.assertEqual(variants["generic_only"], self.clauses[0])
        self.assertEqual(
            variants["malformed"],
            f"<explanation>\n{self.score_text}\n<|im_end|>",
        )

    def test_build_variant_texts_join_mismatch_raises(self):
        with self.assertRaises(ca.StudyError):
            ca.build_variant_texts(self.clauses, "wrong score text")

    def test_build_variant_texts_not_three_clauses_raises(self):
        with self.assertRaises(ca.StudyError):
            ca.build_variant_texts(self.clauses[:2], "\n\n".join(self.clauses[:2]))

    def test_build_variant_texts_final_clause_prefix_raises(self):
        bad = list(self.clauses)
        bad[-1] = "Not final prefix."
        with self.assertRaises(ca.StudyError):
            ca.build_variant_texts(bad, "\n\n".join(bad))

    def test_compare_gate_values_pass_inside_tols(self):
        out = ca.compare_gate_values(1.0, 1.0 + 5e-5, 0.2, 0.2 + 3e-6)
        self.assertTrue(out["passed"])
        self.assertLessEqual(out["mse_delta"], ca.GATE_MSE_TOL)
        self.assertLessEqual(out["jsd_delta"], ca.GATE_JSD_TOL)

    def test_compare_gate_values_fail_outside_tols(self):
        out_mse = ca.compare_gate_values(1.0, 1.01, 0.2, 0.2)
        self.assertFalse(out_mse["passed"])
        out_jsd = ca.compare_gate_values(1.0, 1.0, 0.2, 0.21)
        self.assertFalse(out_jsd["passed"])

    def test_variant_metrics_unique_top1_and_floor(self):
        decoy_mse = [1.0] * 7
        decoy_jsd = [0.5] * 7
        strong = ca._variant_metrics(
            own_mse=0.05,
            own_jsd=0.01,
            decoy_mse=decoy_mse,
            decoy_jsd=decoy_jsd,
            paraphrase_mse=0.5,
            paraphrase_jsd=0.2,
        )
        self.assertTrue(strong["joint_specific"])
        self.assertTrue(strong["strong_ar"])
        self.assertTrue(strong["strong_behavior"])
        self.assertEqual(strong["own_ar_rank"], 1)
        self.assertEqual(strong["own_behavior_rank"], 1)

    def test_variant_metrics_not_unique_top1_fails_joint(self):
        decoy_mse = [0.05, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        decoy_jsd = [0.5] * 7
        weak = ca._variant_metrics(
            own_mse=0.05,
            own_jsd=0.01,
            decoy_mse=decoy_mse,
            decoy_jsd=decoy_jsd,
            paraphrase_mse=0.5,
            paraphrase_jsd=0.2,
        )
        self.assertFalse(weak["strong_ar"])
        self.assertFalse(weak["joint_specific"])


if __name__ == "__main__":
    unittest.main()
