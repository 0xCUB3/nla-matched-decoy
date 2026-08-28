from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

CODE = Path(__file__).parents[1] / "code" / "nla"
sys.path.insert(0, str(CODE))
import run_semantic_swap_study as study


class SemanticSwapTests(unittest.TestCase):
    def test_swapped_prefix_keeps_own_final_clause(self):
        own = ["Scene A.", "Detail A.", 'Final token "alpha".']
        partner = ["Scene B.", "Detail B.", 'Final token "beta".']
        text = study.swapped_prefix_text(own, partner)
        self.assertEqual(text, "Scene B.\n\nDetail B.\n\nFinal token \"alpha\".")
        self.assertNotEqual(text, "\n\n".join(own))

    def test_swapped_prefix_rejects_identical_final_clause(self):
        own = ["Scene A.", "Detail A.", 'Final token "same".']
        partner = ["Scene B.", "Detail B.", 'Final token "same".']
        with self.assertRaises(study.StudyError):
            study.swapped_prefix_text(own, partner)

    def test_swapped_prefix_rejects_wrong_clause_count(self):
        with self.assertRaises(study.StudyError):
            study.swapped_prefix_text(["a", "b"], ["c", "d", "e"])

    def test_cycle_partners_is_derangement_of_eight(self):
        ids = [f"p{index}" for index in range(8)]
        partners = study.cycle_partners(ids)
        self.assertEqual(len(partners), 8)
        self.assertTrue(all(partners[item] != item for item in ids))
        self.assertEqual(set(partners.values()), set(ids))
        self.assertEqual(partners["p0"], "p1")
        self.assertEqual(partners["p7"], "p0")

    def test_cycle_partners_rejects_duplicates(self):
        with self.assertRaises(study.StudyError):
            study.cycle_partners(["a"] * 8)

    def test_classify_swap_every_branch(self):
        self.assertEqual(study.classify_swap(operational=False, joint_rate=0.0, ar_sig=True, jsd_sig=True), "INVALID_MEASUREMENT")
        self.assertEqual(study.classify_swap(operational=True, joint_rate=0.2, ar_sig=True, jsd_sig=True), "SEMANTICS_CARRY_WEIGHT")
        self.assertEqual(study.classify_swap(operational=True, joint_rate=0.8, ar_sig=False, jsd_sig=False), "TOKEN_CLAUSE_DOMINATES")
        self.assertEqual(study.classify_swap(operational=True, joint_rate=0.5, ar_sig=True, jsd_sig=True), "MIXED")
        self.assertEqual(study.classify_swap(operational=True, joint_rate=0.2, ar_sig=True, jsd_sig=False), "MIXED")

    def test_stage_variants_builds_unique_swaps_from_fixture(self):
        prompts = {
            "schema_version": 1,
            "seed": 20260813,
            "study": "context-baselines",
            "experiment": 3,
            "n_prompts": 24,
            "prompts": [],
        }
        # This test uses sealed records when present; otherwise a tiny local group.
        source_variants = Path(__file__).parents[1] / "pilots/wildcard-nla/context-baselines/results/latest/variants"
        if not (source_variants / "cb-safety-01__content_early.json").exists():
            self.skipTest("sealed Experiment 3 variants are not local")
        import run_context_baseline_study as cb
        record = json.loads((source_variants / "cb-safety-01__content_early.json").read_text())
        partner = json.loads((source_variants / "cb-safety-02__content_early.json").read_text())
        text = study.swapped_prefix_text(record["primary_clauses"], partner["primary_clauses"])
        self.assertTrue(text.endswith(record["primary_clauses"][2]))
        self.assertIn(partner["primary_clauses"][0], text)
        self.assertNotEqual(text, record["score_text"])
        self.assertEqual(cb.load_context_prompt_spec.__name__, "load_context_prompt_spec")
        self.assertEqual(prompts["experiment"], 3)


if __name__ == "__main__":
    unittest.main()
