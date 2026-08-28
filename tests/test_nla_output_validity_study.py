import json
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

CODE = Path(__file__).parents[1] / "code" / "nla"
sys.path.insert(0, str(CODE))
import run_output_validity_study as study


class OffsetTokenizer:
    all_special_ids = [3, 4]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        prompt = messages[0]["content"]
        return "<s>" + prompt + "<|assistant|><|eos|>"

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        # The fake fast-tokenizer exposes offsets in rendered-string coordinates.
        prompt_start = 3
        prompt_end = text.index("<|assistant|>")
        ids = [10, 11, 12, 3, 4]
        offsets = [(prompt_start, prompt_start + 3),
                   (prompt_start + 3, prompt_start + 7),
                   (prompt_start + 7, prompt_end),
                   (prompt_end, prompt_end + 13),
                   (prompt_end + 13, len(text))]
        return {"input_ids": ids, "offset_mapping": offsets}

    def decode(self, ids, skip_special_tokens=False):
        return {10: "abc", 11: " DEF", 12: " ghi", 3: "<|assistant|>", 4: "<|eos|>"}[ids[0]]


class OutputValidityTests(unittest.TestCase):
    def test_rendered_offsets_choose_three_distinct_positions(self):
        result = study.rendered_chat_positions(OffsetTokenizer(), "abc DEF ghi")
        positions = result["positions"]
        self.assertEqual([positions[x]["position"] for x in study.POSITION_STRATA], [0, 1, 3])
        self.assertEqual(result["prompt_span"], [3, 14])

    def test_rendered_offsets_fail_on_repeated_prompt(self):
        class Repeated(OffsetTokenizer):
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return messages[0]["content"] + " / " + messages[0]["content"] + "<|assistant|><|eos|>"
        with self.assertRaises(study.StudyError):
            study.rendered_chat_positions(Repeated(), "abc")

    def test_wrapper_validity_and_malformed_score_rule(self):
        valid = study.validate_av_output("<explanation>\nOne.\nTwo.\n</explanation>", 12, True)
        self.assertTrue(valid["wrapper_valid"])
        self.assertTrue(valid["structural_valid"])
        self.assertEqual(valid["score_text"], "One.\nTwo.")
        raw = "prefix <explanation>bad"
        malformed = study.validate_av_output(raw, 12, True)
        self.assertFalse(malformed["wrapper_valid"])
        self.assertEqual(malformed["score_text"], raw)
        self.assertFalse(malformed["structural_valid"])

    def test_literal_special_and_cap_invalidate_structure(self):
        special = study.validate_av_output("<explanation>A.\n<|assistant|>\nB.</explanation>", 20, True)
        self.assertFalse(special["structural_valid"])
        capped = study.validate_av_output("<explanation>A.\nB.</explanation>", 180, False)
        self.assertTrue(capped["hit_token_cap_without_eos"])
        self.assertFalse(capped["structural_valid"])

    def test_parser_cap_deterministically_merges_tail(self):
        text = "\n".join(f"{i}. clause {i}." for i in range(1, 11))
        clauses = study.parse_primary_clauses(text)
        self.assertEqual(len(clauses), 8)
        self.assertEqual(clauses[-1], "clause 8. clause 9. clause 10.")

    def test_candidate_grouping_has_eight_per_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts_path = Path(__file__).parents[1] / "pilots/wildcard-nla/output-validity/prompts.json"
            prompts = json.loads(prompts_path.read_text())["prompts"]
            results = root / "results"
            av_dir = results / "av"
            av_dir.mkdir(parents=True)
            for row in prompts:
                for stratum in study.POSITION_STRATA:
                    stem = f"{row['id']}__{stratum}"
                    (av_dir / f"{stem}.json").write_text(json.dumps({
                        "activation_id": f"{row['id']}::{stratum}",
                        "prompt_id": row["id"], "prompt_index": row["index"],
                        "category": row["category"], "position_stratum": stratum,
                        "position": 5, "checkpoint": {"repo": study.AV_REPO, "revision": study.AV_REVISION},
                        "generated_token_count": 10, "eos_terminated": True,
                        "literal_special_token_strings": [],
                        "raw_text": "<explanation>One.\nTwo.</explanation>",
                    }))
            args = type("Args", (), {"prompts": prompts_path, "results_dir": results})()
            study.stage_validity(args)
            manifest = json.loads((results / "validity" / "manifest.json").read_text())
            self.assertEqual(manifest["count"], 72)
            self.assertEqual(len(manifest["groups"]), 9)
            self.assertTrue(all(group["candidate_count"] == 8 for group in manifest["groups"]))

    def test_rank_ties_and_strong_floors(self):
        self.assertEqual(study.rank_ascending([1.0, 1.0, 2.0], 0), 1)
        self.assertFalse(study.unique_top1([1.0, 1.0, 2.0], 0))
        self.assertTrue(study.unique_top1([1.0, 2.0, 3.0], 0))
        metrics = study._strong_metrics([1.0, 2.0, 2.0, 2.0], [0.1, 0.4, 0.4, 0.4], 0, 1.0, 0.1)
        self.assertEqual(metrics["ar_floor"], 0.001)
        self.assertEqual(metrics["behavior_floor"], 1e-5)
        self.assertTrue(metrics["strong_ar"])
        self.assertTrue(metrics["strong_behavior"])

    def test_top1_uses_explicit_own_index_not_rank_minus_one(self):
        row = {
            "candidate_ids": ["own", "decoy-a", "decoy-b", "decoy-c", "decoy-d", "decoy-e", "decoy-f", "decoy-g"],
            "own_candidate_id": "own", "own_index": 0,
            "ar_mse": [0.2, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        }
        self.assertEqual(study.rank_ascending(row["ar_mse"], row["own_index"]), 2)
        self.assertFalse(study._row_top1(row, "ar_mse"))

    def test_permutation_is_deterministic_and_one_to_one(self):
        matrices = {f"g{i}": [[0.0 if i == j else 1.0 for j in range(8)] for i in range(8)] for i in range(9)}
        first = study.matched_assignment_permutation(matrices, n_permutations=50)
        second = study.matched_assignment_permutation(matrices, n_permutations=50)
        self.assertEqual(first, second)
        self.assertEqual(first["observed_count"], 72)
        self.assertEqual(first["p_value"], 1 / 51)

    def test_prompt_cluster_bootstrap_is_deterministic(self):
        records = [{"prompt_id": f"p{i}", "value": float((i + j) % 2)} for i in range(24) for j in range(3)]
        first = study.prompt_cluster_bootstrap(records, "value", n_resamples=100)
        second = study.prompt_cluster_bootstrap(records, "value", n_resamples=100)
        self.assertEqual(first, second)
        self.assertEqual(first["cluster_count"], 24)

    def test_prompt_cluster_bootstrap_supports_subset_shapes_and_medians(self):
        records = [{"prompt_id": f"p{i}", "value": float(i + j)}
                   for i in range(8) for j in range(1 + (i % 3))]
        first = study.prompt_cluster_bootstrap(records, "value", seed=study.SEED, n_resamples=25, statistic="median")
        second = study.prompt_cluster_bootstrap(records, "value", seed=study.SEED, n_resamples=25, statistic="median")
        self.assertEqual(first, second)
        self.assertEqual(first["cluster_count"], 8)
        self.assertEqual(first["cluster_sizes"], [1, 2, 3, 1, 2, 3, 1, 2])
        self.assertEqual(first["statistic"], "median")

    def test_prompt_cluster_bootstrap_difference_is_deterministic(self):
        valid = [{"prompt_id": f"p{i}", "joint": float(i % 2)} for i in range(12)]
        invalid = [{"prompt_id": f"q{i}", "joint": float((i + 1) % 2)} for i in range(12)]
        first = study.prompt_cluster_bootstrap_difference(valid, invalid, "joint", n_resamples=25)
        second = study.prompt_cluster_bootstrap_difference(valid, invalid, "joint", n_resamples=25)
        self.assertEqual(first, second)
        self.assertEqual(first["cluster_count_a"], 12)
        self.assertEqual(first["cluster_count_b"], 12)

    def test_every_frozen_decision_branch(self):
        rates = {x: 0.3 for x in study.POSITION_STRATA}
        self.assertEqual(study.frozen_classification(True, .001, .001, .2, rates), "LOCALIZED_AND_CAUSAL")
        self.assertEqual(study.frozen_classification(True, .001, .06, .2, rates), "RECONSTRUCTION_NOT_CAUSAL")
        self.assertEqual(study.frozen_classification(True, .2, .001, .2, rates), "NOT_SEMANTICALLY_LOCALIZED")
        self.assertEqual(study.frozen_classification(True, .02, .02, .2, rates), "MIXED_OR_UNDERPOWERED")
        self.assertEqual(study.frozen_classification(False, .001, .001, .2, rates), "INVALID_MEASUREMENT")

    @unittest.skipUnless(_HAS_TORCH, "torch not installed")
    def test_missing_record_and_provenance_fail_loudly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = Path(__file__).parents[1] / "pilots/wildcard-nla/output-validity/prompts.json"
            results = root / "results"
            results.mkdir()
            (results / "validation.json").write_text(json.dumps({"status": "pass"}))
            args = type("Args", (), {"prompts": prompts, "results_dir": results})()
            with self.assertRaises(study.StudyError):
                study.stage_decide(args)


if __name__ == "__main__":
    unittest.main()
