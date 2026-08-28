import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

CODE = Path(__file__).parents[1] / "code" / "nla"
sys.path.insert(0, str(CODE))
import run_context_baseline_study as study


PROMPTS = Path(__file__).parents[1] / "pilots/wildcard-nla/context-baselines/prompts.json"


class ContextBaselineTests(unittest.TestCase):
    def test_frozen_prompt_inventory_matches_hash_and_is_disjoint(self):
        inventory = study.validate_fresh_prompt_inventory(PROMPTS)
        self.assertEqual(inventory["status"], "pass")
        self.assertEqual(inventory["prompt_count"], 24)
        self.assertEqual(inventory["category_counts"], {
            "safety": 8,
            "compositional_planning": 8,
            "social_character_ood": 8,
        })
        self.assertEqual(inventory["prompt_sha256"], study.FROZEN_PROMPT_SHA256)
        self.assertEqual(hashlib.sha256(PROMPTS.read_bytes()).hexdigest(), study.FROZEN_PROMPT_SHA256)
        self.assertTrue(inventory["string_disjoint_from_locked"])

    def test_mutated_prompt_file_fails_hash(self):
        raw = json.loads(PROMPTS.read_text())
        raw["prompts"][0]["prompt"] += " extra"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            path.write_text(json.dumps(raw))
            with self.assertRaises(study.StudyError):
                study.validate_fresh_prompt_inventory(path)

    def test_build_variant_texts_and_baselines_are_distinct(self):
        clauses = [
            "The word cat appears early.",
            "Middle constraint.",
            'Final token "cat" follows from the above.',
        ]
        score_text = "\n\n".join(clauses)
        texts = study.build_variant_texts(
            score_text=score_text,
            primary_clauses=clauses,
            token="cat",
            window="the word cat appears",
        )
        self.assertEqual(texts["nla_full"], score_text)
        self.assertEqual(texts["token_only"], 'Final token "cat".')
        self.assertEqual(texts["local_ctx"], 'Final token "cat". Local token window: "the word cat appears".')
        self.assertEqual(texts["nla_drop_final_sym"], "The word cat appears early.\n\nMiddle constraint.")
        self.assertEqual(
            texts["nla_scrubbed"],
            "The word <TOKEN> appears early.\n\nMiddle constraint.\n\nFinal token \"<TOKEN>\" follows from the above.",
        )
        self.assertNotIn(texts["token_only"], (score_text, texts["nla_drop_final_sym"]))
        self.assertNotIn(texts["local_ctx"], (score_text, texts["nla_drop_final_sym"]))

    def test_scrub_operates_on_full_score_text_not_drop_final(self):
        clauses = ["Generic opener.", "Middle constraint.", 'Final token "zebra".']
        score_text = "\n\n".join(clauses)
        drop = study.drop_final_sym_text(clauses)
        self.assertNotIn("zebra", drop)
        texts = study.build_variant_texts(
            score_text=score_text,
            primary_clauses=clauses,
            token="zebra",
            window="window text",
        )
        self.assertIn("Generic opener.", texts["nla_scrubbed"])
        self.assertIn("Middle constraint.", texts["nla_scrubbed"])
        self.assertIn('Final token "<TOKEN>".', texts["nla_scrubbed"])
        self.assertNotIn("zebra", texts["nla_scrubbed"])
        with self.assertRaises(study.StudyError):
            study.build_variant_texts(
                score_text=score_text,
                primary_clauses=clauses,
                token="absent_token",
                window="window text",
            )

    def test_local_token_window_is_prompt_overlapping_and_centered(self):
        alignment = {
            "prompt_span": [0, 20],
            "token_offsets": [[0, 2], [2, 5], [5, 10], [10, 15], [15, 20], [20, 25]],
            "token_texts": ["aa", "bbb", "ccccc", "ddddd", "eeeee", "<|im_end|>"],
        }
        self.assertEqual(study.prompt_overlapping_indices(alignment), [0, 1, 2, 3, 4])
        self.assertEqual(study.local_token_window(alignment, 2, radius=1), "bbbcccccddddd")
        with self.assertRaises(study.StudyError):
            study.local_token_window(alignment, 5, radius=1)

    def test_payload_excludes_ownership_metadata(self):
        payload = study.model_text_payload({"variant_text": "hello", "own": True, "prompt_id": "x"})
        self.assertEqual(payload, {"explanation": "hello"})
        with self.assertRaises(study.StudyError):
            study.assert_payload_excludes_ownership({"explanation": "hello", "own": True})

    def test_pairwise_distinct_tokens(self):
        study.validate_pairwise_distinct_tokens(list("abcdefgh"), group_id="safety::content_early")
        with self.assertRaises(study.StudyError):
            study.validate_pairwise_distinct_tokens(list("abcdefgg"), group_id="safety::content_early")

    def test_display_token_helper(self):
        self.assertEqual(study.display_token(" metal"), "metal")
        self.assertEqual(study.display_token("metal"), "metal")
        self.assertEqual(study.display_token("  metal \t\n"), "metal")
        with self.assertRaises(study.StudyError):
            study.display_token("")
        with self.assertRaises(study.StudyError):
            study.display_token("   ")
        with self.assertRaises(study.StudyError):
            study.display_token(123)

    def test_leading_whitespace_in_baselines_and_scrub(self):
        token = " metal"
        self.assertEqual(study.token_only_text(token), 'Final token "metal".')
        self.assertEqual(study.local_ctx_text(token, "local window"), 'Final token "metal". Local token window: "local window".')

        # Text with both " metal" (with leading space) and "metal" (without leading space, e.g. in quotes)
        text = 'The word  metal is extracted, and "metal" is in quotes.'
        scrubbed = study.scrub_target_token(text, token)
        self.assertNotIn(" metal", scrubbed)
        self.assertNotIn("metal", scrubbed)
        self.assertEqual(scrubbed, 'The word <TOKEN> is extracted, and "<TOKEN>" is in quotes.')

        # Text with raw token and unquoted surface
        text_both_plain = "The word metal and  metal appear."
        scrubbed_plain = study.scrub_target_token(text_both_plain, token)
        self.assertNotIn(" metal", scrubbed_plain)
        self.assertNotIn("metal", scrubbed_plain)
        self.assertEqual(scrubbed_plain, "The word<TOKEN> and <TOKEN> appear.")

        # Text with only display surface
        text_surface_only = 'The word is "metal".'
        scrubbed_surface = study.scrub_target_token(text_surface_only, token)
        self.assertNotIn("metal", scrubbed_surface)
        self.assertEqual(scrubbed_surface, 'The word is "<TOKEN>".')

        # Text with only raw token
        text_raw_only = "The word is  metal."
        scrubbed_raw = study.scrub_target_token(text_raw_only, token)
        self.assertNotIn(" metal", scrubbed_raw)
        self.assertEqual(scrubbed_raw, "The word is <TOKEN>.")

        # Absent token fails loudly
        with self.assertRaises(study.StudyError):
            study.scrub_target_token("The word is gold.", token)

    def test_pairwise_distinct_tokens_validates_raw_and_surface(self):
        valid = ["a", "b", "c", "d", "e", "f", "g", "h"]
        study.validate_pairwise_distinct_tokens(valid, group_id="safety::content_early")

        # Distinct raw tokens but colliding surface forms
        colliding_surface = ["metal", " metal", "c", "d", "e", "f", "g", "h"]
        with self.assertRaises(study.StudyError):
            study.validate_pairwise_distinct_tokens(colliding_surface, group_id="safety::content_early")

        # Whitespace-only token fails
        whitespace_only = [" ", "b", "c", "d", "e", "f", "g", "h"]
        with self.assertRaises(study.StudyError):
            study.validate_pairwise_distinct_tokens(whitespace_only, group_id="safety::content_early")

    def test_diagnose_token_duplicates_eight_token_fixture(self):
        # 1. Valid distinct 8-token fixture
        clean_eight = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew"]
        diag_clean = study.diagnose_token_duplicates(clean_eight, group_id="safety::content_early")
        self.assertTrue(diag_clean["is_valid"])
        self.assertFalse(diag_clean["has_violations"])
        self.assertEqual(diag_clean["token_count"], 8)
        self.assertEqual(diag_clean["raw_duplicates"], [])
        self.assertEqual(diag_clean["surface_duplicates"], [])
        self.assertEqual(diag_clean["surface_errors"], [])

        # 2. Raw duplicate in 8-token fixture
        raw_dups = ["apple", "banana", "apple", "date", "elderberry", "fig", "grape", "honeydew"]
        diag_raw = study.diagnose_token_duplicates(raw_dups, group_id="safety::content_early")
        self.assertFalse(diag_raw["is_valid"])
        self.assertTrue(diag_raw["has_violations"])
        self.assertEqual(len(diag_raw["raw_duplicates"]), 1)
        self.assertEqual(diag_raw["raw_duplicates"][0]["token"], "apple")
        self.assertEqual(diag_raw["raw_duplicates"][0]["count"], 2)
        self.assertEqual(diag_raw["raw_duplicates"][0]["indices"], [0, 2])

        # 3. Surface-only collision in 8-token fixture (distinct raw tokens, identical surface)
        surface_dups = ["metal", " metal", "cherry", "date", "elderberry", "fig", "grape", "honeydew"]
        diag_surf = study.diagnose_token_duplicates(surface_dups, group_id="safety::content_early")
        self.assertFalse(diag_surf["is_valid"])
        self.assertTrue(diag_surf["has_violations"])
        self.assertEqual(diag_surf["raw_duplicates"], [])
        self.assertEqual(len(diag_surf["surface_duplicates"]), 1)
        self.assertEqual(diag_surf["surface_duplicates"][0]["surface_token"], "metal")
        self.assertEqual(diag_surf["surface_duplicates"][0]["count"], 2)
        self.assertEqual(diag_surf["surface_duplicates"][0]["indices"], [0, 1])

        # 4. Whitespace and empty tokens
        whitespace_eight = [" ", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew"]
        diag_ws = study.diagnose_token_duplicates(whitespace_eight, group_id="safety::content_early")
        self.assertTrue(diag_ws["has_violations"])
        self.assertEqual(len(diag_ws["surface_errors"]), 1)

    def test_build_preflight_diagnostic_contains_required_fields_and_no_av(self):
        prompts = study.load_context_prompt_spec(PROMPTS)
        selected_rows = []
        for prompt in prompts:
            for stratum in study.CONTENT_STRATA:
                raw = f"token_{prompt['id']}_{stratum}"
                # Inject a collision in safety::content_early for prompt 0 and prompt 1
                if prompt["category"] == "safety" and stratum == "content_early" and prompt["index"] in (0, 1):
                    raw = "duplicate_token"
                selected_rows.append({
                    "prompt_id": prompt["id"],
                    "prompt_index": int(prompt["index"]),
                    "category": prompt["category"],
                    "stratum": stratum,
                    "position_stratum": stratum,
                    "position": 5,
                    "token_id": 1234,
                    "raw_token": raw,
                    "raw_decoded_token": raw,
                    "display_token": study.display_token(raw),
                    "display_surface_token": study.display_token(raw),
                    "local_window": f"window_{prompt['id']}_{stratum}",
                })

        self.assertEqual(len(selected_rows), 48)
        diag = study.build_preflight_diagnostic(
            prompt_sha=study.FROZEN_PROMPT_SHA256,
            selected_rows=selected_rows,
        )

        self.assertEqual(diag["prompt_sha256"], study.FROZEN_PROMPT_SHA256)
        self.assertEqual(diag["prompt_sha"], study.FROZEN_PROMPT_SHA256)
        self.assertEqual(diag["status"], "fail")
        self.assertTrue(diag["has_violations"])
        self.assertEqual(len(diag["selected_rows"]), 48)

        # Check that each selected row has all 7 required fields
        for row in diag["selected_rows"]:
            self.assertIn("prompt_id", row)
            self.assertIn("category", row)
            self.assertIn("stratum", row)
            self.assertIn("position", row)
            self.assertIn("raw_decoded_token", row)
            self.assertIn("display_surface_token", row)
            self.assertIn("local_window", row)

        # Check every category x stratum inventory in candidate order
        for category in study.CATEGORIES:
            for stratum in study.CONTENT_STRATA:
                gid = study._group_id(category, stratum)
                self.assertIn(gid, diag["inventories"])
                inv = diag["inventories"][gid]
                self.assertEqual(inv["count"], 8)
                self.assertEqual(len(inv["items"]), 8)
                self.assertEqual(inv["candidate_order_prompt_ids"], [
                    p["id"] for p in prompts if p["category"] == category
                ])

        # Check explicit duplicate records
        self.assertEqual(len(diag["violations"]), 1)
        viol = diag["violations"][0]
        self.assertEqual(viol["group_id"], "safety::content_early")
        self.assertEqual(viol["raw_duplicates"][0]["token"], "duplicate_token")
        self.assertEqual(viol["raw_duplicates"][0]["count"], 2)
        self.assertEqual(viol["raw_duplicates"][0]["indices"], [0, 1])

        # Verify no AV outputs are included in diagnostic payload
        diag_str = json.dumps(diag)
        self.assertNotIn("score_text", diag_str)
        self.assertNotIn("nla_full", diag_str)
        self.assertNotIn("logits", diag_str)
        self.assertNotIn("hidden_states", diag_str)

    def test_stage_decide_regression_successful_path_writes_decision_json(self):
        prompts = study.load_context_prompt_spec(PROMPTS)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_dir = root / "results"
            extract_dir = results_dir / "extract"
            variants_dir = results_dir / "variants"
            ar_dir = results_dir / "ar"
            behavior_dir = results_dir / "behavior"
            bridge_dir = results_dir / "bridge"
            for d in (extract_dir, variants_dir, ar_dir, behavior_dir, bridge_dir):
                d.mkdir(parents=True, exist_ok=True)

            study.json_dump(results_dir / "preflight.json", {
                "status": "pass",
                "seed": study.SEED,
                "prompt_sha256": study.FROZEN_PROMPT_SHA256,
                "activation_count": study.N_CONTENT_TARGETS,
            })
            study.json_dump(bridge_dir / "ar.json", {
                "status": "pass",
                "target_count": len(study.BRIDGE_TARGETS),
                "targets": [
                    {
                        "stem": study._target_stem(pid, s),
                        "activation_id": study._activation_id(pid, s),
                        "recomputed_mse": 0.1,
                        "frozen_mse": 0.1,
                        "token_count": 10,
                        "finite": True,
                    }
                    for pid, s in study.BRIDGE_TARGETS
                ],
            })
            study.json_dump(bridge_dir / "behavior.json", {
                "status": "pass",
                "target_count": len(study.BRIDGE_TARGETS),
                "targets": [
                    {
                        "stem": study._target_stem(pid, s),
                        "activation_id": study._activation_id(pid, s),
                        "recomputed_jsd": 0.05,
                        "frozen_jsd": 0.05,
                        "finite": True,
                    }
                    for pid, s in study.BRIDGE_TARGETS
                ],
            })

            for prompt, stratum in study._iter_content_targets(prompts):
                stem = study._target_stem(prompt["id"], stratum)
                act_id = study._activation_id(prompt["id"], stratum)
                expected_ids = study._expected_candidate_ids(prompts, prompt["category"], stratum)

                study.json_dump(extract_dir / f"{stem}.json", {
                    "activation_id": act_id,
                    "position": 5,
                    "layer_index": study.LAYER_INDEX,
                })
                study.json_dump(variants_dir / f"{stem}.json", {
                    "activation_id": act_id,
                    "token_text": f"token_{prompt['id']}_{stratum}",
                    "structural_valid": True,
                    "variant_texts": {
                        variant: f"{variant}_{prompt['id']}_{stratum}"
                        for variant in study.VARIANT_NAMES
                    },
                })
                study.json_dump(ar_dir / f"{stem}.json", {
                    "activation_id": act_id,
                    "layer_index": study.LAYER_INDEX,
                    "unrelated_mse_nrm": 1.0,
                    "variants": {
                        variant: {
                            "candidate_ids": expected_ids,
                            "candidate_texts": [f"{variant}_{cid.split('::')[0]}_{stratum}" for cid in expected_ids],
                            "candidates": [
                                {
                                    "candidate_id": cid,
                                    "own": cid == act_id,
                                    "mse_nrm": 0.01 if cid == act_id else (0.5 if variant == "nla_full" else 0.4),
                                }
                                for cid in expected_ids
                            ],
                            "paraphrase_mse_nrm": 0.02,
                        }
                        for variant in study.VARIANT_NAMES
                    },
                })
                study.json_dump(behavior_dir / f"{stem}.json", {
                    "activation_id": act_id,
                    "gold_reinjection_jsd": 0.0,
                    "random_direction_jsd": 0.9,
                    "variants": {
                        variant: {
                            "candidate_ids": expected_ids,
                            "candidates": [
                                {
                                    "candidate_id": cid,
                                    "own": cid == act_id,
                                    "jsd": 0.01 if cid == act_id else (0.5 if variant == "nla_full" else 0.4),
                                }
                                for cid in expected_ids
                            ],
                            "paraphrase_jsd": 0.02,
                        }
                        for variant in study.VARIANT_NAMES
                    },
                })

            args = type("Args", (), {
                "prompts": PROMPTS,
                "results_dir": results_dir,
                "frozen_dir": root / "frozen",
            })()
            result = study.stage_decide(args)
            self.assertEqual(result["status"], "pass")
            self.assertTrue(result["operational_validity"])
            self.assertIn("replicated_on_token_only", result)
            self.assertIsInstance(result["replicated_on_token_only"], bool)
            self.assertTrue((results_dir / "decision.json").is_file())
            written = json.loads((results_dir / "decision.json").read_text())
            self.assertEqual(written["status"], "pass")
            self.assertEqual(written["replicated_on_token_only"], result["replicated_on_token_only"])
            self.assertTrue((results_dir / "completion-manifest.json").is_file())
            manifest = json.loads((results_dir / "completion-manifest.json").read_text())
            self.assertEqual(manifest["decision_sha256"], study.sha256_file(results_dir / "decision.json"))
            self.assertEqual(manifest["stage_statuses"]["decide"], "pass")

    def test_compare_bridge_values_tolerances(self):
        inside = study.compare_bridge_values(1.0, 1.0 + 5e-5, 0.2, 0.2 + 3e-6)
        self.assertTrue(inside["passed"])
        outside_mse = study.compare_bridge_values(1.0, 1.01, 0.2, 0.2)
        self.assertFalse(outside_mse["passed"])
        outside_jsd = study.compare_bridge_values(1.0, 1.0, 0.2, 0.21)
        self.assertFalse(outside_jsd["passed"])

    def test_pair_delta_positive_when_prose_is_better(self):
        self.assertAlmostEqual(study.pair_delta(0.4, 0.1), 0.3)
        with self.assertRaises(study.StudyError):
            study.pair_delta(float("nan"), 0.1)

    def test_tournament_margin_advantage_when_own_scores_tie(self):
        own_index = 0
        baseline_errors = [0.20, 0.25, 0.28, 0.30, 0.30, 0.32, 0.35, 0.40]
        prose_errors = [0.20, 0.50, 0.55, 0.60, 0.60, 0.65, 0.70, 0.80]

        margin_baseline = study.tournament_margin(baseline_errors, own_index)
        margin_prose = study.tournament_margin(prose_errors, own_index)

        self.assertAlmostEqual(margin_baseline, 0.10)
        self.assertAlmostEqual(margin_prose, 0.40)

        own_score_delta = baseline_errors[own_index] - prose_errors[own_index]
        self.assertEqual(own_score_delta, 0.0)

        delta = study.pair_delta(margin_prose, margin_baseline)
        self.assertAlmostEqual(delta, 0.30)
        self.assertGreater(delta, 0.0)

    def test_every_context_decision_branch(self):
        # 1. INVALID_MEASUREMENT: operational is False
        self.assertEqual(
            study.classify_context_decision(
                False, 0.8, 38, 0.1, 0.001, 0.1, 0.001, 0.1, 0.001, 0.1, 0.001, -0.1, 0.9, -0.1, 0.9
            ),
            "INVALID_MEASUREMENT",
        )
        # 2. REPLICATION_FAILURE: joint rate <= 0.125 or joint count <= 24
        self.assertEqual(
            study.classify_context_decision(
                True, 0.10, 5, 0.1, 0.001, 0.1, 0.001, 0.1, 0.001, 0.1, 0.001, -0.1, 0.9, -0.1, 0.9
            ),
            "REPLICATION_FAILURE",
        )
        self.assertEqual(
            study.classify_context_decision(
                True, 0.50, 24, 0.1, 0.001, 0.1, 0.001, 0.1, 0.001, 0.1, 0.001, -0.1, 0.9, -0.1, 0.9
            ),
            "REPLICATION_FAILURE",
        )
        # 3. PROSE_EXCEEDS_CONTEXT: primary and secondary both sig positive
        self.assertEqual(
            study.classify_context_decision(
                True, 0.8, 38, 0.1, 0.001, 0.1, 0.001, 0.1, 0.001, 0.1, 0.001, -0.1, 0.9, -0.1, 0.9
            ),
            "PROSE_EXCEEDS_CONTEXT",
        )
        # 4. PROSE_REDUCES_TO_CONTEXT: reversed local_ctx-full sign-flips both sig positive
        self.assertEqual(
            study.classify_context_decision(
                True, 0.8, 38, -0.1, 0.9, -0.1, 0.9, -0.1, 0.9, -0.1, 0.9, 0.1, 0.001, 0.1, 0.001
            ),
            "PROSE_REDUCES_TO_CONTEXT",
        )
        # 5. PROSE_PARTIAL: when neither exceeds nor reduces
        self.assertEqual(
            study.classify_context_decision(
                True, 0.8, 38, 0.1, 0.001, 0.1, 0.05, 0.1, 0.001, 0.1, 0.001, -0.1, 0.9, -0.1, 0.9
            ),
            "PROSE_PARTIAL",
        )
        self.assertEqual(
            study.classify_context_decision(
                True, 0.8, 38, 0.1, 0.001, 0.1, 0.001, 0.1, 0.001, 0.1, 0.05, -0.1, 0.9, -0.1, 0.9
            ),
            "PROSE_PARTIAL",
        )
        self.assertEqual(
            study.classify_context_decision(
                True, 0.8, 38, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5
            ),
            "PROSE_PARTIAL",
        )
        self.assertEqual(set(study.DECISION_LABELS), {
            "PROSE_EXCEEDS_CONTEXT",
            "PROSE_REDUCES_TO_CONTEXT",
            "PROSE_PARTIAL",
            "REPLICATION_FAILURE",
            "INVALID_MEASUREMENT",
        })

    def test_decide_fails_loudly_without_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = type("Args", (), {
                "prompts": PROMPTS,
                "results_dir": root / "results",
                "frozen_dir": root / "frozen",
            })()
            with self.assertRaises(study.StudyError):
                study.stage_decide(args)

    def test_prompt_cluster_sign_flip_is_deterministic_and_clustered(self):
        records = [
            {"prompt_id": "p1", "value": 1.0},
            {"prompt_id": "p1", "value": 1.0},
            {"prompt_id": "p2", "value": 1.0},
            {"prompt_id": "p2", "value": 1.0},
        ]
        first = study.prompt_cluster_sign_flip(records, "value", n_draws=50)
        second = study.prompt_cluster_sign_flip(records, "value", n_draws=50)
        self.assertEqual(first, second)
        self.assertEqual(first["cluster_count"], 2)
        self.assertEqual(first["cluster_sizes"], [2, 2])
        self.assertEqual(first["statistic"], "median")
        self.assertEqual(first["observed"], 1.0)
        self.assertEqual(first["one_sided"], "greater")
        self.assertEqual(first["p_value"], (first["exceedances"] + 1) / 51)

    def test_completion_manifest_inventory_sorted_and_no_self_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_dir = root / "results"
            extract_dir = results_dir / "extract"
            ar_dir = results_dir / "ar"
            extract_dir.mkdir(parents=True, exist_ok=True)
            ar_dir.mkdir(parents=True, exist_ok=True)

            study.json_dump(results_dir / "decision.json", {"status": "pass", "study": "context-baselines"})
            study.json_dump(results_dir / "preflight.json", {"status": "pass"})
            study.json_dump(extract_dir / "target_01.json", {"id": 1})
            study.json_dump(ar_dir / "target_01.json", {"mse": 0.05})
            # Pre-create a completion-manifest to assert that it gets excluded from inventory
            study.json_dump(results_dir / "completion-manifest.json", {"old": "manifest"})

            args = type("Args", (), {
                "prompts": PROMPTS,
                "results_dir": results_dir,
                "study_dir": root,
            })()

            manifest_path = study.write_completion_manifest(args)
            self.assertTrue(manifest_path.is_file())
            manifest = json.loads(manifest_path.read_text())

            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["runner_sha256"], study.sha256_file(Path(study.__file__).resolve()))
            self.assertEqual(manifest["prompt_sha256"], study.sha256_file(PROMPTS))
            self.assertEqual(manifest["decision_sha256"], study.sha256_file(results_dir / "decision.json"))
            self.assertEqual(manifest["result_dir"], str(results_dir.resolve()))
            self.assertEqual(manifest["stage_statuses"], {
                "preflight": "pass",
                "av": "pass",
                "variants": "pass",
                "ar": "pass",
                "behavior": "pass",
                "decide": "pass",
            })

            inventory = manifest["file_inventory"]
            paths = [item["path"] for item in inventory]
            self.assertEqual(paths, sorted(paths))
            self.assertNotIn("completion-manifest.json", paths)
            self.assertFalse(any("completion-manifest.json" in item["path"] for item in inventory))

            expected_paths = {
                "ar/target_01.json",
                "decision.json",
                "extract/target_01.json",
                "preflight.json",
            }
            self.assertEqual(set(paths), expected_paths)

            for item in inventory:
                file_path = results_dir / item["path"]
                self.assertEqual(item["sha256"], study.sha256_file(file_path))
                self.assertEqual(item["bytes"], file_path.stat().st_size)

    def test_slurm_script_provenance_and_atomic_symlink(self):
        slurm_path = Path(__file__).parents[1] / "code" / "run_nla_context_baselines.slurm"
        self.assertTrue(slurm_path.is_file())
        content = slurm_path.read_text()

        self.assertIn('RUN_DIR="$STUDY/results/runs/job-${SLURM_JOB_ID:?}"', content)
        self.assertIn('test ! -e "$RUN_DIR"', content)
        self.assertIn('mkdir -p "$RUN_DIR/provenance"', content)

        self.assertIn('cp "$STUDY/prompts.json" "$STUDY/PROTOCOL.md" "$ROOT/code/nla/run_context_baseline_study.py" "$ROOT/code/run_nla_context_baselines.slurm" "$RUN_DIR/provenance/"', content)
        self.assertIn('sha256sum prompts.json PROTOCOL.md run_context_baseline_study.py run_nla_context_baselines.slurm > input-sha256.txt', content)

        self.assertIn('--results-dir "$RUN_DIR"', content)
        self.assertIn('ln -sfn "runs/job-$SLURM_JOB_ID" "$STUDY/results/latest.new" && mv -Tf "$STUDY/results/latest.new" "$STUDY/results/latest"', content)

        # Assert provenance copy happens before runner execution
        prov_idx = content.index('sha256sum prompts.json PROTOCOL.md')
        runner_idx = content.index('"$PYTHON" "$ROOT/code/nla/run_context_baseline_study.py" --stage all')
        symlink_idx = content.index('ln -sfn "runs/job-$SLURM_JOB_ID"')
        self.assertLess(prov_idx, runner_idx)
        self.assertLess(runner_idx, symlink_idx)

        # Assert no rm of prior root artifacts
        self.assertNotIn("rm -rf", content)
        self.assertNotIn("rm -r", content)


if __name__ == "__main__":
    unittest.main()
