import tempfile
import unittest
from pathlib import Path

from cweval_corrected_analysis_v2 import (
    analyze,
    audit_candidate_provenance,
    audit_manifest_prompts,
    chi_square3_sf,
    checkpoint_audit,
    cochran_q,
    holm_adjust,
    mcnemar_exact,
    read_res_all,
    single_outcome,
    wilson_ci,
)


class CorrectedAnalysisTests(unittest.TestCase):
    def test_wilson_small_sample(self):
        self.assertEqual(wilson_ci(13, 25), [33.5, 69.97])

    def test_mcnemar_exact_two_sided(self):
        # Two improvements and ten regressions among 12 discordant pairs.
        a = [0] * 11 + [1] * 14
        b = [1, 1] + [0] * 19 + [1] * 4
        stat = mcnemar_exact(a, b)
        self.assertEqual(stat["repaired_A_to_B"], 2)
        self.assertEqual(stat["regressed_A_to_B"], 10)
        self.assertEqual(stat["exact_two_sided_p"], 0.03857422)

    def test_zero_baseline_denominator_is_null_not_zero(self):
        stat = mcnemar_exact([1, 1], [1, 0])
        self.assertIsNone(stat["repair_rate_pct"])
        self.assertEqual(stat["regression_rate_pct"], 50.0)

    def test_holm_monotonicity(self):
        corrected = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.20})
        self.assertAlmostEqual(corrected["a"]["holm_adjusted_p"], 0.03)
        self.assertAlmostEqual(corrected["b"]["holm_adjusted_p"], 0.08)
        self.assertAlmostEqual(corrected["c"]["holm_adjusted_p"], 0.20)

    def test_chi_square_df3_survival(self):
        self.assertAlmostEqual(chi_square3_sf(9.0), 0.0292908865, places=8)

    def test_cochran_q_reports_exact_conditional_permutation(self):
        matrix = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 1, 0, 0],
        ]
        result = cochran_q(matrix, exact_permutation=True)
        self.assertGreaterEqual(result["exact_conditional_permutation_p"], 0.0)
        self.assertLessEqual(result["exact_conditional_permutation_p"], 1.0)
        self.assertEqual(result["df"], 3)

    def test_real_cweval_array_schema_false_is_not_truthy_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "res_all.json"
            p.write_text(
                '{"../../../eval_workspace/arm/generated_X/core/py/cwe_020_0_test.py":'
                '{"functional":[false],"secure":[false],"func_secure":[false]}}',
                encoding="utf-8",
            )
            parsed = read_res_all(p)
            value, present = single_outcome(parsed["cwe_020_0"])
            self.assertTrue(present)
            self.assertEqual(value, {"functional": 0, "secure": 0, "func_secure": 0})

    def test_a2_multiple_candidates_are_not_collapsed_to_bool_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "res_all.json"
            p.write_text(
                '{"../../../eval_workspace/arm/generated_X/core/py/cwe_022_0_test.py":'
                '{"functional":[false,true,false],"secure":[false,false,true],"func_secure":[false,false,false]}}',
                encoding="utf-8",
            )
            parsed = read_res_all(p)
            self.assertEqual(any(parsed["cwe_022_0"]["functional"]), True)
            self.assertEqual(any(parsed["cwe_022_0"]["secure"]), True)
            self.assertEqual(any(parsed["cwe_022_0"]["func_secure"]), False)



    def test_manifest_prompt_hash_and_ast_audit(self):
        import hashlib
        import json
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.json"
            prompt = "def f(x):\n    return x\n"
            item = {
                "task_id": "cwe_001_0",
                "prompt_code": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
            path.write_text(json.dumps([item]), encoding="utf-8")
            result = audit_manifest_prompts(path)
            self.assertEqual(result["tasks_checked"], 1)
            self.assertEqual(result["prompt_sha256_mismatches"], [])
            self.assertEqual(result["prompt_ast_syntax_errors"], [])
            self.assertEqual(result["solution_marker_present_in_prompt"], [])

    def test_candidate_provenance_audit_detects_mismatch(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cp = root / "cweval_study" / "checkpoints"
            ev = root / "cweval_study" / "cweval_export" / "evals"
            task = "cwe_001_0"
            cp.mkdir(parents=True)
            telemetry = {task: {"A2_syntax_selected_index": "1"}}
            calls = {
                ("A1", "arm_A1_baseline", 0, f"{task}__A1_draft"),
                ("A2_Index0", "arm_A2_Index0", 0, f"{task}__A2_sample_0"),
                ("A2_SyntaxSelected", "arm_A2_SyntaxSelected", 0, f"{task}__A2_sample_1"),
                ("B", "arm_B_homogeneous", 0, f"{task}__B_merged"),
                ("C", "arm_C_heterogeneous", 0, f"{task}__C_merged"),
            }
            for i in range(4):
                calls.add(("A2_Pass4", "arm_A2_samples4", i, f"{task}__A2_sample_{i}"))
            for arm, folder, idx, call_id in calls:
                sample_suffix = call_id.rsplit("__A2_sample_", 1)
                code_value = int(sample_suffix[1]) if len(sample_suffix) == 2 else 10
                code = f"def f(): return {code_value}\n"
                (cp / f"{call_id}.json").write_text(
                    json.dumps({"call_id": call_id, "extracted_code": code}),
                    encoding="utf-8"
                )
                d = ev / folder / f"generated_{idx}" / "core" / "py"
                d.mkdir(parents=True, exist_ok=True)
                # Deliberately corrupt one candidate so the audit reports it.
                actual = code if not (arm == "B") else "def f(): return 99\n"
                (d / f"{task}_task.py").write_text(actual, encoding="utf-8")
            result = audit_candidate_provenance(root, [task], telemetry)
            self.assertEqual(result["by_arm"]["A1"]["matched"], 1)
            self.assertEqual(result["by_arm"]["B"]["content_mismatch"], 1)
            self.assertEqual(len(result["problems"]), 1)


    def test_checkpoint_audit_counts_empty_fallback_issue_payloads(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "checkpoints"
            cp.mkdir()
            common = {
                "arm": "C",
                "role": "Heterogeneous_Functional_Reviewer",
                "canonical_model": "@cf/qwen/qwen3.8-27b",
                "status": "SUCCESS",
                "output_tokens": 100,
                "gpu_vram_used_mb": 0,
                "report_json_valid": False,
            }
            records = [
                {**common, "call_id": "empty_list", "parsed_report": {
                    "raw_parse_fallback": True, "verdict": "PASS", "issues": []
                }},
                {**common, "call_id": "blank_desc", "parsed_report": {
                    "raw_parse_fallback": True, "verdict": "PASS",
                    "issues": [{"severity": "MEDIUM", "description": ""}]
                }},
                {**common, "call_id": "has_desc", "parsed_report": {
                    "raw_parse_fallback": True, "verdict": "PASS",
                    "issues": [{"severity": "HIGH", "description": "Concrete issue"}]
                }},
            ]
            for record in records:
                (cp / f"{record['call_id']}.json").write_text(
                    json.dumps(record), encoding="utf-8"
                )
            result = checkpoint_audit(cp)
            self.assertEqual(result["checkpoint_records_parsed"], 3)
            self.assertEqual(result["raw_review_parse_fallbacks_by_arm"]["C"], 3)
            self.assertEqual(result["fallback_reviews_defaulted_to_pass_by_arm"]["C"], 3)
            self.assertEqual(result["fallback_pass_with_empty_issue_description_by_arm"]["C"], 2)

    def test_recovered_study_cochran_q_exact_permutation_values(self):
        # Frozen task-level outcomes independently reconstructed from the six
        # raw CWEval res_all.json files; arm order A1, SyntaxSelected, B, C.
        arrays = {
            "functional": [
                [0,0,0,0,1,0,1,1,0,1,1,1,1,0,0,1,1,1,0,0,0,1,0,1,1],
                [0,0,1,0,1,0,1,1,0,1,0,1,1,1,1,1,1,1,0,0,1,1,0,1,1],
                [0,1,0,1,1,0,1,1,1,1,0,1,1,1,0,1,1,0,0,0,0,1,0,0,1],
                [0,0,0,0,1,0,0,1,1,0,1,1,0,1,1,0,0,0,0,0,1,0,0,0,1],
            ],
            "secure": [
                [0,0,0,0,1,0,0,0,0,1,0,1,1,0,0,1,1,1,0,1,0,1,0,0,1],
                [0,0,0,0,1,0,0,0,0,1,0,1,1,1,1,1,1,1,0,1,1,1,0,0,1],
                [0,1,0,1,1,0,1,1,1,1,0,1,1,1,0,1,1,0,0,1,0,1,0,1,1],
                [0,0,0,0,1,0,1,0,0,0,0,1,0,1,1,0,0,0,0,1,1,0,0,0,1],
            ],
            "func_secure": [
                [0,0,0,0,1,0,0,0,0,1,0,1,1,0,0,1,1,1,0,0,0,1,0,0,1],
                [0,0,0,0,1,0,0,0,0,1,0,1,1,1,1,1,1,1,0,0,1,1,0,0,1],
                [0,1,0,1,1,0,1,1,1,1,0,1,1,1,0,1,1,0,0,0,0,1,0,0,1],
                [0,0,0,0,1,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,1,0,0,0,1],
            ],
        }
        expected = {
            "functional": (5.77777778, 0.14675981),
            "secure": (9.0, 0.02736334),
            "func_secure": (9.8, 0.01846344),
        }
        for metric, arm_columns in arrays.items():
            matrix = [[arm_columns[j][i] for j in range(4)] for i in range(25)]
            result = cochran_q(matrix, exact_permutation=True)
            self.assertAlmostEqual(result["Q"], expected[metric][0], places=6)
            self.assertAlmostEqual(result["exact_conditional_permutation_p"], expected[metric][1], places=7)

    def test_end_to_end_fixture_with_real_res_all_shape(self):
        import csv
        import json
        from pathlib import Path

        arms = {
            "A1": "arm_A1_baseline",
            "A2_Index0": "arm_A2_Index0",
            "A2_SyntaxSelected": "arm_A2_SyntaxSelected",
            "B": "arm_B_homogeneous",
            "C": "arm_C_heterogeneous",
            "A2_Pass4": "arm_A2_samples4",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_root = root / "final_paper_results" / "extracted_bundle" / "evals"
            eval_root.mkdir(parents=True)
            (root / "cweval_study").mkdir()
            tasks = [f"cwe_{i:03d}_0" for i in range(25)]
            (root / "cweval_study" / "tasks_25_manifest.json").write_text(
                json.dumps([{"task_id": t} for t in tasks]), encoding="utf-8"
            )
            fields = [
                "task_id", "A1_syntax_valid", "A1_time_sec", "A1_in_tokens", "A1_out_tokens",
                "A2_valid_samples_count", "A2_syntax_selected_index", "A2_total_time_sec",
                "A2_total_in_tokens", "A2_total_out_tokens", "B_func_review_time_sec",
                "B_sec_review_time_sec", "B_merge_time_sec", "B_total_time_sec",
                "B_total_in_tokens", "B_total_out_tokens", "B_func_json_valid", "B_sec_json_valid",
                "B_final_syntax_valid", "C_func_review_time_sec", "C_sec_review_time_sec",
                "C_merge_time_sec", "C_total_time_sec", "C_total_in_tokens", "C_total_out_tokens",
                "C_func_json_valid", "C_sec_json_valid", "C_final_syntax_valid"
            ]
            rows=[]
            for t in tasks:
                rows.append({
                    "task_id":t,"A1_syntax_valid":"1","A1_time_sec":"1","A1_in_tokens":"10","A1_out_tokens":"10",
                    "A2_valid_samples_count":"4","A2_syntax_selected_index":"0","A2_total_time_sec":"4",
                    "A2_total_in_tokens":"40","A2_total_out_tokens":"40","B_func_review_time_sec":"1",
                    "B_sec_review_time_sec":"1","B_merge_time_sec":"1","B_total_time_sec":"3",
                    "B_total_in_tokens":"30","B_total_out_tokens":"30","B_func_json_valid":"1","B_sec_json_valid":"1",
                    "B_final_syntax_valid":"1","C_func_review_time_sec":"1","C_sec_review_time_sec":"1",
                    "C_merge_time_sec":"1","C_total_time_sec":"3","C_total_in_tokens":"30","C_total_out_tokens":"30",
                    "C_func_json_valid":"1","C_sec_json_valid":"1","C_final_syntax_valid":"1"
                })
            telemetry=eval_root.parent / "generation_telemetry_25tasks.csv"
            with telemetry.open("w",encoding="utf-8",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            for arm,folder in arms.items():
                d=eval_root/folder;d.mkdir()
                payload={}
                for i,t in enumerate(tasks):
                    k=4 if arm=="A2_Pass4" else 1
                    vals=[bool((i+j)%3==0) for j in range(k)]
                    payload[f"../../../eval_workspace/{folder}/generated_X/core/py/{t}_test.py"]={
                        "functional":vals,"secure":vals,"func_secure":vals
                    }
                (d/"res_all.json").write_text(json.dumps(payload),encoding="utf-8")
            out=root/"corrected"
            report=analyze(root,out)
            self.assertEqual(report["n_tasks"],25)
            self.assertTrue((out/"corrected_statistical_report.json").exists())
            self.assertTrue((out/"corrected_unified_task_results.csv").exists())
            self.assertTrue((out/"corrected_arm_summary.csv").exists())
            self.assertEqual(report["arm_summaries"][0]["tasks_denominator"],25)
            self.assertEqual(len(report["cochrans_q_holm_adjustment_across_3_outcomes"]), 3)
            self.assertEqual(len(report["holm_adjustment_across_all_15_reported_tests_sensitivity"]), 15)


if __name__ == "__main__":
    unittest.main(verbosity=2)