#!/usr/bin/env python3
"""Corrected, schema-aware CWEval analysis for the 25-task study.

This tool deliberately does NOT run model calls. It reads saved CWEval res_all.json
files and telemetry, preserves missing results as end-to-end failures, and reports
coverage separately. The source repository and manuscript are never modified.

Expected layout (defaults):
  final_paper_results/extracted_bundle/evals/<arm>/res_all.json
  final_paper_results/extracted_bundle/generation_telemetry_25tasks.csv
  cweval_study/tasks_25_manifest.json
  cweval_study/checkpoints/*.json

Usage:
  python cweval_corrected_analysis_v2.py \
    --repo-root . \
    --output-dir final_paper_results_corrected
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TASK_RE = re.compile(r"(cwe_\d+_\d+)_test\.py$")
METRICS = ("functional", "secure", "func_secure")
SINGLE_ARMS = {
    "A1": "arm_A1_baseline",
    "A2_Index0": "arm_A2_Index0",
    "A2_SyntaxSelected": "arm_A2_SyntaxSelected",
    "B": "arm_B_homogeneous",
    "C": "arm_C_heterogeneous",
}
A2_DIR = "arm_A2_samples4"
OMNIBUS_ARMS = ("A1", "A2_SyntaxSelected", "B", "C")
PAIRWISE = (
    ("B_vs_A1", "A1", "B"),
    ("C_vs_A1", "A1", "C"),
    ("C_vs_B", "B", "C"),
    ("C_vs_A2_SyntaxSelected", "A2_SyntaxSelected", "C"),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n <= 0:
        return None
    p = successes / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / den
    return [round(100 * max(0.0, center - half), 2), round(100 * min(1.0, center + half), 2)]


def mcnemar_exact(a: list[int], b: list[int]) -> dict[str, Any]:
    if len(a) != len(b):
        raise ValueError("McNemar inputs must be paired and equal length")
    n00 = sum(x == 0 and y == 0 for x, y in zip(a, b))
    n11 = sum(x == 1 and y == 1 for x, y in zip(a, b))
    n01 = sum(x == 0 and y == 1 for x, y in zip(a, b))
    n10 = sum(x == 1 and y == 0 for x, y in zip(a, b))
    d = n01 + n10
    if d == 0:
        p = 1.0
    else:
        tail = sum(math.comb(d, i) for i in range(min(n01, n10) + 1)) / (2**d)
        p = min(1.0, 2.0 * tail)
    failed_a = n00 + n01
    passed_a = n11 + n10
    return {
        "n_total": len(a),
        "both_fail": n00,
        "both_pass": n11,
        "repaired_A_to_B": n01,
        "regressed_A_to_B": n10,
        "discordant": d,
        "repair_denominator_A_failures": failed_a,
        "regression_denominator_A_passes": passed_a,
        "repair_rate_pct": round(100 * n01 / failed_a, 2) if failed_a else None,
        "regression_rate_pct": round(100 * n10 / passed_a, 2) if passed_a else None,
        "exact_two_sided_p": round(p, 8),
    }


def chi_square3_sf(q: float) -> float:
    """Exact survival function for chi-square with 3 degrees of freedom."""
    return math.erfc(math.sqrt(q / 2.0)) + math.sqrt(2.0 * q / math.pi) * math.exp(-q / 2.0)


def cochran_q(matrix: list[list[int]], exact_permutation: bool = True) -> dict[str, Any]:
    if not matrix:
        raise ValueError("Cochran Q requires at least one task")
    k = len(matrix[0])
    if k < 2 or any(len(row) != k for row in matrix):
        raise ValueError("Cochran Q matrix must be rectangular with at least two arms")
    if any(v not in (0, 1) for row in matrix for v in row):
        raise ValueError("Cochran Q accepts binary outcomes only")
    n = len(matrix)
    row_sums = [sum(row) for row in matrix]
    col_sums = [sum(row[j] for row in matrix) for j in range(k)]
    total = sum(row_sums)
    numerator = k * (k - 1) * sum((c - total / k) ** 2 for c in col_sums)
    denominator = k * total - sum(r * r for r in row_sums)
    q = numerator / denominator if denominator else 0.0
    result: dict[str, Any] = {
        "Q": round(q, 8),
        "df": k - 1,
        "asymptotic_chi_square_p": round(chi_square3_sf(q), 8) if k == 4 else None,
        "column_pass_counts": col_sums,
        "n_tasks": n,
    }
    if exact_permutation and k == 4 and denominator == 0:
        # Every task is either a four-arm all-failure or all-success row;
        # there is no within-task disagreement and the conditional null has
        # exactly one assignment.
        result["exact_conditional_permutation_p"] = 1.0
        result["permutation_assignments"] = 1
    elif exact_permutation and k == 4:
        # Conditional exact test: preserve each task's number of successes and
        # enumerate all possible within-task assignments to the four arms.
        choices_by_r = {
            r: list(itertools.combinations(range(k), r)) for r in range(k + 1)
        }
        dp: dict[tuple[int, ...], int] = {(0, 0, 0, 0): 1}
        for r in row_sums:
            next_dp: dict[tuple[int, ...], int] = defaultdict(int)
            for state, weight in dp.items():
                for cols in choices_by_r[r]:
                    new = list(state)
                    for j in cols:
                        new[j] += 1
                    next_dp[tuple(new)] += weight
            dp = next_dp
        total_weight = sum(dp.values())
        extreme_weight = 0
        for cols, weight in dp.items():
            num = k * (k - 1) * sum((c - total / k) ** 2 for c in cols)
            q_perm = num / denominator
            if q_perm >= q - 1e-12:
                extreme_weight += weight
        result["exact_conditional_permutation_p"] = round(extreme_weight / total_weight, 8)
        result["permutation_assignments"] = total_weight
    return result


def holm_adjust(p_values: dict[str, float]) -> dict[str, dict[str, float]]:
    ordered = sorted(p_values.items(), key=lambda x: x[1])
    m = len(ordered)
    running = 0.0
    out: dict[str, dict[str, float]] = {}
    for i, (name, p) in enumerate(ordered):
        adjusted = min(1.0, (m - i) * p)
        running = max(running, adjusted)
        out[name] = {"raw_p": round(p, 8), "holm_adjusted_p": round(running, 8)}
    return out


def read_telemetry(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Telemetry is empty: {path}")
    data: dict[str, dict[str, str]] = {}
    for row in rows:
        task = row.get("task_id", "").strip()
        if not task or task in data:
            raise ValueError(f"Missing or duplicate task_id in telemetry: {task!r}")
        data[task] = row
    return data


def read_res_all(path: Path) -> dict[str, dict[str, list[bool]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"res_all must be a JSON object: {path}")
    by_task: dict[str, dict[str, list[bool]]] = {}
    for result_path, entry in raw.items():
        match = TASK_RE.search(result_path)
        if not match:
            continue
        task = match.group(1)
        if task in by_task:
            raise ValueError(f"Duplicate res_all entries for {task} in {path}")
        if not isinstance(entry, dict):
            raise ValueError(f"Result for {task} is not an object")
        parsed: dict[str, list[bool]] = {}
        for metric in METRICS:
            values = entry.get(metric)
            if not isinstance(values, list) or any(type(v) is not bool for v in values):
                raise ValueError(
                    f"Expected list[bool] for {metric} at {task} in {path}; "
                    f"got {type(values).__name__}"
                )
            parsed[metric] = values
        sizes = {len(parsed[m]) for m in METRICS}
        if len(sizes) != 1:
            raise ValueError(f"Metric arrays have inconsistent lengths for {task}: {sizes}")
        by_task[task] = parsed
    return by_task


def single_outcome(result: dict[str, list[bool]] | None) -> tuple[dict[str, int], bool]:
    if not result or len(result["functional"]) != 1:
        return {m: 0 for m in METRICS}, False
    return {m: int(result[m][0]) for m in METRICS}, True



def audit_manifest_prompts(manifest_path: Path) -> dict[str, Any]:
    """Check manifest prompt hashes, Python syntax and solution-marker absence."""
    if not manifest_path.exists():
        return {"manifest_present": False}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked = 0
    hash_mismatches: list[str] = []
    syntax_errors: list[dict[str, str]] = []
    solution_marker_present: list[str] = []
    for item in manifest:
        task_id = str(item.get("task_id") or "")
        prompt = str(item.get("prompt_code") or "")
        checked += 1
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if digest != item.get("prompt_sha256"):
            hash_mismatches.append(task_id)
        try:
            ast.parse(prompt)
        except SyntaxError as exc:
            syntax_errors.append({"task_id": task_id, "error": exc.msg})
        if "# BEGIN SOLUTION" in prompt or "BEGIN SOLUTION" in prompt:
            solution_marker_present.append(task_id)
    return {
        "manifest_present": True,
        "tasks_checked": checked,
        "prompt_sha256_mismatches": hash_mismatches,
        "prompt_ast_syntax_errors": syntax_errors,
        "solution_marker_present_in_prompt": solution_marker_present,
        "raw_task_sha256_validation": "not_performed_without_pinned_benchmark_checkout",
    }


def audit_candidate_provenance(
    repo_root: Path, tasks: list[str], telemetry: dict[str, dict[str, str]]
) -> dict[str, Any]:
    """Compare every exported task candidate against its saved generation checkpoint."""
    checkpoint_dir = repo_root / "cweval_study" / "checkpoints"
    eval_root = repo_root / "cweval_study" / "cweval_export" / "evals"
    arm_dirs = {
        "A1": "arm_A1_baseline",
        "A2_Index0": "arm_A2_Index0",
        "A2_SyntaxSelected": "arm_A2_SyntaxSelected",
        "A2_Pass4": "arm_A2_samples4",
        "B": "arm_B_homogeneous",
        "C": "arm_C_heterogeneous",
    }
    outcomes: dict[str, Counter] = {arm: Counter() for arm in arm_dirs}
    problems: list[dict[str, str]] = []

    def verify(arm: str, folder: str, generated_idx: int, task: str, call_id: str) -> None:
        rel = Path(folder) / f"generated_{generated_idx}" / "core" / "py" / f"{task}_task.py"
        candidate_path = eval_root / rel
        checkpoint_path = checkpoint_dir / f"{call_id}.json"
        if not candidate_path.exists():
            outcomes[arm]["missing_exported_candidate"] += 1
            problems.append({"arm": arm, "task_id": task, "status": "missing_exported_candidate", "path": str(rel)})
            return
        if not checkpoint_path.exists():
            outcomes[arm]["missing_checkpoint"] += 1
            problems.append({"arm": arm, "task_id": task, "status": "missing_checkpoint", "path": str(checkpoint_path.relative_to(repo_root))})
            return
        try:
            rec = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            expected = str(rec.get("extracted_code") or "").strip() + "\n"
            actual = candidate_path.read_text(encoding="utf-8")
        except Exception as exc:
            outcomes[arm]["unreadable_source"] += 1
            problems.append({"arm": arm, "task_id": task, "status": "unreadable_source", "error": str(exc)})
            return
        if actual == expected:
            outcomes[arm]["matched"] += 1
        else:
            outcomes[arm]["content_mismatch"] += 1
            problems.append({
                "arm": arm,
                "task_id": task,
                "status": "content_mismatch",
                "candidate_sha256": hashlib.sha256(actual.encode("utf-8")).hexdigest(),
                "checkpoint_code_sha256": hashlib.sha256(expected.encode("utf-8")).hexdigest(),
            })

    for task in tasks:
        verify("A1", arm_dirs["A1"], 0, task, f"{task}__A1_draft")
        verify("A2_Index0", arm_dirs["A2_Index0"], 0, task, f"{task}__A2_sample_0")
        selected_idx = int(telemetry[task]["A2_syntax_selected_index"])
        verify("A2_SyntaxSelected", arm_dirs["A2_SyntaxSelected"], 0, task, f"{task}__A2_sample_{selected_idx}")
        verify("B", arm_dirs["B"], 0, task, f"{task}__B_merged")
        verify("C", arm_dirs["C"], 0, task, f"{task}__C_merged")
        for sample_idx in range(4):
            verify("A2_Pass4", arm_dirs["A2_Pass4"], sample_idx, task, f"{task}__A2_sample_{sample_idx}")

    return {
        "exported_task_candidates_checked": sum(sum(c.values()) for c in outcomes.values()),
        "by_arm": {arm: dict(counts) for arm, counts in outcomes.items()},
        "problems": problems,
    }


def checkpoint_audit(checkpoints_dir: Path) -> dict[str, Any]:
    files = sorted(checkpoints_dir.glob("*.json")) if checkpoints_dir.exists() else []
    models = Counter()
    by_arm_role: dict[str, Counter] = defaultdict(Counter)
    status = Counter()
    output_cap = Counter()
    gpu_nonzero = 0
    json_valid = Counter()
    parse_fallback = Counter()
    syntax_valid = Counter()
    fallback_pass = Counter()
    fallback_pass_empty_issue = Counter()
    unexpected_models_by_role: dict[str, Counter] = defaultdict(Counter)
    finish_reason_fields = 0
    records = 0
    unreadable_json_files = 0
    seen_call_ids: Counter = Counter()
    expected_models = {
        "Writer_Initial": "@cf/qwen/qwen3-30b-a3b-fp8",
        "Writer_Sample_0": "@cf/qwen/qwen3-30b-a3b-fp8",
        "Writer_Sample_1": "@cf/qwen/qwen3-30b-a3b-fp8",
        "Writer_Sample_2": "@cf/qwen/qwen3-30b-a3b-fp8",
        "Writer_Sample_3": "@cf/qwen/qwen3-30b-a3b-fp8",
        "Homogeneous_Functional_Reviewer": "@cf/qwen/qwen3-30b-a3b-fp8",
        "Homogeneous_Security_Auditor": "@cf/qwen/qwen3-30b-a3b-fp8",
        "Heterogeneous_Functional_Reviewer": "@cf/qwen/qwen3.8-27b",
        "Heterogeneous_Security_Auditor": "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
        "Writer_Merger_B": "@cf/qwen/qwen3-30b-a3b-fp8",
        "Writer_Merger_C": "@cf/qwen/qwen3-30b-a3b-fp8",
    }
    for path in files:
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            unreadable_json_files += 1
            continue
        if not isinstance(rec, dict) or "call_id" not in rec:
            continue
        records += 1
        model = str(rec.get("canonical_model") or "(missing)")
        models[model] += 1
        arm = str(rec.get("arm") or "?")
        role = str(rec.get("role") or "?")
        call_id = str(rec.get("call_id") or path.stem)
        seen_call_ids[call_id] += 1
        by_arm_role[f"{arm} / {role}"][model] += 1
        expected = expected_models.get(role)
        if expected is not None and model != expected:
            unexpected_models_by_role[role][model] += 1
        status[str(rec.get("status") or "missing")] += 1
        output_cap[str(rec.get("output_tokens"))] += 1
        if int(rec.get("output_tokens") or 0) == 3072:
            output_cap["at_3072_cap"] += 1
        if float(rec.get("gpu_vram_used_mb") or 0) > 0:
            gpu_nonzero += 1
        if "finish_reason" in rec or "stop_reason" in rec:
            finish_reason_fields += 1
        if rec.get("report_json_valid") is not None:
            json_valid[str(bool(rec["report_json_valid"]))] += 1
        parsed_report = rec.get("parsed_report")
        if isinstance(parsed_report, dict) and parsed_report.get("raw_parse_fallback"):
            parse_fallback[arm] += 1
            if parsed_report.get("verdict") == "PASS":
                fallback_pass[arm] += 1
                issues = parsed_report.get("issues") or []
                descriptions = [
                    str(issue.get("description") or "").strip()
                    for issue in issues
                    if isinstance(issue, dict)
                ]
                # Empty/missing issue arrays are also an absence of an issue
                # description and must not disappear from this quality metric.
                if not any(descriptions):
                    fallback_pass_empty_issue[arm] += 1
        if rec.get("syntax_valid") is not None:
            syntax_valid[str(bool(rec["syntax_valid"]))] += 1
    return {
        "checkpoint_files_found": len(files),
        "checkpoint_records_parsed": records,
        "unreadable_json_files": unreadable_json_files,
        "unique_call_ids": len(seen_call_ids),
        "duplicate_call_ids": {k: v for k, v in seen_call_ids.items() if v > 1},
        "model_id_counts": dict(models),
        "unexpected_model_ids_by_role_vs_declared_protocol": {k: dict(v) for k, v in sorted(unexpected_models_by_role.items())},
        "model_ids_by_arm_and_role": {k: dict(v) for k, v in sorted(by_arm_role.items())},
        "call_status_counts": dict(status),
        "output_token_counts": dict(output_cap),
        "calls_at_3072_output_token_cap": output_cap.get("at_3072_cap", 0),
        "gpu_vram_nonzero_records": gpu_nonzero,
        "review_json_validity_counts": dict(json_valid),
        "raw_review_parse_fallbacks_by_arm": dict(parse_fallback),
        "fallback_reviews_defaulted_to_pass_by_arm": dict(fallback_pass),
        "fallback_pass_with_empty_issue_description_by_arm": dict(fallback_pass_empty_issue),
        "code_syntax_validity_counts": dict(syntax_valid),
        "records_with_finish_or_stop_reason": finish_reason_fields,
        "records_without_finish_or_stop_reason": max(0, records - finish_reason_fields),
    }


def analyze(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    result_dir = repo_root / "final_paper_results"
    extracted = result_dir / "extracted_bundle"
    evals_root = extracted / "evals"
    telemetry_path = extracted / "generation_telemetry_25tasks.csv"
    if not telemetry_path.exists():
        telemetry_path = repo_root / "cweval_study" / "cweval_export" / "generation_telemetry_25tasks.csv"
    manifest_path = repo_root / "cweval_study" / "tasks_25_manifest.json"
    telemetry = read_telemetry(telemetry_path)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tasks = [x["task_id"] for x in manifest]
        if len(tasks) != len(set(tasks)):
            raise ValueError("Duplicate task IDs in task manifest")
        if set(tasks) != set(telemetry):
            raise ValueError("Task manifest and telemetry task IDs do not match")
    else:
        tasks = sorted(telemetry)
    n = len(tasks)
    if n != 25:
        raise ValueError(f"Expected 25 tasks for this protocol; found {n}")

    prompt_audit = audit_manifest_prompts(manifest_path)
    candidate_provenance = audit_candidate_provenance(repo_root, tasks, telemetry)

    raw_by_arm: dict[str, dict[str, dict[str, list[bool]]]] = {}
    input_hashes: dict[str, str] = {str(telemetry_path.relative_to(repo_root)): sha256(telemetry_path)}
    if manifest_path.exists():
        input_hashes[str(manifest_path.relative_to(repo_root))] = sha256(manifest_path)
    for arm, folder in SINGLE_ARMS.items():
        p = evals_root / folder / "res_all.json"
        raw_by_arm[arm] = read_res_all(p)
        input_hashes[str(p.relative_to(repo_root))] = sha256(p)
    a2_path = evals_root / A2_DIR / "res_all.json"
    raw_by_arm["A2_Pass4"] = read_res_all(a2_path)
    input_hashes[str(a2_path.relative_to(repo_root))] = sha256(a2_path)

    missing: dict[str, list[str]] = {}
    coverage: dict[str, int] = {}
    outcomes: dict[str, dict[str, list[int]]] = {
        arm: {m: [] for m in METRICS}
        for arm in (*SINGLE_ARMS.keys(), "A2_Pass4")
    }
    unified_rows: list[dict[str, Any]] = []
    a2_valid_dist = Counter()
    a2_evaluated_total = 0

    for arm in SINGLE_ARMS:
        missing[arm] = sorted(set(tasks) - set(raw_by_arm[arm]))
        coverage[arm] = n - len(missing[arm])

    for task in tasks:
        row: dict[str, Any] = {"task_id": task}
        tel = telemetry[task]
        # One-candidate arms: missing/unparseable outputs remain failures in N=25.
        for arm in SINGLE_ARMS:
            values, evaluated = single_outcome(raw_by_arm[arm].get(task))
            row[f"{arm}_evaluation_present"] = int(evaluated)
            for metric in METRICS:
                outcomes[arm][metric].append(values[metric])
                row[f"{arm}_{metric}"] = values[metric]
        # Four-attempt A2: CWEval arrays include syntactically/evaluably valid
        # candidates only; syntax-invalid attempts are retained as failures in
        # the fixed denominator of four. The task-level pass@4 is OR over all
        # successful results across up to four attempts.
        a2 = raw_by_arm["A2_Pass4"].get(task)
        expected_valid = int(tel["A2_valid_samples_count"])
        a2_lengths = len(a2["functional"]) if a2 else 0
        if a2_lengths != expected_valid:
            raise ValueError(
                f"A2 valid candidate count mismatch for {task}: "
                f"telemetry={expected_valid}, res_all={a2_lengths}"
            )
        if a2_lengths > 4 or expected_valid > 4:
            raise ValueError(f"A2 has more than four candidates for {task}")
        a2_valid_dist[str(expected_valid)] += 1
        a2_evaluated_total += a2_lengths
        for metric in METRICS:
            passes = int(any(a2[metric])) if a2 else 0
            outcomes["A2_Pass4"][metric].append(passes)
            row[f"A2_{metric}_pass_count"] = sum(a2[metric]) if a2 else 0
            row[f"A2_{metric}_pass_at_4"] = passes
        row["A2_attempts"] = 4
        row["A2_syntax_valid_candidates"] = expected_valid
        row["A2_syntax_invalid_attempts"] = 4 - expected_valid
        row["A2_tested_candidates"] = a2_lengths
        # Preserve source telemetry fields, including reviewer format validity.
        row.update(tel)
        unified_rows.append(row)

    # Single-candidate coverage must agree with syntax metadata where a final
    # candidate was reported syntactically invalid.
    unexplained_missing = {
        "A1": [t for t in missing["A1"] if int(telemetry[t]["A1_syntax_valid"]) == 1],
        "B": [t for t in missing["B"] if int(telemetry[t]["B_final_syntax_valid"]) == 1],
        "C": [t for t in missing["C"] if int(telemetry[t]["C_final_syntax_valid"]) == 1],
    }

    # Summary rows. Missing outputs count as failures in the 25-task denominator,
    # while evaluation coverage is reported separately.
    summary: list[dict[str, Any]] = []
    arm_alias = {
        "A1": "A1",
        "A2_Index0": "A2_Index0",
        "A2_SyntaxSelected": "A2_SyntaxSelected",
        "A2_Pass4": "A2_Pass4",
        "B": "B",
        "C": "C",
    }
    for arm in arm_alias:
        source_arm = "A2_Pass4" if arm == "A2_Pass4" else arm
        row: dict[str, Any] = {
            "arm": arm,
            "tasks_denominator": n,
            "evaluation_coverage_n": n if arm == "A2_Pass4" else coverage[arm],
            "missing_evaluation_n": 0 if arm == "A2_Pass4" else len(missing[arm]),
        }
        for metric in METRICS:
            count = sum(outcomes[source_arm][metric])
            row[f"{metric}_pass_n"] = count
            row[f"{metric}_rate_pct"] = round(100 * count / n, 2)
            ci = wilson_ci(count, n)
            row[f"{metric}_wilson95_low_pct"] = ci[0] if ci else None
            row[f"{metric}_wilson95_high_pct"] = ci[1] if ci else None
        summary.append(row)

    # Timing/token summaries, preserving the shared A1 draft accounting.
    def sum_col(name: str, cast: Any = float) -> float:
        return sum(cast(telemetry[t][name]) for t in tasks)
    accounting = {
        "A1": {"input_tokens": int(sum_col("A1_in_tokens", int)), "output_tokens": int(sum_col("A1_out_tokens", int)), "generation_seconds": round(sum_col("A1_time_sec"), 4)},
        "A2_four_attempts": {"input_tokens": int(sum_col("A2_total_in_tokens", int)), "output_tokens": int(sum_col("A2_total_out_tokens", int)), "generation_seconds": round(sum_col("A2_total_time_sec"), 4)},
        "B_end_to_end_including_shared_A1_draft": {"input_tokens": int(sum_col("B_total_in_tokens", int)), "output_tokens": int(sum_col("B_total_out_tokens", int)), "generation_seconds": round(sum_col("B_total_time_sec"), 4)},
        "C_end_to_end_including_shared_A1_draft": {"input_tokens": int(sum_col("C_total_in_tokens", int)), "output_tokens": int(sum_col("C_total_out_tokens", int)), "generation_seconds": round(sum_col("C_total_time_sec"), 4)},
    }
    for arm in ("B", "C"):
        accounting[f"{arm}_incremental_review_and_merge_only"] = {
            "input_tokens": accounting[f"{arm}_end_to_end_including_shared_A1_draft"]["input_tokens"] - accounting["A1"]["input_tokens"],
            "output_tokens": accounting[f"{arm}_end_to_end_including_shared_A1_draft"]["output_tokens"] - accounting["A1"]["output_tokens"],
            "generation_seconds": round(accounting[f"{arm}_end_to_end_including_shared_A1_draft"]["generation_seconds"] - accounting["A1"]["generation_seconds"], 4),
            "note": "Incremental overhead excludes the shared A1 draft; end-to-end cost above includes it.",
        }

    pairwise: dict[str, Any] = {}
    raw_p: dict[str, float] = {}
    for label, a_arm, b_arm in PAIRWISE:
        pairwise[label] = {}
        for metric in METRICS:
            stat = mcnemar_exact(outcomes[a_arm][metric], outcomes[b_arm][metric])
            pairwise[label][metric] = stat
            raw_p[f"{label}.{metric}"] = stat["exact_two_sided_p"]
    holm = holm_adjust(raw_p)

    omnibus: dict[str, Any] = {}
    for metric in METRICS:
        matrix = [
            [outcomes[arm][metric][i] for arm in OMNIBUS_ARMS]
            for i in range(n)
        ]
        omnibus[metric] = cochran_q(matrix, exact_permutation=True)

    omnibus_exact_p = {
        metric: result["exact_conditional_permutation_p"]
        for metric, result in omnibus.items()
        if "exact_conditional_permutation_p" in result
    }
    omnibus_holm = holm_adjust(omnibus_exact_p)
    combined_test_p = dict(raw_p)
    combined_test_p.update({f"Cochran_Q.{metric}": p for metric, p in omnibus_exact_p.items()})
    combined_holm_15 = holm_adjust(combined_test_p)

    telemetry_quality = {
        "A1_syntax_valid_n": sum(int(telemetry[t]["A1_syntax_valid"]) for t in tasks),
        "A2_attempts_total": 4 * n,
        "A2_syntax_valid_candidates_total": sum(int(telemetry[t]["A2_valid_samples_count"]) for t in tasks),
        "A2_syntax_invalid_attempts_total": 4 * n - sum(int(telemetry[t]["A2_valid_samples_count"]) for t in tasks),
        "B_functional_review_json_valid_n": sum(int(telemetry[t]["B_func_json_valid"]) for t in tasks),
        "B_security_review_json_valid_n": sum(int(telemetry[t]["B_sec_json_valid"]) for t in tasks),
        "B_final_syntax_valid_n": sum(int(telemetry[t]["B_final_syntax_valid"]) for t in tasks),
        "C_functional_review_json_valid_n": sum(int(telemetry[t]["C_func_json_valid"]) for t in tasks),
        "C_security_review_json_valid_n": sum(int(telemetry[t]["C_sec_json_valid"]) for t in tasks),
        "C_final_syntax_valid_n": sum(int(telemetry[t]["C_final_syntax_valid"]) for t in tasks),
    }
    checkpoints = checkpoint_audit(repo_root / "cweval_study" / "checkpoints")
    env_path = repo_root / "cweval_study" / "environment_manifest.json"
    stage_path = repo_root / "cweval_study" / "stage_load_metrics.json"
    environment_manifest = json.loads(env_path.read_text(encoding="utf-8")) if env_path.exists() else None
    stage_load_metrics = json.loads(stage_path.read_text(encoding="utf-8")) if stage_path.exists() else None
    for meta_path in (env_path, stage_path):
        if meta_path.exists():
            input_hashes[str(meta_path.relative_to(repo_root))] = sha256(meta_path)
    total_tokens = {
        arm: row["input_tokens"] + row["output_tokens"]
        for arm, row in {
            "A1": accounting["A1"],
            "A2_four_attempts": accounting["A2_four_attempts"],
            "B_end_to_end": accounting["B_end_to_end_including_shared_A1_draft"],
            "C_end_to_end": accounting["C_end_to_end_including_shared_A1_draft"],
        }.items()
    }
    baseline_seconds = accounting["A1"]["generation_seconds"]
    baseline_tokens = total_tokens["A1"]
    cost_ratios = {
        arm: {
            "total_token_ratio_vs_A1": round(tokens / baseline_tokens, 4) if baseline_tokens else None,
            "generation_time_ratio_vs_A1": round(seconds / baseline_seconds, 4) if baseline_seconds else None,
        }
        for arm, tokens, seconds in [
            ("A2_four_attempts", total_tokens["A2_four_attempts"], accounting["A2_four_attempts"]["generation_seconds"]),
            ("B_end_to_end", total_tokens["B_end_to_end"], accounting["B_end_to_end_including_shared_A1_draft"]["generation_seconds"]),
            ("C_end_to_end", total_tokens["C_end_to_end"], accounting["C_end_to_end_including_shared_A1_draft"]["generation_seconds"]),
        ]
    }
    report = {
        "analysis_version": "1.1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_tasks": n,
        "metric_definitions": {
            "functional": "CWEval functional oracle pass",
            "secure": "CWEval security oracle pass",
            "func_secure": "CWEval joint functional AND security pass",
            "A2_Pass4": "At least one passing result among four generation attempts; syntax-invalid attempts count as non-passing; denominator remains all 25 tasks.",
            "missing_results": "Counted as end-to-end failures in the 25-task denominator and reported separately as missing evaluation coverage.",
        },
        "input_sha256": input_hashes,
        "manifest_prompt_audit": prompt_audit,
        "candidate_export_provenance_audit": candidate_provenance,
        "arm_summaries": summary,
        "missing_evaluation_tasks": missing,
        "unexplained_missing_results_despite_syntax_validity": unexplained_missing,
        "a2_valid_syntax_sample_distribution": dict(a2_valid_dist),
        "a2_tasks_with_zero_syntax_valid_samples": a2_valid_dist.get("0", 0),
        "a2_evaluated_candidates_total": a2_evaluated_total,
        "telemetry_quality": telemetry_quality,
        "compute_accounting": accounting,
        "pairwise_mcnemar_exact": pairwise,
        "pairwise_holm_adjustment_across_12_tests": holm,
        "cochrans_q_holm_adjustment_across_3_outcomes": omnibus_holm,
        "holm_adjustment_across_all_15_reported_tests_sensitivity": combined_holm_15,
        "cochrans_q_four_single_output_arms": {"arms": list(OMNIBUS_ARMS), **omnibus},
        "compute_cost_ratios_vs_A1": cost_ratios,
        "environment_manifest": environment_manifest,
        "stage_load_metrics": stage_load_metrics,
        "checkpoint_audit": checkpoints,
        "known_interpretation_limits": [
            "This is a re-analysis of saved outputs; it does not rerun LLM generation or CWEval Docker tests.",
            "CWEval res_all arrays contain results for candidates that reached its test-result aggregation; telemetry is used to reconcile A2 syntax-valid counts.",
            "Provider-reported finish_reason/stop_reason is absent from the current checkpoint schema, so outputs capped at 3072 tokens cannot be conclusively classified as truncated.",
            "GPU utilization and dual-T4 GPU-hours are not inferred from generation wall time; zero measured GPU VRAM in checkpoints is reported as such.",
            "The two so-called manual security overrides are not applied: the pinned upstream CWEval runner collects marker-tagged tests and excludes names matching _unsafe; the candidate security cases are marked on the ordinary parametrized tests, while the *_unsafe_security helper functions are not candidate score oracles.",
            "Call-count matching (four logical model calls per A2/B/C task path, with a shared A1 draft in B/C) is not identical to compute matching: input/output token volumes, model roles, generated reasoning and wall-clock latency differ.",
            "No primary outcome is documented in the manuscript version inspected. Pairwise tests are Holm-adjusted across 12 tests; omnibus tests are also adjusted across the three outcomes, with a combined 15-test sensitivity family reported separately.",
            "The checkpoint audit can only verify exact model fallback counts if all checkpoint JSON files are present locally; canonical_model is the recorded model identifier actually placed in the request payload.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    # Write machine-readable report.
    (output_dir / "corrected_statistical_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    # Write task-level table.
    cols = list(unified_rows[0].keys())
    with (output_dir / "corrected_unified_task_results.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(unified_rows)
    # Write arm summary table.
    with (output_dir / "corrected_arm_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("final_paper_results_corrected"))
    args = parser.parse_args()
    report = analyze(args.repo_root.resolve(), args.output_dir.resolve())
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "n_tasks": report["n_tasks"],
        "arm_summaries": report["arm_summaries"],
        "missing_evaluation_tasks": report["missing_evaluation_tasks"],
        "cochrans_q": report["cochrans_q_four_single_output_arms"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()