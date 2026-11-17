# CWEval research audit tools

This directory contains a schema-aware re-analysis utility, regression tests, and the provisional outputs/status report for the 25-task CWEval study.

## Scope and safety

- Reads the saved CWEval `res_all.json` arrays, telemetry, manifest, and checkpoint files.
- Does **not** call an LLM, regenerate candidates, rerun Docker, alter the source results, or edit the manuscript.
- Writes derived outputs only to the requested output directory.
- Uses a fixed task denominator of 25 and reports missing evaluation coverage separately.
- Treats this as a re-analysis of saved evaluator outputs until a pinned Docker reproduction is completed.

## Run from the repository root

```bash
python research_audit/cweval_corrected_analysis_v2.py \
  --repo-root . \
  --output-dir final_paper_results_corrected
```

Expected inputs in the repository:

```text
final_paper_results/extracted_bundle/evals/<arm>/res_all.json
final_paper_results/extracted_bundle/generation_telemetry_25tasks.csv
cweval_study/tasks_25_manifest.json
cweval_study/cweval_export/evals/<arm>/generated_*/core/py/*.py
cweval_study/checkpoints/*.json
```

Outputs are created under `final_paper_results_corrected/` (or the path supplied with `--output-dir`):

- `corrected_statistical_report.json`
- `corrected_unified_task_results.csv`
- `corrected_arm_summary.csv`

## Run tests

Requires Python 3.10+ and `pytest` for the command below. The analysis script itself uses only the Python standard library.

```bash
python -m pytest -q research_audit/test_cweval_corrected_analysis_v2.py
```

Alternatively, run with the standard library test runner:

```bash
python -m unittest discover -s research_audit -p 'test_*.py' -v
```

## Interpretation limits

- Passing the local tests verifies parser/statistics behavior and synthetic fixtures; it is not equivalent to executing the official CWEval Docker evaluator.
- The full prompt-hash, candidate-to-checkpoint provenance, and 275-call checkpoint audits must be run against a complete local checkout before finalizing the study.
- Equal logical call counts do not establish compute matching. Token use, model identity, and elapsed inference time differ.
- No manuscript edits or branch merges are part of this toolset.