# CWEval Docker reproduction runner — v2 (manifest parsing hardened)
[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$RepoRoot = (Get-Location).Path,

    [Parameter(Mandatory = $false)]
    [ValidateRange(1, 24)]
    [int]$NumProc = 6
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path

# Immutable reproduction identifiers.
$UpstreamCommit = 'e9a2a124c8c53679b6d8d27adfd2f6c40e7576d7'
$Image = 'co1lin/cweval@sha256:2d28261cc26838e20b69ce28a742b376a7c420b2f9591b32b95922e2794413d0'
$PinnedSource = Join-Path $RepoRoot '_cweval_docker_reproduction_v2\CWEval_source_pinned'

# Original inputs are read-only. All evaluator outputs go into a new run directory.
$SourceEvalRoot = Join-Path $RepoRoot 'cweval_study\cweval_export\evals'
$SavedResultsRoot = Join-Path $RepoRoot 'final_paper_results\extracted_bundle\evals'
$ManifestPath = Join-Path $RepoRoot 'cweval_study\tasks_25_manifest.json'
$AnalyzerPath = Join-Path $RepoRoot 'research_audit\cweval_corrected_analysis_v2.py'
$CheckpointRoot = Join-Path $RepoRoot 'cweval_study\checkpoints'
$TelemetryCandidates = @(
    (Join-Path $RepoRoot 'final_paper_results\extracted_bundle\generation_telemetry_25tasks.csv'),
    (Join-Path $RepoRoot 'cweval_study\cweval_export\generation_telemetry_25tasks.csv')
)
$TelemetryPath = $TelemetryCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1

$RunId = Get-Date -Format 'yyyyMMdd_HHmmss'
$RunRoot = Join-Path $RepoRoot (Join-Path '_cweval_docker_reproduction_runs' "run_$RunId")
$EvalRoot = Join-Path $RunRoot 'evals'
$LogsRoot = Join-Path $RunRoot 'logs'
$AnalysisInputRoot = Join-Path $RunRoot 'analysis_input'
$AnalysisOutputRoot = Join-Path $RunRoot 'analysis_output'
$CompareScriptPath = Join-Path $RunRoot 'compare_results.py'
$CompareReportPath = Join-Path $RunRoot 'result_comparison.json'
$MetadataPath = Join-Path $RunRoot 'reproduction_metadata.json'
$ZipPath = Join-Path $RepoRoot "cweval_docker_reproduction_$RunId.zip"

$Arms = @(
    [pscustomobject]@{ Key = 'A1'; DockerFolder = 'audit_A1'; SourceFolder = 'arm_A1_baseline'; SavedFolder = 'arm_A1_baseline'; Generated = 1 },
    [pscustomobject]@{ Key = 'A2_Index0'; DockerFolder = 'audit_A2_Index0'; SourceFolder = 'arm_A2_Index0'; SavedFolder = 'arm_A2_Index0'; Generated = 1 },
    [pscustomobject]@{ Key = 'A2_SyntaxSelected'; DockerFolder = 'audit_A2_SyntaxSelected'; SourceFolder = 'arm_A2_SyntaxSelected'; SavedFolder = 'arm_A2_SyntaxSelected'; Generated = 1 },
    [pscustomobject]@{ Key = 'A2_Pass4'; DockerFolder = 'audit_A2_Pass4'; SourceFolder = 'arm_A2_samples4'; SavedFolder = 'arm_A2_samples4'; Generated = 4 },
    [pscustomobject]@{ Key = 'B'; DockerFolder = 'audit_B'; SourceFolder = 'arm_B_homogeneous'; SavedFolder = 'arm_B_homogeneous'; Generated = 1 },
    [pscustomobject]@{ Key = 'C'; DockerFolder = 'audit_C'; SourceFolder = 'arm_C_heterogeneous'; SavedFolder = 'arm_C_heterogeneous'; Generated = 1 }
)

function Assert-Path([string]$Path, [string]$Label) {
    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path -LiteralPath $Path)) {
        throw "Required $Label not found: $Path"
    }
}

function Invoke-NativeChecked {
    param([string]$Label, [scriptblock]$Command)
    Write-Host "`n===== $Label =====" -ForegroundColor Cyan
    & $Command
    $nativeCode = $LASTEXITCODE
    if ($null -ne $nativeCode -and $nativeCode -ne 0) {
        throw "$Label failed with exit code $nativeCode"
    }
}

Write-Host '===== PREFLIGHT: ORIGINAL INPUTS =====' -ForegroundColor Cyan
Assert-Path $SourceEvalRoot 'exported candidate evaluation root'
Assert-Path $SavedResultsRoot 'saved evaluator results root'
Assert-Path $ManifestPath '25-task manifest'
Assert-Path $AnalyzerPath 'corrected analysis script'
Assert-Path $CheckpointRoot 'generation checkpoints folder'
if (-not $TelemetryPath) { throw 'Could not locate generation_telemetry_25tasks.csv in either expected location.' }

# Confirm the local upstream checkout is pinned correctly; do not modify it.
Assert-Path (Join-Path $PinnedSource '.git') 'pinned upstream Git checkout'
$LocalPinnedCommit = (& git -C $PinnedSource rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $LocalPinnedCommit -ne $UpstreamCommit) {
    throw "Local pinned source mismatch. Expected $UpstreamCommit; found $LocalPinnedCommit"
}

# Parse the manifest with Python's standard JSON parser instead of PowerShell's
# ConvertFrom-Json pipeline enumeration, which can collapse the top-level JSON
# array to a single pipeline object in some Windows PowerShell environments.
$ManifestProbe = @'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("r", encoding="utf-8-sig") as handle:
    items = json.load(handle)
if not isinstance(items, list):
    raise SystemExit(f"Manifest root must be a JSON array; found {type(items).__name__}")
ids = []
for index, item in enumerate(items):
    if not isinstance(item, dict):
        raise SystemExit(f"Manifest entry {index} must be an object")
    task_id = item.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise SystemExit(f"Manifest entry {index} has no valid task_id")
    ids.append(task_id.strip())
if len(ids) != 25 or len(set(ids)) != 25:
    raise SystemExit(f"Expected exactly 25 unique task_id values; found {len(ids)} entries and {len(set(ids))} unique IDs")
for task_id in ids:
    print(task_id)
'@
$ManifestProbePath = Join-Path $env:TEMP "cweval_manifest_probe_$RunId.py"
[System.IO.File]::WriteAllText($ManifestProbePath, $ManifestProbe, [System.Text.UTF8Encoding]::new($false))
try {
    $TaskIds = @(& python $ManifestProbePath $ManifestPath)
    $ManifestProbeExitCode = $LASTEXITCODE
    if ($ManifestProbeExitCode -ne 0) {
        throw "Python could not validate the task manifest (exit code $ManifestProbeExitCode). Check the manifest file and Python output above."
    }
} finally {
    if (Test-Path -LiteralPath $ManifestProbePath) {
        Remove-Item -LiteralPath $ManifestProbePath -Force
    }
}
if ($TaskIds.Count -ne 25) {
    throw "Manifest parser returned $($TaskIds.Count) task IDs instead of 25."
}
$UniqueTaskCount = @($TaskIds | Select-Object -Unique).Count
if ($UniqueTaskCount -ne 25) {
    throw "Manifest must contain exactly 25 unique task_id entries; found $UniqueTaskCount unique IDs."
}
$TaskIdSet = [System.Collections.Generic.HashSet[string]]::new([string[]]$TaskIds)
Write-Host "Manifest validation passed: $($TaskIds.Count) entries; $UniqueTaskCount unique task IDs." -ForegroundColor Green

foreach ($arm in $Arms) {
    $source = Join-Path $SourceEvalRoot $arm.SourceFolder
    $saved = Join-Path $SavedResultsRoot (Join-Path $arm.SavedFolder 'res_all.json')
    Assert-Path $source "source folder for $($arm.Key)"
    Assert-Path $saved "saved res_all.json for $($arm.Key)"

    $generatedFolders = @(Get-ChildItem -LiteralPath $source -Directory | Where-Object { $_.Name -match '^generated_\d+$' })
    if ($generatedFolders.Count -ne $arm.Generated) {
        throw "$($arm.Key): expected $($arm.Generated) generated_* folders, found $($generatedFolders.Count)."
    }
    foreach ($generated in $generatedFolders) {
        $rawFiles = @(Get-ChildItem -LiteralPath $generated.FullName -Recurse -File -Filter '*_raw.py')
        $rawIds = @($rawFiles | ForEach-Object { [System.IO.Path]::GetFileNameWithoutExtension($_.Name) -replace '_raw$', '' })
        if ($rawFiles.Count -ne 25 -or @($rawIds | Select-Object -Unique).Count -ne 25) {
            throw "$($arm.Key)/$($generated.Name): expected 25 unique *_raw.py candidates; found $($rawFiles.Count)."
        }
        foreach ($taskId in $rawIds) {
            if (-not $TaskIdSet.Contains($taskId)) { throw "Unexpected candidate task '$taskId' in $($arm.Key)/$($generated.Name)." }
        }
        $missingRawIds = @($TaskIds | Where-Object { $_ -notin $rawIds })
        if ($missingRawIds.Count -gt 0) {
            throw "$($arm.Key)/$($generated.Name): missing raw candidates: $($missingRawIds -join ', ')"
        }
    }
}

Write-Host 'Preflight passed: 6 arms; 25 tasks; 1/1/1/4/1/1 generated folders.' -ForegroundColor Green
Write-Host "Pinned source: $LocalPinnedCommit"
Write-Host "Docker image: $Image"

Invoke-NativeChecked 'Docker engine' { docker version }
Invoke-NativeChecked 'Pull immutable Docker image' { docker pull $Image }

# Create a new run directory. Existing runs are never deleted or overwritten.
if (Test-Path -LiteralPath $RunRoot) { throw "Run directory already exists; rerun after the timestamp changes: $RunRoot" }
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $RunRoot) | Out-Null
New-Item -ItemType Directory -Force -Path $EvalRoot, $LogsRoot | Out-Null
if ($PSCommandPath) { Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $RunRoot 'CWEval_Docker_Reproduction_ALL_v1.ps1') -Force }

# Prepare disposable copies. The original exported candidates and saved res_all files are untouched.
foreach ($arm in $Arms) {
    $source = Join-Path $SourceEvalRoot $arm.SourceFolder
    $target = Join-Path $EvalRoot $arm.DockerFolder
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    Copy-Item -Path (Join-Path $source '*') -Destination $target -Recurse -Force

    Get-ChildItem -LiteralPath $target -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -in @('res.json', 'res_all.json', 'run_tests.log', 'pass_at_k_report.txt') -or $_.Extension -eq '.pyc' } |
        Remove-Item -Force
    Get-ChildItem -LiteralPath $target -Recurse -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -in @('__pycache__', '.pytest_cache') } |
        Sort-Object { $_.FullName.Length } -Descending |
        Remove-Item -Recurse -Force
}

# Each arm gets a fresh container, following the pinned runner's isolated-container approach.
foreach ($arm in $Arms) {
    $containerName = "cweval-audit-$RunId-$($arm.Key)"
    $containerLog = Join-Path $LogsRoot "$($arm.Key)_docker.log"
    $environmentLog = Join-Path $LogsRoot "$($arm.Key)_environment.log"
    $containerStarted = $false
    $hostArmPath = Join-Path $EvalRoot $arm.DockerFolder

    try {
        Write-Host "`n===== ISOLATED DOCKER EVALUATION: $($arm.Key) =====" -ForegroundColor Cyan
        $StartArgs = @('run', '-d', '--name', $containerName, '--entrypoint', 'bash', $Image, '-lc', 'sleep infinity')
        Invoke-NativeChecked "Start isolated container for $($arm.Key)" { & docker @StartArgs | Out-Null }
        $containerStarted = $true

        $MkdirArgs = @('exec', '--user', 'root', $containerName, 'mkdir', '-p', '/home/ubuntu/CWEval/evals')
        Invoke-NativeChecked "Create evaluator input directory for $($arm.Key)" { & docker @MkdirArgs | Out-Null }

        $CopyInArgs = @('cp', $hostArmPath, "${containerName}:/home/ubuntu/CWEval/evals")
        Invoke-NativeChecked "Copy isolated candidates into container for $($arm.Key)" { & docker @CopyInArgs | Out-Null }

        $ChownArgs = @('exec', '--user', 'root', $containerName, 'chown', '-R', 'ubuntu:ubuntu', "/home/ubuntu/CWEval/evals/$($arm.DockerFolder)")
        Invoke-NativeChecked "Set safe evaluator ownership for $($arm.Key)" { & docker @ChownArgs | Out-Null }

        $CommitArgs = @('exec', '--user', 'ubuntu', $containerName, 'git', '-C', '/home/ubuntu/CWEval', 'rev-parse', 'HEAD')
        $ContainerCommit = (& docker @CommitArgs 2>&1 | Out-String).Trim()
        $CommitExitCode = $LASTEXITCODE
        if ($CommitExitCode -ne 0 -or $ContainerCommit -ne $UpstreamCommit) {
            throw "$($arm.Key): container source commit mismatch. Expected $UpstreamCommit; received $ContainerCommit"
        }

        $VersionArgs = @('exec', '--user', 'ubuntu', $containerName, 'bash', '-lc', 'cd /home/ubuntu/CWEval && source /home/ubuntu/miniforge3/bin/activate && source .env && python --version')
        & docker @VersionArgs 2>&1 | Tee-Object -FilePath $environmentLog
        $VersionExitCode = $LASTEXITCODE
        if ($VersionExitCode -ne 0) { throw "$($arm.Key): failed to verify Python environment inside Docker." }

        $EvalCommand = "cd /home/ubuntu/CWEval && source /home/ubuntu/miniforge3/bin/activate && source .env && python cweval/evaluate.py pipeline --eval_path evals/$($arm.DockerFolder) --num_proc $NumProc --docker False"
        $EvalArgs = @('exec', '--user', 'ubuntu', '--workdir', '/home/ubuntu/CWEval', $containerName, 'bash', '-lc', $EvalCommand)
        & docker @EvalArgs *> $containerLog
        $EvalExitCode = $LASTEXITCODE
        if ($EvalExitCode -ne 0) { throw "$($arm.Key): official CWEval pipeline failed with exit code $EvalExitCode. See $containerLog" }

        $ContainerResultsPath = "${containerName}:/home/ubuntu/CWEval/evals/$($arm.DockerFolder)/."
        $CopyOutArgs = @('cp', $ContainerResultsPath, $hostArmPath)
        Invoke-NativeChecked "Copy fresh evaluator outputs back for $($arm.Key)" { & docker @CopyOutArgs | Out-Null }
        Assert-Path (Join-Path $hostArmPath 'res_all.json') "fresh Docker res_all.json for $($arm.Key)"

    } finally {
        if ($containerStarted) {
            & docker rm -f $containerName 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) { Write-Warning "Could not remove temporary container $containerName" }
        }
    }
}
Set-Content -LiteralPath (Join-Path $RunRoot 'docker_completed.flag') -Value 'ALL_SIX_ARMS_COMPLETED' -Encoding ASCII

# Compare all new Docker res_all arrays with saved arrays by task ID, not JSON key path.
$ComparePy = @'
import argparse, json, re
from pathlib import Path
MAP = {
 "A1": ("audit_A1", "arm_A1_baseline"),
 "A2_Index0": ("audit_A2_Index0", "arm_A2_Index0"),
 "A2_SyntaxSelected": ("audit_A2_SyntaxSelected", "arm_A2_SyntaxSelected"),
 "A2_Pass4": ("audit_A2_Pass4", "arm_A2_samples4"),
 "B": ("audit_B", "arm_B_homogeneous"),
 "C": ("audit_C", "arm_C_heterogeneous"),
}
METRICS = ("functional", "secure", "func_secure")
TASK_RE = re.compile(r"(cwe_\d+_\d+)_test\.py$")
def read(path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict): raise ValueError(f"Not an object: {path}")
    out = {}
    for key, entry in raw.items():
        m = TASK_RE.search(key)
        if not m: raise ValueError(f"Cannot resolve task id: {key} in {path}")
        task = m.group(1)
        if task in out: raise ValueError(f"Duplicate task {task} in {path}")
        if not isinstance(entry, dict): raise ValueError(f"Invalid entry {task} in {path}")
        values = {metric: entry.get(metric) for metric in METRICS}
        if any(not isinstance(v, list) or any(type(x) is not bool for x in v) for v in values.values()):
            raise ValueError(f"Invalid bool arrays for {task} in {path}")
        if len({len(v) for v in values.values()}) != 1: raise ValueError(f"Array length mismatch for {task}")
        out[task] = values
    return out
p = argparse.ArgumentParser()
p.add_argument("--run-root", type=Path, required=True)
p.add_argument("--saved-root", type=Path, required=True)
p.add_argument("--manifest", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
tasks = [x["task_id"] for x in json.loads(a.manifest.read_text(encoding="utf-8-sig"))]
if len(tasks) != 25 or len(set(tasks)) != 25: raise SystemExit("Manifest must contain exactly 25 unique tasks")
report = {"comparison_basis": "task-ID normalized res_all arrays; original files are read-only", "n_manifest_tasks": 25, "arms": {}}
for arm, (newdir, olddir) in MAP.items():
    new = read(a.run_root / "evals" / newdir / "res_all.json")
    old = read(a.saved_root / olddir / "res_all.json")
    unexpected = sorted((set(new) | set(old)) - set(tasks))
    rows = {"new_task_entries": len(new), "saved_task_entries": len(old), "exact_match_on_overlap": True,
            "recovered_missing_tasks": sorted(set(new) - set(old)), "newly_missing_tasks": sorted(set(old) - set(new)),
            "mismatches": [], "unexpected_task_ids": unexpected}
    for task in sorted(set(new) & set(old)):
        if new[task] != old[task]:
            rows["exact_match_on_overlap"] = False
            rows["mismatches"].append({"task_id": task, "saved": old[task], "docker": new[task]})
    rows["full_exact_match"] = (set(new) == set(old) and rows["exact_match_on_overlap"] and not unexpected)
    report["arms"][arm] = rows
    print(f"{arm}: Docker={len(new)}/25, saved={len(old)}/25, overlap_exact={rows['exact_match_on_overlap']}, recovered={len(rows['recovered_missing_tasks'])}, newly_missing={len(rows['newly_missing_tasks'])}, mismatches={len(rows['mismatches'])}")
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Comparison report: {a.output}")
'@
[System.IO.File]::WriteAllText($CompareScriptPath, $ComparePy, [System.Text.UTF8Encoding]::new($false))

Write-Host "`n===== COMPARING DOCKER OUTPUTS WITH SAVED RESULTS =====" -ForegroundColor Cyan
& python $CompareScriptPath --run-root $RunRoot --saved-root $SavedResultsRoot --manifest $ManifestPath --output $CompareReportPath
$CompareExitCode = $LASTEXITCODE
if ($CompareExitCode -ne 0) { throw "Result comparison failed with exit code $CompareExitCode" }

# Build an isolated analysis workspace so the corrected analyzer consumes the new Docker outputs.
$ExtractedRoot = Join-Path $AnalysisInputRoot 'final_paper_results\extracted_bundle'
$AnalysisEvalsRoot = Join-Path $ExtractedRoot 'evals'
$AnalysisStudyRoot = Join-Path $AnalysisInputRoot 'cweval_study'
$AnalysisCandidateRoot = Join-Path $AnalysisStudyRoot 'cweval_export\evals'
$AnalysisCheckpointRoot = Join-Path $AnalysisStudyRoot 'checkpoints'
New-Item -ItemType Directory -Force -Path $AnalysisEvalsRoot, $AnalysisCandidateRoot, $AnalysisCheckpointRoot | Out-Null
Copy-Item -LiteralPath $ManifestPath -Destination (Join-Path $AnalysisStudyRoot 'tasks_25_manifest.json') -Force
Copy-Item -LiteralPath $TelemetryPath -Destination (Join-Path $ExtractedRoot 'generation_telemetry_25tasks.csv') -Force
Copy-Item -Path (Join-Path $CheckpointRoot '*.json') -Destination $AnalysisCheckpointRoot -Force
foreach ($optionalName in @('environment_manifest.json', 'stage_load_metrics.json')) {
    $optionalPath = Join-Path (Join-Path $RepoRoot 'cweval_study') $optionalName
    if (Test-Path -LiteralPath $optionalPath) { Copy-Item -LiteralPath $optionalPath -Destination (Join-Path $AnalysisStudyRoot $optionalName) -Force }
}
foreach ($arm in $Arms) {
    $analysisSavedDir = Join-Path $AnalysisEvalsRoot $arm.SavedFolder
    New-Item -ItemType Directory -Force -Path $analysisSavedDir | Out-Null
    Copy-Item -LiteralPath (Join-Path (Join-Path $EvalRoot $arm.DockerFolder) 'res_all.json') -Destination (Join-Path $analysisSavedDir 'res_all.json') -Force

    $sourceArm = Join-Path $SourceEvalRoot $arm.SourceFolder
    $destArm = Join-Path $AnalysisCandidateRoot $arm.SourceFolder
    foreach ($generated in (Get-ChildItem -LiteralPath $sourceArm -Directory | Where-Object { $_.Name -match '^generated_\d+$' })) {
        $destPy = Join-Path (Join-Path $destArm $generated.Name) 'core\py'
        New-Item -ItemType Directory -Force -Path $destPy | Out-Null
        Get-ChildItem -LiteralPath $generated.FullName -Recurse -File -Filter '*_task.py' |
            Copy-Item -Destination $destPy -Force
    }
}

Write-Host "`n===== RUNNING CORRECTED ANALYSIS ON DOCKER-REPRODUCED RESULTS =====" -ForegroundColor Cyan
$AnalysisLog = Join-Path $LogsRoot 'corrected_analysis.log'
& python $AnalyzerPath --repo-root $AnalysisInputRoot --output-dir $AnalysisOutputRoot 2>&1 | Tee-Object -FilePath $AnalysisLog
$AnalysisExitCode = $LASTEXITCODE
if ($AnalysisExitCode -ne 0) {
    throw "Corrected analysis rejected the Docker results (exit code $AnalysisExitCode). Inspect $AnalysisLog; do not treat the run as final."
}

$Metadata = [ordered]@{
    run_id = $RunId
    completed_local = (Get-Date).ToString('o')
    upstream_repository = 'https://github.com/Co1lin/CWEval'
    upstream_commit = $UpstreamCommit
    docker_image_digest = $Image
    num_proc = $NumProc
    arms = @($Arms | ForEach-Object { [ordered]@{ arm = $_.Key; source_folder = $_.SourceFolder; docker_folder = $_.DockerFolder; generated_folders = $_.Generated } })
    original_inputs_modified = $false
    analysis_basis = 'fresh Docker reproduction res_all.json outputs'
}
$Metadata | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $MetadataPath -Encoding UTF8

$SummaryCsv = Join-Path $AnalysisOutputRoot 'corrected_arm_summary.csv'
$StatsJson = Join-Path $AnalysisOutputRoot 'corrected_statistical_report.json'
Assert-Path $SummaryCsv 'Docker-based corrected arm summary'
Assert-Path $StatsJson 'Docker-based corrected statistical report'
if (Test-Path -LiteralPath $ZipPath) { throw "Refusing to overwrite existing bundle: $ZipPath" }
Compress-Archive -Path (Join-Path $RunRoot '*') -DestinationPath $ZipPath -CompressionLevel Optimal

Write-Host "`n===== REPRODUCTION FINISHED =====" -ForegroundColor Green
Write-Host "Run folder: $RunRoot"
Write-Host "Comparison: $CompareReportPath"
Write-Host "Docker-based analysis: $AnalysisOutputRoot"
Write-Host "Bundle: $ZipPath"
Write-Host "`n===== CORRECTED ARM SUMMARY =====" -ForegroundColor Cyan
Get-Content -LiteralPath $SummaryCsv
Write-Host "`nReview result_comparison.json: any recovered tasks or mismatches must be resolved and described before final manuscript claims are approved." -ForegroundColor Yellow
