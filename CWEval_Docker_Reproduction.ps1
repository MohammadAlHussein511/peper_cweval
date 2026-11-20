[CmdletBinding()]
param(
    [string]$RepoRoot = (Get-Location).Path,
    [int]$NumProc = 6
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path $RepoRoot).Path

$UpstreamRepo = 'https://github.com/Co1lin/CWEval.git'
$UpstreamCommit = 'e9a2a124c8c53679b6d8d27adfd2f6c40e7576d7'
$Image = 'co1lin/cweval@sha256:2d28261cc26838e20b69ce28a742b376a7c420b2f9591b32b95922e2794413d0'

$SourceEvalRoot = Join-Path $RepoRoot 'cweval_study\cweval_export\evals'
$WorkRoot = Join-Path $RepoRoot '_cweval_docker_reproduction'
$PinnedSource = Join-Path $WorkRoot 'CWEval_pinned_source'
$EvalRoot = Join-Path $WorkRoot 'evals'
$PayloadRoot = Join-Path $WorkRoot 'bundle_payload'
$ResultsRoot = Join-Path $PayloadRoot 'docker_results'
$LogsRoot = Join-Path $PayloadRoot 'logs'
$ZipPath = Join-Path $RepoRoot 'cweval_docker_reproduction_bundle.zip'

$Arms = [ordered]@{
    'A1'                  = 'arm_A1_baseline'
    'A2_Index0'           = 'arm_A2_Index0'
    'A2_SyntaxSelected'   = 'arm_A2_SyntaxSelected'
    'A2_Pass4'            = 'arm_A2_samples4'
    'B'                   = 'arm_B_homogeneous'
    'C'                   = 'arm_C_heterogeneous'
}

function Invoke-Checked {
    param([string]$Label, [scriptblock]$Command)
    Write-Host "`n===== $Label =====" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

foreach ($required in @(
    (Join-Path $SourceEvalRoot 'arm_A1_baseline'),
    (Join-Path $RepoRoot 'final_paper_results\extracted_bundle\evals'),
    (Join-Path $RepoRoot 'cweval_study\tasks_25_manifest.json')
)) {
    if (-not (Test-Path $required)) { throw "Required experiment input not found: $required" }
}

# Verify Docker Desktop's engine connection without changing any experiment data.
Invoke-Checked 'Docker engine availability' { docker version }

# Get the exact pinned upstream evaluator source in a separate audit directory.
New-Item -ItemType Directory -Force -Path $WorkRoot | Out-Null
if (-not (Test-Path (Join-Path $PinnedSource '.git'))) {
    if (Test-Path $PinnedSource) { Remove-Item -Recurse -Force $PinnedSource }
    Invoke-Checked 'Clone upstream CWEval source' { git clone $UpstreamRepo $PinnedSource }
}
Push-Location $PinnedSource
try {
    Invoke-Checked 'Fetch pinned upstream commit' { git fetch origin $UpstreamCommit }
    Invoke-Checked 'Checkout pinned upstream commit' { git checkout --detach $UpstreamCommit }
    $ActualCommit = (git rev-parse HEAD).Trim()
    if ($ActualCommit -ne $UpstreamCommit) {
        throw "Upstream source commit mismatch. Expected $UpstreamCommit; got $ActualCommit"
    }
} finally {
    Pop-Location
}

# Pull the exact image digest. The digest is passed to docker run directly below.
Invoke-Checked 'Pull pinned CWEval Docker image' { docker pull $Image }

# Prepare clean, isolated copies of the six candidate folders. The originals remain untouched.
if (Test-Path $EvalRoot) { Remove-Item -Recurse -Force $EvalRoot }
if (Test-Path $PayloadRoot) { Remove-Item -Recurse -Force $PayloadRoot }
New-Item -ItemType Directory -Force -Path $EvalRoot, $ResultsRoot, $LogsRoot | Out-Null

$ExpectedGeneratedCounts = @{
    'A1' = 1; 'A2_Index0' = 1; 'A2_SyntaxSelected' = 1; 'A2_Pass4' = 4; 'B' = 1; 'C' = 1
}

foreach ($entry in $Arms.GetEnumerator()) {
    $arm = $entry.Key
    $sourceFolderName = $entry.Value
    $source = Join-Path $SourceEvalRoot $sourceFolderName
    $targetName = "audit_$arm"
    $target = Join-Path $EvalRoot $targetName

    if (-not (Test-Path $source)) { throw "Candidate export folder missing: $source" }
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    Copy-Item -Path (Join-Path $source '*') -Destination $target -Recurse -Force

    $generated = @(Get-ChildItem -Path $target -Directory -Filter 'generated_*')
    if ($generated.Count -ne $ExpectedGeneratedCounts[$arm]) {
        throw "$arm has $($generated.Count) generated folders; expected $($ExpectedGeneratedCounts[$arm])."
    }
}