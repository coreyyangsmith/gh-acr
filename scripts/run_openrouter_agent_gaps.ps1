<#
.SYNOPSIS
  Re-run agent coverage gaps via OpenRouter for base_a, base_b, and agent.

.DESCRIPTION
  Uses subset CSVs under data/agent_coverage_gaps/subsets/ built from the
  needs_reprocess gap lists. Defaults to methods base_a, base_b, agent.

  Requires OPENROUTER_API_KEY (shell or .env). GITHUB_TOKEN recommended.

.EXAMPLE
  # Build subsets first (once)
  .\.venv\Scripts\python.exe scripts\build_agent_gap_subsets.py

  # GPT-5-nano gaps
  .\scripts\run_openrouter_agent_gaps.ps1 -Model gpt-5-nano

  # Llama gaps, resume-friendly
  .\scripts\run_openrouter_agent_gaps.ps1 -Model llama-3.1-8b -Resume

  # Qwen gaps with lower concurrency
  .\scripts\run_openrouter_agent_gaps.ps1 -Model qwen3-32b -Concurrency 2
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("gpt-5-nano", "llama-3.1-8b", "qwen3-32b")]
    [string]$Model,

    [int]$Concurrency = 4,
    [int]$MethodConcurrency = 3,
    [string[]]$Methods = @("base_a", "base_b", "agent"),
    [switch]$Resume,
    [switch]$SkipBuildSubset,
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$ModelMap = @{
    "gpt-5-nano"  = "openrouter/openai/gpt-5-nano"
    "llama-3.1-8b" = "openrouter/meta-llama/llama-3.1-8b-instruct"
    "qwen3-32b"   = "openrouter/qwen/qwen3-32b"
}

$ModelName = $ModelMap[$Model]
$SubsetCsv = Join-Path $RepoRoot "data\agent_coverage_gaps\subsets\${Model}_agent_gaps_needs_reprocess.csv"
$DateStamp = Get-Date -Format "yyyy_MM_dd"
$ResultsFilename = "${DateStamp}_openrouter_${Model}_agent_gaps_base_agent.csv"

if (-not $Python) {
    $venvPy = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPy) {
        $Python = $venvPy
    }
    elseif (Get-Command uv -ErrorAction SilentlyContinue) {
        $Python = "uv"
    }
    else {
        $Python = "python"
    }
}

if (-not $SkipBuildSubset) {
    Write-Host "Building gap subset CSVs..."
    if ($Python -eq "uv") {
        & uv run python scripts\build_agent_gap_subsets.py
    }
    else {
        & $Python scripts\build_agent_gap_subsets.py
    }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not (Test-Path $SubsetCsv)) {
    throw "Subset CSV not found: $SubsetCsv (run scripts/build_agent_gap_subsets.py)"
}

$scenarioCount = (Import-Csv $SubsetCsv).Count
if (-not $env:OPENROUTER_API_KEY -or [string]::IsNullOrWhiteSpace($env:OPENROUTER_API_KEY)) {
    Write-Warning "OPENROUTER_API_KEY is not set in the shell. Ensure it is present in .env or the run will fail."
}
if (-not $env:GITHUB_TOKEN -or [string]::IsNullOrWhiteSpace($env:GITHUB_TOKEN)) {
    Write-Warning "GITHUB_TOKEN is not set; GitHub clone/fetch may be slower or rate-limited."
}

$env:DATASET_CSV = $SubsetCsv

Write-Host "============================================================"
Write-Host " GH-ACR OpenRouter agent-gap fill"
Write-Host " Model label: $Model"
Write-Host " Model name:  $ModelName"
Write-Host " Methods:     $($Methods -join ', ')"
Write-Host " Scenarios:   $scenarioCount  ($SubsetCsv)"
Write-Host " Concurrency: $Concurrency  method_concurrency=$MethodConcurrency"
Write-Host " Resume:      $Resume"
Write-Host " Results:     data/$ResultsFilename"
Write-Host " Started:     $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "============================================================"

$sw = [System.Diagnostics.Stopwatch]::StartNew()

$cliArgs = @(
    "-m", "src.cli.run_all",
    "--methods"
) + $Methods + @(
    "--mode", "clone",
    "--model-name", $ModelName,
    "--concurrency", "$Concurrency",
    "--method-concurrency", "$MethodConcurrency",
    "--results-filename", $ResultsFilename
)

if ($Resume) {
    $cliArgs += "--resume"
}

if ($Python -eq "uv") {
    & uv run python @cliArgs
}
else {
    & $Python @cliArgs
}

$exitCode = $LASTEXITCODE
$sw.Stop()

Write-Host "============================================================"
Write-Host " Finished:    $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host (" Wall clock:  {0:N1}s ({1})" -f $sw.Elapsed.TotalSeconds, $sw.Elapsed.ToString())
Write-Host " Exit code:   $exitCode"
Write-Host " Results CSV: data/$ResultsFilename"
Write-Host "============================================================"

exit $exitCode
