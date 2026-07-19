<#
.SYNOPSIS
  Example inference run: OpenRouter Llama 3.1 8B Instruct on 1% of the benchmark.

.DESCRIPTION
  Requires OPENROUTER_API_KEY in the environment (or in a root .env loaded by src.startup).
  GITHUB_TOKEN is recommended for faster/more reliable clones.

.EXAMPLE
  .\scripts\run_openrouter_llama_1pct.ps1

.EXAMPLE
  .\scripts\run_openrouter_llama_1pct.ps1 -Concurrency 2 -SamplePercent 1
#>
param(
    [int]$Concurrency = 4,
    [int]$SamplePercent = 1,
    [int]$SampleSeed = 42,
    [string[]]$Methods = @("agent", "bypass7"),
    [string]$ModelName = "openrouter/meta-llama/llama-3.1-8b-instruct",
    [string]$ResultsFilename = "2026_07_17_openrouter_llama31_8b_1pct.csv"
)

$ErrorActionPreference = "Stop"

# Resolve repo root (script lives in scripts/)
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not $env:OPENROUTER_API_KEY -or [string]::IsNullOrWhiteSpace($env:OPENROUTER_API_KEY)) {
    # .env may still supply the key via src.startup; warn only
    Write-Warning "OPENROUTER_API_KEY is not set in the shell. Ensure it is present in .env or the run will fail."
}
if (-not $env:GITHUB_TOKEN -or [string]::IsNullOrWhiteSpace($env:GITHUB_TOKEN)) {
    Write-Warning "GITHUB_TOKEN is not set; GitHub clone/fetch may be slower or rate-limited."
}

Write-Host "============================================================"
Write-Host " GH-ACR OpenRouter Llama example (sample-percent=$SamplePercent)"
Write-Host " Model:       $ModelName"
Write-Host " Methods:     $($Methods -join ', ')"
Write-Host " Concurrency: $Concurrency"
Write-Host " Results:     data/$ResultsFilename"
Write-Host " Started:     $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "============================================================"

$sw = [System.Diagnostics.Stopwatch]::StartNew()

$methodArgs = @()
foreach ($m in $Methods) {
    $methodArgs += $m
}

& uv run python -m src.cli.run_all `
    --methods @methodArgs `
    --mode clone `
    --model-name $ModelName `
    --sample-percent $SamplePercent `
    --sample-seed $SampleSeed `
    --concurrency $Concurrency `
    --results-filename $ResultsFilename

$exitCode = $LASTEXITCODE
$sw.Stop()

Write-Host "============================================================"
Write-Host " Finished:    $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host (" Wall clock:  {0:N1}s ({1})" -f $sw.Elapsed.TotalSeconds, $sw.Elapsed.ToString())
Write-Host " Exit code:   $exitCode"
Write-Host "============================================================"

exit $exitCode
