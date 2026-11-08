# Film trained LLM-GE XQC genes on a Windows laptop (no retraining, no PACE queue).
#
# One-time setup (already done on Alex's laptop):
#   uv venv --python 3.10 $env:USERPROFILE\.xqc-venv
#   uv pip install --python $env:USERPROFILE\.xqc-venv\Scripts\python.exe -r sota\XQC\requirements.txt glfw
#
# For each gene, download from PACE into <Runs>\<gene_id>\ these two files:
#   ~/scratch/xqc-baseline/runs/llmge/<gene_id>/config.json
#   ~/scratch/xqc-baseline/runs/llmge/<gene_id>/actor.pkl
# then, from anywhere:
#   powershell -ExecutionPolicy Bypass -File sota\XQC\make_video_local.ps1 xXxsD5kAqd3JEK13EBiFixC2PTd
#   (several genes: "geneA,geneB"; options: -Steps 300, -Runs <folder>, -Out <folder>)
# The GIF lands in <Out>\<gene_id>\seed0.gif (default: Downloads\xqc_videos).
param(
    # One gene id, or several separated by commas.
    [Parameter(Mandatory = $true, Position = 0)][string]$Genes,
    [int]$Steps = 1000,
    [string]$Runs = "$env:USERPROFILE\Downloads\xqc_runs",
    [string]$Out = "$env:USERPROFILE\Downloads\xqc_videos",
    [string]$Python = "$env:USERPROFILE\.xqc-venv\Scripts\python.exe"
)
$ErrorActionPreference = "Stop"

# Windows renders through GLFW (PACE uses EGL); JAX runs on the CPU.
$env:MUJOCO_GL = "glfw"
$env:JAX_PLATFORMS = "cpu"
$env:XLA_PYTHON_CLIENT_PREALLOCATE = "false"

if (-not (Test-Path $Python)) { throw "No XQC environment at $Python (see the setup lines at the top)." }
$dirs = @(foreach ($g in ($Genes -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })) {
    $d = Join-Path $Runs $g
    foreach ($f in "config.json", "actor.pkl") {
        if (-not (Test-Path (Join-Path $d $f))) { throw "Missing $d\$f - download it from PACE first." }
    }
    $d
})   # @() keeps a single gene an array, so @dirs passes the whole path
# JAX prints harmless warnings on stderr; Windows PowerShell would treat them as
# errors under "Stop", so let them through and judge by the exit code instead.
$ErrorActionPreference = "Continue"
& $Python (Join-Path $PSScriptRoot "render_actor.py") @dirs -video_steps $Steps -out $Out
if ($LASTEXITCODE -ne 0) { throw "render_actor.py failed (exit $LASTEXITCODE)." }
Write-Host "Done: GIFs in $Out"
