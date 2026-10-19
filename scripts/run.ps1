param(
 [ValidateSet('fusion','forecasting')][string]$Module='fusion',
 [ValidateSet('test','execute','catalog')][string]$Task='test',
 [ValidateSet('cpu','gpu')][string]$Runtime='cpu',
 [string]$Script='', [string[]]$ScriptArguments=@(), [string]$Python=''
)
$ErrorActionPreference='Stop'
$suiteRoot=Split-Path $PSScriptRoot -Parent
$moduleRoot=Join-Path $suiteRoot "modules/$Module"
$env:TEMP='E:/Codex/Cache/Temp/portfolio-risk'; $env:TMP=$env:TEMP
$env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONNOUSERSITE='1'; $env:PYTHONUTF8='1'
$env:PYTHONPATH=Join-Path $moduleRoot 'src'
$env:CUDA_CACHE_PATH='E:/Codex/Cache/Dependencies/cuda'
$env:TORCH_HOME='E:/Codex/Cache/Dependencies/torch'
$env:MPLCONFIGDIR='E:/Codex/Cache/Dependencies/matplotlib'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
if (-not $Python) {
 $Python=if ($Runtime -eq 'gpu') {'E:/Codex/Artifacts/ctt-gpu-env/Scripts/python.exe'} else {'E:/Codex/Workspaces/multiscale-risk-fusion/.venv/Scripts/python.exe'}
}
Push-Location $moduleRoot
try {
 if ($Task -eq 'test') { & $Python -B -m pytest tests -q -p no:cacheprovider }
 elseif ($Task -eq 'catalog') { & $Python -B (Join-Path $suiteRoot 'scripts/catalog.py') }
 else {
  if (-not $Script) { throw '-Script is required' }
  $entry=(Resolve-Path (Join-Path $moduleRoot $Script)).Path
  if (-not $entry.StartsWith($moduleRoot.Replace('/','\')+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Script must be inside selected module' }
  & $Python -B $entry @ScriptArguments
 }
 if ($LASTEXITCODE -ne 0) { throw "Task failed: $Task" }
} finally { Pop-Location }
