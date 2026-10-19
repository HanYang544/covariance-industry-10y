param(
 [ValidateSet('report','verify','inspect','train')][string]$Task='report',
 [string]$Config='configs/direct/residual_transformer.json',
 [string]$Name='',
 [ValidateSet('cpu','gpu')][string]$Runtime='gpu'
)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
$env:TEMP='E:/Codex/Cache/Temp/portfolio-risk';$env:TMP=$env:TEMP
$env:PYTHONDONTWRITEBYTECODE='1';$env:PYTHONUTF8='1';$env:PYTHONNOUSERSITE='1'
$env:CUDA_CACHE_PATH='E:/Codex/Cache/Dependencies/cuda'
$env:TORCH_HOME='E:/Codex/Cache/Dependencies/torch'
$env:MPLCONFIGDIR='E:/Codex/Cache/Dependencies/matplotlib'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:PYTHONPATH=Join-Path $projectRoot 'src'
$pythonPath=if($Runtime -eq 'gpu'){'E:/Codex/Artifacts/ctt-gpu-env/Scripts/python.exe'}else{'E:/Codex/Workspaces/multiscale-risk-fusion/.venv/Scripts/python.exe'}
Push-Location $projectRoot
try {
 if($Task -eq 'report'){& $pythonPath -B scripts/build_report.py}
 elseif($Task -eq 'verify'){
  & $pythonPath -B tests/check_modular_equivalence.py
  if($LASTEXITCODE -ne 0){throw 'Model equivalence failed'}
  & $pythonPath -B tests/check_saved_predictions.py
  if($LASTEXITCODE -ne 0){throw 'Archived checkpoint equivalence failed'}
  & $pythonPath -B scripts/check_three_seed_report.py
 }
 elseif($Task -eq 'inspect'){& $pythonPath -B experiments/run.py --config $Config}
 else {
  if(-not $Name){throw 'A new run Name is required'}
  $computeDevice=if($Runtime -eq 'gpu'){'cuda'}else{'cpu'}
  & $pythonPath -B experiments/run.py --config $Config --train --name $Name --device $computeDevice
 }
 if($LASTEXITCODE -ne 0){throw "Task failed: $Task"}
} finally {Pop-Location}
