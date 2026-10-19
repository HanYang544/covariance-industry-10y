$ErrorActionPreference='Stop'
$projectRoot=(Resolve-Path (Split-Path $PSScriptRoot -Parent)).Path
$links=@{
 'data/raw'='modules/fusion/data/raw'
 'data/frozen'='modules/fusion/outputs/industry_10y_aligned'
 'outputs/statistical'='modules/fusion/outputs/industry_10y_aligned'
 'outputs/fusion/gating/transformer'='modules/fusion/outputs/temporal_gate_100'
 'outputs/fusion/gating/mlp'='modules/fusion/outputs/sequence_mlp_100'
 'outputs/fusion/gating/cnn'='modules/fusion/outputs/sequence_cnn_100'
 'outputs/fusion/refinement'='modules/fusion/outputs/sequence_joint_single_100'
 'outputs/direct/feature_heads'='modules/fusion/outputs/direct_spd_100'
 'outputs/direct/covariance_heads'='modules/fusion/outputs/covariance_spd_100'
 'outputs/direct/architectures'='modules/fusion/outputs/residual_architecture_100'
 'outputs/direct/learning_rates'='modules/fusion/outputs/covariance_lr_search'
 'outputs/paper/cab'='modules/fusion/outputs/industry_10y_aligned/cab'
 'outputs/replicates'='modules/fusion/outputs/seed_completion'
}
foreach($entry in $links.GetEnumerator()){
 $target=(Resolve-Path (Join-Path $projectRoot $entry.Value)).Path
 $link=[IO.Path]::GetFullPath((Join-Path $projectRoot $entry.Key))
 if(-not $target.StartsWith($projectRoot+'\') -or -not $link.StartsWith($projectRoot+'\')){throw 'Path outside project'}
 if(Test-Path -LiteralPath $link){
  $item=Get-Item -LiteralPath $link
  if($item.LinkType -ne 'Junction' -or $item.Target -ne $target){throw "Existing nonmatching path: $link"}
 }else{
  New-Item -ItemType Directory -Force -Path (Split-Path $link -Parent) | Out-Null
  New-Item -ItemType Junction -Path $link -Target $target | Out-Null
 }
}
Write-Output "Created/verified $($links.Count) in-project directory views; no artifacts moved."
