param([string]$root='C:\runs\d215_preflight',[string]$prefix='d215/preflight',[switch]$KeepRunning)
$ErrorActionPreference = 'Stop'
$code="$root\code"
$py='C:\venvs\qwen\Scripts\python.exe'
$bucket=$env:VLA_BUCKET
if (-not $bucket) { throw 'VLA_BUCKET required' }
if (Test-Path "$root\started.json") { throw 'Already dispatched; no automatic retry' }
New-Item -ItemType Directory -Force $root | Out-Null
@{started_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content "$root\started.json"
try {
 $env:QWEN_MODEL='C:\models\qwen3vl4b'
 $env:HF_HUB_OFFLINE='1'; $env:TRANSFORMERS_OFFLINE='1'; $env:USE_TF='0'; $env:WANDB_MODE='disabled'
 $env:PYTHONPATH="$code\scripts;$code\scripts\rl"
 $adapter='C:\adapters\d207_stage2_adapter_s8000'
 if (-not (Test-Path "$adapter\adapter_model.safetensors")) {throw 'D208 adapter missing'}
 Set-Location $code
 & $py scripts/rl/aws_preflight_driver.py --root "$root\results"
 $nativeExit=$LASTEXITCODE
 if ($nativeExit -ne 0) {throw "Preflight exited $nativeExit"}
 @{status='complete';finished_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content "$root\completion.json"
} catch { $_ | Out-String | Set-Content "$root\failure.txt" }
finally {
 $uploadErrors=@()
 # Upload diagnostic results first; a locked launcher stream must not hide them.
 $files=Get-ChildItem $root -File -Recurse | Where-Object { $_.FullName -notlike "$code\*" -and $_.Name -notin @('launcher.out','launcher.err','code.zip') } | Sort-Object @{Expression={if ($_.Extension -in @('.json','.log','.txt')) {0} else {1}}},FullName
 foreach ($file in $files) {
  try {
   $relative=$file.FullName.Substring($root.Length+1).Replace('\','/')
   Write-S3Object -BucketName $bucket -Key "$prefix/$relative" -File $file.FullName -Region us-east-1 -ErrorAction Stop
  } catch { $uploadErrors += $_.ToString() }
 }
 if ($uploadErrors.Count) { $uploadErrors | Set-Content "$root\upload_failure.txt" }
 # Session owner controls shutdown when chaining pre-training checks.
 if (-not $KeepRunning) {
 shutdown /a 2>$null
 shutdown /s /f /t 10 /c 'D215 preflight finished'
 }
}
