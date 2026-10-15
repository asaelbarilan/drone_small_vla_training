param([Parameter(Mandatory=$true)][string]$Bucket,[Parameter(Mandatory=$true)][string]$DeadlineUtc)
$ErrorActionPreference='Stop'
$root='C:/runs/d240_resume_review';$code="$root/code";$source='C:/runs/d238_two_task/code'
if((Test-Path "$root/bootstrap_dispatch.json") -or (Test-Path "$root/dispatched.json") -or (Test-Path "$root/results/started.json")){throw 'Existing dispatch marker: monitor only'}
if(Get-Process python,Collection,Collection-Win64-Shipping -ErrorAction SilentlyContinue){throw 'Existing workers: do not duplicate'}
if(!(Test-Path "$source/scripts/rl/d238_job.py")){throw 'Prior code missing'}
if(!(Test-Path 'C:/runs/d238_two_task/results/round_1/training/checkpoints/CURRENT.json')){throw 'Prior committed checkpoint missing'}
$deadline=[DateTime]::Parse($DeadlineUtc).ToUniversalTime();$remaining=[int]($deadline-[DateTime]::UtcNow).TotalSeconds
if($remaining -lt 3600 -or $remaining -gt 7200){throw 'Deadline outside approved remaining budget'}
& shutdown.exe /s /f /t $remaining /c 'D240 hard deadline'
if($LASTEXITCODE -ne 0){throw 'Backup shutdown scheduling failed'}
New-Item -ItemType Directory $code -Force|Out-Null
Copy-Item "$source/*" $code -Recurse
Read-S3Object -BucketName $Bucket -Key 'd240/overlay.zip' -File "$root/overlay.zip" -Region us-east-1|Out-Null
$hash=(Get-FileHash "$root/overlay.zip" -Algorithm SHA256).Hash.ToLowerInvariant()
if($hash -ne '92eb9f0e342049d4056c6765ff3a6915fd7f29d50702f95e754e514dadb490ba'){throw 'Overlay checksum mismatch'}
Expand-Archive "$root/overlay.zip" $code -Force
@{dispatched_utc=[DateTime]::UtcNow.ToString('o');deadline_utc=$DeadlineUtc;source='D238 step1';additional_update_cap=3;baseline_reflights=0}|ConvertTo-Json|Set-Content "$root/bootstrap_dispatch.json"
$proc=Start-Process powershell.exe -WindowStyle Hidden -PassThru -WorkingDirectory $code -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',"$code/scripts/win_eval/run_d240.ps1",'-Bucket',$Bucket,'-DeadlineUtc',$DeadlineUtc) -RedirectStandardOutput "$root/launcher.out" -RedirectStandardError "$root/launcher.err"
@{wrapper_pid=$proc.Id;deadline_utc=$DeadlineUtc}|ConvertTo-Json
