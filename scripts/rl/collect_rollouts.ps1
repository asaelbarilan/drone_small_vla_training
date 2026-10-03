# D172 RL rollout collection on the Windows simulator box.
# Flies every practice task PASSES times with the rollout server sampling
# (--rollout-temperature), using the unmodified official evaluator; one server
# serves all passes, so its log holds every call. Uploads everything to S3 and
# turns the machine off (Stop-Computer BEFORE Stop-Transcript: D168 found the
# reverse order leaves the machine running), with a shutdown /s timer as backup.
# D175: -NoShutdown leaves the machine on (rl_loop.ps1 runs this once per round
# and owns the shutdown); -TaskZipFile takes a local task zip instead of S3;
# -ServerArgs adds server flags (e.g. "--progress").
param([int]$Passes = 8, [double]$Temperature = 1.0, [int]$PerKind = 6,
      [string]$TaskZipKey = "d172/rl_tasks.zip", [string]$TaskZipFile = "",
      [string]$Adapter = "C:\win_bundle\adapter_rl\adapter",
      [string]$Tag = "rl_round1", [int]$HardStopMinutes = 480,
      [switch]$NoShutdown, [string]$ServerArgs = "", [string]$Kinds = "")
# D185: -Temperature 0 = greedy decoding with every photo saved (--save-call-images, for
# DAgger); -Kinds "Land,Pass" keeps only those motion types (with -PerKind > 0).
$ErrorActionPreference = "Continue"
if (-not $NoShutdown) { shutdown /s /t ($HardStopMinutes * 60) /c "D172 rollout hard stop" }
$root = "C:\runs\$Tag"
New-Item -ItemType Directory -Force $root | Out-Null
Start-Transcript -Path "$root\collect.log" -Append
$eval = "C:\win_bundle\UAV-Flow-Eval"
$bucket = $env:VLA_BUCKET  # your S3 bucket (see .env.example)
if (-not $bucket) { throw "set the VLA_BUCKET environment variable to your S3 bucket" }

# Tasks: from $TaskZipKey. $PerKind > 0 takes the first $PerKind of every motion
# type (round 1); 0 takes the whole zip (select_tasks.py already chose them).
$all = "$root\tasks_zip"
New-Item -ItemType Directory -Force $all | Out-Null
if ($TaskZipFile) { Copy-Item $TaskZipFile "$root\rl_tasks.zip" -Force }
else { Read-S3Object -BucketName $bucket -Key $TaskZipKey -File "$root\rl_tasks.zip" -Region us-east-1 | Out-Null }
Expand-Archive -Path "$root\rl_tasks.zip" -DestinationPath $all -Force
$tasks = "$root\tasks"
New-Item -ItemType Directory -Force $tasks | Out-Null
if ($PerKind -gt 0) {
    $wanted = if ($Kinds) { $Kinds.Split(",") } else { @() }
    Get-ChildItem $all -Filter *.json | Sort-Object Name | Group-Object { (Get-Content $_.FullName -Raw | ConvertFrom-Json).motion_kind } |
      Where-Object { $wanted.Count -eq 0 -or $wanted -contains $_.Name } |
      ForEach-Object { $_.Group | Select-Object -First $PerKind | Copy-Item -Destination $tasks }
} else {
    Get-ChildItem $all -Filter *.json | Copy-Item -Destination $tasks
}
"tasks: " + (Get-ChildItem $tasks).Count

$env:UE_MAX_FPS = "10"; $env:UAV_EVAL_OFFSCREEN = "1"; $env:HF_HUB_OFFLINE = "1"; $env:PYTHONWARNINGS = "ignore"
$env:PYTHONPATH = "C:\win_bundle\vla\src;C:\win_bundle\vla\scripts"
$env:QWEN_MODEL = "C:\models\qwen3vl4b"
$port = 5008
$sampling = if ($Temperature -gt 0) { "--rollout-temperature $Temperature" } else { "--save-call-images" }
$sargs = "C:\win_bundle\vla\scripts\uav_flow_eval_server.py --model qwen --path $Adapter --chunk 8 --precision bf16 --port $port --log $root\server_calls.jsonl $sampling $ServerArgs"
$server = Start-Process C:\venvs\qwen\Scripts\python.exe -ArgumentList $sargs -WorkingDirectory "C:\win_bundle\vla" -RedirectStandardOutput "$root\server.out" -RedirectStandardError "$root\server.err" -PassThru -WindowStyle Hidden
for ($i = 0; $i -lt 120; $i++) {
    if ((Test-Path "$root\server.err") -and (Select-String -Path "$root\server.err" -Pattern "Running on" -Quiet)) { break }
    Start-Sleep 5
}
"server up: " + (Select-String -Path "$root\server.err" -Pattern "Running on" -Quiet)
$n = (Get-ChildItem $tasks).Count
for ($p = 1; $p -le $Passes; $p++) {
    $out = "$root\pass$p"
    for ($attempt = 1; $attempt -le 10; $attempt++) {
        $done = (Get-ChildItem "$out\*_2d.png" -ErrorAction SilentlyContinue).Count
        "pass $p attempt $attempt : $done / $n  $(Get-Date)"
        if ($done -ge $n) { break }
        Get-Process Collection -ErrorAction SilentlyContinue | Stop-Process -Force
        Start-Sleep 5
        Push-Location $eval
        & C:\venvs\eval\Scripts\python.exe batch_run_act_all.py -f $tasks -o $out -p $port -m 100 *>> "$root\eval.log"
        Pop-Location
    }
}
Get-Process Collection -ErrorAction SilentlyContinue | Stop-Process -Force
Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
Compress-Archive -Path "$root\pass*", "$root\server_calls.jsonl", "$root\rollout_images", "$root\tasks" -DestinationPath "C:\runs\$Tag.zip" -Force
Write-S3Object -BucketName $bucket -Key "d172/$Tag.zip" -File "C:\runs\$Tag.zip" -Region us-east-1
"ROLLOUTS_DONE $(Get-Date)"
# D181: transcript off before its log is uploaded (locked otherwise); abort the
# backup timer first, Stop-Computer refuses while one is scheduled.
try { Stop-Transcript | Out-Null } catch {}
try { Write-S3Object -BucketName $bucket -Key "d172/$Tag.collect.log" -File "$root\collect.log" -Region us-east-1 } catch {}
if (-not $NoShutdown) { shutdown /a 2>$null; shutdown /s /f /t 5 /c "D172 rollouts done" }
