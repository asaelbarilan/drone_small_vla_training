# D209: one grounded DAgger round and fixed validation; user authorized 2026-10-09.
param([switch]$RetryPreflight)
$ErrorActionPreference = "Stop"
$root = "C:\runs\d209"
$rollout = "C:\runs\d209_r1"
$bucket = $env:VLA_BUCKET
$py = "C:\venvs\qwen\Scripts\python.exe"
$code = "C:\win_bundle\vla"
if (-not $bucket) { throw "VLA_BUCKET is required" }
if (Test-Path "$root\started.json") {
    if (-not $RetryPreflight -or (Test-Path "$root\train.out") -or (Test-Path $rollout)) { throw "D209 already started; refusing a job restart" }
    if (Test-Path "$root\failure.txt") { Move-Item -LiteralPath "$root\failure.txt" -Destination "$root\preflight_failure_1.txt" -ErrorAction Stop }
}
New-Item -ItemType Directory -Force $root | Out-Null
shutdown /s /t 43200 /c "D209 hard stop 12 hours"
@{started_utc=[DateTime]::UtcNow.ToString("o")} | ConvertTo-Json | Set-Content "$root\started.json"
Start-Transcript "$root\run.log"
function RunPy([string]$Arguments, [string]$Name, [int]$Seconds=14400) {
    $exitPath = "$root\$Name.exit"
    if (Test-Path $exitPath) { throw "Exit marker already exists: $Name" }
    $worker = "$root\$Name.worker.ps1"
    $body = "& '$py' $Arguments`r`n" + '$code_ = $LASTEXITCODE' + "`r`n" + "Set-Content -LiteralPath '$exitPath' -Value " + '$code_' + "`r`n" + 'exit $code_'
    [IO.File]::WriteAllText($worker, $body)
    $p = Start-Process powershell.exe -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File $worker" -WorkingDirectory $code -WindowStyle Hidden -PassThru -RedirectStandardOutput "$root\$Name.out" -RedirectStandardError "$root\$Name.err"
    if (-not $p.WaitForExit($Seconds * 1000)) {
        taskkill /PID $p.Id /T /F | Out-Null
        throw "$Name timed out"
    }
    if (-not (Test-Path $exitPath)) { throw "$Name ended without an exit marker" }
    $nativeExit = [int](Get-Content $exitPath -Raw)
    if ($nativeExit -ne 0) { throw "$Name failed exit=$nativeExit; see $Name.err" }
}
try {
    Read-S3Object -BucketName $bucket -Key d209/vla_code.zip -File "$root\code.zip" -Region us-east-1 | Out-Null
    Expand-Archive "$root\code.zip" $code -Force
    Read-S3Object -BucketName $bucket -Key d209/payload.zip -File "$root\payload.zip" -Region us-east-1 | Out-Null
    Expand-Archive "$root\payload.zip" $root -Force
    Read-S3Object -BucketName $bucket -Key d175/anchor_store.tar -File "$root\anchor.tar" -Region us-east-1 | Out-Null
    New-Item -ItemType Directory -Force "$root\anchor" | Out-Null
    tar -xf "$root\anchor.tar" -C "$root\anchor"
    if ($LASTEXITCODE -ne 0) { throw "Anchor extraction failed" }
    Read-S3Object -BucketName $bucket -Key d193/boxes.jsonl -File "$root\boxes.jsonl" -Region us-east-1 | Out-Null
    $env:UAV_FLOW_OFFICIAL_STORE = "$root\anchor"
    $env:QWEN_MODEL = "C:\models\qwen3vl4b"
    $env:HF_HUB_OFFLINE = "1"
    $env:PYTHONPATH = "$code\src;$code\scripts"
    $env:WANDB_MODE = "disabled"
    $adapter = "C:\adapters\d207_stage2_adapter_s8000"
    if (-not (Test-Path "$adapter\adapter_model.safetensors")) { throw "Step8000 adapter missing" }
    if ((Get-Item "$adapter\adapter_model.safetensors").Length -ne 322279296) { throw "Unexpected step8000 weight size" }
    RunPy "-m unittest discover -s tests -p test_dagger_grounding.py" "encoding_tests" 300
    RunPy "scripts\rl\d209_box_check.py --root $root --adapter $adapter --out $root\box_before.json" "box_before" 1800
    & "$code\scripts\rl\collect_rollouts.ps1" -Passes 1 -Temperature 0 -PerKind 0 -TaskZipFile "$root\practice_tasks.zip" -Adapter $adapter -Tag d209_r1 -NoShutdown -ServerArgs "--progress --goal-memory --memory-deadband 1.0,5 --box"
    RunPy "scripts\rl\dagger_relabel.py --flights $rollout\pass1 --calls $rollout\server_calls.jsonl --images $rollout\rollout_images --tasks $rollout\tasks --out $rollout\dagger_rows.jsonl --box-context" "relabel" 300
    $rows = "C:\runs\d188_dagger_r1\dagger_rows.jsonl C:\runs\d190_dagger_r1\dagger_rows.jsonl $rollout\dagger_rows.jsonl"
    RunPy "scripts\rl\d209_gate.py --root $root --rollout $rollout --rows $rows" "gate" 300
    RunPy "scripts\train_uav_flow_vla.py --format official --chunk 8 --precision bf16 --instruction both --progress --extra-rows $rows --extra-repeat 4 --init-adapter $adapter --updates 300 --lr 2e-5 --schedule cosine --warmup 0.05 --batch-size 4 --accum 8 --workers 2 --val-every 100 --val-batches 64 --save-every 100 --max-wall-seconds 10800 --boxes $root\boxes.jsonl --next-box --box-repeat 3 --grounding-mix 0.1 --vision-lr-mult 0.1 --out $root\update" "train" 14400
    $report = Get-Content "$root\update\report.json" -Raw | ConvertFrom-Json
    if ($report.status -ne "complete" -or $report.losses.Count -ne 300 -or -not (Test-Path "$root\update\adapter_s300\adapter_model.safetensors")) { throw "Trainer did not finish all 300 updates" }
    Get-ChildItem "$root\update" -File -Recurse | Where-Object { $_.FullName -notmatch '\\wandb\\' } | ForEach-Object {
        $relative = $_.FullName.Substring(("$root\update\").Length).Replace('\','/')
        Write-S3Object -BucketName $bucket -Key "d209/update/$relative" -File $_.FullName -Region us-east-1 -ErrorAction Stop
    }
    RunPy "scripts\rl\d209_box_check.py --root $root --adapter $root\update\adapter_s300 --out $root\box_after.json" "box_after" 1800
    # eval_variants removes only these fresh task subdirectories; verify resolved scope.
    foreach ($child in @("tasks","tasks_all")) {
        $target=[IO.Path]::GetFullPath("C:\runs\d209_eval\$child")
        if (-not $target.StartsWith("C:\runs\d209_eval\",[StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe evaluation path" }
    }
    & "$code\scripts\win_eval\eval_variants.ps1" -NoShutdown -Tag d209_eval -CodeZipKey d209/vla_code.zip -TaskZipKey d190/check_tasks.zip -Variants "s300|$root\update\adapter_s300|--progress --goal-memory --memory-deadband 1.0,5 --box"
    $done=@(Get-ChildItem C:\runs\d209_eval\s300\flights -Filter *_2d.png).Count
    if ($done -ne 100) { throw "Incomplete validation: $done/100" }
    RunPy "scripts\score_tasks.py --flights C:\runs\d209_eval\s300\flights --tasks C:\runs\d209_eval\tasks --out $root\score.json" "score" 300
    @{status="complete"; updates=300; validation_completed=$done; finished_utc=[DateTime]::UtcNow.ToString("o")} | ConvertTo-Json | Set-Content "$root\completion.json"
} catch {
    $_ | Out-String | Set-Content "$root\failure.txt"
} finally {
    try { Stop-Transcript | Out-Null } catch {}
    foreach ($file in Get-ChildItem $root -File | Where-Object {$_.Extension -in @('.log','.out','.err','.json','.txt')}) {
        try { Write-S3Object -BucketName $bucket -Key "d209/$($file.Name)" -File $file.FullName -Region us-east-1 -ErrorAction Stop } catch {}
    }
    shutdown /a 2>$null
    shutdown /s /f /t 10 /c "D209 finished"
}
