# D178 flight test of several adapters / server options on the Windows simulator box.
# Same 100 tasks as D163-D168 (first 10 of each class, sorted), unmodified official
# evaluator. -Variants is "name|adapter S3 prefix|extra server flags", separated by
# ";", e.g.
#   "s18000|d171/run/adapter_s18000|;final|d171/run/adapter_s30000|--progress;final_mem|d171/run/adapter_s30000|--progress --goal-memory"
# Pulls the current code bundle first (the server with --progress / --goal-memory),
# downloads each adapter once, flies every variant, uploads each variant's flights
# and server log to s3://.../d178/<name>.zip as soon as it is done, and turns the
# machine off (Stop-Computer BEFORE Stop-Transcript, D168), with a shutdown /s timer
# as backup.
param([Parameter(Mandatory = $true)][string]$Variants, [int]$PerClass = 10,
      [string]$CodeZipKey = "d180/vla_code.zip", [int]$HardStopMinutes = 420,
      [switch]$NoShutdown)
$ErrorActionPreference = "Continue"
shutdown /s /t ($HardStopMinutes * 60) /c "D178 flight test hard stop"
$bucket = $env:VLA_BUCKET  # your S3 bucket (see .env.example)
if (-not $bucket) { throw "set the VLA_BUCKET environment variable to your S3 bucket" }
$root = "C:\runs\d178"
New-Item -ItemType Directory -Force $root | Out-Null
Start-Transcript -Path "$root\eval.log" -Append
try {
    Read-S3Object -BucketName $bucket -Key $CodeZipKey -File "$root\code.zip" -Region us-east-1 | Out-Null
    Expand-Archive -Path "$root\code.zip" -DestinationPath "C:\win_bundle\vla" -Force
    "server has --progress: " + (Select-String -Path "C:\win_bundle\vla\scripts\uav_flow_eval_server.py" -Pattern "goal-memory" -Quiet)

    $eval = "C:\win_bundle\UAV-Flow-Eval"
    $tasks = "C:\runs\tasks$($PerClass*10)"
    New-Item -ItemType Directory -Force $tasks | Out-Null
    $classes = Get-Content "$eval\classified_instr.json" -Raw | ConvertFrom-Json
    foreach ($p in $classes.PSObject.Properties) {
        $p.Value | Sort-Object | Select-Object -First $PerClass | ForEach-Object {
            Copy-Item "$eval\test_jsons\$_" $tasks -ErrorAction SilentlyContinue
        }
    }
    $n = (Get-ChildItem $tasks).Count
    "tasks selected: $n"

    $env:UE_MAX_FPS = "10"; $env:UAV_EVAL_OFFSCREEN = "1"; $env:HF_HUB_OFFLINE = "1"; $env:PYTHONWARNINGS = "ignore"
    $env:PYTHONPATH = "C:\win_bundle\vla\src;C:\win_bundle\vla\scripts"
    $env:QWEN_MODEL = "C:\models\qwen3vl4b"
    $port = 5008
    foreach ($v in $Variants.Split(";")) {
        $name, $prefix, $flags = $v.Split("|")
        $adapter = "C:\adapters\" + ($prefix -replace "[/\\]", "_")
        if (-not (Test-Path "$adapter\adapter_config.json")) {
            Read-S3Object -BucketName $bucket -KeyPrefix $prefix -Folder $adapter -Region us-east-1 | Out-Null
        }
        "=== $name : $adapter $flags  $(Get-Date)"
        if (-not (Test-Path "$adapter\adapter_config.json")) { "NO ADAPTER at $prefix - skipped"; continue }
        $out = "$root\$name"
        New-Item -ItemType Directory -Force $out | Out-Null
        $sargs = "C:\win_bundle\vla\scripts\uav_flow_eval_server.py --model qwen --path $adapter --chunk 8 --precision bf16 --port $port --log $out\server_calls.jsonl $flags"
        $server = Start-Process C:\venvs\qwen\Scripts\python.exe -ArgumentList $sargs -WorkingDirectory "C:\win_bundle\vla" `
            -RedirectStandardOutput "$out\server.out" -RedirectStandardError "$out\server.err" -PassThru -WindowStyle Hidden
        for ($i = 0; $i -lt 120; $i++) {
            if ((Test-Path "$out\server.err") -and (Select-String -Path "$out\server.err" -Pattern "Running on" -Quiet)) { break }
            Start-Sleep 5
        }
        "$name server up: " + (Select-String -Path "$out\server.err" -Pattern "Running on" -Quiet)
        for ($attempt = 1; $attempt -le 20; $attempt++) {
            $done = (Get-ChildItem "$out\flights\*_2d.png" -ErrorAction SilentlyContinue).Count
            "$name attempt $attempt : $done / $n  $(Get-Date)"
            if ($done -ge $n) { break }
            Get-Process Collection -ErrorAction SilentlyContinue | Stop-Process -Force
            Start-Sleep 5
            Push-Location $eval
            & C:\venvs\eval\Scripts\python.exe batch_run_act_all.py -f $tasks -o "$out\flights" -p $port -m 100 *>> "$out\eval.log"
            Pop-Location
        }
        Get-Process Collection -ErrorAction SilentlyContinue | Stop-Process -Force
        Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
        Compress-Archive -Path "$out\flights", "$out\server_calls.jsonl", "$out\server.err" -DestinationPath "$root\$name.zip" -Force
        Write-S3Object -BucketName $bucket -Key "d178/$name.zip" -File "$root\$name.zip" -Region us-east-1
        "$name DONE $(Get-Date)"
    }
    "ALL_DONE $(Get-Date)"
} finally {
    # D181: transcript off before its log is uploaded (locked otherwise); abort the
    # backup timer first, Stop-Computer refuses while one is scheduled.
    try { Stop-Transcript | Out-Null } catch {}
    try { Write-S3Object -BucketName $bucket -Key "d178/eval.log" -File "$root\eval.log" -Region us-east-1 } catch { "log upload failed: $_" }
    # -NoShutdown: a wrapper runs more work afterwards and owns the shutdown.
    if (-not $NoShutdown) { shutdown /a 2>$null; shutdown /s /f /t 5 /c "D178 done" }
}
