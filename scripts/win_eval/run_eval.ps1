# D162 Windows evaluation driver (runs detached on the AWS Windows GPU box).
# Selects a stratified 100-task subset (first 10 of each class, sorted), then for
# each model: start its server, run the official evaluator until every task has
# its plots (the evaluator skips finished tasks), stop the server.
param([string]$Models = "openvla,qwen", [int]$PerClass = 10, [string]$Offscreen = "0")
$ErrorActionPreference = "Continue"
Start-Transcript -Path C:\runs\run_eval.log -Append
$eval = "C:\win_bundle\UAV-Flow-Eval"
$tasks = "C:\runs\tasks$($PerClass*10)"
New-Item -ItemType Directory -Force $tasks | Out-Null
$classes = Get-Content "$eval\classified_instr.json" -Raw | ConvertFrom-Json
foreach ($p in $classes.PSObject.Properties) {
    $p.Value | Sort-Object | Select-Object -First $PerClass | ForEach-Object {
        Copy-Item "$eval\test_jsons\$_" $tasks -ErrorAction SilentlyContinue
    }
}
"tasks selected: " + (Get-ChildItem $tasks).Count

$env:UE_MAX_FPS = "10"
$env:UAV_EVAL_OFFSCREEN = $Offscreen
$env:PYTHONPATH = "C:\win_bundle\vla\src;C:\win_bundle\vla\scripts"
$env:PYTHONWARNINGS = "ignore"
$env:HF_HUB_OFFLINE = "1"

foreach ($m in $Models.Split(",")) {
    $out = "C:\runs\$m"
    New-Item -ItemType Directory -Force $out | Out-Null
    if ($m -eq "openvla") {
        $py = "C:\venvs\openvla\Scripts\python.exe"; $port = 5007
        $args = "C:\win_bundle\vla\scripts\uav_flow_eval_server.py --model openvla-uav --path C:\models\openvla-uav --precision bf16 --port $port --log $out\server_calls.jsonl"
    } else {
        $py = "C:\venvs\qwen\Scripts\python.exe"; $port = 5008
        $env:QWEN_MODEL = "C:\models\qwen3vl4b"
        $args = "C:\win_bundle\vla\scripts\uav_flow_eval_server.py --model qwen --path C:\win_bundle\adapter\adapter_s2500 --chunk 8 --precision bf16 --port $port --log $out\server_calls.jsonl"
    }
    $server = Start-Process $py -ArgumentList $args -WorkingDirectory "C:\win_bundle\vla" -RedirectStandardOutput "$out\server.out" -RedirectStandardError "$out\server.err" -PassThru -WindowStyle Hidden
    for ($i = 0; $i -lt 120; $i++) {
        if ((Test-Path "$out\server.err") -and (Select-String -Path "$out\server.err" -Pattern "Running on" -Quiet)) { break }
        Start-Sleep 5
    }
    "$m server up: " + (Select-String -Path "$out\server.err" -Pattern "Running on" -Quiet)
    $n = (Get-ChildItem $tasks).Count
    for ($attempt = 1; $attempt -le 20; $attempt++) {
        $done = (Get-ChildItem "$out\flights\*_2d.png" -ErrorAction SilentlyContinue).Count
        "$m attempt $attempt : $done / $n  $(Get-Date)"
        if ($done -ge $n) { break }
        Get-Process Collection -ErrorAction SilentlyContinue | Stop-Process -Force
        Start-Sleep 5
        Push-Location $eval
        & C:\venvs\eval\Scripts\python.exe batch_run_act_all.py -f $tasks -o "$out\flights" -p $port -m 100 *>> "$out\eval.log"
        Pop-Location
    }
    Get-Process Collection -ErrorAction SilentlyContinue | Stop-Process -Force
    Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
    "$m DONE $(Get-Date)"
}
"ALL_DONE"
Stop-Transcript
