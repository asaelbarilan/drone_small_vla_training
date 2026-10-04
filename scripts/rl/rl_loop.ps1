# D175 RL loop on the Windows simulator box: several rounds on one machine.
# Per round: fly the tasks with collect_rollouts.ps1 (-NoShutdown), update the adapter on
# this box's GPU, upload the round's adapter and report to S3, pick the next tasks. The
# machine turns itself off at the end; a shutdown /s timer ($HardStopMinutes) is the backup.
# Only one G machine fits the GPU quota, so flying and updating share this box.
#
# -Method (D185) picks the update; the others are off:
#   grpo    group z-scored flight rewards (D172), train_grpo_vla.py --objective grpo
#   ppo     per-call advantages from a value head (GAE), --objective ppo; the value head is
#           carried from round to round (value_head.pt)
#   dagger  fly greedily once (no sampling), relabel every visited state with the recorded
#           expert flight (dagger_relabel.py), supervised update on all rows so far (DAgger
#           aggregates) plus the tasks' own flights; same tasks every round
# -Kinds "Land,Pass" restricts the practice tasks to those motion types (collect_rollouts.ps1).
param([int]$Rounds = 3, [int]$Passes = 8, [double]$Temperature = 1.0, [int]$PerKind = 6,
      [int]$Count = 48, [string]$Adapter = "C:\win_bundle\adapter_rl\adapter",
      [string]$Tag = "rl_loop", [string]$CodeZipKey = "d180/vla_code.zip", [string]$AdapterKey = "",
      [string]$AnchorKey = "d175/anchor_store.tar", [string]$TaskZipKey = "d172/rl_tasks.zip",
      [ValidateSet("grpo", "ppo", "dagger")][string]$Method = "grpo", [string]$Kinds = "",
      [int]$DaggerUpdates = 300, [double]$DaggerLr = 2e-5,
      [string]$PriorRows = "",
      [string]$TrainArgs = "", [string]$ServerArgs = "", [int]$HardStopMinutes = 1440,
      [switch]$NoShutdown)
$ErrorActionPreference = "Continue"
# -NoShutdown: a wrapper runs more work afterwards and owns timer and shutdown.
if (-not $NoShutdown) { shutdown /s /t ($HardStopMinutes * 60) /c "D175 RL loop hard stop" }
$bucket = $env:VLA_BUCKET  # your S3 bucket (see .env.example)
if (-not $bucket) { throw "set the VLA_BUCKET environment variable to your S3 bucket" }
$home_ = "C:\runs\$Tag"
New-Item -ItemType Directory -Force $home_ | Out-Null
Start-Transcript -Path "$home_\loop.log" -Append
try {
    # Latest code (trainer, server, RL scripts) and the expert-flight store.
    Read-S3Object -BucketName $bucket -Key $CodeZipKey -File "$home_\code.zip" -Region us-east-1 | Out-Null
    Expand-Archive -Path "$home_\code.zip" -DestinationPath "C:\win_bundle\vla" -Force
    Read-S3Object -BucketName $bucket -Key $AnchorKey -File "$home_\anchor.tar" -Region us-east-1 | Out-Null
    New-Item -ItemType Directory -Force "C:\anchor_store" | Out-Null
    tar -xf "$home_\anchor.tar" -C "C:\anchor_store"
    Read-S3Object -BucketName $bucket -Key $TaskZipKey -File "$home_\all_tasks.zip" -Region us-east-1 | Out-Null
    Expand-Archive -Path "$home_\all_tasks.zip" -DestinationPath "$home_\all_tasks" -Force

    $env:UAV_FLOW_OFFICIAL_STORE = "C:\anchor_store"
    $env:QWEN_MODEL = "C:\models\qwen3vl4b"; $env:HF_HUB_OFFLINE = "1"; $env:PYTHONWARNINGS = "ignore"
    $env:PYTHONPATH = "C:\win_bundle\vla\src;C:\win_bundle\vla\scripts"
    $py = "C:\venvs\qwen\Scripts\python.exe"
    # D180: the starting adapter can come straight from S3 (the training run's sync).
    if ($AdapterKey) {
        $Adapter = "C:\adapters\" + ($AdapterKey -replace "[/\\]", "_")
        if (-not (Test-Path "$Adapter\adapter_config.json")) {
            Read-S3Object -BucketName $bucket -KeyPrefix $AdapterKey -Folder $Adapter -Region us-east-1 | Out-Null
        }
    }
    "start adapter: $Adapter  present: " + (Test-Path "$Adapter\adapter_config.json") + "  method: $Method"
    if (-not (Test-Path "$Adapter\adapter_config.json")) { throw "no start adapter" }
    $current = $Adapter
    $reports = @()
    $daggerRows = @()
    # D190: DAgger rows from earlier runs on this box (comma-separated .jsonl), aggregated too.
    if ($PriorRows) { $daggerRows += $PriorRows.Split(",") | Where-Object { Test-Path $_ } }
    "prior DAgger rows: $($daggerRows -join ' ')"
    $valueHead = ""
    $taskZip = ""
    if ($Method -eq "dagger") { $Passes = 1; $Temperature = 0 }  # greedy, every photo saved
    for ($r = 1; $r -le $Rounds; $r++) {
        $round = "${Tag}_r$r"
        $dir = "C:\runs\$round"
        "=== round $r / $Rounds : adapter $current  $(Get-Date)"
        if ($r -eq 1 -or $Method -eq "dagger") {
            & "C:\win_bundle\vla\scripts\rl\collect_rollouts.ps1" -Passes $Passes -Temperature $Temperature `
                -PerKind $PerKind -Kinds $Kinds -TaskZipKey $TaskZipKey -Adapter $current -Tag $round `
                -NoShutdown -ServerArgs $ServerArgs
        } else {
            & "C:\win_bundle\vla\scripts\rl\collect_rollouts.ps1" -Passes $Passes -Temperature $Temperature `
                -PerKind 0 -TaskZipFile $taskZip -Adapter $current -Tag $round -NoShutdown -ServerArgs $ServerArgs
        }
        $flights = (1..$Passes | ForEach-Object { "$dir\pass$_" }) -join " "
        if ($Method -eq "dagger") {
            & $py "C:\win_bundle\vla\scripts\rl\dagger_relabel.py" --flights $dir\pass1 --calls "$dir\server_calls.jsonl" `
                --images "$dir\rollout_images" --tasks "$dir\tasks" --out "$dir\dagger_rows.jsonl"
            $daggerRows += "$dir\dagger_rows.jsonl"
            $targs = "C:\win_bundle\vla\scripts\train_uav_flow_vla.py --format official --chunk 8 --precision bf16 " +
                     "--instruction both --progress --extra-rows $($daggerRows -join ' ') --init-adapter $current " +
                     "--updates $DaggerUpdates --lr $DaggerLr --schedule cosine --warmup 0.05 --batch-size 4 --accum 8 " +
                     "--val-every 100 --val-batches 64 --max-wall-seconds 14400 --out $dir\update $TrainArgs"
            $next = "$dir\update\adapter_s$DaggerUpdates"
            $report = "$dir\update\report.json"
        } else {
            $vh = if ($Method -eq "ppo" -and $valueHead) { "--value-head $valueHead" } else { "" }
            $targs = "C:\win_bundle\vla\scripts\rl\train_grpo_vla.py --objective $Method --flights $flights " +
                     "--calls $dir\server_calls.jsonl --images $dir\rollout_images --tasks $dir\tasks " +
                     "--init-adapter $current --out $dir\update --precision bf16 $vh $TrainArgs"
            $next = "$dir\update\adapter_$Method"
            $report = "$dir\update\report.json"
        }
        $p = Start-Process $py -ArgumentList $targs -WorkingDirectory "C:\win_bundle\vla" -Wait -PassThru `
            -RedirectStandardOutput "$dir\update.out" -RedirectStandardError "$dir\update.err" -WindowStyle Hidden
        "$Method update exit $($p.ExitCode)  $(Get-Date)"
        Get-Content "$dir\update.out" -Tail 6
        if (-not (Test-Path "$next\adapter_config.json")) { "NO ADAPTER at $next - stopping the loop"; break }
        $pack = @($next, $report)
        if (Test-Path "$dir\update\value_head.pt") { $pack += "$dir\update\value_head.pt"; $valueHead = "$dir\update\value_head.pt" }
        Compress-Archive -Path $pack -DestinationPath "$dir\update.zip" -Force
        Write-S3Object -BucketName $bucket -Key "d185/$round.$Method.zip" -File "$dir\update.zip" -Region us-east-1
        $current = $next
        if ($Method -ne "dagger") {
            $reports += $report
            $taskZip = "$dir\next_tasks.zip"
            & $py "C:\win_bundle\vla\scripts\rl\select_tasks.py" --tasks "$home_\all_tasks" --reports $reports `
                --count $Count --out $taskZip
        }
    }
    "LOOP_DONE $(Get-Date)"
} finally {
    # D181: transcript off before its log is uploaded (locked otherwise); abort the
    # backup timer first, Stop-Computer refuses while one is scheduled.
    try { Stop-Transcript | Out-Null } catch {}
    try { Write-S3Object -BucketName $bucket -Key "d185/$Tag.loop.log" -File "$home_\loop.log" -Region us-east-1 } catch {}
    if (-not $NoShutdown) { shutdown /a 2>$null; shutdown /s /f /t 5 /c "D175 RL loop done" }
}
