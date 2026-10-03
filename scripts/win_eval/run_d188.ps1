# D188 (user-approved 2026-10-03, about $6.30): DAgger round 1, decided on VALIDATION only.
#   1. before : current best model on the 73 Land + Pass validation tasks   (measured, not learned)
#   2. dagger : one greedy pass over the 50 Land + Pass practice (TRAIN) tasks, relabel, 300 updates
#   3. after  : the DAgger model on the same 73 validation tasks             (measured, not learned)
#   4. PPO check: 4 practice tasks x 2 flights, one PPO update (value-head fix on the GPU)
# The benchmark test tasks are never used (rule 1). Needs $env:VLA_BUCKET from the launcher.
# 7 h shutdown timer first; every step runs with -NoShutdown; this wrapper turns the box off.
$ErrorActionPreference = "Continue"
shutdown /s /t 25200 /c "D188 hard stop (7 h)"
$bucket = $env:VLA_BUCKET
New-Item -ItemType Directory -Force C:\runs\d188 | Out-Null
Start-Transcript -Path C:\runs\d188\run.log -Append
try {
    if (-not $bucket) { throw "VLA_BUCKET not set" }
    Read-S3Object -BucketName $bucket -Key "d180/vla_code.zip" -File C:\runs\d188\code.zip -Region us-east-1 | Out-Null
    Expand-Archive -Path C:\runs\d188\code.zip -DestinationPath C:\win_bundle\vla -Force
    $server = "--progress --goal-memory --memory-deadband 1.0,5"
    $start = "d171/run/adapter_s31000"
    "=== 1 before (validation) $(Get-Date)"
    & C:\win_bundle\vla\scripts\win_eval\eval_variants.ps1 -NoShutdown -TaskZipKey "d187/val_tasks.zip" -Kinds "Land,Pass" `
        -Tag d188_before -Variants "before|$start|$server"
    "=== 2 DAgger round 1 (train tasks) $(Get-Date)"
    & C:\win_bundle\vla\scripts\rl\rl_loop.ps1 -Method dagger -Rounds 1 -PerKind 25 -Kinds "Land,Pass" -DaggerUpdates 300 `
        -AdapterKey $start -ServerArgs $server -Tag d188_dagger -NoShutdown
    $dagger = "C:\runs\d188_dagger_r1\update\adapter_s300"
    if (Test-Path "$dagger\adapter_config.json") {
        foreach ($f in Get-ChildItem $dagger -File) {
            Write-S3Object -BucketName $bucket -Key "d188/dagger_r1_adapter/$($f.Name)" -File $f.FullName -Region us-east-1
        }
        "=== 3 after (validation) $(Get-Date)"
        & C:\win_bundle\vla\scripts\win_eval\eval_variants.ps1 -NoShutdown -TaskZipKey "d187/val_tasks.zip" -Kinds "Land,Pass" `
            -Tag d188_after -Variants "after|$dagger|$server"
    } else { "NO DAGGER ADAPTER - skipping the after test" }
    "=== 4 PPO check $(Get-Date)"
    & C:\win_bundle\vla\scripts\rl\rl_loop.ps1 -Method ppo -Rounds 1 -Passes 2 -Temperature 1.3 -PerKind 2 -Kinds "Land,Pass" `
        -AdapterKey $start -ServerArgs $server -TrainArgs "--progress" -Tag d188_ppocheck -NoShutdown
    "=== D188_DONE $(Get-Date)"
} catch {
    "WRAPPER ERROR: $_"
} finally {
    try { Stop-Transcript | Out-Null } catch {}
    try { Write-S3Object -BucketName $bucket -Key "d188/run.log" -File C:\runs\d188\run.log -Region us-east-1 } catch {}
    shutdown /a 2>$null; shutdown /s /f /t 5 /c "D188 done"
}
