# D190 (user-approved 2026-10-04, about $10): DAgger round 2, decided on VALIDATION only.
#   1. before : the round-1 DAgger model on a fixed 100-task VALIDATION check set, every kind
#               (Land 15, Pass 15, Shift 15, A/D 10, Approach/Move 10, Rotate 10, Surround 10,
#               Turn 12, Retreat 3)                                       (measured, not learned)
#   2. dagger : one greedy pass over 150 TRAINING practice tasks (50 new Land, 75 Turn and
#               25 Surround with the object reconstructed, D189), relabel, 300 updates on the
#               new rows + round 1's rows (DAgger aggregates)
#   3. after  : the round-2 model on the same 100 validation tasks       (measured, not learned)
# The benchmark test tasks are never used (rule 1). Needs $env:VLA_BUCKET from the launcher.
# 13 h shutdown timer first; every step runs with -NoShutdown; this wrapper turns the box off.
$ErrorActionPreference = "Continue"
shutdown /s /t 46800 /c "D190 hard stop (13 h)"
$bucket = $env:VLA_BUCKET
$code = "d190/vla_code.zip"
New-Item -ItemType Directory -Force C:\runs\d190 | Out-Null
Start-Transcript -Path C:\runs\d190\run.log -Append
try {
    if (-not $bucket) { throw "VLA_BUCKET not set" }
    Read-S3Object -BucketName $bucket -Key $code -File C:\runs\d190\code.zip -Region us-east-1 | Out-Null
    Expand-Archive -Path C:\runs\d190\code.zip -DestinationPath C:\win_bundle\vla -Force
    $server = "--progress --goal-memory --memory-deadband 1.0,5"
    $start = "d188/dagger_r1_adapter"
    "round-1 rows present: " + (Test-Path C:\runs\d188_dagger_r1\dagger_rows.jsonl)
    "=== 1 before (validation check set) $(Get-Date)"
    & C:\win_bundle\vla\scripts\win_eval\eval_variants.ps1 -NoShutdown -CodeZipKey $code -TaskZipKey "d190/check_tasks.zip" `
        -Tag d190_before -Variants "before|$start|$server"
    "=== 2 DAgger round 2 (training tasks) $(Get-Date)"
    & C:\win_bundle\vla\scripts\rl\rl_loop.ps1 -Method dagger -Rounds 1 -PerKind 0 -TaskZipKey "d190/dagger_tasks.zip" `
        -DaggerUpdates 300 -AdapterKey $start -ServerArgs $server -Tag d190_dagger -CodeZipKey $code `
        -PriorRows "C:\runs\d188_dagger_r1\dagger_rows.jsonl" -NoShutdown
    $dagger = "C:\runs\d190_dagger_r1\update\adapter_s300"
    if (Test-Path "$dagger\adapter_config.json") {
        foreach ($f in Get-ChildItem $dagger -File) {
            Write-S3Object -BucketName $bucket -Key "d190/dagger_r2_adapter/$($f.Name)" -File $f.FullName -Region us-east-1
        }
        "=== 3 after (validation check set) $(Get-Date)"
        & C:\win_bundle\vla\scripts\win_eval\eval_variants.ps1 -NoShutdown -CodeZipKey $code -TaskZipKey "d190/check_tasks.zip" `
            -Tag d190_after -Variants "after|$dagger|$server"
    } else { "NO DAGGER ADAPTER - skipping the after test" }
    "=== D190_DONE $(Get-Date)"
} catch {
    "WRAPPER ERROR: $_"
} finally {
    try { Stop-Transcript | Out-Null } catch {}
    try { Write-S3Object -BucketName $bucket -Key "d190/run.log" -File C:\runs\d190\run.log -Region us-east-1 } catch {}
    shutdown /a 2>$null; shutdown /s /f /t 5 /c "D190 done"
}
