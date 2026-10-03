# D180 (user-approved 2026-10-03): flight test of three variants, then the RL trial round,
# on one start of the Windows box. An 8 h shutdown timer is set first (the inner scripts'
# own timers then fail harmlessly: Windows keeps the first scheduled shutdown); the RL loop
# turns the machine off at its end, and this wrapper does too if anything throws.
$ErrorActionPreference = "Continue"
shutdown /s /t 28800 /c "D180 hard stop (8 h)"
$bucket = $env:VLA_BUCKET  # your S3 bucket (see .env.example)
if (-not $bucket) { throw "set the VLA_BUCKET environment variable to your S3 bucket" }
New-Item -ItemType Directory -Force C:\runs\d180 | Out-Null
Start-Transcript -Path C:\runs\d180\run.log -Append
try {
    Read-S3Object -BucketName $bucket -Key "d180/vla_code.zip" -File C:\runs\d180\code.zip -Region us-east-1 | Out-Null
    Expand-Archive -Path C:\runs\d180\code.zip -DestinationPath C:\win_bundle\vla -Force
    "=== flight test $(Get-Date)"
    & C:\win_bundle\vla\scripts\win_eval\eval_variants.ps1 -NoShutdown -Variants (
        "s18000|d171/run/adapter_s18000|;" +
        "s31000_progress|d171/run/adapter_s31000|--progress;" +
        "s31000_goalmem|d171/run/adapter_s31000|--progress --goal-memory")
    "=== RL trial $(Get-Date)"
    & C:\win_bundle\vla\scripts\rl\rl_loop.ps1 -Rounds 1 -Passes 2 -PerKind 1 -Count 8 `
        -AdapterKey d171/run/adapter_s31000 -ServerArgs "--progress" -TrainArgs "--progress" -Tag rl_trial
} catch {
    "WRAPPER ERROR: $_"
} finally {
    # D181: transcript off before its log is uploaded (locked otherwise); abort the
    # backup timer first, Stop-Computer refuses while one is scheduled.
    try { Stop-Transcript | Out-Null } catch {}
    try { Write-S3Object -BucketName $bucket -Key "d180/run.log" -File C:\runs\d180\run.log -Region us-east-1 } catch {}
    shutdown /a 2>$null; shutdown /s /f /t 5 /c "D180 done"
}
