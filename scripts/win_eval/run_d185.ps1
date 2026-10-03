# D185 check of DAgger and PPO on the Windows box (user-approved 2026-10-03, about $0.60):
# one tiny round each on 4 Land/Pass practice tasks (2 per type), best server setup, then off.
# 3 h shutdown timer first; the loops run with -NoShutdown and this wrapper turns the box off.
$ErrorActionPreference = "Continue"
shutdown /s /t 10800 /c "D185 check hard stop (3 h)"
$bucket = $env:VLA_BUCKET  # your S3 bucket (see .env.example)
if (-not $bucket) { throw "set the VLA_BUCKET environment variable to your S3 bucket" }
New-Item -ItemType Directory -Force C:\runs\d185 | Out-Null
Start-Transcript -Path C:\runs\d185\run.log -Append
try {
    Read-S3Object -BucketName $bucket -Key "d180/vla_code.zip" -File C:\runs\d185\code.zip -Region us-east-1 | Out-Null
    Expand-Archive -Path C:\runs\d185\code.zip -DestinationPath C:\win_bundle\vla -Force
    $server = "--progress --goal-memory --memory-deadband 1.0,5"
    "=== DAgger check $(Get-Date)"
    & C:\win_bundle\vla\scripts\rl\rl_loop.ps1 -Method dagger -Rounds 1 -PerKind 2 -Kinds "Land,Pass" `
        -DaggerUpdates 20 -AdapterKey d171/run/adapter_s31000 -ServerArgs $server -Tag dagger_check -NoShutdown
    "=== PPO check $(Get-Date)"
    & C:\win_bundle\vla\scripts\rl\rl_loop.ps1 -Method ppo -Rounds 1 -Passes 2 -Temperature 1.3 -PerKind 2 -Kinds "Land,Pass" `
        -AdapterKey d171/run/adapter_s31000 -ServerArgs $server -TrainArgs "--progress" -Tag ppo_check -NoShutdown
    "=== CHECKS_DONE $(Get-Date)"
} catch {
    "WRAPPER ERROR: $_"
} finally {
    try { Stop-Transcript | Out-Null } catch {}
    try { Write-S3Object -BucketName $bucket -Key "d185/run.log" -File C:\runs\d185\run.log -Region us-east-1 } catch {}
    shutdown /a 2>$null; shutdown /s /f /t 5 /c "D185 checks done"
}
