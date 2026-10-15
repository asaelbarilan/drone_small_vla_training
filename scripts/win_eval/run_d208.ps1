# D208: user-authorized step8000 validation, same protocol as D206.
$ErrorActionPreference = "Stop"
if (-not $env:VLA_BUCKET) { throw "VLA_BUCKET must be set" }
$root = "C:\runs\d208"
foreach ($child in @("tasks", "tasks_all")) {
    $resolved = [IO.Path]::GetFullPath((Join-Path $root $child))
    if (-not $resolved.StartsWith($root + "\", [StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe task path" }
}
New-Item -ItemType Directory -Force $root | Out-Null
try {
    & C:\win_bundle\vla\scripts\win_eval\eval_variants.ps1 -NoShutdown -Tag d208 -CodeZipKey d204/vla_code.zip -TaskZipKey d190/check_tasks.zip -HardStopMinutes 420 -Variants "s8000|d207/stage2/adapter_s8000|--progress --goal-memory --memory-deadband 1.0,5 --box"
    $tasks = @(Get-ChildItem "$root\tasks" -Filter *.json).Count
    $done = @(Get-ChildItem "$root\s8000\flights" -Filter *_2d.png).Count
    if ($tasks -ne 100 -or $done -ne 100) { throw "Incomplete validation: $done / $tasks" }
    @{status="complete"; tasks=$tasks; completed=$done; adapter="d207/stage2/adapter_s8000"; finished_utc=[DateTime]::UtcNow.ToString("o")} | ConvertTo-Json | Set-Content "$root\completion.json"
    Write-S3Object -BucketName $env:VLA_BUCKET -Key "d208/completion.json" -File "$root\completion.json" -Region us-east-1 -ErrorAction Stop
} catch {
    $_ | Out-String | Set-Content "$root\failure.txt"
    try { Write-S3Object -BucketName $env:VLA_BUCKET -Key "d208/failure.txt" -File "$root\failure.txt" -Region us-east-1 -ErrorAction Stop } catch {}
} finally {
    shutdown /a 2>$null
    shutdown /s /f /t 10 /c "D208 evaluation finished"
}
