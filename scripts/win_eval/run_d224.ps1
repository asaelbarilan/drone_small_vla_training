param([Parameter(Mandatory=$true)][string]$Bucket,[Parameter(Mandatory=$true)][string]$DeadlineUtc,[string]$Root='C:/runs/d224_ppo',[string]$Prefix='d224')
$ErrorActionPreference='Stop'
$root=$Root;$code="$root/code";$result="$root/results"
$deadline=[DateTime]::Parse($DeadlineUtc).ToUniversalTime()
$env:VLA_DEADLINE_UTC=$DeadlineUtc;$env:VLA_BUCKET=$Bucket;$env:USE_TF='0';$env:HF_HUB_OFFLINE='1';$env:TRANSFORMERS_OFFLINE='1';$env:QWEN_MODEL='C:/models/qwen3vl4b';$env:PYTHONPATH="$code/scripts;$code/scripts/rl"
New-Item -ItemType Directory -Path $result -Force|Out-Null
try {
    if(Test-Path "$root/dispatched.json"){throw 'D224 already dispatched; no retry'}
    @{dispatched_utc=[DateTime]::UtcNow.ToString('o');deadline_utc=$DeadlineUtc}|ConvertTo-Json|Set-Content "$root/dispatched.json"
    if(Get-Process python,Collection,Collection-Win64-Shipping -ErrorAction SilentlyContinue){throw 'Existing worker or simulator; refusing duplicate launch'}
    $seconds=[int]($deadline-[DateTime]::UtcNow).TotalSeconds-90
    if($seconds -lt 3600){throw 'Insufficient session budget remaining before launch'}
    $ErrorActionPreference='Continue'
    & C:/venvs/qwen/Scripts/python.exe -u "$code/scripts/rl/reward_gated_job.py" --root $result --wall-seconds $seconds *> "$root/driver.log"
    $nativeExit=$LASTEXITCODE
    $ErrorActionPreference='Stop'
    if($nativeExit -ne 0){throw "Pilot driver exited $nativeExit"}
} catch {$_|Out-String|Set-Content "$root/failure.txt"}
finally {
    try {
        # Closed stage files first; exclude checkpoint directories from compact archive.
        $evidence="$root/evidence";New-Item -ItemType Directory $evidence -Force|Out-Null
        Get-ChildItem $result -File|Where-Object {$_.Name -ne 'admitted_rollouts.json'}|Copy-Item -Destination $evidence
        foreach($name in @('attempts','after')){if(Test-Path "$result/$name"){Copy-Item "$result/$name" "$evidence/$name" -Recurse}}
        foreach($name in @('driver.log','failure.txt','dispatched.json')){if(Test-Path "$root/$name"){Copy-Item "$root/$name" $evidence}}
        if(Test-Path "$result/training/report.json"){Copy-Item "$result/training/report.json" "$evidence/training_report.json"}
        foreach($file in Get-ChildItem $evidence -File){Write-S3Object -BucketName $Bucket -Key "$Prefix/$($file.Name)" -File $file.FullName -Region us-east-1}
        Compress-Archive -Path "$evidence/*" -DestinationPath "$root/evidence.zip"
        Write-S3Object -BucketName $Bucket -Key "$Prefix/evidence.zip" -File "$root/evidence.zip" -Region us-east-1
        if(Test-Path "$result/training/checkpoints/CURRENT.json"){
            $pointer=Get-Content "$result/training/checkpoints/CURRENT.json" -Raw|ConvertFrom-Json
            $checkpoint="$result/training/checkpoints/$($pointer.directory)"
            Write-S3Object -BucketName $Bucket -Key "$Prefix/checkpoints/CURRENT.json" -File "$result/training/checkpoints/CURRENT.json" -Region us-east-1
            foreach($file in Get-ChildItem $checkpoint -File -Recurse){
                $relative=$file.FullName.Substring($checkpoint.Length+1).Replace('\','/')
                Write-S3Object -BucketName $Bucket -Key "$Prefix/checkpoints/$($pointer.directory)/$relative" -File $file.FullName -Region us-east-1
            }
        }
        'uploaded'|Set-Content "$root/uploaded.txt"
        Write-S3Object -BucketName $Bucket -Key "$Prefix/uploaded.txt" -File "$root/uploaded.txt" -Region us-east-1
    } catch {$_|Out-String|Set-Content "$root/upload_failure.txt"}
    # A scheduled backup blocks Stop-Computer; replace it only at final outcome.
    $ErrorActionPreference='Continue'
    & shutdown.exe /a 2>$null
    & shutdown.exe /s /f /t 10 /c 'D224 final outcome: artifacts upload attempted'
    if($LASTEXITCODE -ne 0){'Immediate shutdown failed; EC2 monitor must stop instance'|Set-Content "$root/shutdown_failure.txt"}
}
