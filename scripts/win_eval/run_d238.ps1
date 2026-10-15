param([Parameter(Mandatory=$true)][string]$Bucket,[Parameter(Mandatory=$true)][string]$DeadlineUtc,[string]$Root='C:/runs/d238_two_task',[string]$Prefix='d238')
$ErrorActionPreference='Stop'
$code="$Root/code";$results="$Root/results";$deadline=[DateTime]::Parse($DeadlineUtc).ToUniversalTime()
$env:VLA_BUCKET=$Bucket;$env:VLA_DEADLINE_UTC=$DeadlineUtc;$env:USE_TF='0';$env:HF_HUB_OFFLINE='1';$env:TRANSFORMERS_OFFLINE='1';$env:QWEN_MODEL='C:/models/qwen3vl4b';$env:PYTHONPATH="$code/scripts;$code/scripts/rl"
try {
 if(Test-Path "$Root/dispatched.json"){throw 'Already dispatched; no duplicate or retry'}
 if(Get-Process python,Collection,Collection-Win64-Shipping -ErrorAction SilentlyContinue){throw 'Existing worker; refusing launch'}
 @{deadline_utc=$DeadlineUtc;dispatched_utc=[DateTime]::UtcNow.ToString('o');training_update_cap=4}|ConvertTo-Json|Set-Content "$Root/dispatched.json"
 $seconds=[int]($deadline-[DateTime]::UtcNow).TotalSeconds-90
 if($seconds -lt 3600){throw 'Insufficient deadline remaining'}
 if((Get-PSDrive C).Free -lt 60GB){throw 'Less than60GB free for atomic checkpoints and flight evidence'}
 $ErrorActionPreference='Continue'
 & C:/venvs/qwen/Scripts/python.exe -u "$code/scripts/rl/d238_job.py" --root $results --wall-seconds $seconds *> "$Root/driver.log"
 $exitCode=$LASTEXITCODE;$ErrorActionPreference='Stop'
 if($exitCode -ne 0){throw "D238 driver exited $exitCode"}
} catch {$_|Out-String|Set-Content "$Root/failure.txt"}
finally {
 try {
  $evidence="$Root/evidence";New-Item -ItemType Directory $evidence -Force|Out-Null
  if(Test-Path $results){
   foreach($file in Get-ChildItem $results -File -Recurse){
    $relative=$file.FullName.Substring($results.Length+1)
    if($relative -match '(^|\\)(checkpoints|batch)(\\|$)' -or $file.Name -eq 'admitted_rollouts.json'){continue}
    $dest=Join-Path $evidence $relative;New-Item -ItemType Directory ([IO.Path]::GetDirectoryName($dest)) -Force|Out-Null;Copy-Item -LiteralPath $file.FullName -Destination $dest
   }
  }
  foreach($name in @('driver.log','failure.txt','dispatched.json')){if(Test-Path "$Root/$name"){Copy-Item "$Root/$name" $evidence}}
  foreach($file in Get-ChildItem $evidence -File){Write-S3Object -BucketName $Bucket -Key "$Prefix/$($file.Name)" -File $file.FullName -Region us-east-1}
  Compress-Archive -Path "$evidence/*" -DestinationPath "$Root/evidence.zip"
  Write-S3Object -BucketName $Bucket -Key "$Prefix/evidence.zip" -File "$Root/evidence.zip" -Region us-east-1
  foreach($pointerFile in Get-ChildItem $results -Filter CURRENT.json -Recurse){
   $parent=$pointerFile.Directory.FullName;$pointer=Get-Content $pointerFile.FullName -Raw|ConvertFrom-Json
   $checkpoint=Join-Path $parent $pointer.directory
   if([IO.Path]::GetFullPath($checkpoint).StartsWith($parent+[IO.Path]::DirectorySeparatorChar) -eq $false){throw 'checkpoint escapes parent'}
   $keyRoot=$parent.Substring($results.Length+1).Replace('\','/')
   foreach($file in Get-ChildItem $checkpoint -File -Recurse){$rel=$file.FullName.Substring($checkpoint.Length+1).Replace('\','/');Write-S3Object -BucketName $Bucket -Key "$Prefix/$keyRoot/$($pointer.directory)/$rel" -File $file.FullName -Region us-east-1}
   Write-S3Object -BucketName $Bucket -Key "$Prefix/$keyRoot/CURRENT.json" -File $pointerFile.FullName -Region us-east-1
  }
  'uploaded'|Set-Content "$Root/uploaded.txt";Write-S3Object -BucketName $Bucket -Key "$Prefix/uploaded.txt" -File "$Root/uploaded.txt" -Region us-east-1
 } catch {$_|Out-String|Set-Content "$Root/upload_failure.txt"}
 $ErrorActionPreference='Continue'
 & shutdown.exe /a 2>$null
 & shutdown.exe /s /f /t 10 /c 'D238 final outcome; upload attempted'
 if($LASTEXITCODE -ne 0){'Shutdown failed; monitor must stop EC2'|Set-Content "$Root/shutdown_failure.txt"}
}
