param([Parameter(Mandatory=$true)][string]$Bucket,[Parameter(Mandatory=$true)][string]$DeadlineUtc)
$ErrorActionPreference='Stop'
$Root='C:/runs/d236_reload_verify';$code="$Root/code";$results="$Root/results"
$env:USE_TF='0';$env:HF_HUB_OFFLINE='1';$env:TRANSFORMERS_OFFLINE='1';$env:QWEN_MODEL='C:/models/qwen3vl4b';$env:PYTHONPATH="$code/scripts;$code/scripts/rl"
try {
 if(Test-Path "$Root/dispatched.json"){throw 'Already dispatched; no duplicate'}
 if(Get-Process python,Collection,Collection-Win64-Shipping -ErrorAction SilentlyContinue){throw 'Existing worker; refusing verification'}
 @{deadline_utc=$DeadlineUtc;utc=[DateTime]::UtcNow.ToString('o');training=$false}|ConvertTo-Json|Set-Content "$Root/dispatched.json"
 $ErrorActionPreference='Continue'
 & C:/venvs/qwen/Scripts/python.exe -u "$code/scripts/rl/d236_verify.py" --root $results --deadline $DeadlineUtc *> "$Root/driver.log"
 $exitCode=$LASTEXITCODE;$ErrorActionPreference='Stop'
 if($exitCode -ne 0){throw "Verification driver exited $exitCode"}
} catch {$_|Out-String|Set-Content "$Root/failure.txt"}
finally {
 try {
  $evidence="$Root/evidence";New-Item -ItemType Directory $evidence -Force|Out-Null
  if(Test-Path $results){Copy-Item "$results/*" $evidence -Recurse}
  foreach($n in @('driver.log','failure.txt','dispatched.json')){if(Test-Path "$Root/$n"){Copy-Item "$Root/$n" $evidence}}
  foreach($file in Get-ChildItem $evidence -File){Write-S3Object -BucketName $Bucket -Key "d236/$($file.Name)" -File $file.FullName -Region us-east-1}
  if(Test-Path "$results/fixed_inputs/same_input.json"){Write-S3Object -BucketName $Bucket -Key 'd236/same_input.json' -File "$results/fixed_inputs/same_input.json" -Region us-east-1}
  Compress-Archive -Path "$evidence/*" -DestinationPath "$Root/evidence.zip"
  Write-S3Object -BucketName $Bucket -Key 'd236/evidence.zip' -File "$Root/evidence.zip" -Region us-east-1
  'uploaded'|Set-Content "$Root/uploaded.txt";Write-S3Object -BucketName $Bucket -Key 'd236/uploaded.txt' -File "$Root/uploaded.txt" -Region us-east-1
 } catch {$_|Out-String|Set-Content "$Root/upload_failure.txt"}
 $ErrorActionPreference='Continue'
 & shutdown.exe /a 2>$null
 & shutdown.exe /s /f /t 10 /c 'D236 verification finished; no training'
}
