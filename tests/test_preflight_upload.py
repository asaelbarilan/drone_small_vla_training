import json,shutil,subprocess,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

@unittest.skipUnless(shutil.which('powershell'),'Windows uploader regression')
class UploadTests(unittest.TestCase):
    def test_locked_launcher_excluded_and_other_failure_does_not_hide_results(self):
        wrapper=(ROOT/'scripts/win_eval/run_d215_preflight.ps1').read_text(encoding='utf-8')
        snippet=wrapper[wrapper.index(' $uploadErrors=@()'):wrapper.index(' # Session owner')]
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);run=folder/'run';run.mkdir()
            for name in ('bad.txt','good.json','launcher.err'):(run/name).write_text('fixture')
            script=folder/'check.ps1'
            preamble=r"""param([string]$root)
$ErrorActionPreference='Stop';$code="$root\code";$bucket='mock';$prefix='d215/preflight';$seen=New-Object 'System.Collections.Generic.List[string]'
function Write-S3Object {
 [CmdletBinding()] param($BucketName,$Key,$File,$Region)
 if ([IO.Path]::GetFileName($File) -eq 'bad.txt') {throw 'injected upload failure'}
 $script:seen.Add($Key)
}
$handle=[IO.File]::Open("$root\launcher.err",'Open','ReadWrite','None')
try {
"""
            script.write_text(preamble+snippet+r"""
} finally {$handle.Dispose()}
@{uploaded=@($seen);error_saved=(Test-Path "$root\upload_failure.txt")} | ConvertTo-Json -Compress
""",encoding='utf-8')
            result=subprocess.run(['powershell','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',str(script),str(run)],capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)
            data=json.loads(result.stdout)
            self.assertTrue(data['error_saved'])
            self.assertEqual(data['uploaded'],['d215/preflight/good.json'])

if __name__=='__main__':unittest.main()
