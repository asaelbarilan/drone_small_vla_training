import sys,unittest,tempfile,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rl'))
from bounded_job import run_stages
class JobTests(unittest.TestCase):
    def test_success_requires_real_exit_and_output(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'result.txt'
            stage=dict(name='ok',argv=[sys.executable,'-c',f'from pathlib import Path; Path({str(out)!r}).write_text("done")'],timeout_seconds=5,required_outputs=[str(out)])
            self.assertEqual(run_stages([stage],Path(d)/'run',wall_seconds=10,reserve_seconds=1)['state'],'complete')
    def test_exit7_prevents_next_stage(self):
        with tempfile.TemporaryDirectory() as d:
            stages=[dict(name='fail',argv=[sys.executable,'-c','raise SystemExit(7)'],timeout_seconds=5),
                    dict(name='never',argv=[sys.executable,'-c','print(1)'],timeout_seconds=5)]
            with self.assertRaises(RuntimeError):run_stages(stages,d,wall_seconds=10,reserve_seconds=1)
            self.assertFalse((Path(d)/'01.log').exists());self.assertFalse((Path(d)/'completion.json').exists())
            self.assertIn('exited 7',json.loads((Path(d)/'failure.json').read_text())['error'])
    def test_timeout_and_duplicate_owner(self):
        with tempfile.TemporaryDirectory() as d:
            stage=dict(name='slow',argv=[sys.executable,'-c','import time; time.sleep(10)'],timeout_seconds=.1)
            (Path(d)/'RUNNING.lock').write_text('existing')
            with self.assertRaises(FileExistsError):run_stages([stage],d,wall_seconds=10,reserve_seconds=1)
            (Path(d)/'RUNNING.lock').unlink()
            with self.assertRaises(TimeoutError):run_stages([stage],d,wall_seconds=10,reserve_seconds=1)
            self.assertFalse((Path(d)/'completion.json').exists())
    def test_finished_run_is_not_rewritten(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'failure.json';path.write_text('{"original": true}')
            stage=dict(name='unused',argv=[sys.executable,'-c','pass'],timeout_seconds=1)
            with self.assertRaises(ValueError):run_stages([stage],d,wall_seconds=10,reserve_seconds=1)
            self.assertEqual(path.read_text(),'{"original": true}')
            self.assertFalse((Path(d)/'RUNNING.lock').exists())
    def test_missing_output_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            stage=dict(name='no_output',argv=[sys.executable,'-c','pass'],timeout_seconds=5,required_outputs=[str(Path(d)/'absent')])
            with self.assertRaises(ValueError):run_stages([stage],d,wall_seconds=10,reserve_seconds=1)
if __name__=='__main__':unittest.main()
