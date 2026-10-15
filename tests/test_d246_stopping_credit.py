"""Regression controls for D246; CPU only."""
import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/rl'))
from audit_stopping_credit import run
class StoppingCredit(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.result=run()
 def test_target_stop_and_early_continuation(self):
  rows=self.result['controls'];self.assertEqual(len(rows),40)
  for row in rows:
   with self.subTest(task=row['task'],mode=row['mode'],fraction=row['fraction']):
    at_goal=row['fraction']==1
    self.assertEqual(row['complete'],at_goal)
    self.assertEqual(row['stop_probability_after']>.5,at_goal)
 def test_saved_successful_stop_signs(self):
  self.assertEqual(len(self.result['saved_terminal_chunks']),3)
  for row in self.result['saved_terminal_chunks']:
   self.assertGreater(row['raw_gae'],0);self.assertGreater(row['scale_only'],0)
   self.assertLess(row['centered'],0)
 def test_critic_failure_not_concealed(self):
  self.assertGreater(self.result['critic_counterexample']['early_stop_scale_only_advantage'],0)
if __name__=='__main__':unittest.main()