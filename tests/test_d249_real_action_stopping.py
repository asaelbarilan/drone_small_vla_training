import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts/rl'))
from audit_real_action_stopping import terminal_direction
class RealActionDirectionTests(unittest.TestCase):
 def sample(self,reward):return dict(logps=[-2.]*32,reward_v2=reward,terminated=True,truncated=False)
 def test_accepted_and_failed_stops_have_opposite_gradients(self):
  for value in [-.5,0,.5,.9]:
   self.assertGreater(terminal_direction(self.sample(.974),value)['selected_logp_descent_direction'],0)
   self.assertLess(terminal_direction(self.sample(-.999),value)['selected_logp_descent_direction'],0)
 def test_terminal_gradient_matches_reward_minus_baseline(self):
  result=terminal_direction(self.sample(.98),.8)
  self.assertAlmostEqual(result['raw_terminal_advantage'],.18)
  self.assertAlmostEqual(result['selected_logp_descent_direction'],.18)
 def test_bad_critic_can_reverse_signal_and_is_not_hidden(self):
  self.assertLess(terminal_direction(self.sample(.98),2)['selected_logp_descent_direction'],0)
  self.assertGreater(terminal_direction(self.sample(-1),-2)['selected_logp_descent_direction'],0)
 def test_timeout_cannot_be_checked_as_terminal_stop(self):
  s=self.sample(1);s.update(terminated=False,truncated=True)
  with self.assertRaises(ValueError):terminal_direction(s,0)
if __name__=='__main__':unittest.main()