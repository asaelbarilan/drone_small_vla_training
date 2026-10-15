import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts/rl'))
from audit_critic_stopping import remaining_returns,run
from ppo_math import gae
class CriticStopping(unittest.TestCase):
 def calls(self,truncated=False):
  return [dict(reward_v2=r,terminated=(i==2 and not truncated),truncated=(i==2 and truncated)) for i,r in enumerate([.2,.3,1.])]
 def test_terminal_returns_have_no_bootstrap(self):
  _,mc=remaining_returns(self.calls());np.testing.assert_allclose(mc,[1.5,1.3,1.])
 def test_timeout_not_misreported_as_measured_return(self):
  observed,mc=remaining_returns(self.calls(True));np.testing.assert_allclose(observed,[1.5,1.3,1.]);self.assertEqual(mc,[None]*3)
 def test_interior_boundary_rejected(self):
  calls=self.calls();calls[0]['terminated']=True
  with self.assertRaises(ValueError):remaining_returns(calls)
 def test_multicall_hold_requires_intermediate_value(self):
  a,_=gae([0,.5],[.4,0],[0,0],[False,True],[False,False],1,.95)
  b,_=gae([0,.5],[.4,1],[1,0],[False,True],[False,False],1,.95)
  self.assertAlmostEqual(a[0],.075);self.assertAlmostEqual(b[0],.125)
  self.assertNotAlmostEqual(a[0],.5-.4)
 def test_lambda_one_matches_terminal_mc_and_timeout_bootstrap(self):
  vals=np.array([.8,.6,.4]);rewards=[.2,.3,1.]
  for truncated in [False,True]:
   bootstrap=.7 if truncated else 0
   a,_=gae(rewards,vals,[.6,.4,bootstrap],[False,False,not truncated],[False,False,truncated],1,1)
   np.testing.assert_allclose(a,np.array([1.5,1.3,1.])+bootstrap-vals)
 def test_saved_audit_does_not_claim_critic_repaired(self):
  result=run();self.assertEqual(result['counts']['states'],32)
  self.assertEqual(result['counts']['censored_states'],16)
  self.assertFalse(result['verdict']['paid_training_ready'])
  self.assertFalse(result['verdict']['critic_calibration_established'])
  self.assertEqual(len(result['positive_incomplete_hold']),1)
  self.assertEqual(len(result['negative_complete_hold']),1)
if __name__=='__main__':unittest.main()