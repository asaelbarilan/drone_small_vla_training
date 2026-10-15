import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts/rl'))
from movement_reward import movement_reward
from audit_movement_reward import line
from audit_accepted_goal_stopping import run

def score(p,r,**kw):return movement_reward(p,r,kind=kw.pop('kind','Ascend/Descend'),stopped=kw.pop('stopped',True),stopping_mode='accepted_goal',**kw)
class AcceptedGoalTests(unittest.TestCase):
 def test_nine_execution_branch_pairs(self):
  self.assertEqual(len(run()),9)
 def test_position_and_heading_both_required(self):
  r=line(x=3,z=2,n=20)
  for f,yaw,expected in [(.8,0,False),(1,5.01,False),(.95,4.99,True)]:
   p=line(x=3*f,z=2*f,yaw=yaw,n=20);q=score(p,r)
   self.assertEqual(q['complete'],expected);self.assertEqual(q['terminal_credit'],1 if expected else -1)
 def test_both_turn_directions(self):
  for sign in [-1,1]:
   r=line(yaw=30*sign,n=10);a=line(yaw=28*sign,n=10)
   b=np.vstack([a,[0,0,0,0,30*sign,0]])
   self.assertGreater(score(a,r,kind='Turn')['reward'],score(b,r,kind='Turn')['reward'])
   self.assertEqual(score(line(yaw=26*sign,n=10),r,kind='Turn')['terminal_credit'],-1)
 def test_both_altitude_directions(self):
  for sign in [-1,1]:
   r=line(x=3,z=2*sign,n=20);a=line(x=2.8,z=1.9*sign,n=20)
   b=np.vstack([a,r[-1]])
   self.assertTrue(score(a,r)['complete']);self.assertGreater(score(a,r)['reward'],score(b,r)['reward'])
 def test_departure_and_reentry_never_farm_credit(self):
  r=line(x=3,z=2,n=20);a=line(x=2.8,z=1.9,n=20)
  outside=a[-1].copy();outside[0]-=80
  away=np.vstack([a,outside]);back=np.vstack([away,a[-1]])
  self.assertEqual(score(away,r)['terminal_credit'],-1)
  self.assertGreater(score(a,r)['reward'],score(back,r)['reward'])
 def test_duplicates_and_hold_do_not_earn_extra_credit(self):
  r=line(x=3,z=2,n=20);a=line(x=2.8,z=1.9,n=20)
  base=score(a,r)['reward']
  self.assertAlmostEqual(score(np.repeat(a,3,axis=0),r)['reward'],base,places=9)
  self.assertAlmostEqual(score(np.vstack([a,np.repeat(a[-1:],20,axis=0)]),r)['reward'],base,places=9)
 def test_horizon_not_rewarded_as_stop(self):
  r=line(x=3,z=2,n=20);q=score(r,r,stopped=False)
  self.assertEqual(q['terminal_credit'],0);self.assertFalse(q['complete'])
  self.assertAlmostEqual(score(r,r)['reward']-q['reward'],1)
 def test_collision_invalid_jump_and_landing(self):
  r=line(x=3,z=2,n=20)
  for p,kw in [(r,dict(events={'collision':True})),(r[[0,-1]],{})]:
   q=score(p,r,**kw);self.assertFalse(q['complete']);self.assertLessEqual(q['reward'],-1)
  with self.assertRaises(ValueError):score(r,r,kind='Land')
 def test_additive_transition_accounting(self):
  r=line(x=3,z=2,n=20)
  for p in [r,r[:10],np.vstack([r,r[-2::-1],r])]:
   q=score(p,r);self.assertAlmostEqual(sum(q['transition_rewards']),q['reward']);self.assertEqual(len(q['transition_rewards']),len(p)-1)
 def test_default_is_historical(self):
  r=line(x=3,z=2,n=20)
  self.assertEqual(movement_reward(r,r,kind='Ascend/Descend',stopped=True),movement_reward(r,r,kind='Ascend/Descend',stopped=True,stopping_mode='continuous'))
if __name__=='__main__':unittest.main()