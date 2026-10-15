import sys,json,pathlib,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts/rl'))
from movement_reward import movement_reward
from audit_movement_reward import line,variants,controls
MODE='rational_tolerant_heading'
def score(p,r,**kw):
 return movement_reward(p,r,kind='Ascend/Descend',stopped=kw.pop('stopped',True),altitude_cost_mode=MODE,**kw)
def path(r,amp=0,yaw=0):
 p=r.copy();p[:,1]+=100*amp*np.sin(np.linspace(0,np.pi,len(p)));p[:,4]+=np.linspace(0,yaw,len(p));return p
class AltitudeTradeoff(unittest.TestCase):
 def test_route_preference_with_acceptable_headings(self):
  for length in [3,9]:
   for angle in [-85,-40,30,85]:
    r=line(x=length*np.cos(np.deg2rad(angle)),z=length*np.sin(np.deg2rad(angle)))
    for yaw in [-4.9,-1,0,1,4.9]:
     a=score(path(r,yaw=yaw),r)
     for amp in [.5,2,4]:
      b=score(path(r,amp=amp),r)
      self.assertGreater(a['reward'],b['reward'],(length,angle,yaw,amp))
     self.assertGreater(score(path(r,amp=4,yaw=yaw),r)['reward'],score(line(),r)['reward'])
 def test_endpoint_credit_and_idle(self):
  for length in [.1,.5,3,9]:
   r=line(z=length)
   self.assertAlmostEqual(score(line(),r)['reward'],0,places=12)
   vals=[score(line(z=length*f),r)['terminal_credit'] for f in [0,.25,.5,.75,1]]
   self.assertTrue(all(a<=b for a,b in zip(vals,vals[1:])))
   self.assertAlmostEqual(vals[-1],1,places=12)
 def test_heading_credit_plateau_and_outside_penalty(self):
  r=line(x=4,z=3)
  vals=[score(path(r,yaw=y),r)['terminal_credit'] for y in [0,1,4.9,5,6,10,20]]
  np.testing.assert_allclose(vals[:4],np.ones(4),atol=1e-12)
  self.assertTrue(all(a>b for a,b in zip(vals[3:],vals[4:])))
  for y in [1,5,10]:self.assertAlmostEqual(score(path(r,yaw=y),r)['reward'],score(path(r,yaw=-y),r)['reward'],places=8)
  self.assertLess(abs(score(path(r,yaw=5.0001),r)['reward']-score(path(r,yaw=4.9999),r)['reward']),1e-4)
 def test_arrival_overshoot_and_loops(self):
  for sign in [-1,1]:
   r=line(x=2,z=5*sign);v=variants(r);s={k:score(p,r)['reward'] for k,p in v.items()}
   for k in ['stationary','early_stop','wrong_direction','overshoot','off_path_detour','reversal']:self.assertGreater(s['correct'],s[k])
   self.assertGreater(s['off_path_detour'],s['stationary'])
   self.assertLessEqual(score(np.vstack([r,r[-2::-1]]),r)['reward'],0)
   rr=line(z=9*sign)
   self.assertLess(score(line(z=18*sign),rr,stopped=False)['reward'],score(line(),rr,stopped=False)['reward'])
 def test_resampling_safety_and_stopping(self):
  r=line(x=4,z=3);p=path(r,amp=2,yaw=3);a=score(p,r)
  self.assertAlmostEqual(a['reward'],score(np.repeat(p,3,axis=0),r)['reward'],places=8)
  dense=np.vstack([v+(w-v)*t for v,w in zip(p,p[1:]) for t in [0,.5]]+[p[-1]])
  self.assertLess(abs(a['reward']-score(dense,r)['reward']),.01)
  self.assertEqual(score(p,r,stopped=False)['terminal_credit'],0)
  self.assertLessEqual(score(p,r,events={'collision':True})['reward'],-1)
  self.assertFalse(score(np.vstack([r[0],r[-1]]),r)['complete'])
  self.assertFalse(a['complete']) # reward is not completion certification
 def test_other_movements_unchanged(self):
  for kind,r in controls().items():
   if kind=='Ascend/Descend':continue
   for p in variants(r).values():
    args=dict(kind=kind,stopped=True,events={'landed':True})
    self.assertEqual(movement_reward(p,r,**args)['reward'],movement_reward(p,r,**args,altitude_cost_mode=MODE)['reward'])
 def test_saved_ascent_accounting(self):
  root=ROOT/'reports/d240_resume_review/evidence';vals=[]
  for side in ['before_dev','after_dev']:
   ep=json.loads((root/side/'episodes/sim_2025-03-27_16-41-53_r0.json').read_text());r=json.loads((root/'dev_tasks'/ep['task']).read_text())['reference_path_preprocessed']
   result=score(ep['poses'],r);vals.append(result['reward'])
   self.assertAlmostEqual(sum(result['transition_rewards']),result['reward'],places=12)
   self.assertFalse(result['complete'])
   self.assertAlmostEqual(movement_reward(ep['poses'],r,kind=ep['kind'],stopped=True)['reward'],ep['reward']['reward'],places=9)
  self.assertGreater(vals[0],vals[1])
 def test_prior_saved_arrival_vs_idle(self):
  r=json.loads((ROOT/'reports/d224_reward_gated_ppo/tasks/sim_2025-03-27_19-11-33.json').read_text())['reference_path_preprocessed'];vals=[]
  for i in [0,1]:
   ep=json.loads((ROOT/f'reports/d228_local_numeric_flights/finish/episodes/sim_2025-03-27_19-11-33_r{i}.json').read_text());vals.append(score(ep['poses'],r)['reward'])
  self.assertGreater(score(r,r)['reward'],vals[0]);self.assertGreater(vals[0],vals[1])
if __name__=='__main__':unittest.main()
