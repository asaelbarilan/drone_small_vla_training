import sys,json,pathlib,unittest,tempfile
import numpy as np
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts/rl'))
from movement_reward import movement_reward
from audit_movement_reward import line,variants,controls
from ppo_engine import prepare_advantages,PPOConfig,train_batch
from ppo_math import joint_ppo_loss
import test_ppo_engine
class LocalCandidates(unittest.TestCase):
 def test_arrival_controls(self):
  for z in [-5,5]:
   ref=line(x=1.3,z=z);v=variants(ref);score=lambda p:movement_reward(p,ref,kind='Ascend/Descend',stopped=True,altitude_cost_mode='rational')['reward']
   scores={k:score(p) for k,p in v.items()}
   for k in ['stationary','early_stop','wrong_direction','overshoot','off_path_detour','reversal']:self.assertGreater(scores['correct'],scores[k])
   self.assertGreater(scores['off_path_detour'],scores['stationary'])
   self.assertLessEqual(score(np.vstack([ref,ref[-2::-1]])),0)
   self.assertAlmostEqual(score(np.repeat(v['off_path_detour'],3,axis=0)),scores['off_path_detour'],places=8)
   self.assertLessEqual(movement_reward(v['off_path_detour'],ref,kind='Ascend/Descend',stopped=True,events={'collision':True},altitude_cost_mode='rational')['reward'],-1)
 def test_matched_endpoint_detours(self):
  ref=line(x=4,z=3);scores=[]
  for amp in [0,.5,1,2,4]:
   p=ref.copy();p[:,1]+=100*amp*np.sin(np.linspace(0,np.pi,len(p)))
   scores.append(movement_reward(p,ref,kind='Ascend/Descend',stopped=True,altitude_cost_mode='rational')['reward'])
  self.assertTrue(all(a>b for a,b in zip(scores,scores[1:])))
 def test_other_kinds_unchanged(self):
  for kind,r in controls().items():
   if kind=='Ascend/Descend':continue
   for p in variants(r).values():
    args=dict(kind=kind,stopped=True,events={'landed':True})
    self.assertEqual(movement_reward(p,r,**args)['reward'],movement_reward(p,r,**args,altitude_cost_mode='rational')['reward'])
 def test_signed_signal_and_gradient(self):
  for raw in ([.1,.2,2.],[-.1,-.2,-2.],[.1,-.2,2.]):
   adv=prepare_advantages(raw,'scale_only');self.assertTrue(np.array_equal(np.sign(adv),np.sign(raw)))
   new=torch.zeros((3,4),requires_grad=True);loss,_=joint_ppo_loss(new,torch.zeros_like(new),torch.tensor(adv,dtype=torch.float32));loss.backward()
   self.assertTrue(np.array_equal(np.sign(-new.grad[:,0].numpy()),np.sign(raw)))
 def test_bad_critic_not_hidden(self):
  # A genuinely negative raw estimate stays negative; no success-based override.
  self.assertLess(prepare_advantages([-.4,.1,.2],'scale_only')[0],0)
 def test_scale_only_engine_and_resume(self):
  with tempfile.TemporaryDirectory() as d:
   a,c,o,f,s=test_ppo_engine.EngineTests().make();cfg=PPOConfig(chunk=1,batch_size=2,accum=1,warmup_steps=2,max_updates=2,advantage_mode='scale_only')
   r=train_batch(a,c,o,s,f,None,pathlib.Path(d)/'c',config=cfg,identity='test');self.assertEqual(r['step'],2)
   expected={k:v.clone() for k,v in a.state_dict().items()}
   a2,c2,o2,f2,s2=test_ppo_engine.EngineTests().make()
   split=pathlib.Path(d)/'split'
   train_batch(a2,c2,o2,s2,f2,None,split,config=cfg,identity='test',pause_after_updates=1)
   from checkpoint_state import verified_checkpoint
   a2.load_state_dict(torch.load(verified_checkpoint(split)/'adapter/weights.pt',weights_only=True))
   train_batch(a2,c2,o2,s2,f2,None,split,config=cfg,identity='test',resume=True)
   for k,v in a2.state_dict().items():torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
   with self.assertRaises(ValueError):train_batch(a,c,o,s,f,None,pathlib.Path(d)/'c',config=PPOConfig(chunk=1,batch_size=2,accum=1,warmup_steps=2,max_updates=2),identity='test',resume=True)
if __name__=='__main__':unittest.main()
