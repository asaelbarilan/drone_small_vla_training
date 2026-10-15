import sys,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rl'))
from ppo_engine import train_batch,PPOConfig,write_kl_failure
import test_ppo_engine

class KLDiagnostics(unittest.TestCase):
 def test_pre_step_guard_persists_and_does_not_step(self):
  for delta in (0.2,float('nan'),float('inf')):
   with self.subTest(delta=delta),tempfile.TemporaryDirectory() as d:
    root=Path(d)/'checkpoints';a,c,o,f,s=test_ppo_engine.EngineTests().make()
    initial={k:v.clone() for k,v in a.state_dict().items()}
    def failing(part):
     lp,x=f(part)
     if (root/'CURRENT.json').exists():lp=lp+delta
     return lp,x
    with patch.object(o,'step',wraps=o.step) as step:
     with self.assertRaises(ValueError):
      train_batch(a,c,o,s,failing,None,root,config=PPOConfig(chunk=1,batch_size=1,accum=1,warmup_steps=0,max_updates=1))
     self.assertEqual(step.call_count,0)
    report=json.loads((Path(d)/'kl_failure.json').read_text())
    self.assertEqual(report['last_committed_step'],0)
    self.assertEqual(report['phase'],'pre_update')
    self.assertEqual(report['reason'],'threshold_exceeded' if delta==.2 else 'nonfinite')
    self.assertEqual(len(report['rows'][0]['old_logps']),4)
    self.assertIsNotNone(report['committed_checkpoint'])
    for k,v in a.state_dict().items():self.assertTrue(torch.equal(v,initial[k]))
 def test_post_update_guard_has_full_batch_report(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)/'checkpoints';a,c,o,f,s=test_ppo_engine.EngineTests().make();updated=False
   original=o.step
   def step(*args,**kwargs):
    nonlocal updated
    result=original(*args,**kwargs);updated=True;return result
   def forward(part):
    lp,x=f(part);return (lp+.2 if updated else lp),x
   with patch.object(o,'step',side_effect=step):
    with self.assertRaisesRegex(ValueError,'full-batch KL limit'):
     train_batch(a,c,o,s,forward,None,root,config=PPOConfig(chunk=1,batch_size=1,accum=1,warmup_steps=0,max_updates=1))
   q=json.loads((Path(d)/'kl_failure.json').read_text())
   self.assertEqual(q['aggregation'],'full_batch_mean');self.assertEqual(len(q['rows']),4)
   self.assertIn('step_000000_',q['committed_checkpoint']['directory'])
 def test_low_average_outlier_no_longer_vetoes_next_update(self):
  # The second shuffled sample changes much more than the batch average.
  import random
  order=list(range(4));random.Random(7).shuffle(order);outlier=order[1]
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)/'checkpoints';a,c,o,f,s=test_ppo_engine.EngineTests().make();updated=False
   original=o.step
   def step(*args,**kwargs):
    nonlocal updated
    result=original(*args,**kwargs);updated=True;return result
   def forward(part):
    lp,x=f(part)
    if updated:
     shift=torch.tensor([.1 if int(v['task'])==outlier else 0. for v in part])[:,None]
     lp=lp+shift
    return lp,x
   with patch.object(o,'step',side_effect=step) as opt_step:
    report=train_batch(a,c,o,s,forward,None,root,config=PPOConfig(chunk=1,batch_size=1,accum=1,warmup_steps=0,max_updates=2))
    self.assertEqual(opt_step.call_count,2)
    self.assertEqual(report['step'],2)
   from checkpoint_state import verified_checkpoint
   state=torch.load(verified_checkpoint(root)/'state.pt',weights_only=False)
   logged=state['metadata']['logs'][-1]
   self.assertLess(logged['post_kl'],.05)
   self.assertGreater(logged['post_kl_max'],.05)
   self.assertEqual(logged['post_kl_worst_sample'],outlier)
   self.assertEqual(logged['post_kl_samples_over_limit'],1)
   self.assertFalse((Path(d)/'kl_failure.json').exists())
   self.assertEqual(state['metadata']['kl_policy'],'full_batch_mean_v2')

 def test_nonfinite_during_gradient_pass_never_steps(self):
  with tempfile.TemporaryDirectory() as d:
   a,c,o,f,s=self.fixture();root=Path(d)/'checkpoints'
   def forward(part):
    lp,x=f(part)
    if torch.is_grad_enabled() and (root/'CURRENT.json').exists():lp=lp*float('nan')
    return lp,x
   with patch.object(o,'step',wraps=o.step) as step:
    with self.assertRaises(ValueError):train_batch(a,c,o,s,forward,None,root,config=PPOConfig(chunk=1,warmup_steps=0,max_updates=1))
    self.assertEqual(step.call_count,0)
   self.assertEqual(json.loads((Path(d)/'kl_failure.json').read_text())['phase'],'pre_step')

 def fixture(self):return test_ppo_engine.EngineTests().make()

 def test_resume_adds_exactly_three_updates_and_matches_uninterrupted(self):
  from checkpoint_state import verified_checkpoint
  cfg=PPOConfig(chunk=1,batch_size=1,accum=1,warmup_steps=2,max_updates=4)
  with tempfile.TemporaryDirectory() as d:
   a,c,o,f,s=self.fixture();train_batch(a,c,o,s,f,None,Path(d)/'full',config=cfg)
   expected={k:v.clone() for k,v in a.state_dict().items()}
   a,c,o,f,s=self.fixture();first=train_batch(a,c,o,s,f,None,Path(d)/'split',config=cfg,pause_after_updates=1)
   self.assertEqual(first['step'],1)
   saved=verified_checkpoint(Path(d)/'split');a.load_state_dict(torch.load(saved/'adapter/weights.pt',weights_only=True))
   with patch.object(o,'step',wraps=o.step) as step:
    last=train_batch(a,c,o,s,f,None,Path(d)/'split',config=cfg,resume=True,pause_after_updates=4)
    self.assertEqual(step.call_count,3)
   self.assertEqual(last['step'],4);self.assertEqual(len(last['updates']),4)
   for k,v in a.state_dict().items():torch.testing.assert_close(v,expected[k],rtol=0,atol=0)

if __name__=='__main__':unittest.main()

