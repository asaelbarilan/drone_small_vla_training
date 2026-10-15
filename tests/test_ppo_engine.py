import sys,unittest,tempfile
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rl'))
from ppo_engine import train_batch,PPOConfig
from checkpoint_state import verified_checkpoint

class ToyActor(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layer=torch.nn.Linear(2,2)
    def save_pretrained(self,path):
        path.mkdir();torch.save(self.state_dict(),path/'weights.pt')

class EngineTests(unittest.TestCase):
    def make(self):
        torch.manual_seed(7);actor=ToyActor();critic=torch.nn.Linear(2,1)
        torch.nn.init.zeros_(critic.weight);torch.nn.init.zeros_(critic.bias)
        optimizer=torch.optim.AdamW([{'params':actor.parameters(),'lr':.001},{'params':critic.parameters(),'lr':.01}])
        def forward(samples):
            x=torch.tensor([s['x'] for s in samples]);lp=actor.layer(x).log_softmax(-1)
            ids=torch.tensor([s['ids'] for s in samples]);return lp.gather(1,ids),x
        samples=[]
        for i in range(4):
            s=dict(task=str(i),passage=0,call_index=0,ids=[i%2]*4,temperature=1.,x=[1.,float(i)],
                   context_input_ids=[1],box_text='',progress_text='',reward_v2=float(i%2),
                   reward_spec_approved=True,terminated=True,truncated=False)
            s['logps']=forward([s])[0].detach()[0].tolist();samples.append(s)
        return actor,critic,optimizer,forward,samples
    def test_complete_update_checkpoint_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            actor,critic,opt,forward,samples=self.make()
            cfg=PPOConfig(chunk=1,batch_size=2,accum=1,warmup_steps=2,max_updates=2)
            report=train_batch(actor,critic,opt,samples,forward,None,Path(d)/'ckpt',config=cfg,identity={'test':1})
            self.assertEqual(report['step'],2);self.assertGreater(report['updates'][-1]['grad_norm'],0)
            expected={k:v.clone() for k,v in actor.state_dict().items()}
            folder=verified_checkpoint(Path(d)/'ckpt')
            actor.load_state_dict(torch.load(folder/'adapter/weights.pt',weights_only=True))
            resumed=train_batch(actor,critic,opt,samples,forward,None,Path(d)/'ckpt',config=cfg,resume=True,identity={'test':1})
            self.assertEqual(resumed['step'],2)
            for k,v in actor.state_dict().items():torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
    def test_failure_then_resume_matches_uninterrupted(self):
        cfg=PPOConfig(chunk=1,batch_size=2,accum=1,warmup_steps=2,max_updates=2)
        with tempfile.TemporaryDirectory() as d:
            a,c,o,f,s=self.make();train_batch(a,c,o,s,f,None,Path(d)/'full',config=cfg,identity='toy')
            expected={k:v.clone() for k,v in a.state_dict().items()}
            a,c,o,f,s=self.make()
            def failing(part):
                if (Path(d)/'interrupted/CURRENT.json').exists():
                    import json
                    folder=verified_checkpoint(Path(d)/'interrupted')
                    manifest=json.loads((folder/'manifest.json').read_text())
                    if manifest['step']==1:raise RuntimeError('injected worker failure')
                return f(part)
            with self.assertRaises(RuntimeError):train_batch(a,c,o,s,failing,None,Path(d)/'interrupted',config=cfg,identity='toy')
            folder=verified_checkpoint(Path(d)/'interrupted');a.load_state_dict(torch.load(folder/'adapter/weights.pt',weights_only=True))
            train_batch(a,c,o,s,f,None,Path(d)/'interrupted',config=cfg,resume=True,identity='toy')
            for k,v in a.state_dict().items():torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
    def test_partial_and_remainder_coverage(self):
        for batch_size,limit,expected_unique,complete in [(2,1,2,False),(3,2,4,True)]:
            with self.subTest(batch_size=batch_size), tempfile.TemporaryDirectory() as d:
                a,c,o,f,s=self.make()
                cfg=PPOConfig(chunk=1,batch_size=batch_size,accum=1,warmup_steps=2,max_updates=limit)
                report=train_batch(a,c,o,s,f,None,Path(d)/'ckpt',config=cfg,identity='coverage')
                q=report['coverage']
                self.assertEqual(q['unique_actor_samples'],expected_unique)
                self.assertEqual(q['actor_sample_visits'],expected_unique)
                self.assertEqual(q['schedule_complete'],complete)
                self.assertEqual(len(q['unvisited_sample_indices']),4-expected_unique)
                self.assertEqual(sum(q['actor_visits_per_task'].values()),expected_unique)
                folder=verified_checkpoint(Path(d)/'ckpt')
                a.load_state_dict(torch.load(folder/'adapter/weights.pt',weights_only=True))
                resumed=train_batch(a,c,o,s,f,None,Path(d)/'ckpt',config=cfg,resume=True,identity='coverage')
                self.assertEqual(resumed['coverage'],q)

    def test_planned_review_pause_resume_is_exact(self):
        cfg=PPOConfig(chunk=1,batch_size=2,accum=1,warmup_steps=2,max_updates=2)
        with tempfile.TemporaryDirectory() as d:
            a,c,o,f,s=self.make();train_batch(a,c,o,s,f,None,Path(d)/'full',config=cfg,identity='review')
            expected={k:v.clone() for k,v in a.state_dict().items()}
            a,c,o,f,s=self.make()
            paused=train_batch(a,c,o,s,f,None,Path(d)/'split',config=cfg,identity='review',pause_after_updates=1)
            self.assertEqual(paused['step'],1);self.assertEqual(paused['coverage']['stop_reason'],'intermediate_check')
            folder=verified_checkpoint(Path(d)/'split');a.load_state_dict(torch.load(folder/'adapter/weights.pt',weights_only=True))
            final=train_batch(a,c,o,s,f,None,Path(d)/'split',config=cfg,identity='review',resume=True,pause_after_updates=2)
            self.assertTrue(final['coverage']['schedule_complete'])
            for k,v in a.state_dict().items():torch.testing.assert_close(v,expected[k],rtol=0,atol=0)

    def test_budget_and_gamma_rejection(self):
        with self.assertRaises(ValueError):PPOConfig(gamma=.99)
        with tempfile.TemporaryDirectory() as d:
            a,c,o,f,s=self.make()
            with self.assertRaises(TimeoutError):train_batch(a,c,o,s,f,None,d,config=PPOConfig(chunk=1,max_seconds=1e-12))

if __name__=='__main__':unittest.main()
