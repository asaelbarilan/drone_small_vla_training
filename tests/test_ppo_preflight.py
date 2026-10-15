"""Offline counterexamples and analytic PPO tests; no model downloads/flights."""
import ast
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/rl'))
from reward_v2 import OrbitSpec, orbit_diagnostic, fit_reference_orbit
from ppo_math import gae, joint_ppo_loss, replay_parity, update_batches
from rollout_samples import ppo_advantages


def circle(radius=3, sweep=360, centre=(0,0), altitude=2, n=181):
    t=np.deg2rad(np.linspace(0,sweep,n))
    return np.column_stack([centre[0]+radius*np.cos(t),centre[1]+radius*np.sin(t),np.full(n,altitude)])

class OrbitTests(unittest.TestCase):
    def test_complete_in_both_directions_and_multiple_frames(self):
        for direction in [-1,1]:
            for radius in [1,3,5]:
                for centre in [(0,0),(-20,15)]:
                    with self.subTest(direction=direction,radius=radius,centre=centre):
                        s=OrbitSpec(centre,radius,2,direction)
                        result=orbit_diagnostic(circle(radius,360*direction,centre),s,stopped=True)
                        self.assertTrue(result['geometric_complete'])
                        self.assertFalse(result['semantic_approved'])
    def test_negative_controls(self):
        s=OrbitSpec((0,0),3,2,1)
        arc=circle(sweep=45)
        cases={
          'stationary':np.tile([3,0,2],(181,1)),
          'yaw_only':np.tile([3,0,2],(181,1)),
          'half':circle(sweep=180), 'wrong_direction':circle(sweep=-360),
          'wrong_object':circle(centre=(8,0)), 'wrong_radius':circle(radius=4),
          'wrong_altitude':circle(altitude=4), 'teleport':circle(n=5),
          'repeat_arc':np.concatenate([arc,arc[::-1]]*8),
          'centre_crossing':np.array([[3,0,2],[0,0,2],[3,0,2]]),
        }
        for name,xyz in cases.items():
            with self.subTest(name=name):self.assertFalse(orbit_diagnostic(xyz,s,stopped=True)['geometric_complete'])
        self.assertIn('not_stopped',orbit_diagnostic(circle(),s,stopped=False)['reasons'])
    def test_sweep_without_closure_fails(self):
        s=OrbitSpec((0,0),3,2,1)
        self.assertFalse(orbit_diagnostic(circle(sweep=400),s,stopped=True)['geometric_complete'])
    def test_reference_fit_is_diagnostic_only(self):
        xyz=circle(centre=(-3,0))
        ref=np.column_stack([xyz*100,np.zeros((len(xyz),3))])
        s=fit_reference_orbit(ref)
        np.testing.assert_allclose(s.centre_xy,[-3,0],atol=1e-8)
        self.assertFalse(orbit_diagnostic(xyz,s,stopped=True)['semantic_approved'])
    def test_bad_input_rejected(self):
        for xyz in [[],[[1,2,3]],[[1,2,float('nan')],[2,3,4]]]:
            with self.assertRaises(ValueError):orbit_diagnostic(xyz,OrbitSpec((0,0),3,2,1),stopped=True)
        with self.assertRaises(ValueError):OrbitSpec((0,0),3,2,0)

class PathTests(unittest.TestCase):
    def test_full_tail_matters_and_self_path_is_one(self):
        from reward_v2 import full_path_similarity
        ref=[[i*10,0,0,0,0,0] for i in range(200)]
        flight=[dict(state=[r[:3],r[3:]]) for r in ref]
        self.assertAlmostEqual(full_path_similarity(flight,ref)['similarity'],1)
        self.assertLess(full_path_similarity(flight[:100],ref)['similarity'],.5)
    def test_yaw_direction_matters(self):
        from reward_v2 import full_path_similarity
        ref=[[0,0,0,0,y,0] for y in range(0,91,5)]
        wrong=[dict(state=[[0,0,0],[0,-y,0]]) for y in range(0,91,5)]
        self.assertLess(full_path_similarity(wrong,ref)['similarity'],.5)
    def test_bound_fails_instead_of_truncating(self):
        from reward_v2 import full_path_similarity
        with self.assertRaises(ValueError):
            full_path_similarity([dict(state=[[0,0,0],[0,0,0]])],[[0,0,0,0,0,0],[10000,0,0,0,0,0]],max_points=5)

class PPOTests(unittest.TestCase):
    def test_terminal_by_hand(self):
        a,r=gae([1,2],[.5,.7],[.7,99],[False,True],[False,False],.9,.8)
        np.testing.assert_allclose(a,[2.066,1.3]);np.testing.assert_allclose(r,[2.566,2])
    def test_timeout_by_hand(self):
        a,r=gae([1,2],[.5,.7],[.7,3],[False,False],[False,True],.9,.8)
        np.testing.assert_allclose(a,[4.01,4]);np.testing.assert_allclose(r,[4.51,4.7])
    def test_reset_boundary_breaks_recurrence(self):
        a,_=gae([1,100],[0,0],[3,0],[False,True],[True,False],.9,.8)
        np.testing.assert_allclose(a,[3.7,100])
    def test_boundaries_required(self):
        for term,trunc in [([False],[False]),([True],[True]),([1],[0])]:
            with self.assertRaises(ValueError):gae([1],[0],[0],term,trunc)
    def test_joint_not_per_token_ratio(self):
        old=torch.tensor([[math.log(.5),math.log(.5)]],dtype=torch.float64)
        new=torch.tensor([[math.log(.6),math.log(.6)]],dtype=torch.float64,requires_grad=True)
        loss,ratio=joint_ppo_loss(new,old,torch.tensor([1.]),.05)
        self.assertAlmostEqual(ratio.item(),1.44);self.assertAlmostEqual(loss.item(),-1.05)
    def test_analytic_gradient_and_finite_difference(self):
        logits=torch.tensor([0.,0.],dtype=torch.float64,requires_grad=True)
        old=torch.tensor([[math.log(.5)]],dtype=torch.float64)
        loss,_=joint_ppo_loss(logits.log_softmax(-1)[:1].reshape(1,1),old,torch.tensor([1.]))
        loss.backward();self.assertAlmostEqual(logits.grad[0].item(),-.5)
        eps=1e-6
        def value(x):
            lp=torch.tensor([x,0.],dtype=torch.float64).log_softmax(-1)[:1].reshape(1,1)
            return joint_ppo_loss(lp,old,torch.tensor([1.]))[0].item()
        self.assertAlmostEqual((value(eps)-value(-eps))/(2*eps),-.5,places=6)
        with torch.no_grad():logits-=.01*logits.grad
        self.assertGreater(logits.softmax(-1)[0].item(),.5)
    def test_negative_advantage_clipping(self):
        loss,_=joint_ppo_loss(torch.tensor([[math.log(.1)]]),torch.tensor([[math.log(.5)]]),torch.tensor([-1.]))
        self.assertAlmostEqual(loss.item(),.95,places=6)
    def test_remainder_weights_and_coverage(self):
        for n in [1,17,31,32,33]:
            seen=[]
            for micros in update_batches(list(range(n)),4,8):
                self.assertAlmostEqual(sum(w for _,w in micros),1)
                for ids,w in micros:seen.extend(ids)
            self.assertEqual(seen,list(range(n)))
        with self.assertRaises(ValueError):list(update_batches([],4,8))
    def test_parity_detects_joint_drift_and_cancellation(self):
        replay_parity([[-1]*32],[[-1]*32])
        with self.assertRaises(ValueError):replay_parity([[-1]*32],[[-.999]*32])
        with self.assertRaises(ValueError):replay_parity([[-1,-1]],[[-.9,-1.1]])
    def test_legacy_gae_rejected(self):
        with self.assertRaises(ValueError):ppo_advantages([dict(task='a',passage=0,call_index=0)])
    def test_rollout_timeout_requires_final_observation(self):
        s=dict(task='a',passage=0,call_index=0,reward_v2=2,value_old=.7,
               terminated=False,truncated=True,final_value=3)
        with self.assertRaises(ValueError):ppo_advantages([s])
        s['final_observation_id']='actual_final';ppo_advantages([s],.9,.8)
        self.assertAlmostEqual(s['gae'],4)
    def test_preload_gate_rejects_missing_bootstrap_and_duplicates(self):
        from ppo_math import validate_ppo_samples
        s=dict(task='a',passage=0,call_index=0,ids=[1]*4,logps=[-1.]*4,temperature=1.,
               context_input_ids=[10],box_text='',progress_text='',reward_v2=1.,
               terminated=False,truncated=True,reward_spec_approved=True)
        with self.assertRaises(ValueError):validate_ppo_samples([s],1)
        s.update(final_observation_id='end',final_value=2.)
        self.assertEqual(validate_ppo_samples([s],1),1)
        with self.assertRaises(ValueError):validate_ppo_samples([s,s.copy()],1)
        s['reward_spec_approved']=False
        with self.assertRaises(ValueError):validate_ppo_samples([s],1)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):joint_ppo_loss(torch.tensor([[float('nan')]]),torch.zeros(1,1),torch.ones(1))

class ContextTests(unittest.TestCase):
    def test_production_encoded_preserves_prefix_and_checks_tokens(self):
        path=ROOT/'scripts/rl/train_grpo_vla.py'
        fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='encoded')
        captured={}
        def encode(processor,tokenizer,row,*args,**kwargs):
            captured.update(row)
            return dict(input_ids=torch.tensor([[10,20]]),attention_mask=torch.ones(1,2,dtype=torch.long))
        image=SimpleNamespace(open=lambda p:SimpleNamespace(convert=lambda mode:object()))
        env=dict(torch=torch,Image=image,encode=encode)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),env)
        sample=dict(passage=0,image='unused',prompt='prompt',box_text='Box: 1,2,3,4',progress_text='Left +1',ids=[90],context_input_ids=[10,20])
        batch,length=env['encoded'](None,None,sample,1,[Path('.')])
        self.assertEqual(captured['box_text'],sample['box_text']);self.assertEqual(length,2)
        self.assertEqual(batch['input_ids'].tolist(),[[10,20,90]])
        sample['context_input_ids']=[20,10]
        with self.assertRaises(ValueError):env['encoded'](None,None,sample,1,[Path('.')])

class CheckpointTests(unittest.TestCase):
    def test_committed_optimizer_rng_and_tamper_detection(self):
        import tempfile, random
        from checkpoint_state import save_checkpoint, verified_checkpoint, restore_training_state
        class Actor(torch.nn.Linear):
            def save_pretrained(self,path):
                path.mkdir();torch.save(self.state_dict(),path/'weights.pt')
        with tempfile.TemporaryDirectory() as d:
            actor=Actor(1,1);optim=torch.optim.AdamW(actor.parameters(),lr=.01)
            def step():
                optim.zero_grad();actor(torch.ones(1,1)).sum().backward();optim.step()
            step();rng=random.Random(7)
            folder=save_checkpoint(d,actor,None,optim,step=1,metadata={'policy':'toy'},shuffle_rng=rng)
            expected_random=(random.random(),np.random.rand(),torch.rand(1).item(),rng.random())
            step();expected={k:v.clone() for k,v in actor.state_dict().items()}
            actor.load_state_dict(torch.load(folder/'adapter/weights.pt',weights_only=True))
            state=restore_training_state(d,optim,shuffle_rng=rng)
            self.assertEqual(state['step'],1)
            self.assertEqual(expected_random,(random.random(),np.random.rand(),torch.rand(1).item(),rng.random()))
            step()
            for k,v in actor.state_dict().items():torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
            (folder/'adapter/weights.pt').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):verified_checkpoint(d)
    def test_failed_save_preserves_previous_commit(self):
        import tempfile
        from checkpoint_state import save_checkpoint,verified_checkpoint
        class Actor(torch.nn.Linear):
            fail=False
            def save_pretrained(self,path):
                path.mkdir()
                if self.fail:raise OSError('simulated disk failure')
                torch.save(self.state_dict(),path/'weights.pt')
        with tempfile.TemporaryDirectory() as d:
            actor=Actor(1,1);opt=torch.optim.AdamW(actor.parameters())
            good=save_checkpoint(d,actor,None,opt,step=0,metadata={})
            actor.fail=True
            with self.assertRaises(OSError):save_checkpoint(d,actor,None,opt,step=1,metadata={})
            self.assertEqual(verified_checkpoint(d),good.resolve())

if __name__=='__main__':unittest.main()
