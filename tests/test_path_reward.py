import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rl'))
from path_reward import path_reward,PathRewardConfig

def line(end=10,n=101):
    a=np.zeros((n,6));a[:,0]=np.linspace(0,end*100,n);return a

def circle(sweep=360,n=201):
    a=np.zeros((n,6));t=np.deg2rad(np.linspace(0,sweep,n))
    a[:,0]=300*(np.cos(t)-1);a[:,1]=300*np.sin(t);return a

class PathRewardTests(unittest.TestCase):
    def test_completed_partial_stationary_order(self):
        ref=circle();full=path_reward(ref,ref,stopped=True)
        half=path_reward(ref[:101],ref,stopped=True)
        idle=path_reward(np.repeat(ref[:1],201,axis=0),ref,stopped=True)
        self.assertTrue(full['complete']);self.assertFalse(half['complete']);self.assertFalse(idle['complete'])
        self.assertGreater(full['reward'],half['reward']);self.assertGreater(half['reward'],idle['reward'])
        self.assertEqual(idle['progress_fraction'],0)
    def test_wrong_way_and_spin_do_not_complete(self):
        ref=circle();spin=np.repeat(ref[:1],201,axis=0);spin[:,4]=np.linspace(0,360,201)
        for a in [circle(-360),spin]:self.assertFalse(path_reward(a,ref,stopped=True)['complete'])
    def test_oscillation_does_not_farm_progress(self):
        ref=line();arc=ref[:21];osc=np.concatenate([arc,arc[::-1]]*4)
        r=path_reward(osc,ref,stopped=True)
        self.assertLessEqual(r['reward'],0);self.assertFalse(r['complete'])
    def test_overshoot_penalized(self):
        ref=line();good=path_reward(ref,ref,stopped=True);over=path_reward(line(13,131),ref,stopped=True)
        self.assertLess(over['reward'],good['reward']);self.assertFalse(over['complete'])
    def test_reference_density_not_action_frequency(self):
        flight=line(n=51)
        a=path_reward(flight,line(n=21),stopped=True)
        b=path_reward(flight,line(n=201),stopped=True)
        self.assertAlmostEqual(a['reward'],b['reward'],places=9)
        self.assertTrue(a['complete'])
    def test_action_frequency_on_same_line(self):
        ref=line();a=path_reward(line(n=51),ref,stopped=True);b=path_reward(line(n=201),ref,stopped=True)
        self.assertAlmostEqual(a['reward'],b['reward'],places=9)
    def test_shortcut_jump_fails(self):
        r=path_reward(np.array([circle()[0],circle()[100],circle()[-1]]),circle(),stopped=True)
        self.assertFalse(r['complete']);self.assertGreater(r['invalid_transitions'],0)
    def test_yaw_only_task_supported(self):
        ref=np.zeros((91,6));ref[:,4]=np.arange(91)
        self.assertTrue(path_reward(ref,ref,stopped=True)['complete'])
        wrong=ref.copy();wrong[:,4]*=-1
        self.assertFalse(path_reward(wrong,ref,stopped=True)['complete'])
    def test_turn_overshoot_needs_an_independent_heading_gate(self):
        ref=np.zeros((23,6));ref[:,4]=np.arange(23)
        actual=np.zeros((46,6));actual[:,4]=np.arange(46)
        old=path_reward(actual,ref,stopped=True,config=PathRewardConfig(endpoint_yaw_tolerance_deg=180))
        new=path_reward(actual,ref,stopped=True)
        self.assertTrue(old['complete']);self.assertFalse(new['complete'])
        self.assertAlmostEqual(old['reward']-new['reward'],1)
        self.assertAlmostEqual(new['endpoint_yaw_error_deg'],23)

    def test_heading_wrap_and_five_degree_boundary(self):
        ref=np.zeros((3,6));ref[:,4]=[179,180,181]
        actual=ref.copy();actual[:,4]=[179,-180,-179]
        self.assertTrue(path_reward(actual,ref,stopped=True)['complete'])
        actual=np.zeros((31,6));actual[:,4]=np.linspace(0,28,31)
        ref=np.zeros((23,6));ref[:,4]=np.arange(23)
        self.assertFalse(path_reward(actual,ref,stopped=True)['complete'])

    def test_reference_revisits_a_segment(self):
        forward=line(3,31)
        ref=np.concatenate([forward,forward[-2:9:-1],line(5,51)[11:]])
        r=path_reward(ref,ref,stopped=True)
        self.assertTrue(r['complete'])
        self.assertAlmostEqual(r['reward'],2)

    def test_stopping_bonus_only_at_completion(self):
        ref=line();r=path_reward(ref,ref,stopped=False)
        self.assertFalse(r['complete']);self.assertAlmostEqual(r['reward'],1)
        self.assertAlmostEqual(path_reward(ref,ref,stopped=True)['reward'],2)

if __name__=='__main__':unittest.main()
