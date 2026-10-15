import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rl'))
from movement_reward import movement_reward,MovementRewardConfig
from audit_movement_reward import line,controls
from executed_rollouts import attach_path_rewards

class MovementRewardTests(unittest.TestCase):
    def test_turn_symmetric_stopping(self):
        for goal in [-60,-9,-.1,.1,9,60]:
            ref=line(yaw=goal)
            a=movement_reward(line(yaw=goal*.95),ref,kind='Turn',stopped=True)
            b=movement_reward(line(yaw=goal*1.05),ref,kind='Turn',stopped=True)
            self.assertAlmostEqual(a['reward'],b['reward'],places=8)
            self.assertTrue(a['complete']);self.assertTrue(b['complete'])
            idle=movement_reward(line(),ref,kind='Turn',stopped=True)
            self.assertFalse(idle['complete'])

    def test_reference_resampling_all_kinds(self):
        for kind,ref in controls().items():
            dense=np.vstack([p+(q-p)*t for p,q in zip(ref,ref[1:]) for t in (0,.5)]+[ref[-1]])
            args=dict(kind=kind,stopped=True,events={'landed':True})
            a=movement_reward(ref,ref,**args);b=movement_reward(ref,dense,**args)
            self.assertAlmostEqual(a['reward'],b['reward'],places=7)
            self.assertEqual(a['complete'],b['complete'])

    def test_start_frame_and_events(self):
        ref=line(x=3);wrong=ref.copy();wrong[:,0]+=100
        a=movement_reward(wrong,ref,kind='Shift',stopped=True)
        self.assertGreater(a['invalid_transitions'],0);self.assertFalse(a['complete'])
        with self.assertRaises(ValueError):
            movement_reward(ref,ref,kind='Shift',stopped=True,events={'collision':'false'})

    def test_reject_bad_inputs(self):
        ref=line(x=3)
        for poses,kind,stop in [(ref,'unknown',True),(ref,'Shift',1),(ref*np.nan,'Shift',True)]:
            with self.assertRaises(ValueError):movement_reward(poses,ref,kind=kind,stopped=stop)
        with self.assertRaises(ValueError):MovementRewardConfig(integration_spacing=0)
        with self.assertRaises(ValueError):
            movement_reward(ref,ref,kind='Shift',stopped=True,
                config=MovementRewardConfig(max_integrated_points=2))

    def test_land_requires_contact_even_at_correct_endpoint(self):
        ref=line(z=-3)
        for events in ({},{'landed':False},{'landed':True,'collision':True}):
            r=movement_reward(ref,ref,kind='Land',stopped=True,events=events)
            self.assertFalse(r['complete']);self.assertEqual(r['terminal_credit'],0)
        self.assertTrue(movement_reward(ref,ref,kind='Land',stopped=True,
                                       events={'landed':True})['complete'])

    def test_downward_motion_and_wrong_altitude(self):
        ref=line(x=2,z=-3)
        good=movement_reward(ref,ref,kind='Ascend/Descend',stopped=True)
        wrong=ref.copy();wrong[:,2]*=-1
        bad=movement_reward(wrong,ref,kind='Ascend/Descend',stopped=True)
        self.assertTrue(good['complete']);self.assertFalse(bad['complete'])
        self.assertGreater(good['reward'],bad['reward'])

    def test_chunk_attachment_preserves_total(self):
        ref=line(x=1,n=11)
        def call(start,end):
            return dict(proprio=[*ref[start,:3],ref[start,4]],
                        poses=[[ *r[:3],np.deg2rad(r[4])] for r in ref[start+1:end+1]])
        for counts in ([10],[4,6]):
            calls=[];offset=0
            for n in counts:calls.append(call(offset,offset+n));offset+=n
            samples,r=attach_path_rewards(calls,ref,counts,ref,end_reason='model_stop',kind='Shift')
            self.assertAlmostEqual(sum(x['reward_v2'] for x in samples),r['reward'])
            self.assertEqual(r['version'],'movement-reward-candidate-v5')
            self.assertTrue(all(not x['reward_spec_approved'] for x in samples))

if __name__=='__main__':unittest.main()
