import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rl'))
from flight_collector import collect_episode

class FakeEnv:
    def __init__(self):self.pose=[0.,0.,0.,0.,0.,0.];self.commands=0
    def observe(self):return dict(pose=self.pose.copy(),pose_source='simulator_readback',id=str(self.commands))
    def execute(self,c):self.pose=[c[0],c[1],c[2],0.,0.,0.];self.commands+=1

def proposal(o,delta=10):
    x=o['pose'][0]
    return dict(proprio=[x,0,0,0],poses=[[x+(i+1)*delta,0,0,0] for i in range(8)],ids=list(range(32)))

class CollectionTests(unittest.TestCase):
    def test_new_reward_opt_in_preserves_horizon_contract(self):
        env=FakeEnv()
        r=collect_episode(env,proposal,lambda o:dict(id=o['id']),
            [[0,0,0,0,0,0],[200,0,0,0,0,0]],max_steps=10,reward_kind='Shift')
        self.assertEqual(r['reward']['version'],'movement-reward-candidate-v5')
        self.assertFalse(r['reward']['complete'])
        self.assertEqual(r['reward']['terminal_credit'],0)
        self.assertTrue(r['samples'][-1]['truncated'])
        self.assertAlmostEqual(sum(s['reward_v2'] for s in r['samples']),r['reward']['reward'])
        self.assertTrue(all(not s['reward_spec_approved'] for s in r['samples']))

    def test_exact_horizon_and_real_final_context(self):
        env=FakeEnv();seen=[]
        def context(o):seen.append(o['id']);return dict(id=o['id'])
        r=collect_episode(env,proposal,context,[[0,0,0,0,0,0],[200,0,0,0,0,0]],max_steps=10)
        self.assertEqual(r['executed_counts'],[8,2]);self.assertEqual(env.commands,10)
        self.assertEqual(seen,['10']);self.assertTrue(r['samples'][-1]['truncated'])
        self.assertEqual(len(r['samples'][-1]['ids']),32)
    def test_stop_in_middle_of_chunk(self):
        env=FakeEnv()
        r=collect_episode(env,lambda o:proposal(o,0),lambda o:self.fail('no bootstrap on terminal'),
                          [[0,0,0,0,0,0],[200,0,0,0,0,0]],max_steps=100)
        self.assertEqual(r['executed_counts'],[8,2]);self.assertTrue(r['samples'][-1]['terminated'])
        self.assertEqual(env.commands,10)
    def test_measured_yaw_tolerance_has_a_strict_bound(self):
        from executed_rollouts import attach_path_rewards
        call=proposal(dict(pose=[0,0,0,0,0,0]))
        actual=[[0]*6]+[[10*(i+1),0,0,0,.02,0] for i in range(8)]
        attach_path_rewards([call],actual,[8],[[0]*6,[100,0,0,0,0,0]],end_reason='model_stop')
        actual[-1][4]=.06
        with self.assertRaisesRegex(ValueError,'actual yaw mismatch'):
            attach_path_rewards([call],actual,[8],[[0]*6,[100,0,0,0,0,0]],end_reason='model_stop')

    def test_command_log_cannot_claim_readback(self):
        env=FakeEnv();env.observe=lambda:dict(pose=env.pose,pose_source='command')
        with self.assertRaisesRegex(ValueError,'readback'):
            collect_episode(env,proposal,lambda o:o,[[0]*6,[100,0,0,0,0,0]],max_steps=10)
    def test_transport_failure_is_not_a_training_terminal(self):
        env=FakeEnv()
        def fail(c):raise ConnectionError('lost simulator')
        env.execute=fail
        with self.assertRaises(ConnectionError):
            collect_episode(env,proposal,lambda o:o,[[0]*6,[100,0,0,0,0,0]],max_steps=10)

class FrameTests(unittest.TestCase):
    def test_rotated_world_and_mesh_heading_roundtrip(self):
        import numpy as np
        from simulator_transport import UnrealReadback
        class API:
            location=[100,200,300];yaw=-90
            def get_obj_location(self,a):return self.location
            def get_obj_rotation(self,a):return [0,self.yaw,0]
            def set_obj_location(self,a,p):self.location=p
            def set_rotation(self,a,y):self.yaw=y
            def get_image(self,*a):return np.zeros((2,2,3),dtype=np.uint8)
        api=API();env=UnrealReadback(api,'drone',[100,200,300,0,90,0],lambda:None,settle_seconds=0)
        env.execute([20,10,-5,np.pi/2]);p=env.observe()['pose']
        np.testing.assert_allclose(api.location,[90,220,295],atol=1e-6)
        np.testing.assert_allclose(p,[20,10,-5,0,90,0],atol=1e-6)

    def test_blueprint_command_offset_is_not_measured_heading(self):
        import numpy as np
        from simulator_transport import UnrealReadback
        class API:
            location=[100,200,300];yaw=20.
            def get_obj_location(self,a):return self.location
            def get_obj_rotation(self,a):return [0,self.yaw,0]
            def set_obj_location(self,a,p):self.location=p
            def set_rotation(self,a,y):self.yaw=y+180.233
            def get_image(self,*a):return np.zeros((2,2,3),dtype=np.uint8)
        api=API();env=UnrealReadback(api,'drone',[100,200,300,0,20,0],lambda:None,settle_seconds=0,initial_command_yaw=20-180.233)
        env.execute([0,0,0,np.deg2rad(10)])
        self.assertAlmostEqual(env.observe()['pose'][4],10)

class FeedbackTests(unittest.TestCase):
    def test_small_settled_residual_is_corrected_without_relaxing_tolerance(self):
        from unittest.mock import patch
        from simulator_transport import UnrealReadback
        class API:
            yaw=0.
            def get_obj_location(self,a):return [0,0,0]
            def get_obj_rotation(self,a):return [0,self.yaw,0]
            def set_obj_location(self,a,p):pass
            def set_rotation(self,a,y):self.yaw=y+.012
        api=API();env=UnrealReadback(api,'drone',[0]*6,lambda:None)
        with patch('simulator_transport.time.sleep'),patch('simulator_transport.time.monotonic',side_effect=[0]+[2]*20):
            env.execute([0,0,0,0])
        self.assertLess(abs(api.yaw),.005)
        self.assertEqual(len(env.rotation_corrections),1)

    def test_feedback_does_not_chase_a_moving_heading(self):
        from unittest.mock import patch
        from simulator_transport import UnrealReadback
        class API:
            readings=iter([0,.05,.04,.03,.02,.01,0])
            commands=[]
            def get_obj_location(self,a):return [0,0,0]
            def get_obj_rotation(self,a):return [0,next(self.readings),0]
            def set_obj_location(self,a,p):pass
            def set_rotation(self,a,y):self.commands.append(y)
        api=API();env=UnrealReadback(api,'drone',[0]*6,lambda:None)
        with patch('simulator_transport.time.sleep'),patch('simulator_transport.time.monotonic',side_effect=[0]+[2]*20):
            env.execute([0,0,0,0])
        self.assertEqual(len(api.commands),1)
        self.assertEqual(env.rotation_corrections,[])

    def test_large_heading_error_fails_with_measured_evidence(self):
        from unittest.mock import patch
        from simulator_transport import UnrealReadback
        class API:
            yaw=0.
            def get_obj_location(self,a):return [0,0,0]
            def get_obj_rotation(self,a):return [0,self.yaw,0]
            def set_obj_location(self,a,p):pass
            def set_rotation(self,a,y):self.yaw=y+10
        api=API();env=UnrealReadback(api,'drone',[0]*6,lambda:None)
        with patch('simulator_transport.time.sleep'),patch('simulator_transport.time.monotonic',side_effect=[0,6,6]):
            with self.assertRaisesRegex(TimeoutError,'actual=10.0 error=-10.0'):
                env.execute([0,0,0,0])
        self.assertEqual(env.rotation_corrections,[])

if __name__=='__main__':unittest.main()
