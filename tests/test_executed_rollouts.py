import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rl'))
from executed_rollouts import attach_path_rewards

class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.ref=[[i*10,0,0,0,0,0] for i in range(11)]
        self.calls=[dict(proprio=[0,0,0,0],poses=[[i*10,0,0,0] for i in range(1,9)],ids=[1]*32),
                    dict(proprio=[80,0,0,0],poses=[[i*10,0,0,0] for i in range(9,17)],ids=[1]*32)]
    def test_partial_last_chunk_counts_actual_moves_only(self):
        samples,r=attach_path_rewards(self.calls,self.ref,[8,2],self.ref,end_reason='model_stop')
        self.assertEqual([s['executed_count'] for s in samples],[8,2])
        self.assertEqual(len(samples[1]['ids']),32)
        self.assertAlmostEqual(sum(s['reward_v2'] for s in samples),2)
        self.assertFalse(samples[0]['terminated']);self.assertTrue(samples[1]['terminated'])
    def test_truncation_has_no_completion_bonus(self):
        samples,r=attach_path_rewards(self.calls,self.ref,[8,2],self.ref,end_reason='horizon',final_observation={'id':'actual_end'})
        self.assertTrue(samples[-1]['truncated']);self.assertFalse(samples[-1]['terminated'])
        self.assertAlmostEqual(r['reward'],1)
    def test_wrong_counts_and_missing_final_image_rejected(self):
        for counts,reason in [([8,8],'model_stop'),([8,2],'horizon')]:
            with self.assertRaises(ValueError):attach_path_rewards(self.calls,self.ref,counts,self.ref,end_reason=reason)
    def test_wrong_frame_or_infrastructure_failure_rejected(self):
        with self.assertRaises(ValueError):attach_path_rewards(self.calls,self.ref,[8,2],self.ref,end_reason='renderer_crash')
        self.calls[-1]['proprio'][0]=0
        with self.assertRaises(ValueError):attach_path_rewards(self.calls,self.ref,[8,2],self.ref,end_reason='model_stop')
    def test_chunking_preserves_undiscounted_reward(self):
        one=[dict(proprio=[0,0,0,0],poses=[[i*10,0,0,0] for i in range(1,11)],ids=[1]*40)]
        a,_=attach_path_rewards(one,self.ref,[10],self.ref,end_reason='model_stop')
        b,_=attach_path_rewards(self.calls,self.ref,[8,2],self.ref,end_reason='model_stop')
        self.assertAlmostEqual(sum(s['reward_v2'] for s in a),sum(s['reward_v2'] for s in b))

if __name__=='__main__':unittest.main()
