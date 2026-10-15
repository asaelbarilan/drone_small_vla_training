import unittest,sys,random
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import uav_flow_eval_server as server

class SeededResetTest(unittest.TestCase):
    def setUp(self):
        server.STATE.update(episode=0,call=9,goal=3,yaw_total=12,last_yaw=8)
        self.client=server.app.test_client()
    def draw(self,seed):
        result=self.client.post('/reset',json={'seed':seed})
        self.assertEqual(result.status_code,200)
        self.assertEqual(result.json['seed'],seed)
        return random.random(),np.random.random(),torch.rand(4)
    def test_repeat_and_different_seed(self):
        a=self.draw(7);b=self.draw(7);c=self.draw(8)
        self.assertEqual(a[:2],b[:2]);self.assertTrue(torch.equal(a[2],b[2]))
        self.assertFalse(torch.equal(a[2],c[2]))
        self.assertEqual(server.STATE['call'],0);self.assertIsNone(server.STATE['goal'])
    def test_invalid_seed(self):
        for seed in (-1,2**32,True,'7'):
            self.assertEqual(self.client.post('/reset',json={'seed':seed}).status_code,400)
    def test_legacy_empty_reset(self):
        self.assertEqual(self.client.post('/reset').status_code,200)
