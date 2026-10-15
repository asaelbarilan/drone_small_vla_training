import sys,unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts/rl'))
from state_critic import StateCritic,fit_state_critic,select_state_critic,flight_weights
class StateCriticTests(unittest.TestCase):
    def data(self):
        rng=np.random.default_rng(253);x=rng.normal(size=(15,8));z=rng.normal(size=(15,3));g=np.repeat(np.arange(3),5);y=z[:,0]+.2*x[:,0]
        return x,z,y,g
    def test_training_only_normalization(self):
        x,z,y,g=self.data();z[g==2]=10000;train=g!=2
        h=fit_state_critic(x[train],z[train],y[train],g[train],.1,1)
        np.testing.assert_allclose(h.state_mean,flight_weights(g[train])@z[train],atol=1e-6)
        self.assertLess(float(h.state_mean.abs().max()),2.)
    def test_cv_leaves_whole_flight_out(self):
        x,z,y,g=self.data();_,cv=select_state_critic(x,z,y,g,[.1,1.],[1.])
        for row in cv['candidates']:
            for fold in row['folds']:self.assertNotIn(fold['heldout_group'],fold['training_groups'])
    def test_state_changes_values_without_actor_gradients(self):
        x,z,y,g=self.data();h=fit_state_critic(x,z,y,g,.01,1.)
        a=torch.tensor(x,dtype=torch.float32,requires_grad=True);b=torch.tensor(z,dtype=torch.float32,requires_grad=True)
        v=h(a,b);v.sum().backward();self.assertIsNone(a.grad);self.assertIsNone(b.grad)
        self.assertFalse(torch.equal(v,h(a,b+1)))
    def test_exact_reload(self):
        x,z,y,g=self.data();h=fit_state_critic(x,z,y,g,.1,1.)
        other=StateCritic(8,3);other.load_state_dict(h.state_dict())
        self.assertTrue(torch.equal(h(torch.tensor(x),torch.tensor(z)),other(torch.tensor(x),torch.tensor(z))))
    def test_equal_flight_weighting(self):
        x,z,y,g=self.data();a=fit_state_critic(x,z,y,g,.1,1.);idx=np.r_[np.arange(15),np.arange(5)]
        b=fit_state_critic(x[idx],z[idx],y[idx],g[idx],.1,1.)
        for n,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[n],atol=1e-6,rtol=1e-6)
    def test_constant_features_and_finite_checks(self):
        x,z,y,g=self.data();z[:,2]=5;h=fit_state_critic(x,z,y,g,.1,1.);self.assertEqual(float(h.state_scale[2]),1.)
        z[0,0]=np.nan
        with self.assertRaises(ValueError):fit_state_critic(x,z,y,g,.1,1.)
if __name__=='__main__':unittest.main()
