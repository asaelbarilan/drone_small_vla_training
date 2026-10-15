import sys,unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rl'))
from regularized_critic import fit_ridge,select_ridge
class CriticTests(unittest.TestCase):
 def data(self):
  rng=np.random.default_rng(2);x=rng.normal(size=(12,6)).astype('float32');g=np.repeat(np.arange(3),4);y=x[:,0]*.2
  return x,y,g
 def test_folds_never_contain_validation_flight(self):
  x,y,g=self.data();_,r=select_ridge(x,y,g,[.01,1.])
  for a in r['candidates']:
   for f in a['folds']:self.assertNotIn(f['heldout_group'],f['training_groups'])
 def test_features_unchanged_and_no_gradient_path(self):
  x,y,g=self.data();old=x.copy();h=fit_ridge(x,y,g,.1);np.testing.assert_array_equal(x,old)
  self.assertEqual(set(h.state_dict()),{'weight','bias'})
 def test_exact_reload(self):
  x,y,g=self.data();h=fit_ridge(x,y,g,.1);other=torch.nn.Linear(6,1);other.load_state_dict(h.state_dict())
  self.assertTrue(torch.equal(h(torch.tensor(x)),other(torch.tensor(x))))
 def test_regularization_limits_interpolation(self):
  x,y,g=self.data();a=fit_ridge(x,y,g,.001);b=fit_ridge(x,y,g,100.)
  self.assertGreater(a.weight.norm(),b.weight.norm())
 def test_rejects_nonfinite_targets(self):
  x,y,g=self.data();y[0]=np.nan
  with self.assertRaises(ValueError):fit_ridge(x,y,g,.1)
 def test_flight_weighting_resists_duplicate_calls(self):
  x,y,g=self.data();a=fit_ridge(x,y,g,.1);idx=np.r_[np.arange(12),np.arange(4)]
  b=fit_ridge(x[idx],y[idx],g[idx],.1)
  np.testing.assert_allclose(a.weight.detach(),b.weight.detach(),atol=1e-6)
class MetricTests(unittest.TestCase):
 def test_equal_flight_metric_does_not_overweight_long_flight(self):
  from d251_evaluate_critic import weighted_mse
  flights=[dict(samples=[dict(mc_target=1.,value_old=0.)]*20),dict(samples=[dict(mc_target=0.,value_old=0.)])]
  self.assertAlmostEqual(weighted_mse(flights),.5)
 def test_constants_compared_to_same_targets(self):
  from d251_evaluate_critic import weighted_mse
  flights=[dict(samples=[dict(mc_target=1.,value_old=99.)])]
  self.assertEqual(weighted_mse(flights,0.),1.)
  self.assertEqual(weighted_mse(flights,1.),0.)
if __name__=='__main__':unittest.main()
