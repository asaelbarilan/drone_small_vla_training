import json,sys,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts/rl'))
from critic_gate_metrics import numeric_criteria
class UnchangedGateTests(unittest.TestCase):
 def test_d251_verdicts_and_numbers_preserved(self):
  folder=ROOT/'reports/d251_critic_gate'
  data=torch.load(folder/'test_features.pt',weights_only=False,map_location='cpu')
  fit=json.loads((folder/'fit.json').read_text(encoding='utf-8'));old=json.loads((folder/'result.json').read_text(encoding='utf-8'))
  now=numeric_criteria(data['flights'],fit)
  for k,v in now['checks'].items():self.assertEqual(v,old['criteria'][k],k)
  for k in ['mse','zero_baseline_mse','training_mean_baseline_mse','terminal_mae','terminal_max_error']:self.assertAlmostEqual(now['metrics'][k],old[k],places=12)
  self.assertEqual(now['metrics']['state_rates'],old['state_rates'])
 def test_protocol_criteria_equal_verbatim(self):
  a=json.loads((ROOT/'reports/d251_critic_gate/protocol.json').read_text(encoding='utf-8'))
  b=json.loads((ROOT/'reports/d253_state_critic/protocol.json').read_text(encoding='utf-8'))
  self.assertEqual(a['criteria'],b['criteria']);self.assertFalse(b['aws_allowed'])
if __name__=='__main__':unittest.main()
