import sys,tempfile,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rl'))
from train_path_ppo import load_initial_critic
class InitialCriticTests(unittest.TestCase):
 def test_exact_weights_without_actor_access(self):
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder)/'head.pt';src=torch.nn.Linear(4,1);torch.save(src.state_dict(),path)
   dst=torch.nn.Linear(4,1);digest=load_initial_critic(dst,path)
   self.assertEqual(len(digest),64)
   for k,v in src.state_dict().items():self.assertTrue(torch.equal(v,dst.state_dict()[k]))
 def test_nonfinite_rejected(self):
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder)/'head.pt';src=torch.nn.Linear(4,1);state=src.state_dict();state['bias'][0]=float('nan');torch.save(state,path)
   with self.assertRaisesRegex(ValueError,'nonfinite'):load_initial_critic(torch.nn.Linear(4,1),path)
 def test_wrong_architecture_rejected(self):
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder)/'head.pt';torch.save(torch.nn.Linear(8,1).state_dict(),path)
   with self.assertRaisesRegex(ValueError,'architecture'):load_initial_critic(torch.nn.Linear(4,1),path)
if __name__=='__main__':unittest.main()
