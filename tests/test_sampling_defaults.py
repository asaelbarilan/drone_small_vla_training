"""Real Transformers generation regression, CPU tiny random model, no downloads."""
import ast,sys,unittest
from pathlib import Path
import torch
from transformers import GPT2Config,GPT2LMHeadModel,GenerationConfig,LogitsProcessor,LogitsProcessorList
ROOT=Path(__file__).resolve().parents[1]
class SamplingDefaultsTest(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(5)
  self.model=GPT2LMHeadModel(GPT2Config(vocab_size=320,n_positions=32,n_embd=16,n_layer=1,n_head=2,bos_token_id=1,eos_token_id=2)).eval()
  self.model.generation_config=GenerationConfig(do_sample=True,temperature=.7,top_p=.8,top_k=20,bos_token_id=1,eos_token_id=2,pad_token_id=2,transformers_version='4.56.0')
  self.cfg=GenerationConfig(do_sample=True,temperature=1.,top_p=1.,top_k=0,repetition_penalty=1.,min_new_tokens=4,max_new_tokens=4,return_dict_in_generate=True,output_logits=True,output_scores=True,pad_token_id=2)
 def test_reproduce_override_and_fix(self):
  old,_=self.model._prepare_generation_config(self.cfg,None)
  fixed,_=self.model._prepare_generation_config(self.cfg,False)
  self.assertEqual((old.temperature,old.top_p),(.7,.8))
  self.assertEqual((fixed.temperature,fixed.top_p,fixed.top_k),(1.,1.,0))
 def test_actual_sampling_scores_equal_recorded_distribution(self):
  class Mask(LogitsProcessor):
   def __call__(self,ids,scores):
    out=torch.full_like(scores,float('-inf'));out[:,64:320]=scores[:,64:320];return out
  with torch.inference_mode():
   out=self.model.generate(torch.tensor([[1,3]]),attention_mask=torch.ones((1,2),dtype=torch.long),generation_config=self.cfg,use_model_defaults=False,logits_processor=LogitsProcessorList([Mask()]))
  for raw,sampled in zip(out.logits,out.scores):
   self.assertTrue(torch.isneginf(sampled[:,:64]).all())
   torch.testing.assert_close(raw[:,64:].log_softmax(-1),sampled[:,64:].log_softmax(-1),rtol=0,atol=0)
 def test_server_cannot_omit_opt_out(self):
  tree=ast.parse((ROOT/'scripts/uav_flow_eval_server.py').read_text())
  calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and any(k.arg=='generation_config' and isinstance(k.value,ast.Name) and k.value.id=='sampling' for k in n.keywords)]
  self.assertEqual(len(calls),1)
  self.assertTrue(any(k.arg=='use_model_defaults' and isinstance(k.value,ast.Constant) and k.value.value is False for k in calls[0].keywords))
class SamplingContractTest(unittest.TestCase):
 def test_reject_old_or_unverified_samples(self):
  sys.path.insert(0,str(ROOT/'scripts/rl'))
  from sampling_contract import validate_sampling_contract
  with self.assertRaises(ValueError):validate_sampling_contract([{'temperature':1.}])
  good=dict(sampling_contract='action-categorical-v2',policy_provenance={'sampling_contract':'action-categorical-v2'},temperature=1.,top_k=0,top_p=1.,use_model_defaults=False,sampler_logp_max_error=0.)
  self.assertEqual(validate_sampling_contract([good]),1)
  for change in [dict(top_p=.8),dict(temperature=float('nan')),dict(sampler_logp_max_error=.01),dict(use_model_defaults=True)]:
   with self.subTest(change=change),self.assertRaises(ValueError):validate_sampling_contract([good|change])

if __name__=='__main__':unittest.main()
