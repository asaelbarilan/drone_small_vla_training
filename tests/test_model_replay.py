import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rl'))
from model_replay import replay_cached_tokens

class CachedGradientTests(unittest.TestCase):
    def test_cached_gradient_matches_full_float32_reference(self):
        from transformers import GPT2Config,GPT2LMHeadModel
        torch.manual_seed(19)
        model=GPT2LMHeadModel(GPT2Config(n_layer=1,n_head=2,n_embd=16,vocab_size=32,n_positions=32,
             bos_token_id=0,eos_token_id=None,pad_token_id=0,attn_pdrop=0,resid_pdrop=0,embd_pdrop=0)).eval()
        prefix=torch.tensor([[2,3,4]]);ids=[20,22,25,23]
        lp,feature=replay_cached_tokens(model,dict(input_ids=prefix,attention_mask=torch.ones_like(prefix)),ids,
                    low=16,high=32,temperature=.8,pad_token_id=0)
        self.assertTrue(lp.requires_grad);self.assertFalse(feature.requires_grad)
        weights=torch.tensor([.2,-.4,.7,.3]);(lp*weights).sum().backward()
        gradients={n:p.grad.clone() for n,p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        all_ids=torch.cat([prefix,torch.tensor([ids])],1)
        logits=model(input_ids=all_ids,use_cache=False).logits[0,2:6,16:32].float()/.8
        expected=logits.log_softmax(-1).gather(1,(torch.tensor(ids)-16)[:,None]).flatten()
        torch.testing.assert_close(lp.detach(),expected.detach(),rtol=1e-5,atol=1e-6)
        (expected*weights).sum().backward()
        self.assertGreater(sum(float(g.abs().sum()) for g in gradients.values()),0)
        for n,p in model.named_parameters():torch.testing.assert_close(gradients[n],p.grad,rtol=1e-4,atol=2e-6)

if __name__=='__main__':unittest.main()
