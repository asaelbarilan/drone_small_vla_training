"""CPU integration using actual installed PEFT; no downloaded model."""
import sys,unittest,tempfile
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rl'))
from composite_policy import CompositePolicy
try:
    from peft import LoraConfig,get_peft_model
    from transformers import GPT2Config,GPT2LMHeadModel
    HAVE_PEFT=True
except ImportError:HAVE_PEFT=False

@unittest.skipUnless(HAVE_PEFT,'requires existing Qwen/PEFT environment')
class CompositeTests(unittest.TestCase):
    def test_prefix_immutable_through_action_update_and_switches(self):
        torch.manual_seed(7)
        base=GPT2LMHeadModel(GPT2Config(n_layer=1,n_head=2,n_embd=16,vocab_size=64,n_positions=32))
        model=get_peft_model(base,LoraConfig(r=2,lora_alpha=2,target_modules=['c_attn'],lora_dropout=.3),adapter_name='action')
        model.add_adapter('prefix',LoraConfig(r=2,lora_alpha=2,target_modules=['c_attn'],lora_dropout=.3))
        policy=CompositePolicy(model,train_action=True)
        tokens=torch.tensor([[1,2,3]])
        prefix=policy.activate('prefix')(tokens).logits.detach().clone()
        initial=policy.activate('action')(tokens).logits.detach().clone()
        torch.testing.assert_close(prefix,initial)
        frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
        opt=torch.optim.AdamW(policy.action_parameters(),lr=.02)
        loss=-model(tokens).logits[0,-1].log_softmax(-1)[4];loss.backward();opt.step()
        changed=model(tokens).logits.detach()
        self.assertFalse(torch.equal(initial,changed))
        for _ in range(3):
            torch.testing.assert_close(policy.activate('prefix')(tokens).logits,prefix,rtol=0,atol=0)
            self.assertFalse(any(p.requires_grad for p in model.parameters()))
            torch.testing.assert_close(policy.activate('action')(tokens).logits,changed,rtol=0,atol=0)
        for n,p in model.named_parameters():
            if n in frozen:torch.testing.assert_close(p,frozen[n],rtol=0,atol=0)
        self.assertFalse(model.training)
        with tempfile.TemporaryDirectory() as d:
            policy.save_pretrained(Path(d)/'adapter')
            self.assertTrue((Path(d)/'adapter/action/adapter_config.json').exists())

    def test_bf16_base_checkpoint_preserves_small_fp32_updates(self):
        torch.manual_seed(17)
        base=GPT2LMHeadModel(GPT2Config(n_layer=1,n_head=2,n_embd=16,vocab_size=64,n_positions=32)).to(torch.bfloat16)
        import copy
        clean_base=copy.deepcopy(base)
        model=get_peft_model(base,LoraConfig(r=2,lora_alpha=2,target_modules=['c_attn']),adapter_name='action')
        policy=CompositePolicy(model,train_action=True)
        with torch.no_grad():
            for p in policy.action_parameters():p.add_(0.0000137)
        tokens=torch.tensor([[1,2,3]])
        expected=model(tokens).logits.detach().clone()
        original={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        with tempfile.TemporaryDirectory() as d:
            policy.save_pretrained(d)
            saved=Path(d)/'action'
            model.delete_adapter('action')
            model.load_adapter(str(saved),adapter_name='action',is_trainable=True)
            policy.restore_action_weights_exact(saved)
            for n,p in model.named_parameters():
                if n in original:torch.testing.assert_close(p,original[n],rtol=0,atol=0)
            torch.testing.assert_close(model(tokens).logits,expected,rtol=0,atol=0)
            fresh=CompositePolicy.load(clean_base,saved,saved,train_action=False)
            torch.testing.assert_close(fresh.model(tokens).logits,expected,rtol=0,atol=0)

    def test_unmarked_baseline_exact_reload_and_prefix_preserved(self):
        from peft.utils.save_and_load import get_peft_model_state_dict
        from safetensors.torch import load_file
        import copy
        torch.manual_seed(19)
        base=GPT2LMHeadModel(GPT2Config(n_layer=1,n_head=2,n_embd=16,vocab_size=64,n_positions=32)).to(torch.bfloat16)
        clean=copy.deepcopy(base)
        raw=get_peft_model(base,LoraConfig(r=2,lora_alpha=2,target_modules=['c_attn']))
        with torch.no_grad():
            for n,v in raw.named_parameters():
                if 'lora_' in n:v.add_(0.0000137)
        with tempfile.TemporaryDirectory() as d:
            raw.save_pretrained(d)
            self.assertFalse((Path(d)/'exact_restore.json').exists())
            policy=CompositePolicy.load(clean,d,d,train_action=True)
            disk=load_file(str(Path(d)/'adapter_model.safetensors'))
            for k,v in get_peft_model_state_dict(policy.model,adapter_name='action').items():
                torch.testing.assert_close(v.cpu(),disk[k],rtol=0,atol=0)
            tokens=torch.tensor([[1,2,3]])
            original=policy.activate('action')(tokens).logits.detach().clone()
            prefix={k:v.detach().clone() for k,v in get_peft_model_state_dict(policy.model,adapter_name='prefix').items()}
            policy.reload_action_exact(d)
            torch.testing.assert_close(policy.model(tokens).logits,original,rtol=0,atol=0)
            for k,v in get_peft_model_state_dict(policy.model,adapter_name='prefix').items():
                torch.testing.assert_close(v,prefix[k],rtol=0,atol=0)
            self.assertTrue(policy.action_parameters())
            self.assertFalse(any(p.requires_grad for n,p in policy.model.named_parameters() if '.prefix.' in n))


if __name__=='__main__':unittest.main()


class RuntimeMetadataTests(unittest.TestCase):
    def test_bf16_runtime_without_optional_quantization_package(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from importlib.metadata import PackageNotFoundError
        from composite_policy import runtime_identity
        model=SimpleNamespace(config=SimpleNamespace(to_dict=lambda:{'model_type':'fixture'}))
        processor=SimpleNamespace(tokenizer=SimpleNamespace(get_vocab=lambda:{'token':1}))
        def version(name):
            if name=='bitsandbytes':raise PackageNotFoundError(name)
            return 'test-version'
        with patch('importlib.metadata.version',side_effect=version):
            result=runtime_identity(model,processor,{'low':[0],'high':[1]})
        self.assertIsNone(result['runtime_versions']['bitsandbytes'])
        self.assertEqual(result['runtime_versions']['torch'],'test-version')
    def test_missing_required_runtime_still_fails(self):
        from unittest.mock import patch
        from importlib.metadata import PackageNotFoundError
        from composite_policy import runtime_identity
        with patch('importlib.metadata.version',side_effect=PackageNotFoundError('torch')):
            with self.assertRaises(PackageNotFoundError):runtime_identity(None,None,{})
