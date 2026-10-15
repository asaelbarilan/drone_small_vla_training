"""D215 one-base-model, frozen-prefix/trainable-action adapter policy."""
import hashlib
from pathlib import Path


def adapter_digest(path):
    path=Path(path)
    files=sorted(p for p in path.iterdir() if p.name=='adapter_config.json' or
                 p.name.startswith('adapter_model.') or p.name=='action_stats.json')
    if not any(p.name=='adapter_config.json' for p in files) or not any(p.name.startswith('adapter_model.') for p in files):
        raise ValueError('adapter weights/config missing: '+str(path))
    h=hashlib.sha256()
    for p in files:
        h.update(p.name.encode())
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


class CompositePolicy:
    def __init__(self,model,*,train_action=False):
        self.model=model
        self.train_action=train_action
        self.activate('action')

    @classmethod
    def load(cls,base,action_path,prefix_path,*,train_action=False):
        from peft import PeftModel
        model=PeftModel.from_pretrained(base,str(action_path),adapter_name='action',is_trainable=train_action)
        model.load_adapter(str(prefix_path),adapter_name='prefix',is_trainable=False)
        obj=cls(model,train_action=train_action)
        obj.restore_action_weights_exact(action_path)  # Baselines need exact loading too, without a marker.
        obj.provenance=dict(action_adapter_sha256=adapter_digest(action_path),
                            prefix_adapter_sha256=adapter_digest(prefix_path),
                            policy_type='frozen-prefix-action-v1',vision_frozen=True)
        return obj

    def activate(self,name):
        if name not in ('prefix','action'):raise ValueError('unknown policy adapter')
        self.model.set_adapter(name)
        # PEFT set_adapter can re-enable parameters. Reassert after EVERY switch.
        for key,p in self.model.named_parameters():
            p.requires_grad_(bool(self.train_action and name=='action' and '.action.' in key
                                  and 'lora_' in key and not any(x in key for x in ('.visual.','.vision_'))))
        self.model.eval()  # dropout-free likelihood replay; gradients still work
        return self.model

    def action_parameters(self):
        self.activate('action')
        params=[p for p in self.model.parameters() if p.requires_grad]
        if self.train_action and not params:raise ValueError('no trainable action LoRA parameters')
        return params

    def restore_action_weights_exact(self,path):
        """Restore saved FP32 LoRA after PEFT allocates modules on a BF16 base.

        PEFT load_adapter may first copy through base dtype before upcasting;
        that discards small trained updates. Restore into the allocated FP32
        parameters and check every saved tensor before optimizer recovery.
        """
        import torch
        from safetensors.torch import load_file
        from peft.utils.save_and_load import get_peft_model_state_dict,set_peft_model_state_dict
        state=load_file(str(Path(path)/'adapter_model.safetensors'))
        current=get_peft_model_state_dict(self.model,adapter_name='action')
        if set(current)!=set(state):raise ValueError('checkpoint action tensor keys differ')
        if any(current[k].dtype!=v.dtype or current[k].shape!=v.shape for k,v in state.items()):
            raise ValueError('checkpoint action tensor dtype/shape differs')
        set_peft_model_state_dict(self.model,state,adapter_name='action')
        restored=get_peft_model_state_dict(self.model,adapter_name='action')
        if any(not torch.equal(restored[k].detach().cpu(),v) for k,v in state.items()):
            raise ValueError('checkpoint action weights not restored exactly')
        self.activate('action')

    def reload_action_exact(self,path):
        """Reload any saved action adapter without a lossy BF16 round trip.

        Invoke between optimizer lifetimes only: deleting modules invalidates
        optimizer parameter references. Prefix loading semantics stay unchanged.
        """
        self.model.delete_adapter('action')
        self.model.load_adapter(str(path),adapter_name='action',is_trainable=self.train_action)
        self.restore_action_weights_exact(path)

    def save_pretrained(self,path):
        # PEFT writes named adapters to a subfolder. Export one standard adapter
        # directory, not a silently merged composite policy.
        self.model.save_pretrained(path,selected_adapters=['action'])
        import json
        (Path(path)/'action/exact_restore.json').write_text(json.dumps({'version':1,'preserve_saved_tensor_precision':True}),encoding='utf-8')


def runtime_identity(model,processor,stats):
    """Pin the encoding/numerical runtime in addition to adapter/image bytes."""
    import json,importlib.metadata
    def digest(value):
        raw=json.dumps(value,sort_keys=True,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))
        return hashlib.sha256(raw.encode()).hexdigest()
    versions={k:importlib.metadata.version(k) for k in ('torch','transformers','peft')}
    # BF16 does not depend on bitsandbytes. Preserve its absence in provenance;
    # do not install an unused quantization backend just to collect metadata.
    try:versions['bitsandbytes']=importlib.metadata.version('bitsandbytes')
    except importlib.metadata.PackageNotFoundError:versions['bitsandbytes']=None
    config=model.config.to_dict()
    return dict(model_config_sha256=digest(config),tokenizer_vocab_sha256=digest(processor.tokenizer.get_vocab()),
                action_stats_sha256=digest(stats),
                runtime_versions=versions)
