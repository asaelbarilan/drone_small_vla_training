"""Local audit helpers. Preserve effective loaded weights, not just file identity."""
import hashlib,json
from pathlib import Path
import torch
from peft.utils.save_and_load import get_peft_model_state_dict,set_peft_model_state_dict

def snapshot(model,name='action'):
    return {k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model,adapter_name=name).items()}

def describe(state):
    return {k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()) for k,v in state.items()}

def compare(a,b):
    if set(a)!=set(b):return dict(equal=False,key_mismatch=True)
    changed=[]
    for k in a:
        if a[k].dtype!=b[k].dtype or a[k].shape!=b[k].shape or not torch.equal(a[k],b[k]):
            changed.append(dict(key=k,before_dtype=str(a[k].dtype),after_dtype=str(b[k].dtype),max_abs=float((a[k].float()-b[k].float()).abs().max()) if a[k].shape==b[k].shape else None))
    return dict(equal=not changed,changed=changed)

def restore_snapshot(model,reference):
    current=snapshot(model)
    if set(current)!=set(reference) or any(current[k].dtype!=v.dtype or current[k].shape!=v.shape for k,v in reference.items()):
        raise ValueError('effective adapter snapshot dtype/shape/keys differ')
    set_peft_model_state_dict(model,reference,adapter_name='action')
    if not compare(reference,snapshot(model))['equal']:raise ValueError('effective adapter snapshot restore failed')

def save_evidence(out,raw,report):
    import numpy as np
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'same_input_logits.npz',**raw)
    temp=out/'same_input.json.tmp'
    temp.write_text(json.dumps(report,indent=2),encoding='utf-8')
    temp.replace(out/'same_input.json')
