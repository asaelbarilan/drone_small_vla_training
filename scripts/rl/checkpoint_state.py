"""Atomic committed checkpoints for a local synchronous trainer.

A failed write never replaces CURRENT.json. Incomplete directories remain for
inspection. No cloud operations, process restart or automatic resume.
"""
import hashlib
import json
import os
import random
import uuid
from pathlib import Path
import numpy as np
import torch


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def save_checkpoint(root, actor, critic, optimizer, *, step, metadata, scheduler=None, scaler=None, shuffle_rng=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if step < 0:raise ValueError('negative update step')
    name=f'step_{step:06d}_{uuid.uuid4().hex}'
    folder=root/name;folder.mkdir()
    actor.save_pretrained(folder/'adapter')
    state=dict(version=1,step=step,metadata=metadata,
               critic=None if critic is None else critic.state_dict(),optimizer=optimizer.state_dict(),
               scheduler=None if scheduler is None else scheduler.state_dict(),
               scaler=None if scaler is None else scaler.state_dict(),
               python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
               cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
               shuffle_rng=shuffle_rng.getstate() if shuffle_rng else None)
    torch.save(state,folder/'state.pt')
    files=sorted(x for x in folder.rglob('*') if x.is_file())
    if not any((folder/'adapter') in x.parents for x in files):raise ValueError('empty adapter checkpoint')
    manifest=dict(version=1,step=step,files={str(x.relative_to(folder)):sha256(x) for x in files})
    manifest_path=folder/'manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    pointer=root/('CURRENT.'+uuid.uuid4().hex+'.tmp')
    with pointer.open('w',encoding='utf-8') as f:
        json.dump(dict(directory=name,manifest_sha256=sha256(manifest_path)),f)
        f.flush();os.fsync(f.fileno())
    os.replace(pointer,root/'CURRENT.json')
    return folder


def verified_checkpoint(root):
    root=Path(root).resolve()
    pointer=json.loads((root/'CURRENT.json').read_text(encoding='utf-8'))
    folder=(root/pointer['directory']).resolve()
    if folder.parent!=root:raise ValueError('checkpoint path escapes root')
    manifest_path=folder/'manifest.json'
    if sha256(manifest_path)!=pointer['manifest_sha256']:raise ValueError('manifest checksum mismatch')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    for relative,expected in manifest['files'].items():
        path=(folder/relative).resolve()
        if not path.is_relative_to(folder) or sha256(path)!=expected:
            raise ValueError('checkpoint file checksum/path mismatch')
    return folder


def restore_training_state(root, optimizer, critic=None, scheduler=None, scaler=None, shuffle_rng=None, expected_metadata=None):
    """Trusted local checkpoints only. Load actor adapter before calling.

    The caller must reconstruct identical optimizer groups and validate metadata.
    This helper does not authorize resuming a simulator or partially flown batch.
    """
    folder=verified_checkpoint(root)
    state=torch.load(folder/'state.pt',map_location='cpu',weights_only=False)
    if expected_metadata is not None and any(state['metadata'].get(k)!=v for k,v in expected_metadata.items()):
        raise ValueError('resume configuration/identity mismatch')
    for key,obj in [('critic',critic),('scheduler',scheduler),('scaler',scaler)]:
        if (state[key] is None)!=(obj is None):raise ValueError('checkpoint component mismatch: '+key)
        if obj is not None:obj.load_state_dict(state[key])
    optimizer.load_state_dict(state['optimizer'])
    random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng'] is not None:
        if not torch.cuda.is_available():raise ValueError('CUDA RNG state requires CUDA')
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    if state['shuffle_rng'] is not None:
        if shuffle_rng is None:raise ValueError('shuffle RNG missing')
        shuffle_rng.setstate(state['shuffle_rng'])
    return state
