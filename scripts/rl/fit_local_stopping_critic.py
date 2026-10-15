"""D250 frozen real-VLA features; CPU critic fit, held-out whole flights, no cloud."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import json,time,sys,hashlib,copy,gc
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'scripts/rl'))
from checkpoint_state import sha256
from composite_policy import CompositePolicy,adapter_digest,runtime_identity
from executed_rollouts import attach_path_rewards
from rollout_samples import ppo_advantages
from ppo_engine import prepare_advantages,normalized_features
OUT=ROOT/'reports/d250_local_critic_fit'
ADAPTER=Path('D:/drone_vla_pilot/adapters/d207_s8000')

def write(name,x):
    OUT.mkdir(exist_ok=True);(OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def prepare():
    flights=[];excluded=[];source={}
    for group in ['retry1','finish']:
        folder=ROOT/'reports/d228_local_numeric_flights'/group
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        assert adapter_digest(ADAPTER)==manifest['adapter_sha256']
        for path in sorted((folder/'episodes').glob('*_r?.json')):
            e=json.loads(path.read_text(encoding='utf-8'));source[str(path.relative_to(ROOT))]=sha256(path)
            if e['end_reason']!='model_stop':excluded.append(dict(path=str(path.relative_to(ROOT)),reason='censored_timeout'));continue
            tpath=ROOT/'reports/d224_reward_gated_ppo/tasks'/e['task'];assert sha256(tpath)==manifest['tasks'][e['task']]
            t=json.loads(tpath.read_text(encoding='utf-8'))
            samples,reward=attach_path_rewards(e['samples'],e['poses'],e['executed_counts'],t['reference_path_preprocessed'],kind=e['kind'],end_reason=e['end_reason'],stopping_mode='accepted_goal')
            returns=np.cumsum([s['reward_v2'] for s in samples][::-1])[::-1]
            for s,target in zip(samples,returns):
                s['mc_target']=float(target);s['reward_spec_approved']=False
                image=(folder/'rollout_images'/s['image']).resolve();assert image.is_relative_to((folder/'rollout_images').resolve())
                assert sha256(image)==s['image_sha256'];s['local_image_path']=str(image)
            split='train' if e['seed']==7 else 'heldout';assert e['seed'] in [7,8]
            flights.append(dict(id=group+'/'+path.stem,split=split,seed=e['seed'],task=e['task'],complete=reward['complete'],samples=samples,reward=reward['reward']))
    assert len(flights)==5 and sum(f['split']=='train' for f in flights)==2
    return flights,excluded,source

def fingerprint(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode());h.update(str(p.dtype).encode())
        flat=p.detach().reshape(-1)
        for start in range(0,flat.numel(),1000000):h.update(flat[start:start+1000000].cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def fit_head(x,y,steps=200,lr=1e-4):
    torch.manual_seed(250);head=torch.nn.Linear(x.shape[1],1)
    torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)
    opt=torch.optim.AdamW(head.parameters(),lr=lr);log=[]
    for step in range(steps):
        opt.zero_grad();loss=((head(x).flatten()-y)**2).mean()
        if not torch.isfinite(loss):raise ValueError('nonfinite critic loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);opt.step()
        if (step+1)%20==0:log.append(dict(step=step+1,training_mse=float(loss.detach())))
    return head,log

def main():
    start=time.monotonic();torch.set_num_threads(4);flights,excluded,source=prepare()
    protocol=dict(date='2026-10-15',training='seed7 complete flights only',heldout='seed8 complete flights, never used for model selection',excluded=excluded,
      flights=[{k:v for k,v in f.items() if k!='samples'}|dict(calls=len(f['samples'])) for f in flights],
      critic='zero-initialized linear head over layer-normalized final prompt hidden state',optimizer='AdamW',lr=1e-4,weight_decay=.01,steps=200,grad_clip=1.,seed=250,
      actor_frozen=True,features_precision='NF4/BF16 as recorded',target='undiscounted Monte Carlo return from actual terminal episodes; no bootstrap',
      acceptance='finite outputs, frozen actor fingerprint, heldout MSE better than training-mean constant, all3heldout terminal raw and train-centered advantage signs match completion; otherwise stop. No tuning after viewing heldout.',
      source_hashes=source,script_sha256=sha256(Path(__file__)))
    if (OUT/'report.json').exists():raise ValueError('completed fit exists; no automatic refit')
    write('protocol.json',protocol);print('PROTOCOL_SAVED',flush=True)
    from train_uav_flow_vla import setup,official_stats,encode
    from uav_flow_action_tokenizer import ActionTokenizer
    from model_replay import replay_cached_tokens
    from PIL import Image
    api,processor,base,_=setup('qwen','nf4')
    policy=CompositePolicy.load(base,ADAPTER,ADAPTER,train_action=False);policy.activate('action')
    policy.model.gradient_checkpointing_disable();policy.model.requires_grad_(False)
    assert not any(p.requires_grad for p in policy.model.parameters())
    runtime=runtime_identity(policy.model,processor,official_stats());saved=flights[0]['samples'][0]['policy_provenance']
    differences={k:dict(saved=saved.get(k),actual=v) for k,v in runtime.items() if saved.get(k)!=v}
    write('runtime.json',dict(differences=differences,actual=runtime))
    if differences:raise ValueError('recorded runtime differs; inspect runtime.json before extracting features')
    before=fingerprint(policy.model);tokenizer=ActionTokenizer(processor.tokenizer,official_stats());features=[];parity=[]
    for f in flights:
        for s in f['samples']:
            if time.monotonic()-start>3600:raise TimeoutError('one-hour local extraction cap')
            image=Image.open(s['local_image_path']).convert('RGB')
            row={k:s[k] for k in ['prompt','box_text','progress_text']}
            batch=encode(processor,tokenizer,row,8,with_answer=False,image=image)
            assert batch['input_ids'][0].tolist()==s['context_input_ids']
            with torch.no_grad():
                lp,feature=replay_cached_tokens(policy.model.get_base_model(),api.cuda(batch),s['ids'][:1],low=tokenizer.vocab_size-tokenizer.n_bins,high=tokenizer.vocab_size,temperature=s['temperature'],pad_token_id=processor.tokenizer.pad_token_id)
                error=abs(float(lp[0])-s['logps'][0])
                if not np.isfinite(error) or error>.01:raise ValueError('first-token probability replay mismatch '+str(error))
                vector=normalized_features(feature).cpu();assert torch.isfinite(vector).all()
                s['feature_index']=len(features);features.append(vector)
            parity.append(error);write('progress.json',dict(features=len(features),flight=f['id'],call=s['call_index'],max_first_token_error=max(parity),elapsed_s=time.monotonic()-start))
            print('FEATURE',len(features),f['id'],s['call_index'],'parity',error,flush=True)
    after=fingerprint(policy.model);assert before==after and not any(p.requires_grad for p in policy.model.parameters())
    x=torch.stack(features);assert not x.requires_grad
    torch.save(dict(features=x,flights=flights,actor_sha256=before),OUT/'frozen_features.pt')
    del policy,base;gc.collect();torch.cuda.empty_cache()
    train=[s for f in flights if f['split']=='train' for s in f['samples']];test=[s for f in flights if f['split']=='heldout' for s in f['samples']]
    train_idx=[s['feature_index'] for s in train];test_idx=[s['feature_index'] for s in test]
    assert not set(train_idx)&set(test_idx)
    y=torch.tensor([s['mc_target'] for s in train],dtype=torch.float32)
    head,log=fit_head(x[train_idx],y);torch.save(head.state_dict(),OUT/'critic.pt')
    # Checkpoint is locked BEFORE the first heldout prediction/metric.
    critic_hash=sha256(OUT/'critic.pt')
    with torch.no_grad():pred=head(x).flatten().tolist()
    for f in flights:
        for s in f['samples']:s['value_old']=pred[s['feature_index']]
        ppo_advantages(f['samples'],1,.95)
    train_raw=np.array([s['gae'] for s in train]);mean=float(train_raw.mean());std=float(train_raw.std())
    terminal=[]
    for f in flights:
        s=f['samples'][-1];raw=s['gae'];center=(raw-mean)/(std+1e-8)
        terminal.append(dict(flight=f['id'],split=f['split'],complete=f['complete'],prediction=s['value_old'],terminal_reward=s['reward_v2'],raw_gae=raw,train_centered=center,sign_correct=(raw>0)==f['complete'],centered_sign_correct=(center>0)==f['complete']))
    def metrics(samples):
        target=np.array([s['mc_target'] for s in samples]);prediction=np.array([s['value_old'] for s in samples]);baseline=float(y.mean())
        return dict(samples=len(samples),mse=float(np.mean((target-prediction)**2)),constant_training_mean_mse=float(np.mean((target-baseline)**2)),zero_mse=float(np.mean(target**2)))
    train_keys={s['image_sha256']+str(s['context_input_ids']) for s in train};test_keys={s['image_sha256']+str(s['context_input_ids']) for s in test}
    heldout=metrics(test);stop_checks=[r for r in terminal if r['split']=='heldout']
    passed=heldout['mse']<heldout['constant_training_mean_mse'] and all(r['sign_correct'] and r['centered_sign_correct'] for r in stop_checks) and not train_keys&test_keys
    write('report.json',dict(passed=passed,training=metrics(train),heldout=heldout,terminal_signals=terminal,training_curve=log,train_advantage_mean=mean,train_advantage_std=std,exact_context_overlap=len(train_keys&test_keys),
       actor_frozen=True,actor_before_sha256=before,actor_after_sha256=after,critic_sha256=critic_hash,actor_updates=0,critic_updates=200,
       first_token_replay_max_error=max(parity),elapsed_s=time.monotonic()-start,limitations=['Only2trainingand3heldoutflights, same scene; no generalization proof.','NF4 local feature/critic diagnostic is not BF16 AWS qualification.','No action model update or new flight; heldout signal is for recorded actions only.']))
    write('predictions.json',[dict(flight=f['id'],split=f['split'],call=s['call_index'],target=s['mc_target'],prediction=s['value_old'],raw_gae=s['gae']) for f in flights for s in f['samples']])
    print('FIT_RESULT',json.dumps(dict(passed=passed,heldout=heldout,terminal_signals=terminal)),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        write('failure.json',dict(error=type(e).__name__,message=str(e),aws_used=False));raise