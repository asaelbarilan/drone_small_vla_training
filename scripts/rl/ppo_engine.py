"""D215 bounded synchronous PPO update engine, testable without a VLM.

forward(samples) -> action log probabilities, detached-capable context features.
bootstrap(sample) -> final-observation context features for truncated episodes.
The caller owns exact encoding/provenance and actor adapter restoration.
"""
import copy
import json
import math
import os
from pathlib import Path
import random
import time
from dataclasses import dataclass,asdict
import numpy as np
import torch
from checkpoint_state import save_checkpoint,restore_training_state,verified_checkpoint
from ppo_math import joint_ppo_loss,replay_parity,update_batches,validate_ppo_samples
from rollout_samples import ppo_advantages

@dataclass(frozen=True)
class PPOConfig:
    chunk:int=8
    batch_size:int=1
    accum:int=8
    epochs:int=1
    warmup_steps:int=8
    gamma:float=1.0
    gae_lambda:float=.95
    clip:float=.05
    value_coef:float=.5
    max_grad_norm:float=1.0
    max_updates:int=16
    max_seconds:float=1800
    hard_kl:float=.05
    seed:int=7
    advantage_mode:str="centered"

    def __post_init__(self):
        if self.advantage_mode not in ('centered','scale_only'):raise ValueError('unknown advantage mode')
        if min(self.chunk,self.batch_size,self.accum,self.epochs,self.max_updates)<=0 or self.warmup_steps<0:
            raise ValueError('invalid PPO counts')
        if self.gamma!=1.0:raise ValueError('geometry pilot requires gamma=1; discount changes need separate reward tests')
        if not all(np.isfinite(x) and x>0 for x in (self.max_seconds,self.hard_kl,self.clip,self.max_grad_norm)):
            raise ValueError('invalid numerical limits')


def prepare_advantages(advantages, mode="centered"):
    a=np.asarray(advantages,dtype=float)
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all():raise ValueError("finite advantages required")
    if mode not in ("centered","scale_only"):raise ValueError("unknown advantage mode")
    spread=a.std()
    if spread<1e-8:raise ValueError("zero advantage variance; no informative policy update")
    return (a-a.mean() if mode=="centered" else a)/(spread+1e-8)


def normalized_features(features):
    return torch.nn.functional.layer_norm(features.float().detach(),(features.shape[-1],))


def write_kl_failure(root, *, phase, step, window, indices, samples, new, old, threshold, estimate, aggregate=None, aggregation=None):
    """Persist strict JSON and raw selected-token probabilities before raising."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    def safe(x):
        if isinstance(x, float) and not math.isfinite(x):return str(x)
        if isinstance(x, list):return [safe(v) for v in x]
        return x
    with torch.no_grad():
        delta=new.detach()-old.detach();joint=delta.sum(-1);ratios=joint.exp();k3=ratios-1-joint
        rows=[dict(sample_index=int(i),task=samples[i].get('task'),passage=samples[i].get('passage'),call_index=samples[i].get('call_index'),image_sha256=samples[i].get('image_sha256'),token_ids=samples[i].get('ids'),old_logps=safe(old[j].detach().cpu().tolist()),new_logps=safe(new[j].detach().cpu().tolist()),token_logp_delta=safe(delta[j].cpu().tolist()),joint_log_ratio=safe(float(joint[j])),joint_ratio=safe(float(ratios[j])),sample_k3=safe(float(k3[j]))) for j,i in enumerate(indices)]
    value=float(estimate)
    report=dict(schema='ppo-kl-failure-v1',phase=phase,last_committed_step=step,attempted_step=step+1,window_index=window,threshold=threshold,estimate=safe(value),reason='nonfinite' if not math.isfinite(value) else 'threshold_exceeded',aggregation=aggregation or ('microbatch_mean' if phase=='pre_step' else 'full_batch_mean'),aggregate=safe(aggregate),rows=rows)
    pointer=root/'CURRENT.json'
    report['committed_checkpoint']=json.loads(pointer.read_text()) if pointer.exists() else None
    dest=root.parent/'kl_failure.json';tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False));os.replace(tmp,dest)
    print('PPO_KL_FAILURE '+json.dumps({k:v for k,v in report.items() if k!='rows'},allow_nan=False),flush=True)
    return dest


def train_batch(actor,critic,optimizer,samples,forward,bootstrap,checkpoint_dir,*,config=None,resume=False,identity=None,pause_after_updates=None):
    c=config or PPOConfig();start=time.monotonic();deadline=start+c.max_seconds
    rng=random.Random(c.seed);samples=copy.deepcopy(samples)
    validate_ppo_samples(samples,c.chunk,require_final_value=False)
    if any(s.get('discount_per_call',1.0)!=1.0 for s in samples):raise ValueError('reward discount mismatch')
    all_params=[p for group in optimizer.param_groups for p in group['params']]
    logs=[];step=0;cursor=0
    saved_config=asdict(c)
    # Preserve legacy centered checkpoint identity; explicit alternatives are pinned.
    if c.advantage_mode=="centered":saved_config.pop("advantage_mode")
    metadata_base=dict(config=saved_config,identity=identity)
    actor_model=getattr(actor,'model',actor)
    actor_model.eval()

    def time_guard():
        if time.monotonic()>=deadline:raise TimeoutError('PPO wall budget exhausted')

    def values_and_bootstrap():
        with torch.no_grad():
            for i in range(0,len(samples),c.batch_size):
                part=samples[i:i+c.batch_size];_,features=forward(part)
                values=critic(normalized_features(features)).flatten().tolist()
                for s,v in zip(part,values,strict=True):s['value_old']=v
            for s in samples:
                if s['truncated']:
                    s['final_value']=float(critic(normalized_features(bootstrap(s))).reshape(-1)[0])

    if resume:
        state=restore_training_state(checkpoint_dir,optimizer,critic=critic,shuffle_rng=rng,expected_metadata=metadata_base)
        m=state['metadata']
        if m['config']!=saved_config or m['identity']!=identity:raise ValueError('resume configuration/identity mismatch')
        samples=m['samples'];orders=m['orders'];cursor=m['cursor'];logs=m['logs'];step=state['step']
    else:
        cached=[]
        with torch.no_grad():
            for i in range(0,len(samples),c.batch_size):
                time_guard();part=samples[i:i+c.batch_size];lp,features=forward(part)
                replay_parity([s['logps'] for s in part],lp.cpu().tolist())
                cached.extend(normalized_features(features).unbind(0))
        values_and_bootstrap();ppo_advantages(samples,c.gamma,c.gae_lambda)
        targets=torch.tensor([s['return'] for s in samples],device=cached[0].device)
        features=torch.stack(cached)
        before={n:p.detach().cpu().clone() for n,p in actor_model.named_parameters() if p.requires_grad}
        for _ in range(c.warmup_steps):
            time_guard();optimizer.zero_grad(set_to_none=True)
            prediction=critic(features).flatten();loss=((prediction-targets)**2).mean()
            if not torch.isfinite(loss):raise ValueError('nonfinite critic warmup')
            loss.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),c.max_grad_norm,error_if_nonfinite=True);optimizer.step()
        for n,p in actor_model.named_parameters():
            if n in before and not torch.equal(p.detach().cpu(),before[n]):raise ValueError('critic warmup changed actor')
        values_and_bootstrap();ppo_advantages(samples,c.gamma,c.gae_lambda)
        advantages=np.array([s['gae'] for s in samples]);spread=advantages.std()
        if spread<1e-8:raise ValueError('zero advantage variance; no informative policy update')
        for s,a in zip(samples,prepare_advantages(advantages,c.advantage_mode),strict=True):s['advantage']=float(a)
        orders=[]
        for _ in range(c.epochs):
            order=list(range(len(samples)));rng.shuffle(order);orders.extend(list(update_batches(order,c.batch_size,c.accum)))
        save_checkpoint(checkpoint_dir,actor,critic,optimizer,step=0,
                        metadata=metadata_base|dict(samples=samples,orders=orders,cursor=0,logs=[]),shuffle_rng=rng)

    # Legacy checkpoints preserve optimizer/RNG/sample order; the explicit policy
    # transition changes only how finite KL estimates are aggregated.
    metadata_base['kl_policy']='full_batch_mean_v2'
    if resume and m.get('kl_policy') not in (None,'full_batch_mean_v2'):
        raise ValueError('unsupported checkpoint KL policy')

    def audit_batch_kl(phase,window):
        total=0.;rows=[]
        with torch.no_grad():
            for j in range(0,len(samples),c.batch_size):
                time_guard();part=samples[j:j+c.batch_size];lp,_=forward(part)
                old=torch.tensor([s['logps'] for s in part],device=lp.device,dtype=lp.dtype)
                delta=(lp-old).sum(-1);estimate=(delta.exp()-1-delta).mean()
                ids=list(range(j,j+len(part)))
                if not torch.isfinite(estimate):
                    path=write_kl_failure(checkpoint_dir,phase=phase,step=step,window=window,indices=ids,samples=samples,new=lp,old=old,threshold=c.hard_kl,estimate=estimate,aggregation='microbatch_mean')
                    raise ValueError('nonfinite PPO KL; diagnostics: '+str(path))
                total+=len(part)/len(samples)*float(estimate)
                rows.append((ids,lp.detach().cpu(),old.detach().cpu()))
        if total>c.hard_kl:
            path=write_kl_failure(checkpoint_dir,phase=phase,step=step,window=window,indices=[i for ids,_,_ in rows for i in ids],samples=samples,new=torch.cat([x for _,x,_ in rows]),old=torch.cat([x for _,_,x in rows]),threshold=c.hard_kl,estimate=total,aggregate=total,aggregation='full_batch_mean')
            raise ValueError('PPO full-batch KL limit reached; candidate not committed; diagnostics: '+str(path))
        return total,rows

    # Validate the actual restored model before any new optimizer step. Later
    # windows reuse the preceding post-update acceptance (weights unchanged).
    pre_kl,_=audit_batch_kl('pre_update',cursor)
    print('PPO_KL_ADMISSION',json.dumps(dict(step=step,mean=pre_kl,threshold=c.hard_kl,policy=metadata_base['kl_policy'])),flush=True)
    committed_cursor=cursor
    for index in range(cursor,len(orders)):
        time_guard()
        if step>=c.max_updates or (pause_after_updates is not None and step>=pause_after_updates):break
        optimizer.zero_grad(set_to_none=True);record=dict(pg=0.,value=0.,kl=0.,clip_fraction=0.)
        for indices,weight in orders[index]:
            time_guard();part=[samples[j] for j in indices];new,features=forward(part)
            old=torch.tensor([s['logps'] for s in part],device=new.device,dtype=new.dtype)
            adv=torch.tensor([s['advantage'] for s in part],device=new.device,dtype=new.dtype)
            log_ratio=(new-old).sum(-1)
            ratio=log_ratio.exp()
            # A finite microbatch outlier is diagnostic, not a batch-level veto.
            kl=(ratio-1-log_ratio).mean()
            if not torch.isfinite(kl):
                path=write_kl_failure(checkpoint_dir,phase='pre_step',step=step,window=index,indices=indices,samples=samples,new=new,old=old,threshold=c.hard_kl,estimate=kl)
                optimizer.zero_grad(set_to_none=True);raise ValueError('PPO KL limit reached; diagnostics: '+str(path))
            pg,ratio=joint_ppo_loss(new,old,adv,c.clip)
            predicted=critic(normalized_features(features)).flatten()
            target=torch.tensor([s['return'] for s in part],device=predicted.device)
            value=((predicted-target)**2).mean();loss=pg+c.value_coef*value
            if not torch.isfinite(loss):raise ValueError('nonfinite PPO loss')
            (weight*loss).backward()
            for key,val in [('pg',pg),('value',value),('kl',kl),('clip_fraction',((ratio-1).abs()>c.clip).float().mean())]:
                record[key]+=weight*float(val.detach())
        norm=torch.nn.utils.clip_grad_norm_(all_params,c.max_grad_norm,error_if_nonfinite=True)
        optimizer.step()
        post_kl,post_records=audit_batch_kl('post_update',index)
        post_d=torch.cat([(new-old).sum(-1) for _,new,old in post_records])
        post_samples=post_d.exp()-1-post_d
        worst=int(post_samples.argmax())
        step+=1;record.update(step=step,grad_norm=float(norm),post_kl=post_kl,
            post_kl_max=float(post_samples[worst]),post_kl_worst_sample=worst,
            post_kl_samples_over_limit=int((post_samples>c.hard_kl).sum()));logs.append(record)
        save_checkpoint(checkpoint_dir,actor,critic,optimizer,step=step,
                        metadata=metadata_base|dict(samples=samples,orders=orders,cursor=index+1,logs=logs),shuffle_rng=rng)
        committed_cursor=index+1
        print('PPO_COMMITTED',record,flush=True)
    if step==0:raise ValueError('no policy updates completed')
    # Only committed windows count; forward-only KL/critic passes are not actor coverage.
    consumed=[j for window in orders[:committed_cursor] for indices,_ in window for j in indices]
    unique=set(consumed);per_task={}
    for j in consumed:
        task=samples[j]['task'];per_task[task]=per_task.get(task,0)+1
    coverage=dict(total_samples=len(samples),scheduled_updates=len(orders),
                  committed_windows=committed_cursor,actor_sample_visits=len(consumed),
                  unique_actor_samples=len(unique),unvisited_sample_indices=sorted(set(range(len(samples)))-unique),
                  actor_visits_per_task=per_task,schedule_complete=committed_cursor==len(orders),
                  stop_reason='schedule_exhausted' if committed_cursor==len(orders) else ('intermediate_check' if pause_after_updates is not None and step>=pause_after_updates else 'max_updates'))
    return dict(updates=logs,step=step,elapsed_s=time.monotonic()-start,
                checkpoint=str(verified_checkpoint(checkpoint_dir)),config=asdict(c),coverage=coverage,kl_policy=metadata_base['kl_policy'])


