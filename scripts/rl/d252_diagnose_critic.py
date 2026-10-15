"""Reconstruct D251 failure on CPU; do not refit, retest, or launch anything."""
import copy,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/rl'))
from checkpoint_state import sha256
from executed_rollouts import attach_path_rewards
from rollout_samples import ppo_advantages
from critic_observation import critic_observation
from movement_reward import movement_reward
OUT=ROOT/'reports/d252_local_critic_cause'

def write(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def target_influence(x,groups,alpha,query):
    """Algebraic influence of each OLD target in the locked centered ridge fit."""
    x=np.asarray(x,dtype=float);groups=np.asarray(groups)
    weights=np.array([1/(len(np.unique(groups))*sum(groups==g)) for g in groups])
    scale=np.sqrt(x.shape[1]);x=x/scale;query=np.asarray(query)/scale
    mx=weights@x;z=(x-mx)*np.sqrt(weights[:,None])
    # No model selection, targets, fitted checkpoint, or new model is produced.
    h=np.linalg.solve(z@z.T+alpha*np.eye(len(x)),z@(query-mx).T).T*np.sqrt(weights)
    return weights+h-h.sum(axis=-1,keepdims=True)*weights

def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(2)
    old=ROOT/'reports/d251_critic_gate';devroot=ROOT/'reports/d250_local_critic_fit'
    pinned=['reports/d251_critic_gate/protocol.json','reports/d251_critic_gate/CRITERIA.md',
            'reports/d251_critic_gate/critic.pt','reports/d251_critic_gate/result.json',
            'scripts/rl/accepted_goal_reward.py','scripts/rl/movement_reward.py',
            'scripts/rl/regularized_critic.py','scripts/rl/train_path_ppo.py','scripts/rl/ppo_engine.py']
    hashes={n:sha256(ROOT/n) for n in pinned};fit=json.loads((old/'fit.json').read_text(encoding='utf-8'))
    assert hashes['reports/d251_critic_gate/protocol.json']==fit['protocol_sha256']
    assert hashes['reports/d251_critic_gate/critic.pt']==fit['critic_sha256']
    for n in ['scripts/rl/accepted_goal_reward.py','scripts/rl/movement_reward.py','scripts/rl/regularized_critic.py']:
        assert hashes[n]==fit['source_hashes'][n]
    assert sha256(devroot/'frozen_features.pt')==fit['development_cache_sha256']
    dev=torch.load(devroot/'frozen_features.pt',map_location='cpu',weights_only=False)
    test=torch.load(old/'test_features.pt',map_location='cpu',weights_only=False)
    x=dev['features'].numpy();q=test['features'].numpy()
    ds=[s for f in dev['flights'] for s in f['samples']];ts=[s for f in test['flights'] for s in f['samples']]
    assert [s['feature_index'] for s in ds]==list(range(len(ds)))
    assert [s['feature_index'] for s in ts]==list(range(len(ts)))
    y=np.array([s['mc_target'] for s in ds]);g=np.concatenate([np.full(len(f['samples']),i) for i,f in enumerate(dev['flights'])])
    head=torch.nn.Linear(x.shape[1],1);head.load_state_dict(torch.load(old/'critic.pt',map_location='cpu',weights_only=True))
    with torch.no_grad():pred=head(test['features']).flatten().numpy()
    stored=np.array([s['value_old'] for s in ts]);assert np.max(np.abs(pred-stored))<1e-5
    coefficients=target_influence(x,g,fit['cv']['selected_alpha'],q)
    reconstruction=coefficients@y;reconstruction_error=float(np.max(np.abs(reconstruction-stored)))
    assert reconstruction_error<1e-5
    coverage={}
    for f in dev['flights']:
        row=coverage.setdefault(f['task'],dict(flights=0,success=0,failure=0,calls=0))
        row['flights']+=1;row['success' if f['complete'] else 'failure']+=1;row['calls']+=len(f['samples'])
    observed=[];terminal_observations={};reward_errors=[];gae_errors=[]
    for split,flights in [('development',dev['flights']),('diagnostic_former_test',test['flights'])]:
        for f in flights:
            if split=='development':
                group,stem=f['id'].split('/')
                ep=ROOT/'reports/d228_local_numeric_flights'/group/'episodes'/(stem+'.json')
                task=ROOT/'reports/d224_reward_gated_ppo/tasks'/f['task']
            else:
                ep=old/'test_flights/episodes'/(f['id']+'.json');task=old/'test_flights/tasks'/f['task']
            e=json.loads(ep.read_text(encoding='utf-8'));t=json.loads(task.read_text(encoding='utf-8'))
            samples,reward=attach_path_rewards(e['samples'],e['poses'],e['executed_counts'],t['reference_path_preprocessed'],kind=e['kind'],end_reason=e['end_reason'],stopping_mode='accepted_goal')
            target=np.cumsum([s['reward_v2'] for s in samples][::-1])[::-1]
            reward_errors.extend(abs(float(a)-s['mc_target']) for a,s in zip(target,f['samples'],strict=True))
            if split=='diagnostic_former_test':
                replay=copy.deepcopy(f['samples']);ppo_advantages(replay,1.,.95)
                gae_errors.extend(abs(a['gae']-b['gae']) for a,b in zip(replay,f['samples'],strict=True))
            offset=0
            for i,s in enumerate(f['samples']):
                obs=critic_observation(e['poses'],t['reference_path_preprocessed'],executed_moves=offset,kind=e['kind'],max_moves=128)
                observed.append(dict(split=split,flight=f['id'],call=i,observation=obs))
                if i==len(f['samples'])-1:terminal_observations[f'{split}/{f["id"]}']=obs
                offset+=e['executed_counts'][i]
    assert max(reward_errors)<1e-9 and max(gae_errors)<1e-9
    cases=[]
    for f in test['flights'][-2:]:
        s=f['samples'][-1];i=s['feature_index'];a=coefficients[i]
        distances=np.linalg.norm((x-q[i])/np.sqrt(x.shape[1]),axis=1)
        key='diagnostic_former_test/'+f['id']
        cases.append(dict(flight=f['id'],complete=f['complete'],reward=s['reward_v2'],value=s['value_old'],raw_gae=s['gae'],
                          reconstruction=float(reconstruction[i]),below_lowest_training_target=bool(reconstruction[i]<y.min()),
                          minimum_training_target=float(y.min()),signed_weight_sum=float(a.sum()),negative_weight_mass=float(a[a<0].sum()),
                          target_contributions=[dict(flight=z['id'],weight=float(a[g==j].sum()),contribution=float(a[g==j]@y[g==j])) for j,z in enumerate(dev['flights'])],
                          nearest=[dict(flight=dev['flights'][g[j]]['id'],call=ds[j]['call_index'],distance=float(distances[j]),target=float(y[j]),progress_text=ds[j]['progress_text']) for j in np.argsort(distances)[:5]],
                          pre_action_observation=terminal_observations[key]))
    keys=lambda data:{(s['image_sha256'],tuple(s['context_input_ids'])) for f in data['flights'] for s in f['samples']}
    exact_overlap=len(keys(dev)&keys(test));assert exact_overlap==0
    # Constructed geometry branch from the real failed pre-action state.
    # Not a model flight or estimate of the current policy's expected return.
    f=test['flights'][-2];offset=sum(f['counts'][:-1]);prefix=np.asarray(f['poses'][:offset+1])
    reference=f['reference'];kw=dict(kind=f['kind'],stopping_mode='accepted_goal')
    prefix_reward=movement_reward(prefix,reference,stopped=False,**kw)['reward']
    stop_path=np.vstack([prefix,np.repeat(prefix[-1:],10,axis=0)])
    tail=np.linspace(prefix[-1],reference[-1],21)[1:]
    finish_path=np.vstack([prefix,tail,np.repeat([reference[-1]],10,axis=0)])
    control=dict(source_flight=f['id'],call=len(f['samples'])-1,executed_prefix_moves=offset,
                 type='constructed geometry control, not VLA or physical simulator flight',branches={})
    for name,path in [('stop',stop_path),('continue',finish_path)]:
        score=movement_reward(path,reference,stopped=True,**kw)
        control['branches'][name]=dict(remaining_return=score['reward']-prefix_reward,
            complete=score['complete'],invalid=score['invalid_transitions'],max_deviation=score['max_deviation'],
            endpoint_error_m=score['endpoint_position_error_m'],heading_error_deg=score['endpoint_yaw_error_deg'],
            total_moves=len(path)-1,poses=path.tolist())
    assert control['branches']['continue']['complete'] and not control['branches']['stop']['complete']
    assert control['branches']['continue']['remaining_return']>control['branches']['stop']['remaining_return']
    assert control['branches']['continue']['total_moves']<=128
    write('geometry_control.json',control)
    original=json.loads((old/'result.json').read_text(encoding='utf-8'));assert not original['passed']
    assert hashes=={n:sha256(ROOT/n) for n in pinned}
    result=dict(scope='CPU diagnosis and causal input repair only; not a new final test',
        actor_updates=0,new_flights=0,new_critic_checkpoint=False,aws_used=False,
        locked_criteria_unchanged=True,reward_unchanged=True,original_gate_passed=False,
        source_hashes=hashes,cache_hashes={str(p.relative_to(ROOT)):sha256(p) for p in [devroot/'frozen_features.pt',old/'test_features.pt']},
        development_flights=5,development_calls=len(ds),feature_dimension=x.shape[1],development_coverage=coverage,
        all48_prediction_reconstruction_max_error=reconstruction_error,mc_recompute_max_error=max(reward_errors),gae_recompute_max_error=max(gae_errors),
        exact_context_overlap=exact_overlap,cases=cases,
        development_late_failed_descent=terminal_observations['development/finish/sim_2025-03-27_19-11-33_r0'],
        input_candidate='critic_observation.py exposes existing measured prefix geometry, acceptance history, quiet counter and remaining budget to a future critic; not wired into the actor or training',
        limitations=['Same-state policy continuation distribution was not sampled; positive individual advantage alone does not prove an incorrect policy expectation.',
                     'No successful steep-descent development flight. Coverage confound is observed, not a causal retraining ablation.',
                     'The old failed development descent reached the endpoint but violated route history; current image/pose/progress do not explicitly encode that history.',
                     'Feature repair is not a fitted or validated critic. D251 remains failed; no training readiness is claimed.',
                     'D251 flights are now diagnostic data and cannot be relabelled untouched validation. No additional final-test seeds or refit were run.'])
    write('diagnosis.json',result);write('critic_observations.json',observed)
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_hashes','cache_hashes','cases','limitations','development_late_failed_descent']},indent=2))
if __name__=='__main__':main()
