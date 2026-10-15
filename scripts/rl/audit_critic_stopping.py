"""D247: audit recorded behavior-critic values against measured and constructed returns.
No VLA forward, cloud, flight, optimizer or checkpoint mutation.
"""
import copy,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/rl'))
from checkpoint_state import verified_checkpoint,sha256
from movement_reward import movement_reward
from rollout_samples import ppo_advantages

def remaining_returns(calls):
    if not calls or not (calls[-1]['terminated'] or calls[-1]['truncated']):
        raise ValueError('explicit final boundary required')
    if any(s['terminated'] or s['truncated'] for s in calls[:-1]):
        raise ValueError('interior boundary')
    if calls[-1]['terminated'] and calls[-1]['truncated']:raise ValueError('ambiguous boundary')
    observed=np.cumsum([s['reward_v2'] for s in calls][::-1])[::-1]
    # A timeout's missing future cannot be called a measured Monte Carlo target.
    return observed,[float(x) for x in observed] if calls[-1]['terminated'] else [None]*len(calls)

def run(root=ROOT):
    cp=verified_checkpoint(root/'reports/d239_kl_diagnosis/checkpoints')
    state=torch.load(cp/'state.pt',map_location='cpu',weights_only=False)
    samples=state['metadata']['samples'];assert len(samples)==32
    check=copy.deepcopy(samples);ppo_advantages(check,1,.95)
    assert all(abs(a['gae']-b['gae'])<1e-9 and abs(a['return']-b['return'])<1e-9 for a,b in zip(samples,check))
    base=root/'reports/d238_two_task/evidence/round_1';rows=[];sources={};episodes=[]
    for p in sorted((base/'attempts/episodes').glob('*_r?.json')):
        e=json.loads(p.read_text());taskfile=base/'tasks'/e['task'];task=json.loads(taskfile.read_text())
        sources[str(p.relative_to(root))]=sha256(p);sources[str(taskfile.relative_to(root))]=sha256(taskfile)
        g=[s for s in samples if s['task']==e['task'] and s['passage']==e['repeat']]
        assert len(g)==len(e['samples'])
        observed,mc=remaining_returns(g);offset=0
        for j,(s,original) in enumerate(zip(g,e['samples'])):
            for key in ['ids','context_input_ids','image_sha256','executed_count']:
                assert s[key]==original[key]
            assert abs(s['reward_v2']-original['reward_v2'])<1e-9
            poses=np.asarray(e['poses'],float);prefix=poses[:offset+1]
            # Ideal stationary counterfactual at a real recorded state. No future
            # pixels or action tokens are fabricated or admitted to PPO.
            hold=np.vstack([prefix,np.repeat(prefix[-1:],10,axis=0)])
            ref=task['reference_path_preprocessed'];kw=dict(kind=e['kind'])
            stop=movement_reward(hold,ref,stopped=True,**kw)
            base_reward=movement_reward(hold,ref,stopped=False,**kw)['reward']
            assert not stop['invalid_transitions']
            assert abs(base_reward-sum(t['reward_v2'] for t in g[:j]))<1e-8
            qstop=stop['reward']-base_reward;v=s['value_old']
            row=dict(task=e['task'],kind=e['kind'],seed=e['seed'],call=j,
              value_used_for_advantages=v,ideal_hold_remaining_return=qstop,
              ideal_hold_mc_advantage=qstop-v,hold_meets_geometry=stop['complete'],
              endpoint_m=stop['endpoint_position_error_m'],heading_error_deg=stop['endpoint_yaw_error_deg'],heading_tolerance_deg=stop['heading_tolerance_deg'],
              observed_reward_to_boundary=float(observed[j]),measured_continuation_return=mc[j],
              measured_mc_advantage=None if mc[j] is None else mc[j]-v,
              actual_raw_gae=s['gae'],actual_centered_advantage=s['advantage'],
              timeout_censored=e['end_reason']=='horizon',executed_count=s['executed_count'])
            rows.append(row);offset+=s['executed_count']
        assert offset==len(e['poses'])-1
        episodes.append(dict(task=e['task'],seed=e['seed'],calls=len(g),end_reason=e['end_reason'],complete=e['reward']['complete']))
    assert len(rows)==32
    early=[r for r in rows if not r['hold_meets_geometry']];target=[r for r in rows if r['hold_meets_geometry']]
    wrong_early=[r for r in early if r['ideal_hold_mc_advantage']>0]
    negative_target=[r for r in target if r['ideal_hold_mc_advantage']<0]
    measured=[r for r in rows if r['measured_continuation_return'] is not None]
    sensitivity=[]
    for lam in [.9,.95,1.]:
        x=copy.deepcopy(samples);ppo_advantages(x,1,lam)
        sensitivity.append(dict(lam=lam,positive_actual_advantages=sum(s['gae']>0 for s in x),values=[s['gae'] for s in x]))
    for s,r in zip(samples,check):assert abs(s['gae']-r['gae'])<1e-9
    result=dict(scope='Stored pre-actor-update critic outputs that generated D238/D240 advantages; not final step4 critic inference.',
      checkpoint_step=state['step'],checkpoint_state_sha256=sha256(cp/'state.pt'),sources=sources,episodes=episodes,rows=rows,
      counts=dict(states=len(rows),early_states=len(early),early_negative=sum(r['ideal_hold_mc_advantage']<0 for r in early),target_states=len(target),target_positive=sum(r['ideal_hold_mc_advantage']>0 for r in target),measured_terminal_states=len(measured),censored_states=len(rows)-len(measured)),
      positive_incomplete_hold=wrong_early,negative_complete_hold=negative_target,
      measured_return_rmse=float(np.sqrt(np.mean([(r['value_used_for_advantages']-r['measured_continuation_return'])**2 for r in measured]))),
      gae_sensitivity=sensitivity,
      verdict=dict(gae_reproduction_passed=True,critic_bug_established=False,critic_calibration_established=False,paid_training_ready=False,
        reason='One observed continuation per distinct state and no sampled premature terminal stop. Ideal holds lack sampled probabilities and intermediate critic states; their MC credit is not actual multi-call GAE.'),
      interpretation=['V(s) is a state baseline, not an action-specific Q predictor. Subtracting it cannot reverse the order of two returns at the same state.',
        'A positive hypothetical incomplete-stop advantage is a diagnostic warning, not proof of biased expected policy gradient.',
        'Stationary continuation takes up to two chunks; no claim of exact counterfactual GAE without intermediate contexts/values.',
        'Terminal Monte Carlo returns are realized samples, not unbiased certainty about conditional expected value; no values were fitted to the desired action signs.',
        'Censored rollout values and lambda targets cannot validate themselves.'],
      source_sha256=sha256(Path(__file__)))
    return result

def main():
    out=ROOT/'reports/d247_critic_stopping_audit';out.mkdir(exist_ok=True)
    result=run();(out/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (out/'states.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=result['rows'][0]);writer.writeheader();writer.writerows(result['rows'])
    print(json.dumps({k:result[k] for k in ['counts','positive_incomplete_hold','negative_complete_hold','measured_return_rmse','verdict']},indent=2))
if __name__=='__main__':main()