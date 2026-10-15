"""Independent saved-array verification and readable D253 report; no fitting."""
import json,sys,copy,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/rl'))
from state_critic import StateCritic
from checkpoint_state import sha256
from rollout_samples import ppo_advantages
from critic_observation import critic_observation
OUT=ROOT/'reports/d253_state_critic'

def main():
    r=json.loads((OUT/'partial_result.json').read_text());fit=json.loads((OUT/'fit.json').read_text());p=json.loads((OUT/'protocol.json').read_text())
    old=json.loads((ROOT/'reports/d251_critic_gate/protocol.json').read_text())
    assert p['criteria']==old['criteria']
    assert sha256(OUT/'protocol.json')==fit['protocol_sha256']
    assert sha256(OUT/'critic.pt')==fit['critic_sha256']
    for name,digest in fit['source_hashes'].items():assert sha256(ROOT/name)==digest
    assert sha256(OUT/'development_features.pt')==fit['development_cache_sha256']
    cache=torch.load(OUT/'partial_test_features.pt',map_location='cpu',weights_only=False);dev=torch.load(OUT/'development_features.pt',map_location='cpu',weights_only=False)
    head=StateCritic(fit['hidden'],len(fit['state_names']));head.load_state_dict(torch.load(OUT/'critic.pt',map_location='cpu',weights_only=True))
    with torch.no_grad():values=head(cache['features'],cache['state']).flatten().numpy()
    samples=[s for f in cache['flights'] for s in f['samples']]
    predicted=np.array([s['value_old'] for s in samples]);assert np.max(np.abs(values-predicted))<1e-6
    counts=[];state_errors=[];gae_errors=[];target_errors=[];means=[];zeros=[];constants=[]
    for f in cache['flights']:
        offset=0
        for s,count in zip(f['samples'],f['counts']):
            obs=critic_observation(f['poses'],f['reference'],executed_moves=offset,kind=f['kind'],max_moves=128)['features']
            z=np.array([obs[n] for n in fit['state_names']],dtype=np.float32)
            state_errors.append(float(np.max(np.abs(z-cache['state'][s['feature_index']].numpy()))));offset+=count
        if f['end_reason']!='model_stop':continue
        targets=np.cumsum([s['reward_v2'] for s in f['samples']][::-1])[::-1]
        target_errors.extend(abs(t-s['mc_target']) for t,s in zip(targets,f['samples']))
        replay=copy.deepcopy(f['samples']);ppo_advantages(replay,1.,.95)
        gae_errors.extend(abs(a['gae']-b['gae']) for a,b in zip(replay,f['samples']))
        v=np.array([s['value_old'] for s in f['samples']]);means.append(float(np.mean((targets-v)**2)));zeros.append(float(np.mean(targets**2)));constants.append(float(np.mean((targets-fit['development_target_mean'])**2)))
        counts.append(dict(flight=f['id'],seed=f['seed'],calls=len(f['samples']),complete=f['complete']))
    assert max(state_errors)==0 and max(target_errors,default=0)<1e-9 and max(gae_errors,default=0)<1e-9
    if counts:
        assert abs(np.mean(means)-r['mse'])<1e-10 and abs(np.mean(zeros)-r['zero_baseline_mse'])<1e-10 and abs(np.mean(constants)-r['training_mean_baseline_mse'])<1e-10
    key=lambda s:(s['image_sha256'],tuple(s['context_input_ids']))
    overlap=len({key(s) for f in dev['flights'] for s in f['samples']}&{key(s) for s in samples});assert overlap==r['exact_context_overlap']
    assert len(cache['flights'])==5 and {(f['task'],f['seed']) for f in cache['flights']}==({(task,seed) for task in p['test']['tasks'] for seed in p['test']['seeds']}-{('sim_2025-03-27_19-11-33.json',202)})
    assert r['partial_evidence'] and not r['criteria']['C2_coverage'] and not r['passed']
    assert r['passed']==all(r['criteria'].values())
    # Same fresh data, OLD fixed critic as descriptive comparison only. No selection.
    oldhead=torch.nn.Linear(fit['hidden'],1);oldhead.load_state_dict(torch.load(ROOT/'reports/d251_critic_gate/critic.pt',map_location='cpu',weights_only=True))
    with torch.no_grad():oldv=oldhead(cache['features']).flatten().numpy()
    oldmse=[];oldterms=[]
    for f in cache['flights']:
        if f['end_reason']!='model_stop':continue
        oldmse.append(float(np.mean([(s['mc_target']-float(oldv[s['feature_index']]))**2 for s in f['samples']])))
        s=f['samples'][-1];oldterms.append(dict(flight=f['id'],value=float(oldv[s['feature_index']]),stop_credit=s['mc_target']-float(oldv[s['feature_index']])))
    verify=dict(passed_verification=True,gate_passed=r['passed'],calls=len(samples),flights=counts,criteria_unchanged=True,reward_hashes_unchanged=True,critic_sha256=fit['critic_sha256'],result_sha256=sha256(OUT/'partial_result.json'),state_reconstruction_max_error=max(state_errors),target_recompute_max_error=max(target_errors,default=0),gae_recompute_max_error=max(gae_errors,default=0),prediction_reload_max_error=float(np.max(np.abs(values-predicted))),exact_context_overlap=overlap,actor_updates=0,aws_used=False,old_critic_same_test_mse=float(np.mean(oldmse)) if oldmse else None,old_critic_same_test_terminals=oldterms)
    (OUT/'partial_verification_local.json').write_text(json.dumps(verify,indent=2)+'\n',encoding='utf-8')
    lines=['# D253: local state-aware critic refit','',f"**Overall: {'PASS' if r['passed'] else 'FAIL / not ready'} under the unchanged seven criteria. No AWS and no action-model updates.**",'',
        'All eleven previously inspected flights (eight successes, three failures, 89 calls) were development only. The new inputs expose measured route/acceptance history, stopping counter, precise goal errors and remaining budget before each action. A linear ridge head and train-fold-only standardization were selected by whole-flight CV using the fixed grid, then locked before six new flights on seeds 201/202. No counterfactual controls were fitted. No test-based tuning, reward changes, value clamp or extra seeds. User-authorized recovery after an unclean reboot preserved one original flight and collected four more; the final flight remained unfinished at the fixed collection cap. No further flights were launched. Recovered flights used fresh simulator sessions with the same scene seed; this procedural difference is recorded.', '',
        f"Selected alpha {fit['cv']['selected_alpha']}; state weight {fit['cv']['selected_state_weight']}. Development CV MSE {fit['cv']['validation_mse']:.6f}; development advantage std {fit['development_advantage_std']:.6f}.",'',
        '| Criterion | Result |','|---|---|']
    for k,v in r['criteria'].items():lines.append(f"| {k} | {'PASS' if v else 'FAIL'} |")
    lines+=['','Acceptance text is unchanged verbatim from D251; see CRITERIA.md and protocol.json. Missing required outcomes count as coverage failure/inconclusive, never an invitation to reroll seeds.','',
        f"Five saved local flights; one missing. Saved outcomes: {r['terminal_flights']} terminal; {r['successes']} successful stops; {r['failed_stops']} failed stops; {r['premature_stops']} failed outside position/heading tolerance.",'',
        f"Equal-flight MSE: new critic {r['mse']}; zero baseline {r['zero_baseline_mse']}; development-mean baseline {r['training_mean_baseline_mse']}. Old D251 critic on these SAME new flights: {verify['old_critic_same_test_mse']} (descriptive comparison, not an added gate).",
        f"Terminal MAE {r['terminal_mae']}; largest terminal error {r['terminal_max_error']}.",'',
        '| Flight | Successful | Last reward / return | New value | New stopping credit | Old stopping credit |','|---|---|---:|---:|---:|---:|']
    oldterm={a['flight']:a for a in oldterms}
    for t in r['terminals']:lines.append(f"| {t['flight']} | {t['complete']} | {t['target']:.4f} | {t['value']:.4f} | {t['advantage']:.4f} | {oldterm[t['flight']]['stop_credit']:.4f} |")
    lines+=['','State and continuation checks: '+json.dumps(r['state_rates'])+'.','','Advantage scale: '+json.dumps(r['advantage_scale'])+'.','',
        f"Frozen actor fingerprint and exact critic reload checked; all {len(samples)} first-token replay errors <=0.01 (actual max {r['first_token_max_error']}); zero exact image/context overlap with development. Independent CPU reload verifies saved predictions, pre-action state features, MC returns, GAE and weighted metrics.",'',
        'The critic uses simulator-derived state during training only; action prompts/model weights were not changed. The candidate is not integrated into production PPO. This is a local learning-signal test, not improved flight behavior. All flights use the same frozen NF4 policy. Whole-flight Monte Carlo returns are noisy individual outcomes, not exact conditional expected values. No unsuccessful turn/vertical terminal development examples were available; steep descent now includes both outcomes. Censored horizon flights do not become made-up terminal training targets.', '',
        'Decision: NO-GO for AWS or further PPO on this critic. Coverage failed; partial-data metrics below do not establish performance on the missing outcome. Stop this procedure rather than start another patch-and-retry cycle. No AWS is authorized by the current request, even if all criteria pass. Preserve D251 and D253 outcomes; no automatic refit or new test seeds.']
    (OUT/'partial_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(verify,indent=2))
if __name__=='__main__':main()
