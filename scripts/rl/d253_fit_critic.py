"""One D253 development-only fit, then lock before fresh evaluation flights."""
import json,sys,copy,datetime
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/rl'))
from state_critic import select_state_critic
from checkpoint_state import sha256
from rollout_samples import ppo_advantages
OUT=ROOT/'reports/d253_state_critic'

def main():
    torch.set_num_threads(2)
    if (OUT/'critic.pt').exists():raise ValueError('already locked, no refit')
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));assert sha256(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
    for n,h in p['development_hashes'].items():assert sha256(ROOT/n)==h
    observations=json.loads((ROOT/'reports/d252_local_critic_cause/critic_observations.json').read_text(encoding='utf-8'))
    obs={(r['split'],r['flight'],r['call']):r['observation']['features'] for r in observations}
    flights=[];features=[];state=[];names=None
    for split,path in zip(['development','diagnostic_former_test'],p['development']):
        cache=torch.load(ROOT/path,map_location='cpu',weights_only=False)
        for f in cache['flights']:
            f=copy.deepcopy(f);old_id=f['id'];f['id']=split+'/'+old_id
            for i,s in enumerate(f['samples']):
                row=obs[(split,old_id,i)];names=names or list(row);assert names==list(row)
                features.append(cache['features'][s['feature_index']]);state.append([row[n] for n in names]);s['feature_index']=len(features)-1
            flights.append(f)
    x=torch.stack(features).numpy();z=np.array(state);samples=[s for f in flights for s in f['samples']]
    assert len(flights)==11 and len(samples)==89 and len(names)==22
    y=np.array([s['mc_target'] for s in samples]);groups=np.concatenate([np.full(len(f['samples']),i) for i,f in enumerate(flights)])
    model,cv=select_state_critic(x,z,y,groups,p['critic_fix']['alphas'],p['critic_fix']['state_weights'])
    torch.save(model.state_dict(),OUT/'critic.pt')
    with torch.no_grad():values=model(torch.tensor(x),torch.tensor(z,dtype=torch.float32)).flatten().tolist()
    for s,v in zip(samples,values):s['value_old']=v
    for f in flights:ppo_advantages(f['samples'],1.,.95)
    raw=np.array([s['gae'] for s in samples]);mean=float(np.mean([np.mean([s['mc_target'] for s in f['samples']]) for f in flights]))
    actor=json.loads((ROOT/'reports/d251_critic_gate/fit.json').read_text())['actor_sha256']
    fit=dict(locked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),critic_sha256=sha256(OUT/'critic.pt'),protocol_sha256=sha256(OUT/'protocol.json'),criteria_sha256=p['criteria_sha256'],development_hashes=p['development_hashes'],observations_sha256=sha256(ROOT/'reports/d252_local_critic_cause/critic_observations.json'),actor_sha256=actor,hidden=x.shape[1],state_names=names,cv=cv,development_advantage_std=float(raw.std()),development_target_mean=mean,actor_updates=0,aws_used=False,test_seen=False,development_flights=[dict(id=f['id'],task=f['task'],calls=len(f['samples']),complete=f['complete']) for f in flights],source_hashes={n:sha256(ROOT/n) for n in ['scripts/rl/state_critic.py','scripts/rl/critic_observation.py','scripts/rl/d253_fit_critic.py','scripts/rl/d253_evaluate_critic.py','scripts/rl/critic_gate_metrics.py','scripts/rl/accepted_goal_reward.py','scripts/rl/movement_reward.py','scripts/rl/flight_collector.py']})
    torch.save(dict(features=torch.tensor(x),state=torch.tensor(z,dtype=torch.float32),flights=flights),OUT/'development_features.pt')
    fit['development_cache_sha256']=sha256(OUT/'development_features.pt')
    (OUT/'fit.json').write_text(json.dumps(fit,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(stage='critic_locked',alpha=cv['selected_alpha'],state_weight=cv['selected_state_weight'],development_cv_mse=cv['validation_mse'],development_advantage_std=float(raw.std()),critic_sha256=fit['critic_sha256']),indent=2))
if __name__=='__main__':main()
