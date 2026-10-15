"""Fit and lock D251 critic using development flights only. No test collection."""
import sys,json,hashlib,copy
from pathlib import Path
import torch
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/rl'))
from regularized_critic import select_ridge
from rollout_samples import ppo_advantages
from checkpoint_state import sha256
OUT=ROOT/'reports/d251_critic_gate'
def main():
 if (OUT/'critic.pt').exists():raise ValueError('critic already locked; no refit')
 protocol=json.loads((OUT/'protocol.json').read_text());assert sha256(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
 prior=ROOT/'reports/d250_local_critic_fit';old=json.loads((prior/'report.json').read_text())
 assert old['actor_before_sha256']==old['actor_after_sha256'] and old['first_token_replay_max_error']==0
 cache=torch.load(prior/'frozen_features.pt',map_location='cpu',weights_only=False)
 x=cache['features'].numpy();flights=cache['flights'];samples=[s for f in flights for s in f['samples']]
 assert len(flights)==5 and len(samples)==41 and cache['actor_sha256']==old['actor_before_sha256']
 y=np.array([s['mc_target'] for s in samples]);groups=np.concatenate([np.full(len(f['samples']),i) for i,f in enumerate(flights)])
 head,cv=select_ridge(x,y,groups,protocol['critic_fix']['alphas'])
 torch.save(head.state_dict(),OUT/'critic.pt')
 with torch.no_grad():values=head(torch.tensor(x)).flatten().tolist()
 for s,v in zip(samples,values):s['value_old']=v
 for f in flights:ppo_advantages(f['samples'],1,.95)
 raw=np.array([s['gae'] for s in samples]);training_mean=float(np.mean([np.mean([s['mc_target'] for s in f['samples']]) for f in flights]))
 record=dict(critic_sha256=sha256(OUT/'critic.pt'),protocol_sha256=sha256(OUT/'protocol.json'),development_cache_sha256=sha256(prior/'frozen_features.pt'),actor_sha256=cache['actor_sha256'],cv=cv,development_advantage_std=float(raw.std()),development_target_mean=training_mean,actor_updates=0,test_seen=False,development_flights=[dict(id=f['id'],calls=len(f['samples']),complete=f['complete']) for f in flights],source_hashes={n:sha256(ROOT/n) for n in ['scripts/rl/regularized_critic.py','scripts/rl/d251_fit_critic.py','scripts/rl/accepted_goal_reward.py','scripts/rl/movement_reward.py','scripts/rl/flight_collector.py']})
 (OUT/'fit.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(dict(stage='critic_locked',alpha=cv['selected_alpha'],development_cv_mse=cv['validation_mse'],development_advantage_std=float(raw.std()),critic_sha256=record['critic_sha256']),indent=2))
if __name__=='__main__':main()
