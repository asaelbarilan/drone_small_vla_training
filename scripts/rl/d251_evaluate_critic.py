"""D251 one locked final test. Never fits a critic or updates the actor."""
import os,sys,json,hashlib,time,gc,copy
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'scripts/rl'))
from checkpoint_state import sha256
from composite_policy import CompositePolicy,adapter_digest,runtime_identity
from executed_rollouts import attach_path_rewards
from movement_reward import movement_reward
from rollout_samples import ppo_advantages
from ppo_engine import normalized_features
from fit_local_stopping_critic import fingerprint
OUT=ROOT/'reports/d251_critic_gate';ADAPTER=Path('D:/drone_vla_pilot/adapters/d207_s8000')
def save(name,data):
 (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def weighted_mse(flights,constant=None):
 return float(np.mean([np.mean([(s['mc_target']-(s['value_old'] if constant is None else constant))**2 for s in f['samples']]) for f in flights])) if flights else None
def main():
 start=time.monotonic();torch.set_num_threads(4)
 assert sha256(Path(__file__))==(OUT/'evaluator.sha256').read_text().strip()
 if (OUT/'result.json').exists():raise ValueError('final test already evaluated; no retest or tuning')
 p=json.loads((OUT/'protocol.json').read_text());fit=json.loads((OUT/'fit.json').read_text())
 assert sha256(OUT/'protocol.json')==fit['protocol_sha256']==(OUT/'protocol.sha256').read_text().strip()
 assert sha256(OUT/'critic.pt')==fit['critic_sha256']
 for name,digest in fit['source_hashes'].items():assert sha256(ROOT/name)==digest
 folder=OUT/'test_flights';done=json.loads((folder/'episodes/completion.json').read_text());assert done['episodes']==6
 manifest=json.loads((folder/'manifest.json').read_text());assert adapter_digest(ADAPTER)==manifest['adapter_sha256']
 flights=[];sources={}
 for file in sorted((folder/'episodes').glob('*_r?.json')):
  e=json.loads(file.read_text(encoding='utf-8'));sources[str(file.relative_to(ROOT))]=sha256(file)
  assert e['task'] in p['test']['tasks'] and e['seed'] in p['test']['seeds']
  taskpath=folder/'tasks'/e['task'];assert sha256(taskpath)==manifest['tasks'][e['task']]
  task=json.loads(taskpath.read_text(encoding='utf-8'))
  samples,reward=attach_path_rewards(e['samples'],e['poses'],e['executed_counts'],task['reference_path_preprocessed'],kind=e['kind'],end_reason=e['end_reason'],final_observation=e.get('final_observation'),stopping_mode='accepted_goal')
  target=np.cumsum([s['reward_v2'] for s in samples][::-1])[::-1] if e['end_reason']=='model_stop' else [None]*len(samples)
  for sample,t in zip(samples,target):sample['mc_target']=None if t is None else float(t)
  flights.append(dict(id=file.stem,task=e['task'],seed=e['seed'],kind=e['kind'],end_reason=e['end_reason'],complete=reward['complete'],reward=reward,samples=samples,poses=e['poses'],counts=e['executed_counts'],reference=task['reference_path_preprocessed']))
 assert len(flights)==6 and len({(f['task'],f['seed']) for f in flights})==6
 from train_uav_flow_vla import setup,official_stats,encode
 from uav_flow_action_tokenizer import ActionTokenizer
 from model_replay import replay_cached_tokens
 from PIL import Image
 api,processor,base,_=setup('qwen','nf4');policy=CompositePolicy.load(base,ADAPTER,ADAPTER,train_action=False);policy.activate('action')
 policy.model.gradient_checkpointing_disable();policy.model.requires_grad_(False)
 assert not any(v.requires_grad for v in policy.model.parameters());before=fingerprint(policy.model)
 assert before==fit['actor_sha256']
 runtime=runtime_identity(policy.model,processor,official_stats());tokenizer=ActionTokenizer(processor.tokenizer,official_stats());features=[];errors=[]
 for f in flights:
  for s in f['samples']:
   assert all(s['policy_provenance'].get(k)==v for k,v in runtime.items()),'test runtime identity mismatch'
   imagepath=(folder/'rollout_images'/s['image']).resolve();assert imagepath.is_relative_to((folder/'rollout_images').resolve()) and sha256(imagepath)==s['image_sha256']
   with Image.open(imagepath) as im:im=im.convert('RGB');batch=encode(processor,tokenizer,{k:s[k] for k in ['prompt','box_text','progress_text']},8,with_answer=False,image=im)
   assert batch['input_ids'][0].tolist()==s['context_input_ids']
   with torch.no_grad():
    lp,h=replay_cached_tokens(policy.model.get_base_model(),api.cuda(batch),s['ids'][:1],low=tokenizer.vocab_size-tokenizer.n_bins,high=tokenizer.vocab_size,temperature=s['temperature'],pad_token_id=processor.tokenizer.pad_token_id)
    err=abs(float(lp[0])-s['logps'][0]);assert np.isfinite(err) and err<=.01
    s['feature_index']=len(features);features.append(normalized_features(h).cpu());errors.append(err)
   save('evaluation_progress.json',dict(features=len(features),flight=f['id'],call=s['call_index']))
 after=fingerprint(policy.model);assert before==after and not any(v.requires_grad for v in policy.model.parameters())
 del policy,base;gc.collect();torch.cuda.empty_cache()
 x=torch.stack(features);assert torch.isfinite(x).all()
 head=torch.nn.Linear(x.shape[1],1);head.load_state_dict(torch.load(OUT/'critic.pt',map_location='cpu',weights_only=True))
 with torch.no_grad():values=head(x).flatten().tolist()
 reload=torch.nn.Linear(x.shape[1],1);reload.load_state_dict(torch.load(OUT/'critic.pt',map_location='cpu',weights_only=True))
 with torch.no_grad():assert torch.equal(head(x),reload(x))
 for f in flights:
  for s in f['samples']:s['value_old']=values[s['feature_index']]
  if f['end_reason']=='model_stop':ppo_advantages(f['samples'],1,.95)
 torch.save(dict(features=x,flights=flights),OUT/'test_features.pt')
 terminal=[f for f in flights if f['end_reason']=='model_stop'];good=[f for f in terminal if f['complete']];bad=[f for f in terminal if not f['complete']]
 premature=[f for f in bad if f['reward']['endpoint_position_error_m']>.5 or f['reward']['endpoint_yaw_error_deg']>f['reward']['heading_tolerance_deg']]
 coverage=len(terminal)>=4 and len({f['task'] for f in terminal})==3 and len(good)>=2 and len(premature)>=1
 const=fit['development_target_mean'];mse=weighted_mse(terminal);zero=weighted_mse(terminal,0.);mean=weighted_mse(terminal,const)
 bykind={k:dict(mse=weighted_mse([f for f in terminal if f['kind']==k]),zero=weighted_mse([f for f in terminal if f['kind']==k],0.),mean=weighted_mse([f for f in terminal if f['kind']==k],const)) for k in ['Turn','Ascend/Descend']}
 terms=[dict(flight=f['id'],kind=f['kind'],complete=f['complete'],target=f['samples'][-1]['mc_target'],value=f['samples'][-1]['value_old'],advantage=f['samples'][-1]['gae']) for f in terminal]
 terr=[abs(t['target']-t['value']) for t in terms];mae=float(np.mean(terr)) if terr else None;maxerror=max(terr) if terr else None
 calibration=bool(terminal) and mse<=.9*zero and mse<=.9*mean and all(z['mse'] is not None and z['mse']<=min(z['zero'],z['mean']) for z in bykind.values()) and mae<=.25 and maxerror<=.5
 stopcheck=bool(terms) and all(t['advantage']>=.01 if t['complete'] else t['advantage']<=-.01 for t in terms)
 states=[];continuation=[]
 for f in terminal:
  offset=0;poses=np.array(f['poses']);ref=f['reference'];kw=dict(kind=f['kind'],stopping_mode='accepted_goal')
  for i,s in enumerate(f['samples']):
   prefix=poses[:offset+1];hold=np.vstack([prefix,np.repeat(prefix[-1:],10,axis=0)])
   stop=movement_reward(hold,ref,stopped=True,**kw);qstop=stop['reward']-movement_reward(hold,ref,stopped=False,**kw)['reward'];a=qstop-s['value_old']
   states.append(dict(flight=f['id'],call=i,accepted=stop['complete'],value=s['value_old'],stop_return=qstop,stop_advantage=a,correct=bool(a>=.01 if stop['complete'] else a<=-.01)))
   end=offset+f['counts'][i];prefix2=poses[:end+1];hold2=np.vstack([prefix2,np.repeat(prefix2[-1:],10,axis=0)]);endq=movement_reward(hold2,ref,stopped=True,**kw)
   de=stop['endpoint_position_error_m']-endq['endpoint_position_error_m'];dy=stop['endpoint_yaw_error_deg']-endq['endpoint_yaw_error_deg']
   improving=(de>=.02 or dy>=.2) and de>=-.02 and dy>=-.2
   if f['complete'] and i<len(f['samples'])-1 and not stop['complete'] and improving:
    surplus=s['mc_target']-s['value_old'];gap=s['mc_target']-qstop
    continuation.append(dict(flight=f['id'],kind=f['kind'],call=i,remaining_return=s['mc_target'],value=s['value_old'],surplus=surplus,stop_return=qstop,gap=gap,correct=bool(surplus>=.01 and gap>=.10)))
   offset=end
 early=[a for a in states if not a['accepted']];arrived=[a for a in states if a['accepted']]
 rate=lambda rows:float(np.mean([s['correct'] for s in rows])) if rows else 0.
 count_by_kind={k:sum(a['kind']==k for a in continuation) for k in ['Turn','Ascend/Descend']}
 statecheck=bool(early and arrived) and rate(early)>=.95 and rate(arrived)>=.95 and all(n>=2 for n in count_by_kind.values()) and rate(continuation)>=.90
 raw=np.array([s['gae'] for f in terminal for s in f['samples']]);devstd=fit['development_advantage_std'];teststd=float(raw.std()) if len(raw) else 0.;scaledmax=float(np.abs(raw/devstd).max()) if len(raw) and devstd>0 else None
 scalecheck=devstd>=.05 and .25*devstd<=teststd<=4*devstd and scaledmax is not None and scaledmax<=10
 assert sha256(ROOT/'reports/d250_local_critic_fit/frozen_features.pt')==fit['development_cache_sha256']
 development=torch.load(ROOT/'reports/d250_local_critic_fit/frozen_features.pt',map_location='cpu',weights_only=False)
 key=lambda s:s['image_sha256']+str(s['context_input_ids'])
 dkeys={key(s) for f in development['flights'] for s in f['samples']};tkeys={key(s) for f in flights for s in f['samples']};overlap=len(dkeys&tkeys)
 regressions=json.loads((OUT/'regressions.json').read_text())
 checks=dict(C1_identity=overlap==0,C2_coverage=bool(coverage),C3_value=bool(calibration),C4_actual_stop=bool(stopcheck),C5_state_and_continuation=bool(statecheck),C6_scale=bool(scalecheck),C7_regressions=regressions['passed'])
 result=dict(passed=all(checks.values()),criteria=checks,failed=[k for k,v in checks.items() if not v],protocol_sha256=fit['protocol_sha256'],critic_sha256=fit['critic_sha256'],actor_frozen=before==after,actor_updates=0,aws_used=False,all_flights=6,terminal_flights=len(terminal),successes=len(good),failed_stops=len(bad),premature_stops=len(premature),mse=mse,zero_baseline_mse=zero,training_mean_baseline_mse=mean,per_kind=bykind,terminal_mae=mae,terminal_max_error=maxerror,terminals=terms,state_rates=dict(outside=rate(early),accepted=rate(arrived),continuation=rate(continuation),continuation_counts=count_by_kind),advantage_scale=dict(development_std=devstd,test_std=teststd,max_test_scaled_magnitude=scaledmax),exact_context_overlap=overlap,first_token_max_error=max(errors),source_hashes=sources,elapsed_s=time.monotonic()-start)
 save('state_checks.json',dict(stops=states,continuation=continuation));save('result.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as exc:
  save('evaluation_failure.json',dict(error=type(exc).__name__,message=str(exc),passed=False,actor_updates=0,aws_used=False));raise
