"""D253 one locked final test. Never fits a critic or updates the actor."""
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
from state_critic import StateCritic
from critic_observation import critic_observation
from critic_gate_metrics import numeric_criteria
OUT=ROOT/'reports/d253_state_critic';ADAPTER=Path('D:/drone_vla_pilot/adapters/d207_s8000')
def save(name,data):
 (OUT/('partial_'+name)).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def weighted_mse(flights,constant=None):
 return float(np.mean([np.mean([(s['mc_target']-(s['value_old'] if constant is None else constant))**2 for s in f['samples']]) for f in flights])) if flights else None
def main():
 start=time.monotonic();torch.set_num_threads(4)
 assert sha256(ROOT/'scripts/rl/d253_evaluate_critic.py')==(OUT/'evaluator.sha256').read_text().strip()
 if (OUT/'partial_result.json').exists():raise ValueError('final test already evaluated; no retest or tuning')
 p=json.loads((OUT/'protocol.json').read_text());fit=json.loads((OUT/'fit.json').read_text())
 assert sha256(OUT/'protocol.json')==fit['protocol_sha256']==(OUT/'protocol.sha256').read_text().strip()
 assert sha256(OUT/'critic.pt')==fit['critic_sha256']
 for name,digest in fit['source_hashes'].items():assert sha256(ROOT/name)==digest
 folder=OUT/'test_flights';assert (OUT/'recovery/failure.json').exists()
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
 assert len(flights)==5 and len({(f['task'],f['seed']) for f in flights})==5
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
 state=[]
 for f in flights:
  offset=0
  for count in f['counts']:
   row=critic_observation(f['poses'],f['reference'],executed_moves=offset,kind=f['kind'],max_moves=p['test']['max_moves'])['features']
   assert list(row)==fit['state_names']
   state.append([row[n] for n in fit['state_names']]);offset+=count
 z=torch.tensor(state,dtype=torch.float32);assert torch.isfinite(z).all()
 head=StateCritic(x.shape[1],len(fit['state_names']));head.load_state_dict(torch.load(OUT/'critic.pt',map_location='cpu',weights_only=True))
 with torch.no_grad():values=head(x,z).flatten().tolist()
 reload=StateCritic(x.shape[1],len(fit['state_names']));reload.load_state_dict(torch.load(OUT/'critic.pt',map_location='cpu',weights_only=True))
 with torch.no_grad():assert torch.equal(head(x,z),reload(x,z))
 for f in flights:
  for s in f['samples']:s['value_old']=values[s['feature_index']]
  if f['end_reason']=='model_stop':ppo_advantages(f['samples'],1,.95)
 torch.save(dict(features=x,state=z,flights=flights),OUT/'partial_test_features.pt')
 audit=numeric_criteria(flights,fit)
 assert sha256(OUT/'development_features.pt')==fit['development_cache_sha256']
 development=torch.load(OUT/'development_features.pt',map_location='cpu',weights_only=False)
 key=lambda s:(s['image_sha256'],tuple(s['context_input_ids']))
 dkeys={key(s) for f in development['flights'] for s in f['samples']};tkeys={key(s) for f in flights for s in f['samples']};overlap=len(dkeys&tkeys)
 regressions=json.loads((OUT/'regressions.json').read_text(encoding='utf-8'))
 original=json.loads((ROOT/'reports/d251_critic_gate/protocol.json').read_text(encoding='utf-8'))
 assert p['criteria']==original['criteria']
 audit['checks']['C2_coverage']=False
 checks=dict(C1_identity=overlap==0,**audit['checks'],C7_regressions=regressions['passed'])
 result=dict(partial_evidence=True,missing_flights=1,coverage_note='One interrupted flight missing; two completed horizons; no claimed full pass',passed=all(checks.values()),criteria=checks,failed=[k for k,v in checks.items() if not v],protocol_sha256=fit['protocol_sha256'],criteria_sha256=fit['criteria_sha256'],critic_sha256=fit['critic_sha256'],actor_frozen=before==after,actor_updates=0,aws_used=False,all_flights=len(flights),**audit['metrics'],exact_context_overlap=overlap,first_token_max_error=max(errors),source_hashes=sources,elapsed_s=time.monotonic()-start)
 save('state_checks.json',audit['state_checks']);save('result.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as exc:
  save('evaluation_failure.json',dict(error=type(exc).__name__,message=str(exc),passed=False,actor_updates=0,aws_used=False));raise
