"""D218 saved-policy verification and comparison. Never constructs an optimizer."""
import argparse,json,hashlib,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
OLD=Path('C:/runs/d217_ppo_r2/results')
def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['verify','compare']);p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;r.mkdir(parents=True,exist_ok=True)
 if a.stage=='compare':
  rows=[]
  for f in sorted((r/'after/episodes').glob('sim_*_r*.json')):
   before=json.loads((OLD/'before/episodes'/f.name).read_text());after=json.loads(f.read_text())
   if before['task']!=after['task'] or before['seed']!=7 or after['seed']!=7:raise ValueError('comparison mismatch')
   row=dict(task=after['task'],kind=after['kind'])
   for name,ep in [('before',before),('after',after)]:
    if ep['reward']['invalid_transitions']:raise ValueError('invalid execution')
    row[name]=dict(steps=sum(ep['executed_counts']),end_reason=ep['end_reason'],**{k:ep['reward'][k] for k in ['complete','reward','progress_fraction','endpoint_error','endpoint_yaw_error_deg','max_deviation']})
   row['reward_delta']=row['after']['reward']-row['before']['reward'];rows.append(row)
  if len(rows)!=4 or len({x['kind'] for x in rows})!=4:raise ValueError('four matched tasks required')
  save(r/'comparison.json',dict(status='complete',updates_added=0,checkpoint_step=3,seed=7,rows=rows,before_complete=sum(x['before']['complete'] for x in rows),after_complete=sum(x['after']['complete'] for x in rows),scope='four practice development tasks, not benchmark',verification=json.loads((r/'verification.json').read_text())))
  return
 import torch
 from safetensors.torch import load_file
 from checkpoint_state import verified_checkpoint,sha256
 from composite_policy import CompositePolicy,adapter_digest,runtime_identity
 from train_path_ppo import make_forward
 from train_uav_flow_vla import setup,official_stats
 from uav_flow_action_tokenizer import ActionTokenizer
 start=time.monotonic();cp=verified_checkpoint(OLD/'training/checkpoints')
 if sha256(cp/'state.pt')!='398e60091dc266216353fc30e441d7a1af44b8e1388ff14776737b61cd9e96e5':raise ValueError('unexpected checkpoint')
 state=torch.load(cp/'state.pt',map_location='cpu',weights_only=False);m=state['metadata'];samples=m['samples']
 if state['step']!=3 or m['cursor']!=3:raise ValueError('unexpected step/cursor')
 initial=list((OLD/'training/checkpoints').glob('step_000000_*'))
 if len(initial)!=1:raise ValueError('initial checkpoint missing/ambiguous')
 sm=json.loads((initial[0]/'manifest.json').read_text());initial_weights=initial[0]/'adapter/action/adapter_model.safetensors'
 if sha256(initial_weights)!=sm['files']['adapter\\action\\adapter_model.safetensors']:raise ValueError('initial weights integrity')
 w0=load_file(str(initial_weights));w3=load_file(str(cp/'adapter/action/adapter_model.safetensors'))
 if w0.keys()!=w3.keys():raise ValueError('tensor keys differ')
 vision=[k for k in w0 if 'visual' in k or 'vision_' in k]
 if not vision or any(not torch.equal(w0[k],w3[k]) for k in vision):raise ValueError('vision changed or unverifiable')
 changed=[k for k in w0 if not torch.equal(w0[k],w3[k])]
 if not changed:raise ValueError('no action update')
 del w0,w3
 prefix=Path('C:/adapters/d207_stage2_adapter_s8000')
 if adapter_digest(prefix)!=m['identity']['prefix_adapter_sha256']:raise ValueError('prefix identity differs')
 api,processor,base,_=setup('qwen','bf16');policy=CompositePolicy.load(base,cp/'adapter/action',prefix,train_action=True)
 runtime=runtime_identity(policy.model,processor,official_stats())
 if any(m['identity'][k]!=v for k,v in runtime.items()):raise ValueError('runtime mismatch')
 if any(p.requires_grad and ('.action.' not in n or 'visual' in n or 'vision_' in n) for n,p in policy.model.named_parameters()):raise ValueError('unexpected trainable weights')
 tokenizer=ActionTokenizer(processor.tokenizer,official_stats());forward,_=make_forward(api,processor,tokenizer,policy,OLD/'batch/images',8)
 records=[]
 for i,s in enumerate(samples):
  # Same gradient-enabled forward as the original pre-step guard; no backward.
  lp,_=forward([s]);old=torch.tensor([s['logps']],device=lp.device,dtype=lp.dtype);d=(lp-old).sum(-1);kl=(d.exp()-1-d).mean()
  if not torch.isfinite(kl):raise ValueError('nonfinite replay at sample '+str(i))
  records.append(dict(index=i,task=s['task'],passage=s['passage'],call=s['call_index'],kl=float(kl.detach()),log_ratio=float(d.detach()[0])))
  del lp,d,kl
  if i%8==0:print('REPLAY',i,flush=True)
 average=sum(x['kl'] for x in records)/len(records)
 if abs(average-m['logs'][-1]['post_kl'])>1e-6:raise ValueError('checkpoint probability average not reproduced')
 order=[i for group,w in m['orders'][m['cursor']] for i in group];failing=next((records[i] for i in order if records[i]['kl']>m['config']['hard_kl']),None)
 if failing is None:raise ValueError('original guard failure not reproduced')
 with torch.no_grad():expected=forward(samples[:1])[0].cpu()
 policy.model.delete_adapter('action');policy.model.load_adapter(str(cp/'adapter/action'),adapter_name='action',is_trainable=True);policy.restore_action_weights_exact(cp/'adapter/action')
 with torch.no_grad():actual=forward(samples[:1])[0].cpu()
 torch.testing.assert_close(actual,expected,rtol=0,atol=0)
 save(r/'verification.json',dict(status='passed',checkpoint=str(cp),checkpoint_step=3,updates_added=0,vision_tensors_unchanged=len(vision),changed_action_tensors=len(changed),prefix_identity_verified=True,exact_reload_logps=True,full_batch_kl=average,first_guard_trigger=failing,next_order=order,records=records,elapsed_seconds=time.monotonic()-start))
 print('VERIFIED',failing,flush=True)
if __name__=='__main__':main()
