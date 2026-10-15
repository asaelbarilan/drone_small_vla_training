"""D234 full-pass PPO session. One owner, fixed deadline, no retries."""
import argparse,os,sys,time,math,hashlib,collections,traceback
from pathlib import Path
from bounded_job import run_stages
from checkpoint_state import verified_checkpoint
from d234_support import read,save,materialize,score_controls,admission,comparison
CODE=Path(__file__).resolve().parents[2]
BASE=Path('C:/adapters/d207_stage2_adapter_s8000')

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--wall-seconds',type=float,required=True);a=p.parse_args()
 r=a.root;r.mkdir(parents=True,exist_ok=True)
 if (r/'started.json').exists():raise ValueError('already dispatched; no retry')
 started=time.monotonic();save(r/'started.json',dict(wall_seconds=a.wall_seconds,pid=os.getpid()))
 partition=read(CODE/'reports/d234_full_ppo_preparation/partitions.json')
 completed=[];rounds=[];action=BASE
 def remaining():return a.wall_seconds-(time.monotonic()-started)
 def stage(name,args,limit,required,reserve=900):
  if remaining()<=reserve+60:raise TimeoutError('session reserve reached')
  save(r/'status.json',dict(name=name,state='running',completed=completed,remaining_seconds=remaining()))
  run_stages([dict(name=name,argv=[sys.executable,'-u',*map(str,args)],cwd=str(CODE),timeout_seconds=limit,required_outputs=[str(required)])],r/'stages'/name,wall_seconds=remaining(),reserve_seconds=reserve,env=os.environ.copy())
  completed.append(name)
 def flights(name,tasks,adapter,repeats=1,seed=7,limit=9000,validation=False,max_steps=None,reserve=900):
  out=r/name
  args=['scripts/rl/d234_flights.py','--root',out,'--tasks',tasks,'--action-adapter',adapter,'--repeats',repeats,'--seed',seed,'--wall-seconds',max(60,int(min(limit,remaining()-reserve))-420)]
  if validation:args+=['--validation-only']
  if max_steps:args+=['--max-steps',max_steps]
  stage(name,args,limit,out/'completion.json',reserve)
  return out
 try:
  sets=[{x['source'] for x in partition[k]} for k in ('training','development','validation')]
  if any(sets[i]&sets[j] for i in range(3) for j in range(i)):raise ValueError('partition overlap')
  materialize(CODE,partition['development'],r/'dev_tasks');materialize(CODE,partition['validation'],r/'validation_tasks')
  stage('deployment_assets',['scripts/rl/d234_assets.py','--root',r],60,r/'assets.json')
  stage('regressions',['-m','unittest','discover','-s','tests','-p','test_*reward.py'],120,CODE/'reports/d231_overshoot_ppo/comparison.json')
  stage('bf16_fixed_inputs',['scripts/rl/audit_d232_fixed_inputs.py','--precision','bf16','--baseline',BASE,'--checkpoint',r.parent/'diagnostic_checkpoint','--evidence',r.parent/'diagnostic_evidence','--output',r/'fixed_inputs'],1800,r/'fixed_inputs/same_input.json')
  before_dev=flights('before_dev',r/'dev_tasks',BASE,limit=3000)
  # User requires behavioral evidence before the lengthy validation phase.
  grouped=collections.defaultdict(list)
  for row in partition['training']:grouped[row['kind']].append(row)
  for k in grouped:grouped[k].sort(key=lambda x:hashlib.sha256(('d234-round-v1:'+x['source']).encode()).hexdigest())
  # Reserve candidate development, all85 matched validation and upload before any update.
  evaluation_reserve=21000
  for index in range(2):
   if remaining()<evaluation_reserve+6500:
    if not rounds:raise TimeoutError('insufficient budget for a full training round plus evaluation')
    break
   rr=r/f'round_{index+1}';rr.mkdir()
   chosen=[grouped[k][index] for k in sorted(grouped)]
   materialize(CODE,chosen,rr/'tasks')
   save(rr/'controls.json',score_controls(rr/'tasks'))
   attempts=flights(f'round_{index+1}/attempts',rr/'tasks',action,repeats=2,seed=11+index*2,limit=4200,reserve=evaluation_reserve+1800)
   # New policy must exercise an actual horizon observation, including after round1.
   materialize(CODE,[chosen[0]],rr/'probe_tasks')
   probe=flights(f'round_{index+1}/probe',rr/'probe_tasks',action,limit=600,max_steps=2,reserve=evaluation_reserve+1200)
   stage(f'round_{index+1}/replay',['scripts/rl/audit_live_replay.py','--run',attempts,'--probe-run',probe,'--expected-episodes','16','--adapter',action,'--prefix-adapter',BASE,'--out',rr/'replay.json'],2400,rr/'replay.json',evaluation_reserve+600)
   n=admission(attempts,rr/'tasks',rr/'replay.json',rr,16);updates=math.ceil(n/8)
   available=min(10800,remaining()-evaluation_reserve-120)
   if available<2400:raise TimeoutError('insufficient full-pass training budget')
   save(rr/'schedule.json',dict(samples=n,updates=updates,epochs=1,actor_lr=2.5e-7,value_lr=1e-4,training_seconds=available,selection=chosen,goal='every admitted chunk once; KL stop is not a completed pass'))
   for check_step in range(4,updates+4,4):
    check_step=min(check_step,updates)
    stage(f'round_{index+1}/training_to_{check_step}',['scripts/rl/train_path_ppo.py','--rollouts',rr/'admitted_rollouts.json','--images',rr/'batch/images','--action-adapter',action,'--prefix-adapter',BASE,'--out',rr/'training','--max-updates',updates,'--max-seconds',int(available)-60,'--lr','2.5e-7','--value-lr','1e-4','--batch-size','1','--accum','8','--pause-after-updates',check_step,*(['--resume'] if check_step>4 else [])],available,rr/'training/report.json',evaluation_reserve)
    segment=read(rr/'training/report.json')
    candidate=verified_checkpoint(rr/'training/checkpoints')/'adapter/action'
    intermediate=flights(f'round_{index+1}/check_{check_step}',r/'dev_tasks',candidate,limit=3000,reserve=evaluation_reserve)
    check=comparison(before_dev,intermediate,rr/f'intermediate_{check_step}.json',8)
    from d234_support import behavior_gate
    decision=behavior_gate(before_dev,intermediate,check)
    save(rr/f'decision_{check_step}.json',decision)
    save(r/'latest_intermediate.json',dict(round=index+1,step=check_step,decision=decision,comparison=check))
    if not decision['continue_training']:
     save(r/'review_required.json',dict(reason='No demonstrated behavioral improvement or regression at scheduled checkpoint',round=index+1,step=check_step,decision=decision))
     raise RuntimeError('Intermediate behavioral gate requires review; no further training')
   report=read(rr/'training/report.json')
   if not report['coverage']['schedule_complete'] or report['coverage']['unique_actor_samples']!=n or not report['frozen_adapters_unchanged'] or not report['reload_logps_exact']:raise ValueError('full-pass or invariant check failed')
   action=verified_checkpoint(rr/'training/checkpoints')/'adapter/action'
   cmp=check
   save(rr/'development_comparison.json',cmp)
   before_count=sum(x['before']['geometric_complete'] for x in cmp['rows']);after_count=sum(x['after']['geometric_complete'] for x in cmp['rows'])
   rounds.append(dict(round=index+1,report=str(rr/'training/report.json'),updates=report['step'],samples=n,development_before=before_count,development_after=after_count))
   save(r/'rounds.json',rounds)
   if after_count<before_count:
    save(r/'regression.json',dict(reason='development geometric completion regressed',round=index+1));break
  if not rounds:raise ValueError('no completed full training pass')
  before_val=flights('before_validation',r/'validation_tasks',BASE,limit=10500,validation=True,reserve=11000)
  after_val=flights('after_validation',r/'validation_tasks',action,limit=10500,validation=True)
  comparison(before_val,after_val,r/'validation_comparison.json',85)
  save(r/'completion.json',dict(state='complete',rounds=rounds,checkpoint=str(action),elapsed_seconds=time.monotonic()-started,validation_pairs=85,automatic_promotion=False,regression_observed=(r/'regression.json').exists()))
 except Exception as e:
  save(r/'failure.json',dict(state='failed',error=str(e),traceback=traceback.format_exc(),completed=completed,rounds=rounds,elapsed_seconds=time.monotonic()-started));raise
if __name__=='__main__':main()
