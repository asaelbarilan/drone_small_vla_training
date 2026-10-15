"""D237 four-update PPO review. Unconditional stop after comparison; no continuation."""
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
  materialize(CODE,partition['development'],r/'dev_tasks')
  stage('deployment_assets',['scripts/rl/d234_assets.py','--root',r],60,r/'assets.json')
  stage('regressions',['-m','unittest','discover','-s','tests','-p','test_*reward.py'],120,CODE/'reports/d231_overshoot_ppo/comparison.json')
  stage('bf16_fixed_inputs',['scripts/rl/audit_d232_fixed_inputs.py','--precision','bf16','--baseline',BASE,'--checkpoint',r.parent/'diagnostic_checkpoint','--evidence',r.parent/'diagnostic_evidence','--output',r/'fixed_inputs'],1800,r/'fixed_inputs/same_input.json')
  before_dev=flights('before_dev',r/'dev_tasks',BASE,limit=3000)
  grouped=collections.defaultdict(list)
  for row in partition['training']:grouped[row['kind']].append(row)
  for k in grouped:grouped[k].sort(key=lambda x:hashlib.sha256(('d234-round-v1:'+x['source']).encode()).hexdigest())
  rr=r/'round_1';rr.mkdir()
  chosen=[grouped[k][0] for k in sorted(grouped)]
  materialize(CODE,chosen,rr/'tasks')
  save(rr/'controls.json',score_controls(rr/'tasks'))
  # Always reserve time for the user-requested paired comparison and upload.
  reserve=3300
  attempts=flights('round_1/attempts',rr/'tasks',BASE,repeats=2,seed=11,limit=4200,reserve=reserve+3000)
  materialize(CODE,[chosen[0]],rr/'probe_tasks')
  probe=flights('round_1/probe',rr/'probe_tasks',BASE,limit=600,max_steps=2,reserve=reserve+2400)
  stage('round_1/replay',['scripts/rl/audit_live_replay.py','--run',attempts,'--probe-run',probe,'--expected-episodes','16','--adapter',BASE,'--prefix-adapter',BASE,'--out',rr/'replay.json'],2400,rr/'replay.json',reserve+1800)
  n=admission(attempts,rr/'tasks',rr/'replay.json',rr,16);planned_updates=math.ceil(n/8)
  if planned_updates<4:raise ValueError('Not enough fresh samples for four actor updates')
  available=min(7200,remaining()-reserve-120)
  if available<1800:raise TimeoutError('Insufficient training time with comparison reserve')
  save(rr/'schedule.json',dict(samples=n,planned_full_pass_updates=planned_updates,pause_after_updates=4,actor_lr=2.5e-7,value_lr=1e-4,training_seconds=available,selection=chosen,automatic_continuation=False))
  stage('round_1/training_to_4',['scripts/rl/train_path_ppo.py','--rollouts',rr/'admitted_rollouts.json','--images',rr/'batch/images','--action-adapter',BASE,'--prefix-adapter',BASE,'--out',rr/'training','--max-updates',planned_updates,'--max-seconds',int(available)-60,'--lr','2.5e-7','--value-lr','1e-4','--batch-size','1','--accum','8','--pause-after-updates','4'],available,rr/'training/report.json',reserve)
  report=read(rr/'training/report.json')
  if report['step']!=4 or not report['frozen_adapters_unchanged'] or not report['reload_logps_exact']:raise ValueError('Four-update or invariant check failed')
  candidate=verified_checkpoint(rr/'training/checkpoints')/'adapter/action'
  after=flights('after_dev',r/'dev_tasks',candidate,limit=3000,reserve=180)
  check=comparison(before_dev,after,r/'comparison.json',8)
  from d234_support import behavior_gate
  diagnostic_decision=behavior_gate(before_dev,after,check)
  save(r/'review_required.json',dict(reason='User must see four-update comparison before any continuation',step=4,automatic_continuation=False,diagnostic_decision=diagnostic_decision))
  save(r/'completion.json',dict(state='awaiting_user_review',updates=4,comparison_pairs=8,checkpoint=str(candidate),automatic_continuation=False,elapsed_seconds=time.monotonic()-started))
 except Exception as e:
  save(r/'failure.json',dict(state='failed',error=str(e),traceback=traceback.format_exc(),completed=completed,elapsed_seconds=time.monotonic()-started));raise
if __name__=='__main__':main()
