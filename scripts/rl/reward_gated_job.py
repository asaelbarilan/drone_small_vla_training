"""One bounded D224 run. Failed rank/variation/replay gate prevents learning."""
import argparse,os,sys
from pathlib import Path
from bounded_job import run_stages
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--wall-seconds',type=float,required=True);a=p.parse_args()
r=a.root;code=Path(__file__).resolve().parents[2]
def stage(name,args,limit,out):
 return dict(name=name,argv=[sys.executable,*args],cwd=str(code),timeout_seconds=limit,required_outputs=[str(r/out)])
gate=['scripts/rl/reward_ranking_gate.py']
run_stages([
 stage('controlled_reward_order',gate+['controls','--root',str(r)],60,'controls.json'),
 stage('eight_fresh_attempts',['scripts/rl/reward_gated_flight_driver.py','--root',str(r/'attempts'),'--seed','8','--repeats','2'],2400,'attempts/completion.json'),
 stage('independent_ranking_and_variation',gate+['gate','--root',str(r)],120,'ranking_gate.json'),
 stage('no_gradient_replay',['scripts/rl/audit_live_replay.py','--run',str(r/'attempts'),'--probe-run','C:/runs/d223_preflight/results/probe','--expected-episodes','8','--out',str(r/'replay.json')],1500,'replay.json'),
 stage('admission',gate+['admit','--root',str(r)],120,'admission.json'),
 stage('PPO_at_most_two_updates',['scripts/rl/train_path_ppo.py','--rollouts',str(r/'admitted_rollouts.json'),'--images',str(r/'batch/images'),'--action-adapter','C:/adapters/d207_stage2_adapter_s8000','--prefix-adapter','C:/adapters/d207_stage2_adapter_s8000','--out',str(r/'training'),'--max-updates','2','--max-seconds','3500','--lr','1e-6','--value-lr','1e-4','--batch-size','1','--accum','8'],3600,'training/report.json'),
 stage('matched_candidate',['scripts/rl/reward_gated_outcome.py','after','--root',str(r)],1500,'after/completion.json'),
 stage('comparison',['scripts/rl/reward_gated_outcome.py','compare','--root',str(r)],60,'comparison.json')
],r,wall_seconds=a.wall_seconds,reserve_seconds=600,env=os.environ.copy())

