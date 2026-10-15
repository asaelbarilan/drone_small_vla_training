"""D217 one bounded learning pilot, no retry or automatic extension."""
import argparse,os,sys
from pathlib import Path
from bounded_job import run_stages
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--wall-seconds',type=float,required=True);a=p.parse_args();r=a.root;py=sys.executable
code=Path(__file__).resolve().parents[2]
def stage(name,args,seconds,required):return dict(name=name,argv=[py,*args],timeout_seconds=seconds,cwd=str(code),required_outputs=[str(r/x) for x in required])
common=['scripts/rl/pilot_stage.py']
run_stages([
 stage('assets_gate',common+['assets','--root',str(r)],60,['assets_gate.json']),
 stage('cpu_checks',['-m','unittest','discover','-s','tests','-v'],180,[]),
 stage('seed_sampling_gate',common+['seed','--root',str(r)],480,['seed_gate.json']),
 stage('baseline_development',['scripts/rl/practice_preflight_driver.py','--root',str(r/'before'),'--repeats','1','--seed','7','--selection-index','1'],2100,['before/completion.json']),
 stage('admit_verified_D216_batch',common+['admit','--root',str(r)],60,['admission.json']),
 stage('PPO_four_updates',['scripts/rl/train_path_ppo.py','--rollouts',str(r/'admitted_rollouts.json'),'--images',str(r/'batch/images'),'--action-adapter','C:/adapters/d207_stage2_adapter_s8000','--prefix-adapter','C:/adapters/d207_stage2_adapter_s8000','--out',str(r/'training'),'--max-updates','4','--max-seconds','5300','--batch-size','1','--accum','8'],5400,['training/report.json']),
 stage('candidate_development',common+['after','--root',str(r)],2100,['after/completion.json']),
 stage('comparison',common+['compare','--root',str(r)],60,['comparison.json'])
],r,wall_seconds=a.wall_seconds,reserve_seconds=300,env=os.environ.copy())
