import argparse,os,sys,json
from pathlib import Path
from bounded_job import run_stages
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--wall-seconds',type=float,required=True);a=p.parse_args();r=a.root;code=Path(__file__).resolve().parents[2]
checkpoint=Path('C:/runs/d217_ppo_r2/results/training/checkpoints');pointer=json.loads((checkpoint/'CURRENT.json').read_text());adapter=checkpoint/pointer['directory']/'adapter/action'
def stage(name,args,limit,output):return dict(name=name,argv=[sys.executable,*args],cwd=str(code),timeout_seconds=limit,required_outputs=[str(r/output)])
run_stages([
 stage('verify_saved_candidate',['scripts/rl/evaluate_saved_ppo.py','verify','--root',str(r)],1500,'verification.json'),
 stage('four_matched_candidate_flights',['scripts/rl/practice_preflight_driver.py','--root',str(r/'after'),'--repeats','1','--seed','7','--selection-index','1','--action-adapter',str(adapter)],2100,'after/completion.json'),
 stage('matched_comparison',['scripts/rl/evaluate_saved_ppo.py','compare','--root',str(r)],90,'comparison.json')
],r,wall_seconds=a.wall_seconds,reserve_seconds=240,env=os.environ.copy())
