"""Bounded D223: real policy collection and no-gradient replay only."""
import argparse,os,sys
from pathlib import Path
from bounded_job import run_stages
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--wall-seconds',type=float,required=True);a=p.parse_args()
r=a.root;code=Path(__file__).resolve().parents[2]
def stage(name,args,limit,out):
 return dict(name=name,argv=[sys.executable,*args],cwd=str(code),timeout_seconds=limit,required_outputs=[str(r/out)])
run_stages([
 stage('policy_nonlanding',['scripts/rl/nonlanding_preflight_driver.py','--root',str(r/'policy'),'--seed','7','--repeats','1'],5100,'policy/completion.json'),
 stage('two_move_horizon',['scripts/rl/nonlanding_preflight_driver.py','--root',str(r/'probe'),'--seed','7','--repeats','1','--probe-only'],600,'probe/completion.json'),
 stage('no_gradient_replay',['scripts/rl/audit_live_replay.py','--run',str(r/'policy'),'--probe-run',str(r/'probe'),'--expected-episodes','13','--out',str(r/'replay.json')],2400,'replay.json'),
 stage('evidence_checks',['scripts/rl/nonlanding_check.py','--root',str(r)],90,'checks.json')
],r,wall_seconds=a.wall_seconds,reserve_seconds=240,env=os.environ.copy())

