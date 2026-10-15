"""D215 authorized bounded saved-observation preflight, no simulator flights."""
import sys
import argparse
from pathlib import Path
from bounded_job import run_stages
parser=argparse.ArgumentParser()
parser.add_argument('--root',default='C:/runs/d215_preflight/results')
root=Path(parser.parse_args().root)
py=sys.executable
run_stages([
 dict(name='cpu_invariants',argv=[py,'-m','unittest','discover','-s','tests','-v'],timeout_seconds=180),
 dict(name='D208_bf16_8action_update_reload',argv=[py,'scripts/rl/local_ppo_smoke.py',
      '--adapter','C:/adapters/d207_stage2_adapter_s8000','--precision','bf16','--chunk','8',
      '--max-seconds','1200','--out',str(root/'model')],timeout_seconds=1500,
      required_outputs=[str(root/'model/report.json')])
],root,wall_seconds=1800,reserve_seconds=120)
