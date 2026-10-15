"""D224 matched same-task diagnostic comparison, not a benchmark."""
import argparse,json,subprocess,sys
from pathlib import Path
from checkpoint_state import verified_checkpoint
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['after','compare']);p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
 report=json.loads((r/'training/report.json').read_text())
 assert 1<=report['step']<=2 and report['frozen_adapters_unchanged'] and report['reload_logps_exact']
 if a.stage=='after':
  action=verified_checkpoint(r/'training/checkpoints')/'adapter/action'
  subprocess.run([sys.executable,str(ROOT/'scripts/rl/reward_gated_flight_driver.py'),'--root',str(r/'after'),'--seed','7','--repeats','1','--action-adapter',str(action)],check=True);return
 rows=[]
 for f in sorted((r/'after/episodes').glob('sim_*_r*.json')):
  after=json.loads(f.read_text());before=json.loads((Path('C:/runs/d223_preflight/results/policy/episodes')/f.name).read_text())
  assert before['task']==after['task'] and before['seed']==after['seed']==7
  def compact(ep):
   return dict(end_reason=ep['end_reason'],moves=sum(ep['executed_counts']),**{k:ep['reward'][k] for k in ['reward','complete','progress_fraction','endpoint_position_error_m','endpoint_yaw_error_deg']})
  rows.append(dict(task=after['task'],kind=after['kind'],before=compact(before),after=compact(after)))
 assert len(rows)==4
 (r/'comparison.json').write_text(json.dumps(dict(scope='Same training tasks at reserved seed7; diagnostic only, no generalization claim',rows=rows,training=report,automatic_promotion=False),indent=2))
if __name__=='__main__':main()

