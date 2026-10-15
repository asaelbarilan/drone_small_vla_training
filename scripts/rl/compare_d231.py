"""D231 matched practice comparison; rescores both sides with identical v5."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/rl'))
from movement_reward import movement_reward
OUT=ROOT/'reports/d231_overshoot_ppo'
def main():
 report=json.loads((OUT/'training/report.json').read_text())
 assert report['step']==2 and report['reload_logps_exact'] and report['frozen_adapters_unchanged']
 rows=[]
 for p in sorted((OUT/'after/episodes').glob('sim_*_r0.json')):
  after=json.loads(p.read_text());matches=list((ROOT/'reports/d228_local_numeric_flights').glob('*/episodes/'+p.name));assert len(matches)==1
  before=json.loads(matches[0].read_text());assert before['task']==after['task'] and before['seed']==after['seed']==7
  task=json.loads((ROOT/'reports/d224_reward_gated_ppo/tasks'/after['task']).read_text());row=dict(task=after['task'],kind=task['motion_kind'])
  for name,ep in [('before',before),('after',after)]:
   q=movement_reward(ep['poses'],task['reference_path_preprocessed'],kind=task['motion_kind'],stopped=ep['end_reason']=='model_stop');assert not q['invalid_transitions']
   row[name]=dict(moves=sum(ep['executed_counts']),end=ep['end_reason'],**{k:q[k] for k in ['reward','complete','endpoint_position_error_m','endpoint_yaw_error_deg','max_deviation']})
  rows.append(row)
 assert len(rows)==4
 (OUT/'comparison_flights.json').write_text(json.dumps(dict(rows=rows,scope='four same-task diagnostic repetitions; images may differ; no generalization claim',updates=2),indent=2))
if __name__=='__main__':main()
