"""D223 read-only evidence checks; never admits training data."""
import argparse,json,math,hashlib
from pathlib import Path
from movement_reward import movement_reward
from vertical_tasks import vertical_reward
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 code=Path(__file__).resolve().parents[2];tasks=code/'reports/d223_nonlanding_preflight/tasks'
 gate=json.loads((tasks.parent/'selection_gate.json').read_text());rows=[]
 paths=sorted((a.root/'policy/episodes').glob('sim_*_r*.json'))
 assert len(paths)==len(gate['tasks'])==13
 expected={x['task']:x for x in gate['tasks']};seen=set()
 for f in paths:
  ep=json.loads(f.read_text());name=ep['task'];assert name not in seen;seen.add(name)
  tfile=tasks/name;assert hashlib.sha256(tfile.read_bytes()).hexdigest()==expected[name]['sha256']
  task=json.loads(tfile.read_text(encoding='utf-8-sig'));assert task['motion_kind']!='Land'
  assert ep['seed']==7 and ep['pose_source']=='simulator_readback'
  assert len(ep['poses'])==sum(ep['executed_counts'])+1
  stopped=ep['end_reason']=='model_stop'
  score=vertical_reward(ep['poses'],task,stopped=stopped) if task['motion_kind']=='Ascend/Descend' else movement_reward(ep['poses'],task['reference_path_preprocessed'],kind=task['motion_kind'],stopped=stopped)
  assert not score['invalid_transitions']
  total=sum(s['reward_v2'] for s in ep['samples'])
  assert math.isclose(total,score['reward'],abs_tol=1e-9)
  assert math.isclose(ep['reward']['reward'],score['reward'],abs_tol=1e-9)
  for i,s in enumerate(ep['samples']):
   assert s['reward_version']==score['version'] and s['reward_spec_approved'] is False
   assert len(s['ids'])==len(s['logps'])==32
   assert all(math.isfinite(x) for x in s['logps'])
   assert s['terminated']==(i==len(ep['samples'])-1 and stopped)
   assert s['truncated']==(i==len(ep['samples'])-1 and not stopped)
  rows.append(dict(task=name,kind=task['motion_kind'],calls=len(ep['samples']),moves=sum(ep['executed_counts']),end_reason=ep['end_reason'],reward=score['reward'],progress=score['progress_fraction'],complete=score['complete'],endpoint_m=score['endpoint_position_error_m'],endpoint_yaw_deg=score['endpoint_yaw_error_deg']))
 assert seen==set(expected)
 replay=json.loads((a.root/'replay.json').read_text());assert replay['status']=='passed'
 result=dict(status='passed',optimizer_updates=0,training_approved=False,episodes=rows,replayed_calls=len(replay['records']),bootstrap_records=len(replay['bootstrap_records']),limitations=['Geometry does not verify target identity, passing side or clearance.','Landing excluded; no contact telemetry.','One seed per task is a preflight, not evidence of improvement.'])
 (a.root/'checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()

