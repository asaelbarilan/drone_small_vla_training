"""D228 independent geometry/ranking analysis; never admits data to training."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/rl'))
from movement_reward import movement_reward
from reward_ranking_gate import costs,dominates,controls
from sampling_contract import validate_sampling_contract
OUT=ROOT/'reports/d228_local_numeric_flights/retry1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
 report=dict(status='blocked',training_started=False,tasks=[],blockers=[],controls=controls())
 for taskfile in sorted((ROOT/'reports/d224_reward_gated_ppo/tasks').glob('*.json')):
  task=read(taskfile);eps=[];rows=[]
  for repeat in [0,1]:
   f=OUT/'episodes'/(taskfile.stem+f'_r{repeat}.json')
   if not f.exists():f=OUT.parent/'finish/episodes'/f.name
   if not f.exists():continue
   ep=read(f);assert ep['seed']==7+repeat
   assert ep['end_reason'] in ['model_stop','horizon']
   validate_sampling_contract(ep['samples'])
   score=movement_reward(ep['poses'],task['reference_path_preprocessed'],kind=task['motion_kind'],stopped=ep['end_reason']=='model_stop')
   assert abs(score['reward']-sum(s['reward_v2'] for s in ep['samples']))<1e-8
   assert score['invalid_transitions']==0
   assert all(s['policy_provenance']['goal_mode']=='explicit-numeric-v1' for s in ep['samples'])
   for s in ep['samples']:
    assert hashlib.sha256((f.parent.parent/'rollout_images'/s['image']).read_bytes()).hexdigest()==s['image_sha256']
   c=costs(ep['poses'],task).tolist()
   rows.append(dict(seed=ep['seed'],steps=sum(ep['executed_counts']),calls=len(ep['samples']),end_reason=ep['end_reason'],reward=score['reward'],complete=score['complete'],costs=c,endpoint_error_m=score['endpoint_position_error_m'],yaw_error_deg=score['endpoint_yaw_error_deg'],max_deviation=score['max_deviation'],final_pose=ep['poses'][-1],sampler_logp_max_error=max(s['sampler_logp_max_error'] for s in ep['samples']),first_image_sha256=ep['samples'][0]['image_sha256']))
   eps.append(ep)
  pairs=[];variation=None
  if len(eps)==2:
   ids=[[token for s in ep['samples'] for token in s['ids']] for ep in eps]
   n=min(len(ids[0]),len(ids[1]));variation=dict(action_tokens_differ=ids[0]!=ids[1],common_prefix_token_difference_fraction=sum(a!=b for a,b in zip(*ids))/max(n,1),first_chunk_tokens_different=eps[0]['samples'][0]['ids']!=eps[1]['samples'][0]['ids'],endpoint_separation_m=float(np.linalg.norm(np.asarray(eps[0]['poses'][-1])[:3]-np.asarray(eps[1]['poses'][-1])[:3])/100),initial_image_identical=rows[0]['first_image_sha256']==rows[1]['first_image_sha256'])
   for i,j in [(0,1),(1,0)]:
    a,b=rows[i],rows[j]
    if a['end_reason']==b['end_reason'] and dominates(np.array(a['costs']),np.array(b['costs']),task['motion_kind']=='Turn/Rotate'):
     ok=a['reward']>b['reward']+1e-6;pairs.append(dict(better_seed=a['seed'],worse_seed=b['seed'],correctly_ranked=ok))
     if not ok:report['blockers'].append(taskfile.name+': reward inversion against independent geometry')
   if not pairs:report['blockers'].append(taskfile.name+': insufficient meaningful comparable variation')
  else:report['blockers'].append(taskfile.name+': missing paired flights')
  report['tasks'].append(dict(task=taskfile.name,instruction=task['instruction'],attempts=rows,variation=variation,comparisons=pairs))
 if not report['blockers']:report['status']='ranking_checks_passed_not_training_approval'
 (OUT/'analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
