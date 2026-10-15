import argparse,json,sys,hashlib,shutil
from pathlib import Path
import numpy as np
from movement_reward import movement_reward
from reward_ranking_gate import controls,costs,dominates
from ppo_math import validate_ppo_samples
from sampling_contract import validate_sampling_contract
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['controls','admit']);p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;r.mkdir(parents=True,exist_ok=True)
 if a.stage=='controls':
  q=controls();(r/'controls.json').write_text(json.dumps(q,indent=2));probe=r/'probe_tasks';probe.mkdir();src=sorted((ROOT/'reports/d224_reward_gated_ppo/tasks').glob('*.json'))[0];shutil.copyfile(src,probe/src.name);return
 assert json.loads((r/'replay.json').read_text())['status']=='passed'
 samples=[];rows=[];images=r/'batch/images';images.mkdir(parents=True,exist_ok=False);groups={}
 for f in sorted((r/'attempts/episodes').glob('sim_*_r*.json')):
  ep=json.loads(f.read_text());task=json.loads((ROOT/'reports/d224_reward_gated_ppo/tasks'/ep['task']).read_text());assert task['motion_kind']!='Land'
  score=movement_reward(ep['poses'],task['reference_path_preprocessed'],kind=task['motion_kind'],stopped=ep['end_reason']=='model_stop');assert not score['invalid_transitions']
  assert abs(score['reward']-sum(x['reward_v2'] for x in ep['samples']))<1e-9
  row=dict(task=ep['task'],seed=ep['seed'],reward=score['reward'],costs=costs(ep['poses'],task).tolist(),end=ep['end_reason'],tokens=[i for x in ep['samples'] for i in x['ids']]);groups.setdefault(ep['task'],[]).append(row);rows.append(row)
  for sample in ep['samples']:
   assert sample['reward_version']=='movement-reward-candidate-v5'
   for obs in [sample]+([sample['final_observation']] if sample.get('final_observation') else []):
    src=(r/'attempts/rollout_images'/obs['image']).resolve();assert src.is_relative_to((r/'attempts/rollout_images').resolve());assert hashlib.sha256(src.read_bytes()).hexdigest()==obs['image_sha256']
    dest=images/obs['image'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
   sample['reward_spec_approved']=True;samples.append(sample)
 assert len(rows)==8 and len(groups)==4
 comparisons=[]
 for task,pair in groups.items():
  assert len(pair)==2
  for x in pair:
   for y in pair:
    if x is not y and x['end']==y['end'] and dominates(np.array(x['costs']),np.array(y['costs']),'15-41-24' in task):
     assert x['reward']>y['reward']+1e-6;comparisons.append(dict(task=task,better=x['seed'],worse=y['seed']))
 assert any(pair[0]['tokens']!=pair[1]['tokens'] for pair in groups.values()),'No action variation'
 assert np.std([x['reward'] for x in rows])>1e-6,'No reward variation'
 assert validate_ppo_samples(samples,8,require_final_value=False)==8;validate_sampling_contract(samples)
 (r/'admitted_rollouts.json').write_text(json.dumps(dict(schema='executed-path-ppo-v1',chunk=8,preflight_passed=True,reward_version='movement-reward-candidate-v5',samples=samples)))
 (r/'admission.json').write_text(json.dumps(dict(status='passed',episodes=8,calls=len(samples),comparisons=comparisons,rows=rows,scope='bounded diagnostic; no every-task preference certification'),indent=2))
if __name__=='__main__':main()
