"""D231 local admission after complete probability replay; never launches AWS."""
import json,sys,hashlib,copy,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/rl'))
from movement_reward import movement_reward
from ppo_math import validate_ppo_samples
from sampling_contract import validate_sampling_contract
OUT=ROOT/'reports/d231_overshoot_ppo'
def main():
 replay=json.loads((OUT/'replay.json').read_text())
 assert replay['status']=='passed' and len(replay['records'])==89
 rows=json.loads((OUT/'rescoring.json').read_text())
 assert len(rows)==8
 by={(x['task'],x['seed']):x['new'] for x in rows}
 assert by[('sim_2025-03-27_19-11-33.json',7)]>by[('sim_2025-03-27_19-11-33.json',8)]
 assert by[('sim_2025-03-27_17-35-01.json',8)]>0>by[('sim_2025-03-27_17-35-01.json',7)]
 assert by[('sim_2025-03-27_17-33-04.json',8)]>by[('sim_2025-03-27_17-33-04.json',7)]
 images=OUT/'images';images.mkdir(exist_ok=False);samples=[];evidence=[]
 for folder in ('retry1','finish'):
  run=ROOT/f'reports/d228_local_numeric_flights/{folder}'
  for p in sorted((run/'episodes').glob('sim_*_r*.json')):
   ep=json.loads(p.read_text());task=json.loads((ROOT/'reports/d224_reward_gated_ppo/tasks'/ep['task']).read_text())
   assert task['motion_kind']!='Land'
   score=movement_reward(ep['poses'],task['reference_path_preprocessed'],kind=task['motion_kind'],stopped=ep['end_reason']=='model_stop')
   assert not score['invalid_transitions'];offset=0;current=[]
   assert len(ep['samples'])==len(ep['executed_counts'])
   for s,count in zip(ep['samples'],ep['executed_counts']):
    s=copy.deepcopy(s);s['reward_v2']=sum(score['transition_rewards'][offset:offset+count]);offset+=count
    s['reward_version']=score['version'];s['reward_spec_approved']=True
    for obs in [s]+([s['final_observation']] if s.get('final_observation') else []):
     source=(run/'rollout_images'/obs['image']).resolve();assert source.is_relative_to((run/'rollout_images').resolve())
     assert hashlib.sha256(source.read_bytes()).hexdigest()==obs['image_sha256']
     dest=images/folder/obs['image'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
     obs['image']=str(Path(folder)/obs['image'])
    samples.append(s);current.append(s)
   assert offset==len(ep['poses'])-1
   assert abs(sum(s['reward_v2'] for s in current)-score['reward'])<1e-9
   evidence.append(dict(source=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),reward=score['reward']))
 assert len(evidence)==8 and len(samples)==89
 assert validate_ppo_samples(samples,8,require_final_value=False)==8
 validate_sampling_contract(samples)
 artifact=dict(schema='executed-path-ppo-v1',chunk=8,preflight_passed=True,reward_version='movement-reward-candidate-v5',samples=samples,scope='D231 local NF4 two-update authorized diagnostic only',sources=evidence)
 (OUT/'admitted.json').write_text(json.dumps(artifact,indent=2))
 print('ADMITTED',len(samples),flush=True)
if __name__=='__main__':main()
