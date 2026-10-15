"""D224 independent straight-task ranking gate. No optimizer imports."""
import argparse,json,hashlib,shutil
from pathlib import Path
import numpy as np
from movement_reward import movement_reward
ROOT=Path(__file__).resolve().parents[2]
TASKS=ROOT/'reports/d224_reward_gated_ppo/tasks'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def costs(poses,task):
 a=np.asarray(poses,float);ref=np.asarray(task['reference_path_preprocessed'],float)
 xyz=a[:,:3]/100;goal=ref[-1,:3]/100
 yaw=np.rad2deg(np.unwrap(np.deg2rad(a[:,4])));target=ref[-1,4]
 if task['motion_kind']=='Turn/Rotate':
  return np.array([abs(yaw[-1]-target),np.linalg.norm(xyz,axis=1).max(),np.maximum(-np.diff(yaw)*np.sign(target),0).sum()])
 unit=goal/np.linalg.norm(goal);along=xyz@unit;side=np.linalg.norm(xyz-along[:,None]*unit,axis=1)
 return np.array([np.linalg.norm(xyz[-1]-goal),side.max(),abs(yaw[-1]-target)])
def dominates(a,b,turn):
 tolerance=np.array([0.,.03,.1]) if turn else np.array([0.,.03,.5])
 meaningful=.5 if turn else .1
 return bool(a[0]+meaningful<b[0] and np.all(a<=b+tolerance))
def controls():
 rows=[]
 for f in sorted(TASKS.glob('*.json')):
  t=read(f);ref=np.asarray(t['reference_path_preprocessed'],float);goal=ref[-1];scores=[]
  # These selected numeric tasks have a straight XYZ or pure yaw reference.
  direction=goal[:3];den=max(float(direction@direction),1e-10)
  if t['motion_kind']!='Turn/Rotate':
   assert np.linalg.norm(ref[:,:3]-(ref[:,:3]@direction/den)[:,None]*direction,axis=1).max()<2
  for fraction in [.0,.1,.25,.5,.75,.9,1.]:
   path=np.linspace(ref[0],goal*fraction,201)
   score=movement_reward(path,ref,kind=t['motion_kind'],stopped=True)
   assert not score['invalid_transitions'];scores.append(score['reward'])
  assert all(b>a+1e-6 for a,b in zip(scores,scores[1:]))
  wrong=np.linspace(ref[0],-goal,201)
  bad=movement_reward(wrong,ref,kind=t['motion_kind'],stopped=True)['reward']
  assert scores[-1]>bad
  rows.append(dict(task=f.name,progress_fractions=[0,.1,.25,.5,.75,.9,1],rewards=scores,wrong_direction_reward=bad))
 return dict(status='passed',rows=rows)
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['controls','gate','admit']);p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;r.mkdir(parents=True,exist_ok=True)
 if a.stage=='controls':save(r/'controls.json',controls());return
 if a.stage=='gate':
  report=dict(status='blocked',optimizer_updates=0,tasks=[],blockers=[]);allnames=[]
  assert read(r/'controls.json')['status']=='passed'
  for f in sorted(TASKS.glob('*.json')):
   t=read(f);attempts=[];allnames.append(f.name)
   sources=[Path('C:/runs/d223_preflight/results/policy/episodes')/(f.stem+'_r0.json')]+[r/'attempts/episodes'/(f.stem+'_r'+str(i)+'.json') for i in (0,1)]
   for expected_seed,path in zip((7,8,9),sources):
    ep=read(path);assert ep['task']==f.name and ep['seed']==expected_seed
    assert ep['end_reason'] in ('model_stop','horizon') and not ep['reward']['invalid_transitions']
    score=movement_reward(ep['poses'],t['reference_path_preprocessed'],kind=t['motion_kind'],stopped=ep['end_reason']=='model_stop')
    assert abs(score['reward']-sum(s['reward_v2'] for s in ep['samples']))<1e-9
    attempts.append(dict(seed=ep['seed'],reward=score['reward'],costs=costs(ep['poses'],t).tolist(),end_reason=ep['end_reason']))
   pairs=[]
   for i,x in enumerate(attempts):
    for j,y in enumerate(attempts):
     if i!=j and x['end_reason']==y['end_reason'] and dominates(np.array(x['costs']),np.array(y['costs']),t['motion_kind']=='Turn/Rotate'):
      ok=x['reward']>y['reward']+1e-6;pairs.append(dict(better_seed=x['seed'],worse_seed=y['seed'],correctly_ranked=ok))
      if not ok:report['blockers'].append(f.name+': reward contradicts independent geometric comparison')
   if not pairs:report['blockers'].append(f.name+': no meaningfully better comparable attempt; learning signal not established')
   report['tasks'].append(dict(task=f.name,attempts=attempts,comparisons=pairs))
  actual=list((r/'attempts/episodes').glob('sim_*_r*.json'));assert len(actual)==8
  if not report['blockers']:report['status']='passed'
  save(r/'ranking_gate.json',report);print(json.dumps(report),flush=True)
  if report['status']!='passed':raise SystemExit(2)
  return
 import os,datetime
 remaining=(datetime.datetime.fromisoformat(os.environ['VLA_DEADLINE_UTC'].replace('Z','+00:00'))-datetime.datetime.now(datetime.timezone.utc)).total_seconds()
 if remaining<5700:raise ValueError('Insufficient remaining budget for training, matched flights and upload')
 assert read(r/'ranking_gate.json')['status']=='passed'
 assert read(r/'replay.json')['status']=='passed'
 from ppo_math import validate_ppo_samples
 from composite_policy import adapter_digest
 samples=[];images=r/'batch/images';images.mkdir(parents=True,exist_ok=False)
 identity=adapter_digest(Path('C:/adapters/d207_stage2_adapter_s8000'))
 for epfile in sorted((r/'attempts/episodes').glob('sim_*_r*.json')):
  ep=read(epfile)
  for sample in ep['samples']:
   assert sample['reward_version']=='movement-reward-candidate-v3'
   assert sample['policy_provenance']['action_adapter_sha256']==identity
   assert sample['policy_provenance']['prefix_adapter_sha256']==identity
   for item in [sample]+([sample['final_observation']] if sample.get('final_observation') else []):
    src=(r/'attempts/rollout_images'/item['image']).resolve()
    assert src.is_relative_to((r/'attempts/rollout_images').resolve())
    assert hashlib.sha256(src.read_bytes()).hexdigest()==item['image_sha256']
    dest=images/item['image'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
   sample['reward_spec_approved']=True;samples.append(sample)
 assert validate_ppo_samples(samples,8,require_final_value=False)==8
 data=dict(schema='executed-path-ppo-v1',chunk=8,preflight_passed=True,samples=samples,scope='D224 only: four numeric nonlanding practice tasks; at most two updates',reward_version='movement-reward-candidate-v3')
 save(r/'admitted_rollouts.json',data);save(r/'admission.json',dict(status='passed',samples=len(samples),episodes=8,max_updates=2,ranking_gate='ranking_gate.json',replay='replay.json'))
if __name__=='__main__':main()

