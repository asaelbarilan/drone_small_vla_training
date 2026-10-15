"""D234 prepared-pool integrity, goal/reward admission and matched comparisons."""
import json,hashlib,shutil,re,collections
from pathlib import Path
import numpy as np
from movement_reward import movement_reward
from ppo_math import validate_ppo_samples
from sampling_contract import validate_sampling_contract

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def materialize(code,rows,dest):
 dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
 for row in rows:
  src=Path(code)/row['path']
  if hashlib.sha256(src.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('task manifest mismatch: '+row['task'])
  if read(src)['motion_kind']=='Land':raise ValueError('landing excluded')
  shutil.copyfile(src,dest/row['task'])

def episodes(run):
 return [read(f) for f in sorted((Path(run)/'episodes').glob('sim_*_r*.json'))]

def score_controls(tasks):
 rows=[]
 for f in sorted(Path(tasks).glob('*.json')):
  t=read(f);ref=np.asarray(t['reference_path_preprocessed'],float)
  clean=movement_reward(ref,ref,kind=t['motion_kind'],stopped=True)
  idle=movement_reward(np.repeat(ref[:1],2,axis=0),ref,kind=t['motion_kind'],stopped=True)
  if clean['invalid_transitions'] or not clean['geometric_complete'] or clean['reward']<=idle['reward']+1e-6:raise ValueError('clean/inaction reward ordering failed '+f.name)
  rows.append(dict(task=f.name,clean=clean['reward'],inaction=idle['reward']))
 return dict(status='passed',scope='reference versus inaction control only; not semantic certification',rows=rows)

def admission(run,tasks,replay,out,expected):
 run=Path(run);tasks=Path(tasks);out=Path(out)
 if read(run/'episodes/usage.json').get('training_eligible') is not True:raise ValueError('validation-only flights cannot train')
 if read(replay)['status']!='passed':raise ValueError('replay not passed')
 eps=episodes(run)
 if len(eps)!=expected:raise ValueError('episode count')
 allowed={p.name for p in tasks.glob('*.json')};actual={e['task'] for e in eps}
 if actual!=allowed:raise ValueError('task coverage mismatch')
 images=out/'batch/images';images.mkdir(parents=True,exist_ok=False)
 samples=[];rows=[];groups=collections.defaultdict(list)
 for ep in eps:
  task=read(tasks/ep['task']);ref=task['reference_path_preprocessed']
  if ep['kind']=='Land':raise ValueError('landing excluded')
  score=movement_reward(ep['poses'],ref,kind=task['motion_kind'],stopped=ep['end_reason']=='model_stop')
  if score['invalid_transitions']:raise ValueError('invalid executed movement')
  if abs(score['reward']-sum(s['reward_v2'] for s in ep['samples']))>1e-8:raise ValueError('reward sum mismatch')
  first=ep['samples'][0];nums=[float(x) for x in re.findall(r'[+-]?\d+(?:\.\d+)?',first['progress_text'])]
  if len(nums)!=4:raise ValueError('malformed goal context '+ep['task'])
  if all(x==0 for x in nums):raise ValueError('initial generated goal says already done: '+ep['task'])
  row=dict(task=ep['task'],kind=ep['kind'],seed=ep['seed'],end=ep['end_reason'],reward=score['reward'],endpoint=score['endpoint_position_error_m'],heading=score['endpoint_yaw_error_deg'],deviation=score['max_deviation'],progress=score['progress_fraction'],initial_progress=first['progress_text'],tokens=[i for s in ep['samples'] for i in s['ids']])
  groups[ep['task']].append(row);rows.append(row)
  for sample in ep['samples']:
   if sample['reward_version']!='movement-reward-candidate-v5':raise ValueError('reward version')
   for obs in [sample]+([sample['final_observation']] if sample.get('final_observation') else []):
    src=(run/'rollout_images'/obs['image']).resolve()
    if not src.is_relative_to((run/'rollout_images').resolve()) or hashlib.sha256(src.read_bytes()).hexdigest()!=obs['image_sha256']:raise ValueError('image identity')
    dst=images/obs['image'];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
   sample['reward_spec_approved']=True;samples.append(sample)
 comparisons=[]
 for task,pair in groups.items():
  if len(pair)!=2 or len({x['seed'] for x in pair})!=2:raise ValueError('two seeds per task required')
  for a in pair:
   for b in pair:
    if a is b or a['end']!=b['end']:continue
    # Reject clear observed geometric dominance inversions; absence is not certification.
    better=(a['endpoint']+.1<b['endpoint'] and a['heading']<=b['heading']+.5 and a['deviation']<=b['deviation']+.03 and a['progress']>=b['progress']-.001)
    if better:
     comparisons.append(dict(task=task,better_seed=a['seed'],worse_seed=b['seed']))
     if a['reward']<=b['reward']+1e-6:raise ValueError('observed reward inversion '+task)
 if not any(p[0]['tokens']!=p[1]['tokens'] for p in groups.values()) or np.std([x['reward'] for x in rows])<1e-6:raise ValueError('no action/reward variation')
 if validate_ppo_samples(samples,8,require_final_value=False)!=expected:raise ValueError('sample episode count')
 validate_sampling_contract(samples)
 artifact=dict(schema='executed-path-ppo-v1',chunk=8,preflight_passed=True,reward_version='movement-reward-candidate-v5',samples=samples)
 save(out/'admitted_rollouts.json',artifact)
 save(out/'admission.json',dict(status='passed',calls=len(samples),episodes=expected,comparisons=comparisons,rows=rows,scope='clean/idle controls, observed dominance screen, exact replay; not semantic ground truth'))
 return len(samples)

def comparison(before,after,out,expected):
 b={(e['task'],e['seed']):e for e in episodes(before)};a={(e['task'],e['seed']):e for e in episodes(after)}
 if set(a)!=set(b) or len(a)!=expected:raise ValueError('matched comparison identity/count')
 def compact(e):return dict(end_reason=e['end_reason'],moves=sum(e['executed_counts']),**{k:e['reward'][k] for k in ('reward','geometric_complete','progress_fraction','endpoint_position_error_m','endpoint_yaw_error_deg','max_deviation')})
 rows=[dict(task=k[0],seed=k[1],kind=a[k]['kind'],before=compact(b[k]),after=compact(a[k])) for k in sorted(a)]
 result=dict(scope='matched tasks and policy seeds; simulator images may differ; historically exposed validation',rows=rows,automatic_promotion=False)
 save(out,result);return result

def behavior_gate(before,after,comparison_result):
 """Conservative early decision from matched development flights; no efficacy claim."""
 b={(e['task'],e['seed']):e for e in episodes(before)};a={(e['task'],e['seed']):e for e in episodes(after)}
 if set(a)!=set(b) or not a:raise ValueError('behavior gate identity mismatch')
 changed=[];goal_confounds=[]
 for key in sorted(a):
  old=[i for s in b[key]['samples'] for i in s['ids']];new=[i for s in a[key]['samples'] for i in s['ids']]
  if old!=new:changed.append(key[0])
  parse=lambda e:np.array([float(x) for x in re.findall(r'[+-]?\d+(?:\.\d+)?',e['samples'][0]['progress_text'])])
  x,y=parse(b[key]),parse(a[key])
  if len(x)!=4 or len(y)!=4 or np.linalg.norm(x[:3]-y[:3])>.1 or abs(x[3]-y[3])>1:goal_confounds.append(key[0])
 rows=comparison_result['rows'];old=sum(x['before']['geometric_complete'] for x in rows);new=sum(x['after']['geometric_complete'] for x in rows)
 reward_delta=float(np.mean([x['after']['reward']-x['before']['reward'] for x in rows]))
 endpoint_gain=float(np.mean([x['before']['endpoint_position_error_m']-x['after']['endpoint_position_error_m'] for x in rows]))
 demonstrated=(new>old or (new==old and reward_delta>.05 and endpoint_gain>.1))
 return dict(continue_training=bool(changed and not goal_confounds and new>=old and demonstrated),changed_action_tasks=changed,initial_goal_confounds=goal_confounds,before_complete=old,after_complete=new,mean_reward_delta=reward_delta,mean_endpoint_improvement_m=endpoint_gain,scope='Small historically exposed development set; cannot establish general improvement. Continue only with changed actions, comparable initial goals and completion gain or joint reward/endpoint gain.')
