"""D220 local counterexamples and saved-flight rescoring; no simulator/network."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/rl'))
from movement_reward import movement_reward,KINDS
from path_reward import path_reward

OUT=ROOT/'reports/d220_reward_audit'
TASKS=ROOT/'reports/d210_dagger_audit/practice/tasks'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def compact(x):return {k:v for k,v in x.items() if k not in
    ('transition_rewards','progress_trace','deviation_trace','config','notes')}
def line(x=0,y=0,z=0,yaw=0,n=101):
    a=np.zeros((n,6));a[:,:3]=np.linspace([0,0,0],[x*100,y*100,z*100],n)
    a[:,4]=np.linspace(0,yaw,n);return a
def controls():
    theta=np.linspace(0,2*np.pi,201);circle=line(n=201)
    circle[:,0]=300*np.sin(theta);circle[:,1]=300*(1-np.cos(theta))
    t=np.linspace(0,1,101);pas=line(x=6);pas[:,1]=-200*np.sin(np.pi*t)
    return {'Approach/Move':line(x=5),'Retreat':line(x=-5),'Shift':line(y=-4),
            'Ascend/Descend':line(x=3,z=2),'Land':line(z=-3),
            'Turn':line(yaw=-9),'Turn/Rotate':line(yaw=60),'Pass':pas,'Surround':circle}
def variants(ref):
    ref=np.asarray(ref,float);half=ref[:max(2,len(ref)//2)]
    wrong=2*ref[0]-ref
    over=np.vstack([ref,ref[-1]+np.linspace(0,1,31)[1:,None]*(ref[-1]-ref[-max(2,len(ref)//5)])])
    deviation=ref.copy()
    xyz=ref[:,:3]-ref[0,:3]
    normal=np.linalg.svd(xyz,full_matrices=False)[2][-1]
    deviation[:,:3]+=150*np.sin(np.linspace(0,np.pi,len(ref)))[:,None]*normal
    shortcut=np.linspace(ref[0],ref[-1],len(ref))
    reversal=np.vstack([ref[:len(ref)//2],ref[len(ref)//2-1:0:-1],ref])
    return dict(correct=ref,early_stop=half,stationary=np.repeat(ref[:1],2,axis=0),
                wrong_direction=wrong,overshoot=over,off_path_detour=deviation,
                shortcut=shortcut,reversal=reversal)
def main():
    OUT.mkdir(parents=True,exist_ok=True);checks=[];synthetic=[]
    def check(name,ok):checks.append(dict(name=name,passed=bool(ok)))
    for kind,ref in controls().items():
        scores={}
        for name,poses in variants(ref).items():
            events={'landed':True} if kind=='Land' else {}
            scores[name]=dict(old=compact(path_reward(poses,ref,stopped=True)),
                             new=compact(movement_reward(poses,ref,kind=kind,stopped=True,events=events)))
        good=scores['correct']['new']
        check(kind+': correct geometry completes',good['complete'])
        for name in ('early_stop','stationary','wrong_direction','overshoot','off_path_detour','reversal'):
            check(kind+': correct outranks '+name,good['reward']>scores[name]['new']['reward'])
        for name in ('early_stop','stationary','wrong_direction','overshoot','off_path_detour'):
            check(kind+': rejects '+name,not scores[name]['new']['complete'])
        if kind in ('Pass','Surround'):
            check(kind+': rejects shortcut',not scores['shortcut']['new']['complete'])
        event={'landed':True} if kind=='Land' else {}
        duplicate=np.repeat(ref,3,axis=0)
        repeated=movement_reward(duplicate,ref,kind=kind,stopped=True,events=event)
        check(kind+': duplicate photos invariant',abs(repeated['reward']-good['reward'])<1e-8)
        dense=np.vstack([p+(q-p)*t for p,q in zip(ref,ref[1:]) for t in (0,.5)]+[ref[-1]])
        twice=movement_reward(dense,ref,kind=kind,stopped=True,events=event)
        check(kind+': same geometry resampling stable',abs(twice['reward']-good['reward'])<.01)
        timeout=movement_reward(ref,ref,kind=kind,stopped=False,events=event)
        check(kind+': horizon has no completion credit',not timeout['complete'] and timeout['terminal_credit']==0)
        collision=movement_reward(ref,ref,kind=kind,stopped=True,events={'collision':True,**event})
        check(kind+': collision earns no positive return',collision['reward']<=-1+1e-9 and not collision['complete'])
        check(kind+': rewards attach to actual transitions',len(repeated['transition_rewards'])==len(duplicate)-1)
        synthetic.append(dict(kind=kind,scores=scores))
    # Symmetric heading error, including crossing the old 98% threshold.
    ref=line(yaw=-9)
    under=movement_reward(line(yaw=-8.4),ref,kind='Turn',stopped=True)
    over=movement_reward(line(yaw=-9.6),ref,kind='Turn',stopped=True)
    check('Turn: equal under/overshoot get equal credit',abs(under['reward']-over['reward'])<1e-8)
    left=movement_reward(line(yaw=-8.8199),ref,kind='Turn',stopped=True)
    right=movement_reward(line(yaw=-8.8201),ref,kind='Turn',stopped=True)
    check('Turn: no 98 percent bonus cliff',abs(left['reward']-right['reward'])<.001)
    land=movement_reward(line(z=-3),line(z=-3),kind='Land',stopped=True)
    check('Land: missing contact never certifies landing',not land['complete'] and land['terminal_credit']==0)
    # Wrap-around equivalence.
    a=line(yaw=20);a[:,4]+=170;b=a.copy();b[:,4]=(b[:,4]+180)%360-180
    check('Turn: yaw wrapping invariant',abs(movement_reward(a,a,kind='Turn',stopped=True)['reward']-
                                            movement_reward(b,a,kind='Turn',stopped=True)['reward'])<1e-8)
    for kind in KINDS:
        r=controls()[kind]
        # Separate validity from interpolation: a large single move must fail.
        if kind not in ('Turn','Surround'):
            jump=np.vstack([r[0],r[-1]])
            j=movement_reward(jump,r,kind=kind,stopped=True,events={'landed':True})
            check(kind+': original jump not hidden by interpolation',j['invalid_transitions']>0 and not j['complete'])
    inventory=[]
    for f in sorted(TASKS.glob('*.json')):
        task=read(f);ref=task['reference_path_preprocessed'];kind=task['motion_kind']
        row=dict(task=f.name,kind=kind,instruction=task['instruction'],
                 source_sha256=hashlib.sha256(f.read_bytes()).hexdigest())
        try:
            row['self_control']=compact(movement_reward(ref,ref,kind=kind,stopped=True))
            row['stationary']=compact(movement_reward([ref[0],ref[0]],ref,kind=kind,stopped=True))
            check(f.name+': reference outranks stationary',row['self_control']['reward']>row['stationary']['reward'])
            check(f.name+': reference geometric self-control',row['self_control']['geometric_complete'])
            row['counterexamples']={}
            for name,poses in variants(ref).items():
                if name not in ('early_stop','wrong_direction','off_path_detour'): continue
                score=movement_reward(poses,ref,kind=kind,stopped=True)
                row['counterexamples'][name]=compact(score)
                check(f.name+': reference outranks '+name,row['self_control']['reward']>score['reward'])
                check(f.name+': rejects '+name,not score['complete'])
        except ValueError as e:row['error']=str(e);check(f.name+': reference accepted',False)
        row['semantic_status']='unverified_reference_only'
        if kind=='Land':row['semantic_status']='blocked_task_intent_and_contact'
        if kind=='Ascend/Descend':row['semantic_status']='blocked_altitude_vs_travel_distance_ambiguity'
        inventory.append(row)
    saved=[]
    for label,folder in [('before',ROOT/'reports/d217_ppo_pilot/r2/evidence/before/episodes'),
                         ('after',ROOT/'reports/d218_saved_ppo_eval/evidence/after/episodes')]:
        for f in sorted(folder.glob('sim_*_r*.json')):
            ep=read(f);task=read(TASKS/ep['task'])
            new=movement_reward(ep['poses'],task['reference_path_preprocessed'],
                kind=task['motion_kind'],stopped=ep['end_reason']=='model_stop')
            check(label+' '+ep['task']+': transition sum',abs(sum(new['transition_rewards'])-new['reward'])<1e-9)
            saved.append(dict(phase=label,task=ep['task'],kind=task['motion_kind'],
                              old=compact(ep['reward']),new=compact(new),source=str(f.relative_to(ROOT))))
    report=dict(scope='Offline synthetic controls, 100 practice references and 8 saved flights. No flights/training.',
                checks=checks,passed=sum(x['passed'] for x in checks),total=len(checks),
                synthetic=synthetic,inventory=inventory,saved=saved,
                source_sha256=hashlib.sha256((ROOT/'scripts/rl/movement_reward.py').read_bytes()).hexdigest())
    (OUT/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],total=report['total'],failed=[c for c in checks if not c['passed']],
                          saved=[dict(kind=r['kind'],phase=r['phase'],old=r['old']['reward'],new=r['new']['reward'],
                                      complete=r['new']['complete']) for r in saved]),indent=2))
    return report
if __name__=='__main__':
    result=main()
    raise SystemExit(0 if result['passed']==result['total'] else 1)
