"""D221 explicit practice-task repairs and fail-closed landing evidence.
No network, simulator launch or automatic approval of PPO training.
"""
import copy,hashlib,json,re
import numpy as np
from movement_reward import movement_reward

def reference_hash(reference):
    return hashlib.sha256(json.dumps(reference,separators=(',',':')).encode()).hexdigest()

def repair_vertical_task(task):
    """Create a NEW practice instruction consistent with the normalized command
    and reference. Never assert that this was the original author's intent.
    """
    t=copy.deepcopy(task);kind=t['motion_kind']
    spec=dict(version='vertical-task-v1',source_instruction=t['instruction'],
              source_unified=t.get('instruction_unified'),training_approved=False)
    ref=np.asarray(t['reference_path_preprocessed'],float)
    if ref.ndim!=2 or ref.shape[1]!=6 or len(ref)<2 or not np.isfinite(ref).all():
        raise ValueError('finite reference required')
    if np.linalg.norm(ref[0])>.01:raise ValueError('reference must start at origin')
    unified=t.get('instruction_unified','')
    slope=re.fullmatch(r'(Ascend|Descend) ([0-9.]+) meters at ([0-9.]+) degrees',unified)
    vertical=re.fullmatch(r'Land ([0-9.]+) meters',unified)
    if slope:
        direction,distance,angle=slope.groups();d=float(distance);a=float(angle)
        if d<=0 or not 0<a<=90:raise ValueError('invalid slope')
        sign=1 if direction=='Ascend' else -1
        expected=np.array([d*np.cos(np.deg2rad(a)),0,sign*d*np.sin(np.deg2rad(a))])
        instruction=f'Travel forward {d:g} metres along a straight {a:g}-degree '+('upward' if sign>0 else 'downward')+' slope, then stop. The distance is along the slope, not a target altitude.'
        mode='slope_distance'
    elif kind=='Land' and vertical:
        d=float(vertical.group(1))
        if d<=0:raise ValueError('positive descent distance required')
        expected=np.array([0,0,-d]);mode='relative_vertical'
        instruction=f'Descend vertically by {d:g} metres from the starting height, then stop. This is a height change, not a landing command.'
    elif kind=='Land':
        spec.update(status='blocked_contact_and_landing_site',requires=['validated_landing_site','measured_stable_contact'])
        return t,spec
    else:raise ValueError('unsupported or ambiguous vertical command')
    xyz=ref[:,:3]/100
    error=float(np.linalg.norm(xyz[-1]-expected))
    projection=np.clip(xyz@expected/(expected@expected),0,1)
    residual=float(np.linalg.norm(xyz-projection[:,None]*expected,axis=1).max())
    if error>.02 or residual>.02 or np.any(np.diff(projection)<-.001):
        raise ValueError('normalized command and reference disagree; manual review required')
    t.update(instruction=instruction,instruction_unified=instruction,motion_kind='Ascend/Descend')
    spec.update(status='local_geometry_verified',mode=mode,target_delta_m=expected.tolist(),
                reference_endpoint_error_m=error,reference_line_error_m=residual,
                reference_sha256=reference_hash(t['reference_path_preprocessed']),
                repaired_instruction=instruction,requires=['simulator_frame_and_endpoint_check'])
    t['reward_task_spec']=spec
    return t,spec

def landing_evidence(samples, *, surface_id, now_s, dwell_s=1., max_age_s=.5):
    """Require a fresh, sustained contact on a declared landing surface.

    Samples must come from a validated simulator contact/clearance provider, not
    inference from pose. Thresholds are provisional and must be calibrated.
    """
    if not surface_id or not np.isfinite([now_s,dwell_s,max_age_s]).all() or min(dwell_s,max_age_s)<=0:
        raise ValueError('explicit site and valid timing bounds required')
    if len(samples)<3:return dict(landed=False,reason='insufficient_contact_samples')
    times=[]
    for s in samples:
        try:
            vals=[s['time_s'],s['clearance_m'],s['speed_m_s']]
            if not np.isfinite(vals).all() or s['speed_m_s']<0:raise ValueError()
            if type(s['contact']) is not bool or type(s['collision']) is not bool:raise ValueError()
            times.append(s['time_s'])
        except (KeyError,TypeError,ValueError):
            return dict(landed=False,reason='invalid_telemetry')
    if np.any(np.diff(times)<=0) or now_s<times[-1] or now_s-times[-1]>max_age_s:
        return dict(landed=False,reason='stale_or_invalid_timestamps')
    valid=[]
    for s in samples:
        valid.append(s.get('source')=='validated_simulator_contact' and
                     s.get('surface_id')==surface_id and s['contact'] and not s['collision'] and
                     abs(s['clearance_m'])<=.1 and s['speed_m_s']<=.1)
    start=len(valid)-1
    if not valid[-1]:return dict(landed=False,reason='not_stable_contact')
    while start>0 and valid[start-1] and times[start]-times[start-1]<=max_age_s:
        start-=1
    ok=len(valid)-start>=3 and times[-1]-times[start]>=dwell_s
    return dict(landed=bool(ok),reason='verified_contact' if ok else 'contact_not_sustained')

def vertical_reward(poses, task, *, stopped):
    spec=task.get('reward_task_spec',{})
    if spec.get('status')!='local_geometry_verified' or task['instruction']!=spec.get('repaired_instruction'):
        raise ValueError('explicit repaired task specification required')
    if reference_hash(task['reference_path_preprocessed'])!=spec.get('reference_sha256'):
        raise ValueError('reference changed after verification')
    # Admission remains separate; a local scorer is not a training permit.
    return movement_reward(poses,task['reference_path_preprocessed'],kind='Ascend/Descend',stopped=stopped)
