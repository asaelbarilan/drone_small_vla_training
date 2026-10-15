"""D215 attach route reward to measured execution, never proposed suffixes."""
import copy
import math
import numpy as np
from path_reward import path_reward

# Declared kinematic heading agreement tolerance; 0.05 deg is 0.87 mm at a 1 m yaw scale.
EXECUTED_YAW_TOLERANCE_DEG = 0.05


def attach_path_rewards(calls, actual_poses, executed_counts, reference, *, end_reason, final_observation=None, config=None, kind=None, events=None, stopping_mode="continuous"):
    """actual_poses includes start, in cm/degrees [xyz,roll,yaw,pitch].

    Counts must come from the evaluator after execution, not server proposals.
    The full sampled chunk remains the policy macro-action even if its suffix
    was not executed. Rewards only sum executed moves. gamma=1 per call is the
    initial finite-episode choice, preserving the geometric total across chunking.
    kind explicitly opts into the D220 candidate; omitted preserves legacy scoring.
    Neither branch approves samples for training.
    """
    if end_reason not in ('model_stop','horizon'):
        raise ValueError('infrastructure/unknown ending is not a training episode')
    if not calls or len(calls)!=len(executed_counts):raise ValueError('call/count mismatch')
    if any(type(n) is not int or n<=0 for n in executed_counts):
        raise ValueError('each sampled call must have an explicit positive executed count')
    a=np.asarray(actual_poses,float)
    if a.ndim!=2 or a.shape[1]!=6 or not np.isfinite(a).all() or sum(executed_counts)!=len(a)-1:
        raise ValueError('executed counts do not cover actual trajectory exactly')
    if end_reason=='horizon' and not final_observation:
        raise ValueError('horizon requires real final observation for bootstrap')
    if kind is None and stopping_mode!='continuous':raise ValueError('stopping mode requires movement kind')
    if kind is None:
        reward=path_reward(actual_poses,reference,stopped=end_reason=='model_stop',config=config)
    else:
        from movement_reward import movement_reward
        reward=movement_reward(actual_poses,reference,kind=kind,
            stopped=end_reason=='model_stop',events=events,config=config,stopping_mode=stopping_mode)
    if reward['invalid_transitions']:raise ValueError('invalid executed movement: quarantine episode')
    samples=[];offset=0
    for index,(call,count) in enumerate(zip(calls,executed_counts,strict=True)):
        if count>len(call['poses']):raise ValueError('executed more actions than proposed')
        expected=np.asarray(call['proprio'],float)
        if not np.allclose(expected[:3],a[offset,:3],atol=.05,rtol=0):
            raise ValueError('call observation not at actual execution boundary')
        yaw_error=(expected[3]-a[offset,4]+180)%360-180
        if abs(yaw_error)>.01:raise ValueError('call yaw/frame mismatch')
        proposed=np.asarray(call['poses'][:count],float)
        # Current kinematic simulator should measure the commanded positions.
        # Dynamics-aware collection must declare a different execution contract.
        if not np.allclose(proposed[:,:3],a[offset+1:offset+count+1,:3],atol=.05,rtol=0):
            raise ValueError('actual positions disagree with proposed kinematic execution')
        yaw=np.rad2deg(proposed[:,3]);actual_yaw=a[offset+1:offset+count+1,4]
        if np.any(np.abs((yaw-actual_yaw+180)%360-180)>EXECUTED_YAW_TOLERANCE_DEG):raise ValueError('actual yaw mismatch')
        s=copy.deepcopy(call);last=index==len(calls)-1
        s.update(call_index=index,executed_count=count,
                 reward_v2=float(sum(reward['transition_rewards'][offset:offset+count])),
                 terminated=last and end_reason=='model_stop',truncated=last and end_reason=='horizon',
                 reward_version=reward['version'],discount_per_call=1.0,
                 reward_spec_approved=False)
        if last and end_reason=='horizon':
            s['final_observation']=copy.deepcopy(final_observation)
            s['final_observation_id']=final_observation['id']
        samples.append(s);offset+=count
    return samples,reward
