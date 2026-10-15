"""D248 opt-in numeric stopping objective; no policy or simulator launch."""
import numpy as np
from path_reward import features

def accepted_goal_reward(poses,reference,*,kind,stopped,events=None,config=None,altitude_cost_mode='legacy_exp'):
    from movement_reward import movement_reward,MovementRewardConfig
    if kind not in ('Ascend/Descend','Turn','Turn/Rotate'):
        raise ValueError('accepted_goal is validated only for numeric altitude/turn tasks; landing excluded')
    c=config or MovementRewardConfig()
    kwargs=dict(kind=kind,events=events,config=c,altitude_cost_mode=altitude_cost_mode)
    final=movement_reward(poses,reference,stopped=stopped,**kwargs)
    a=np.asarray(poses,float);first=None
    # Measured execution boundaries, not reference photo indices.
    for end in range(2,len(a)+1):
        prefix=movement_reward(a[:end],reference,stopped=True,**kwargs)
        if prefix['complete']:
            first=end-1;break
    rewards=np.asarray(final['transition_rewards'],float).copy()
    rewards[-1]-=final['terminal_credit']
    post_cost=0.
    if first is not None:
        pre=movement_reward(a[:first+1],reference,stopped=False,**kwargs)
        rewards[:first]=pre['transition_rewards']
        ref=features(reference,1.);actual=features(a,1.,ref[0,3])
        scale=float(np.linalg.norm(np.diff(ref,axis=0),axis=1).sum())
        costs=np.linalg.norm(np.diff(actual,axis=0),axis=1)[first:]/scale
        rewards[first:]=-costs;post_cost=float(costs.sum())
    terminal=(1. if final['complete'] else -1.) if stopped else 0.
    rewards[-1]+=terminal
    if final['invalid_transitions'] or (events or {}).get('collision') is True:
        rewards[-1]-=max(0.,float(rewards.sum()))+1.
    return final|dict(version='movement-reward-accepted-goal-v8-local',
        reward=float(rewards.sum()),transition_rewards=rewards.tolist(),terminal_credit=terminal,
        first_accepted_pose=first,post_acceptance_motion_cost=post_cost,
        stopping_mode='accepted_goal',training_ready=False,
        notes=final['notes']+['D248 local opt-in: +1 accepted stop, -1 premature stop; pre-arrival shaping frozen at first acceptance, then normalized extra-motion cost. No per-photo/time penalty.'])
