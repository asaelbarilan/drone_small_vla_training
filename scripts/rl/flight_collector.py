"""Explicit execution loop, independent of simulator transport.

No launch/reset logic. The supplied environment MUST return read-back start-frame
poses in centimetres/degrees, including images captured at the same boundary.
A separate observation callback generates context without executing an action.
"""
import copy
import numpy as np
from executed_rollouts import attach_path_rewards


def collect_episode(env, policy, observe_context, reference, *, max_steps, stop_window=10,
                    stop_position_cm=3., stop_yaw_deg=1., reward_config=None, reward_kind=None, reward_stopping_mode="continuous"):
    if max_steps<=0 or stop_window<=0:raise ValueError('positive execution bounds required')
    calls=[];counts=[];poses=[];quiet=0;steps=0
    observation=env.observe()
    if observation.get('pose_source')!='simulator_readback':
        raise ValueError('collector requires independent simulator pose readback')
    poses.append(observation['pose']);end_reason=None
    while steps<max_steps:
        call=copy.deepcopy(policy(observation))
        if not call.get('poses'):raise ValueError('empty proposal is not a valid terminal action')
        calls.append(call);executed=0
        for command in call['poses']:
            if steps>=max_steps:break
            env.execute(command)
            # Read every executed transition, including early-stop/horizon tail.
            current=env.observe()
            if current.get('pose_source')!='simulator_readback':raise ValueError('missing pose readback')
            old=np.asarray(poses[-1],float);new=np.asarray(current['pose'],float)
            if new.shape!=(6,) or not np.isfinite(new).all():raise ValueError('invalid readback')
            small=np.max(np.abs(new[:3]-old[:3]))<stop_position_cm
            small=small and abs((new[4]-old[4]+180)%360-180)<stop_yaw_deg
            quiet=quiet+1 if small else 0
            poses.append(new.tolist());observation=current;executed+=1;steps+=1
            if quiet>=stop_window:
                end_reason='model_stop';break
        counts.append(executed)
        if end_reason:break
    end_reason=end_reason or 'horizon'
    final=observe_context(observation) if end_reason=='horizon' else None
    samples,reward=attach_path_rewards(calls,poses,counts,reference,end_reason=end_reason,
                                      final_observation=final,config=reward_config,kind=reward_kind,stopping_mode=reward_stopping_mode)
    return dict(samples=samples,reward=reward,poses=poses,executed_counts=counts,
                end_reason=end_reason,pose_source='simulator_readback',final_observation=final)
