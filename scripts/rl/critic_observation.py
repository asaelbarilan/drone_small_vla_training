"""D252 local candidate: critic-only facts available BEFORE an action.

No value prediction, reward change, actor input change, or training integration.
The reference is the task specification, not the subsequently executed path.
Never pass final-episode collision labels as collision_before.
"""
import numpy as np
from movement_reward import MovementRewardConfig, movement_reward
from path_reward import features


def critic_observation(poses, reference, *, executed_moves, kind, max_moves,
                       collision_before=None, config=None, stop_window=10,
                       stop_position_cm=3., stop_yaw_deg=1.):
    if type(executed_moves) is not int or type(max_moves) is not int:
        raise ValueError('integer execution boundary and move budget required')
    if not 0 <= executed_moves < max_moves or len(poses) <= executed_moves:
        raise ValueError('boundary must be a live pre-action state')
    if type(stop_window) is not int or stop_window <= 0:
        raise ValueError('positive integer stop window required')
    if not all(np.isfinite(v) and v > 0 for v in (stop_position_cm, stop_yaw_deg)):
        raise ValueError('positive finite stop thresholds required')
    if collision_before is not None and type(collision_before) is not bool:
        raise ValueError('collision_before must be measured bool or unknown')
    # Slice before inspecting values: even corrupted/future suffixes are irrelevant.
    prefix=np.array(poses[:executed_moves+1],dtype=float,copy=True)
    if prefix.shape!=(executed_moves+1,6) or not np.isfinite(prefix).all():
        raise ValueError('finite measured cm/degree prefix required')
    c=config or MovementRewardConfig()
    quiet=0
    for old,new in zip(prefix[:-1],prefix[1:]):
        small=np.max(np.abs(new[:3]-old[:3]))<stop_position_cm
        small=small and abs((new[4]-old[4]+180)%360-180)<stop_yaw_deg
        quiet=quiet+1 if small else 0
        if quiet>=stop_window:raise ValueError('prefix already ended under collector stop rule')
    # Geometry requires two poses; duplicate only for metric evaluation at start.
    measured=prefix if len(prefix)>1 else np.repeat(prefix,2,axis=0)
    events={} if collision_before is None else {'collision':collision_before}
    q=movement_reward(measured,reference,kind=kind,stopped=True,events=events,
                      config=c,stopping_mode='accepted_goal')
    pre=movement_reward(measured,reference,kind=kind,stopped=False,events=events,
                        config=c,stopping_mode='accepted_goal')
    ref=features(reference,1.);a=features(measured,1.,ref[0,3])
    remaining=ref[-1]-a[-1]
    values=dict(
        remaining_x_m=float(remaining[0]), remaining_y_m=float(remaining[1]),
        remaining_z_m=float(remaining[2]), remaining_yaw_deg=float(np.rad2deg(remaining[3])),
        endpoint_error_m=q['endpoint_position_error_m'],
        heading_error_deg=q['endpoint_yaw_error_deg'],
        heading_tolerance_deg=q['heading_tolerance_deg'],
        ordered_progress=q['progress_fraction'], max_route_deviation=q['max_deviation'],
        reversal_fraction=q['reverse_fraction'], invalid_transitions=float(q['invalid_transitions']),
        accepted_now=float(q['complete']), accepted_previously=float(q['first_accepted_pose'] is not None),
        motion_cost_since_acceptance=q['post_acceptance_motion_cost'],
        observed_prefix_reward=pre['reward'], quiet_moves=float(quiet),
        remaining_moves=float(max_moves-executed_moves),
        collision_known=float(collision_before is not None), collision_observed=float(collision_before is True),
        task_is_turn=float(kind in ('Turn','Turn/Rotate')),
        goal_distance_m=float(np.linalg.norm(ref[-1,:3]-ref[0,:3])),
        goal_yaw_deg=float(np.rad2deg(ref[-1,3]-ref[0,3])))
    if not np.isfinite(list(values.values())).all():raise ValueError('nonfinite critic observation')
    return dict(schema='critic-observed-prefix-v1-local',executed_moves=executed_moves,
                features=values,training_ready=False)
