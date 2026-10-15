"""D220 locally audited candidate: geometry and stopping, not semantic certification.

cm/degree start-frame poses, including start; gamma=1. Reference labels remain
untrusted task specifications. No camera-count or wall-time term. Historical
path_reward is preserved for reproducibility. Not enabled in paid pilots.
"""
from dataclasses import dataclass, asdict
import numpy as np
from path_reward import path_reward, PathRewardConfig, features

KINDS = ('Approach/Move','Retreat','Shift','Ascend/Descend','Land',
         'Turn','Turn/Rotate','Pass','Surround')

@dataclass(frozen=True)
class MovementRewardConfig:
    altitude_route_budget: float = .5
    corridor_metric: float = .75
    endpoint_m: float = .5
    heading_deg: float = 5.
    turn_relative_tolerance: float = .1
    max_translation_m: float = 1.
    max_rotation_deg: float = 45.
    integration_spacing: float = .05
    max_integrated_points: int = 20000

    def __post_init__(self):
        if not all(np.isfinite(x) and x > 0 for x in asdict(self).values()):
            raise ValueError('finite positive configuration required')
        if self.max_rotation_deg >= 180 or self.turn_relative_tolerance > 1:
            raise ValueError('rotation bound must resolve winding; relative tolerance <= 1')
        if type(self.max_integrated_points) is not int:
            raise ValueError('integer integration bound required')

def _densify(poses, spacing, limit):
    # Interpolate along measured segments, not reference photo indices.
    # Preserve unwrapped yaw and mapping back to each executed transition.
    a=np.asarray(poses,float).copy()
    a[:,4]=np.rad2deg(np.unwrap(np.deg2rad(a[:,4])))
    metric=features(a,1.)
    counts=np.maximum(1,np.ceil(np.linalg.norm(np.diff(metric,axis=0),axis=1)/spacing).astype(int))
    if int(counts.sum())+1 > limit:
        raise ValueError('trajectory exceeds integration bound; never truncate')
    dense=[a[0]]
    for start,end,n in zip(a,a[1:],counts):
        dense.extend(start+(end-start)*t for t in np.arange(1,n+1)/n)
    return np.asarray(dense),counts

def movement_reward(poses, reference, *, kind, stopped, events=None, config=None, altitude_cost_mode="legacy_exp", stopping_mode="continuous"):
    """Continuous terminal credit; separate geometric completion and task truth.

    events is optional trusted simulator telemetry, NOT inferred from images or
    the category label. Land completion requires landed=True. A collision=True
    invalidates completion and removes positive episode credit.
    """
    if stopping_mode == 'accepted_goal':
        from accepted_goal_reward import accepted_goal_reward
        return accepted_goal_reward(poses,reference,kind=kind,stopped=stopped,events=events,config=config,altitude_cost_mode=altitude_cost_mode)
    if stopping_mode != 'continuous':raise ValueError('unknown stopping mode')
    if altitude_cost_mode not in ('legacy_exp','rational','rational_tolerant_heading'):
        raise ValueError('unknown altitude cost mode')
    if kind not in KINDS: raise ValueError('unknown movement kind')
    if type(stopped) is not bool: raise ValueError('explicit stopped boolean required')
    c=config or MovementRewardConfig()
    events={} if events is None else events
    for key in ('landed','collision'):
        if key in events and type(events[key]) is not bool:
            raise ValueError('events require measured boolean values')
    ref=features(reference,1.)
    actual=features(poses,1.,ref[0,3])
    delta=np.diff(actual,axis=0)
    invalid=int(np.sum((np.linalg.norm(delta[:,:3],axis=1)>c.max_translation_m+1e-9) |
                       (np.abs(np.rad2deg(delta[:,3]))>c.max_rotation_deg+1e-9)))
    start_mismatch=bool(np.linalg.norm(actual[0]-ref[0])>.01)
    invalid+=int(start_mismatch)
    dense,counts=_densify(poses,c.integration_spacing,c.max_integrated_points)
    old=path_reward(dense,reference,stopped=False,config=PathRewardConfig(
        corridor=c.corridor_metric, endpoint_tolerance=c.endpoint_m,
        endpoint_yaw_tolerance_deg=c.heading_deg))
    pos_error=float(np.linalg.norm(actual[-1,:3]-ref[-1,:3]))
    yaw_error=float(abs(np.rad2deg(actual[-1,3]-ref[-1,3])))
    length=old['reference_metric_length']
    yaw_goal=float(np.rad2deg(ref[-1,3]-ref[0,3]))
    notes=['Reference geometry is not independent verification of instruction/object identity.',
           'Thresholds are provisional engineering settings, not calibrated success labels.']
    if kind in ('Turn','Turn/Rotate'):
        if abs(yaw_goal)<1e-6 or np.max(np.linalg.norm(ref[:,:3]-ref[0,:3],axis=1))>.05:
            raise ValueError('Turn reference must be a nonzero in-place rotation')
        tolerance=min(c.heading_deg,abs(yaw_goal)*c.turn_relative_tolerance)
        # Equal undershoot/overshoot get equal heading utility; no 98% cliff.
        yaw=np.rad2deg(np.unwrap(np.deg2rad(dense[:,4])))
        yaw += 360*round((np.rad2deg(ref[0,3])-yaw[0])/360)
        error=np.abs(yaw-np.rad2deg(ref[-1,3]))
        utility=1-error/abs(yaw_goal)
        drift=np.linalg.norm(dense[:,:3]/100-ref[0,:3],axis=1)
        drift_cost=drift/c.corridor_metric
        # Potential differences telescope; repeated images earn no credit.
        rewards=np.diff(utility-drift_cost)
        motion=np.linalg.norm(np.diff(features(dense,1.),axis=0),axis=1)
        rewards-=.5*(drift_cost[:-1]+drift_cost[1:])/2*motion/max(abs(np.deg2rad(yaw_goal)),1.)
        reversal=float(np.maximum(-np.diff(yaw)*np.sign(yaw_goal),0).sum()/abs(yaw_goal))
        rewards-=.25*np.maximum(-np.diff(yaw)*np.sign(yaw_goal),0)/abs(yaw_goal)
        quality=float(np.exp(-.5*(yaw_error/tolerance)**2-.5*(pos_error/c.endpoint_m)**2
                             -.5*(float(drift.max())/c.corridor_metric)**2))
        geometric=bool(yaw_error<=tolerance and drift.max()<=c.corridor_metric
                       and pos_error<=c.endpoint_m)
        progress=float(np.clip(1-yaw_error/abs(yaw_goal),0,1))
        max_dev=float(drift.max())
    else:
        tolerance=c.heading_deg
        rewards=np.asarray(old['transition_rewards'],float)
        progress=old['progress_fraction'];max_dev=old['max_deviation']
        reversal=old['reverse_fraction']
        quality=float(progress*np.exp(-.5*(pos_error/c.endpoint_m)**2
                      -.5*(yaw_error/c.heading_deg)**2
                      -.5*(max_dev/c.corridor_metric)**2))
        if kind == 'Ascend/Descend':
            # D229: reaching a valid altitude target must beat staying still.
            # Bound cumulative route/backtracking cost, not each move's cost:
            # B(C)=budget*(1-exp(-C/budget)). Differences telescope, so loops
            # cannot earn credit and duplicate observations cost nothing.
            increments=np.diff(np.r_[0.,old['progress_trace']])
            raw_cost=np.maximum(0.,increments-rewards)
            cumulative=np.r_[0.,np.cumsum(raw_cost)]
            bounded=(c.altitude_route_budget*cumulative/(c.altitude_route_budget+cumulative)
                     if altitude_cost_mode in ('rational','rational_tolerant_heading') else
                     -c.altitude_route_budget*np.expm1(-cumulative/c.altitude_route_budget))
            # D231: route projection alone retains full credit past the end.
            # Cap its potential by remaining endpoint distance. Signed changes
            # remove credit on overshoot and telescope through any recovery.
            start_distance=float(np.linalg.norm(ref[-1,:3]-ref[0,:3]))
            if start_distance < 1e-6:
                raise ValueError('Altitude reference requires nonzero endpoint displacement')
            endpoint_trace=np.linalg.norm(dense[:,:3]/100-ref[-1,:3],axis=1)
            potential=np.minimum(np.r_[0.,old['progress_trace']],
                                 1-endpoint_trace/start_distance)
            rewards=np.diff(potential)-np.diff(bounded)
            # Route history is charged above; do not suppress arrival again.
            # D243: for altitude tasks, heading inside the existing acceptance
            # tolerance gets equal terminal credit. Outside it, smoothly charge
            # excess error. Route cost still accounts for the executed geometry.
            terminal_yaw_error=(max(0.,yaw_error-c.heading_deg)
                                if altitude_cost_mode=='rational_tolerant_heading'
                                else yaw_error)
            # Arrival is endpoint-based for the local candidate. Ordered route
            # projection remains in the dense potential/cost and completion gate;
            # multiplying it here double-counts near-end projection differences.
            if altitude_cost_mode=='rational_tolerant_heading':
                # Subtract initial endpoint utility: doing nothing earns zero
                # arrival credit, including goals closer than endpoint tolerance.
                start_exponent=-.5*(start_distance/c.endpoint_m)**2
                end_exponent=-.5*(pos_error/c.endpoint_m)**2
                arrival=max(0.,float((np.exp(end_exponent)-np.exp(start_exponent))
                                    /(-np.expm1(start_exponent))))
                quality=float(arrival*np.exp(-.5*(terminal_yaw_error/c.heading_deg)**2))
            else:
                quality=float(progress*np.exp(-.5*(pos_error/c.endpoint_m)**2
                              -.5*(terminal_yaw_error/c.heading_deg)**2))
            if altitude_cost_mode=='rational_tolerant_heading':
                notes.append('Local v7: altitude terminal heading has a tolerance plateau; not enabled in training.')
            notes.append('Altitude arrival-priority shaping; route completion still requires the original corridor. Route cost saturates; collision telemetry remains necessary.')
        # Symmetric endpoint tolerance allows a small shortfall, not overshoot
        # preference. Loops retain >=98% ordered coverage to forbid shortcuts.
        remaining=(1-progress)*length
        allowed_remaining=min(c.endpoint_m,.1*length)
        if kind=='Surround': allowed_remaining=.02*length
        geometric=bool(remaining<=allowed_remaining+1e-9 and pos_error<=c.endpoint_m
                       and yaw_error<=c.heading_deg and max_dev<=c.corridor_metric)
    # The terminal component has no discontinuity at the completion threshold.
    # Landing is special: contact cannot be reconstructed from pose alone.
    terminal_allowed=stopped and invalid==0 and events.get('collision') is not True
    if kind=='Land':
        terminal_allowed=terminal_allowed and events.get('landed') is True
        notes.append('Category includes altitude-only instructions. Confirm task intent and contact telemetry before training.')
    terminal_credit=quality if terminal_allowed else 0.
    rewards[-1]+=terminal_credit
    if invalid or events.get('collision') is True:
        # Do not reward an invalid jump or a collision even if later recovered.
        rewards[-1]-=max(0.,float(rewards.sum()))+1.
    ends=np.cumsum(counts);starts=np.r_[0,ends[:-1]]
    transitions=[float(rewards[s:e].sum()) for s,e in zip(starts,ends)]
    complete=bool(geometric and terminal_allowed)
    return dict(version=('movement-reward-candidate-v6-local' if altitude_cost_mode in ('rational','rational_tolerant_heading') else 'movement-reward-candidate-v5'),kind=kind,
                reward=float(sum(transitions)),transition_rewards=transitions,
                progress_fraction=progress,terminal_credit=terminal_credit,
                endpoint_position_error_m=pos_error,endpoint_yaw_error_deg=yaw_error,
                heading_tolerance_deg=tolerance,max_deviation=max_dev,
                reverse_fraction=reversal,invalid_transitions=invalid,
                geometric_complete=bool(geometric and stopped and invalid==0
                                        and events.get('collision') is not True),
                complete=complete,semantic_complete=None,
                events=events,training_ready=False,notes=notes,config=asdict(c))
