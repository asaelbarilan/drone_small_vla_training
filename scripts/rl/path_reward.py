"""D214 candidate route-following reward. Offline tested, not launch-approved.

Input: start-frame [x,y,z,roll,yaw,pitch], centimetres/degrees, including start.
The reference is a continuous polyline in metres plus unwrapped yaw scaled by
metres/radian. Recording points are NOT commanded action targets. Roll/pitch are
ignored because the current action policy commands xyz and yaw only.
"""
from dataclasses import dataclass, asdict
import numpy as np

@dataclass(frozen=True)
class PathRewardConfig:
    yaw_scale: float = 1.0
    corridor: float = 0.75
    endpoint_tolerance: float = 0.5
    endpoint_yaw_tolerance_deg: float = 5.0
    advance_ratio: float = 1.5
    max_transition: float = 1.0
    completion_fraction: float = 0.98
    deviation_weight: float = 0.5
    reverse_weight: float = 0.25
    completion_bonus: float = 1.0

    def __post_init__(self):
        v=asdict(self)
        if not all(np.isfinite(x) for x in v.values()) or min(v.values())<=0:
            raise ValueError('finite positive configuration required')
        if not 0<self.completion_fraction<=1 or self.advance_ratio<1 or self.endpoint_yaw_tolerance_deg>180:
            raise ValueError('invalid completion fraction or advance ratio')

def features(poses, yaw_scale, initial_yaw=None):
    a=np.asarray(poses,float)
    if a.ndim!=2 or a.shape[1]!=6 or len(a)<2 or not np.isfinite(a).all():
        raise ValueError('at least two finite six-dimensional poses including start required')
    yaw=np.unwrap(np.deg2rad(a[:,4]))
    if initial_yaw is not None:yaw += 2*np.pi*round((initial_yaw-yaw[0])/(2*np.pi))
    return np.column_stack([a[:,:3]/100,yaw*yaw_scale])

def path_reward(poses, reference, *, stopped, config=None):
    """Return additive transition rewards and interpretable geometric evidence.

    History-constrained projection prevents nearest-point jumps from loop start
    to loop end. Progress changes are signed (no free repeated arc reward).
    Reach grows only with actual motion; stationary samples cannot creep along
    the route. Large transitions are invalid and earn no progress.
    Completion bonus is paid ONCE at the final stopped state, after ordered
    progress and whole-trajectory corridor checks. No image-based score.
    """
    c=config or PathRewardConfig()
    if not isinstance(stopped,bool):raise ValueError('explicit stopped boolean required')
    ref=features(reference,c.yaw_scale)
    actual=features(poses,c.yaw_scale,ref[0,3]/c.yaw_scale)
    seg=np.diff(ref,axis=0);lengths=np.linalg.norm(seg,axis=1)
    keep=np.r_[True,lengths>1e-10];ref=ref[keep]
    if len(ref)<2:raise ValueError('reference has no commanded movement/rotation')
    seg=np.diff(ref,axis=0);lengths=np.linalg.norm(seg,axis=1)
    cumulative=np.r_[0,np.cumsum(lengths)];total=float(cumulative[-1])
    s=0.;deviation=float(np.linalg.norm(actual[0]-ref[0]));max_dev=deviation
    progress=[];rewards=[];deviations=[deviation];reverse=0.;penalty=0.;invalid=0
    for prev,point in zip(actual,actual[1:]):
        motion=float(np.linalg.norm(point-prev));old_s=s
        if motion>c.max_transition+1e-9:
            invalid+=1
            # Preserve phase and penalize invalid motion; never silently certify.
            deviation=float(np.linalg.norm(point-(ref[np.searchsorted(cumulative,s,side='right')-1])))
            delta=0.
        elif motion<1e-12:
            delta=0.
        else:
            lo=max(0,s-c.advance_ratio*motion);hi=min(total,s+c.advance_ratio*motion)
            candidates=np.flatnonzero((cumulative[:-1]<=hi)&(cumulative[1:]>=lo))
            starts=ref[candidates];vectors=seg[candidates];lens=lengths[candidates]
            t=((point-starts)*vectors).sum(axis=1)/(lens*lens)
            t=np.maximum((lo-cumulative[candidates])/lens,np.minimum((hi-cumulative[candidates])/lens,t))
            t=np.clip(t,0,1)
            projected=starts+t[:,None]*vectors
            distances=np.linalg.norm(projected-point,axis=1)
            # Revisited points can have two equally close route phases. Keep
            # temporal continuity toward the physically reachable forward phase
            # instead of choosing the earliest segment in array order.
            phases=cumulative[candidates]+t*lens
            tied=np.flatnonzero(distances <= distances.min()+1e-9)
            best=int(tied[np.argmin(np.abs(phases[tied]-(old_s+motion)))])
            s=float(cumulative[candidates[best]]+t[best]*lens[best])
            deviation=float(distances[best]);delta=s-old_s
        # Distance-weighted integral: no reward dependence on image count.
        drift=0.5*(deviations[-1]+deviation)*motion/(total*c.corridor)
        back=max(-delta,0)/total
        reward=delta/total-c.deviation_weight*drift-c.reverse_weight*back
        reverse+=back;penalty+=c.deviation_weight*drift
        progress.append(s/total);deviations.append(deviation);rewards.append(float(reward))
        max_dev=max(max_dev,deviation)
    endpoint=float(np.linalg.norm(actual[-1]-ref[-1]))
    endpoint_yaw_error=float(abs(np.rad2deg((actual[-1,3]-ref[-1,3])/c.yaw_scale)))
    complete=bool(stopped and invalid==0 and s/total>=c.completion_fraction
                  and endpoint<=c.endpoint_tolerance and max_dev<=c.corridor
                  and endpoint_yaw_error<=c.endpoint_yaw_tolerance_deg)
    if complete:rewards[-1]+=c.completion_bonus
    return dict(version='path-reward-candidate-v2',reward=float(sum(rewards)),
                transition_rewards=rewards,progress_fraction=float(s/total),
                progress_trace=progress,deviation_trace=deviations,
                deviation_penalty=penalty,reverse_fraction=reverse,
                max_deviation=max_dev,endpoint_error=endpoint,endpoint_yaw_error_deg=endpoint_yaw_error,complete=complete,
                invalid_transitions=invalid,reference_metric_length=total,
                config=asdict(c),training_ready=False)
