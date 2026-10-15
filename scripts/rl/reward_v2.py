"""D213 offline orbit diagnostic. NOT an approved semantic training reward.

All positions are metres in one declared frame. No automatic task/instruction
parsing: a reference-fitted centre measures reference agreement, not object truth.
Historical reward_uav_flow.py remains unchanged.
"""
from dataclasses import dataclass
import numpy as np

VERSION = "orbit-diagnostic-v2.0"

@dataclass(frozen=True)
class OrbitSpec:
    centre_xy: tuple
    radius_m: float
    altitude_m: float
    direction: int
    radial_tolerance_m: float = 0.5
    altitude_tolerance_m: float = 0.5
    max_move_m: float = 0.5
    max_angular_step_deg: float = 20.0
    min_sweep_deg: float = 350.0
    max_reverse_deg: float = 15.0
    closure_m: float = 0.5

    def __post_init__(self):
        nums = [*self.centre_xy, self.radius_m, self.altitude_m,
                self.radial_tolerance_m, self.altitude_tolerance_m,
                self.max_move_m, self.max_angular_step_deg, self.min_sweep_deg,
                self.max_reverse_deg, self.closure_m]
        if len(self.centre_xy) != 2 or not np.isfinite(nums).all():
            raise ValueError("finite centre and tolerances required")
        if self.direction not in (-1, 1) or self.radius_m <= 0:
            raise ValueError("positive radius and direction +/-1 required")
        if min(nums[4:]) <= 0 or self.radial_tolerance_m >= self.radius_m:
            raise ValueError("positive tolerances smaller than radius required")
        if not 0 < self.max_angular_step_deg < 180 or not 0 < self.min_sweep_deg <= 360:
            raise ValueError("invalid angular specification")


def orbit_diagnostic(positions_m, spec, *, stopped):
    """Conservative whole-flight check; starting pose MUST be included.

    Reject jumps, wrong direction, radius/altitude departures and incomplete
    stopping. Segment clearance is checked against the inner radial corridor.
    This is geometry only: no object identity, collision or physical dynamics.
    """
    xyz = np.asarray(positions_m, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 2 or not np.isfinite(xyz).all():
        raise ValueError("at least two finite xyz poses including start required")
    if not isinstance(stopped, bool):
        raise ValueError("explicit boolean stopped required")
    xy = xyz[:, :2] - np.asarray(spec.centre_xy)
    radii = np.linalg.norm(xy, axis=1)
    if np.any(radii < 1e-8):
        return dict(version=VERSION, geometric_complete=False, reasons=["centre_crossing"],
                    semantic_approved=False)
    angles = np.arctan2(xy[:, 1], xy[:, 0])
    delta = np.rad2deg(np.angle(np.exp(1j * np.diff(angles)))) * spec.direction
    sweep = float(delta.sum())
    reverse = float(-np.minimum(delta, 0).sum())
    moves = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    segment = np.diff(xy, axis=0)
    denom = (segment * segment).sum(axis=1)
    projection = np.divide(-(xy[:-1] * segment).sum(axis=1), denom,
                           out=np.zeros_like(denom), where=denom > 0).clip(0, 1)
    inner = np.linalg.norm(xy[:-1] + projection[:, None] * segment, axis=1)
    closure = float(np.linalg.norm(xyz[-1] - xyz[0]))
    checks = {
        "insufficient_sweep": sweep + 1e-7 < spec.min_sweep_deg,
        "reverse_motion": reverse > spec.max_reverse_deg,
        "radial_departure": bool(np.any(np.abs(radii - spec.radius_m) > spec.radial_tolerance_m)),
        "inner_shortcut": bool(np.any(inner < spec.radius_m - spec.radial_tolerance_m)),
        "altitude_departure": bool(np.any(np.abs(xyz[:, 2] - spec.altitude_m) > spec.altitude_tolerance_m)),
        "translation_jump": bool(np.any(moves > spec.max_move_m)),
        "angular_jump": bool(np.any(np.abs(delta) > spec.max_angular_step_deg)),
        "not_closed": closure > spec.closure_m,
        "not_stopped": not stopped,
    }
    reasons = [k for k, failed in checks.items() if failed]
    return dict(version=VERSION, geometric_complete=not reasons, semantic_approved=False,
                reasons=reasons, signed_sweep_deg=sweep, reverse_deg=reverse,
                closure_m=closure, max_move_m=float(moves.max()),
                max_radius_error_m=float(np.abs(radii-spec.radius_m).max()))


def fit_reference_orbit(reference_cm_deg):
    """Fit in the reference start frame, for diagnostics ONLY; never task truth."""
    ref = np.asarray(reference_cm_deg, dtype=float)
    if ref.ndim != 2 or ref.shape[1] != 6 or len(ref) < 20 or not np.isfinite(ref).all():
        raise ValueError("finite six-dimensional reference required")
    xyz = ref[:, :3] / 100.0
    xy = xyz[:, :2]
    design = np.column_stack([2*xy, np.ones(len(xy))])
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError("degenerate reference circle")
    solution = np.linalg.lstsq(design, (xy*xy).sum(axis=1), rcond=None)[0]
    centre = solution[:2]
    radius = float(np.sqrt(solution[2] + centre@centre))
    phase = np.unwrap(np.arctan2(xy[:, 1]-centre[1], xy[:, 0]-centre[0]))
    sweep = float(np.rad2deg(phase[-1]-phase[0]))
    spec = OrbitSpec(tuple(centre), radius, float(np.median(xyz[:, 2])), 1 if sweep > 0 else -1)
    if not 350 <= abs(sweep) <= 380:
        raise ValueError("reference does not describe one complete circle")
    check = orbit_diagnostic(xyz, spec, stopped=True)
    if not check['geometric_complete']:
        raise ValueError("reference fails geometric controls: " + ','.join(check['reasons']))
    return spec


def full_path_similarity(flight, reference, spacing=0.25, max_points=2048):
    """New diagnostic, NOT historical benchmark nDTW or semantic success.

    Resample the WHOLE trajectories in a metric of metres plus sine/cosine
    angles (radians encoded as unit vectors). Both endpoints always included.
    Preserves yaw-only motion and angle sign; fails rather than clipping paths.
    """
    if spacing <= 0 or max_points < 2:
        raise ValueError("positive spacing and at least two points required")
    def resample(poses):
        a=np.asarray(poses,float)
        if a.ndim!=2 or a.shape[1]!=6 or len(a)==0 or not np.isfinite(a).all():
            raise ValueError("finite nonempty six-dimensional path required")
        angles=np.deg2rad(a[:,3:])
        features=np.column_stack([a[:,:3]/100,np.sin(angles),np.cos(angles)])
        distances=np.r_[0,np.cumsum(np.linalg.norm(np.diff(features,axis=0),axis=1))]
        keep=np.r_[True,np.diff(distances)>1e-10]
        distances,features=distances[keep],features[keep]
        length=float(distances[-1])
        if length==0:return features[:1],length
        count=int(np.ceil(length/spacing))+1
        if count>max_points:raise ValueError("path exceeds diagnostic resource bound; no truncation allowed")
        grid=np.linspace(0,length,count)
        return np.column_stack([np.interp(grid,distances,features[:,i]) for i in range(features.shape[1])]),length
    model,_=resample([list(x['state'][0])+list(x['state'][1]) for x in flight])
    truth,length=resample(reference)
    previous=np.full(len(model)+1,np.inf);previous[0]=0
    for point in truth:
        current=np.full(len(model)+1,np.inf)
        costs=np.linalg.norm(model-point,axis=1)
        for j,cost in enumerate(costs,1):current[j]=cost+min(previous[j],current[j-1],previous[j-1])
        previous=current
    score=float(np.exp(-previous[-1]/max(length,spacing)))
    return dict(version="full-path-diagnostic-v2.0",similarity=score,
                reference_points=len(truth),model_points=len(model),reference_metric_length=length)
