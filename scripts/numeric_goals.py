"""D226 explicit numeric goals. No reference trajectories or model calls.

Opt-in narrow grammar; unsupported/ambiguous instructions fail closed.
All targets are relative to the episode start frame (metres, accumulated yaw).
"""
import math
import re

VERSION = 'explicit-numeric-v1'
NUMBER = r'(\d+(?:\.\d+)?)'


def numeric_goal(instruction):
    text = ' '.join(instruction.strip().split())
    flags = re.IGNORECASE
    slope = re.fullmatch(r'Travel forward '+NUMBER+r' metres along a straight '+NUMBER+r'-degree (upward|downward) slope, then stop\. The distance is along the slope, not a target altitude\.', text, flags)
    vertical = re.fullmatch(r'(Ascend|Descend) vertically by '+NUMBER+r' metres from the starting height, then stop\.(?: This is a height change, not a landing command\.)?', text, flags)
    turn = re.fullmatch(r'(?:Rotate|Turn) '+NUMBER+r' degrees to the (right|left)\.?', text, flags)
    if slope:
        distance, angle, direction = slope.groups(); distance=float(distance); angle=float(angle)
        if not 1 <= distance <= 99 or not 0 < angle <= 90:
            raise ValueError('numeric slope outside supported range')
        sign=1 if direction.lower()=='upward' else -1
        return [distance*math.cos(math.radians(angle)), 0., sign*distance*math.sin(math.radians(angle)), 0.]
    if vertical:
        direction,distance=vertical.groups(); distance=float(distance)
        if not 1 <= distance <= 99:raise ValueError('numeric height change outside supported range')
        return [0.,0.,distance*(1 if direction.lower()=='ascend' else -1),0.]
    if turn:
        angle,direction=turn.groups();angle=float(angle)
        if not 5 <= angle <= 360:raise ValueError('numeric turn outside supported range')
        return [0.,0.,0.,angle*(1 if direction.lower()=='right' else -1)]
    raise ValueError('unsupported explicit numeric instruction; landing and ambiguous commands are excluded')


def numeric_progress(instruction, position_m, yaw_total_deg):
    """Preserve signed residual, including overshoot, until a compatible tolerance.

    10 cm / 1 degree are stricter than the pilot's 50 cm / 5 degree endpoint
    criteria. This is goal arrival only, not path/semantic task certification.
    """
    goal=numeric_goal(instruction)
    if len(position_m)<3 or not all(math.isfinite(float(v)) for v in [*position_m[:3],yaw_total_deg]):
        raise ValueError('finite start-frame state required')
    position_m=[float(v) for v in position_m[:3]]
    yaw_total_deg=float(yaw_total_deg)
    dx,dy,dz=[goal[i]-position_m[i] for i in range(3)]
    yaw=math.radians(yaw_total_deg);c,s=math.cos(yaw),math.sin(yaw)
    remaining=[c*dx+s*dy,-s*dx+c*dy,dz,goal[3]-yaw_total_deg]
    position_tolerance=min(.1, .1*math.hypot(*goal[:3])) if math.hypot(*goal[:3]) else .1
    yaw_tolerance=min(1., .1*abs(goal[3])) if goal[3] else 1.
    arrived=math.hypot(dx,dy,dz)<=position_tolerance and abs(remaining[3])<=yaw_tolerance
    used=[0.,0.,0.,0.] if arrived else remaining
    return used,dict(goal_mode=VERSION,goal_start_frame=goal,remaining_before_tolerance=remaining,
                     goal_arrived=arrived,arrival_position_m=position_tolerance,arrival_yaw_deg=yaw_tolerance)


def numeric_action_instruction(instruction):
    """Express a supported slope as simultaneous axis distances, not forward range.

    Derived only from instruction text. Non-slope wording stays unchanged.
    """
    goal=numeric_goal(instruction)
    if not instruction.strip().lower().startswith('travel forward '):
        return instruction
    direction='up' if goal[2]>0 else 'down'
    return f'Move {goal[0]:.3f} metres forward and {abs(goal[2]):.3f} metres {direction} simultaneously, then stop.'


def instruction_goal_mode(instruction, *, strict=False, automatic=False):
    """Choose from instruction text only; never consult reference trajectories."""
    if strict:
        numeric_goal(instruction)
        return 'numeric'
    if automatic:
        try:
            numeric_goal(instruction)
            return 'numeric'
        except ValueError:
            pass
    return 'generated'
