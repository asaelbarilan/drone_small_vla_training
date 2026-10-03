"""D172 RL: flight reward, following WorldVLN's reward_uavflow.py (task part).

WorldVLN (github.com/EmbodiedCity/WorldVLN.code, Worldmodel/runtime/tools/GRPO/
reward_uavflow.py, CC BY 4.0) scores a rollout by its END pose against the
reference end:

  dense    = 1 / (1 + (pos_err_m / 2)^2 + (yaw_err_deg / 10)^2)
  success  = pos_err_m <= 3 and yaw_err_deg <= 10
  reward   = 0.85 * dense + 0.15 * success

Poses are in the start frame, centimetres and degrees, as written by the
UAV-Flow-Eval evaluator (flight json: [{"state": [[x, y, z], [roll, yaw, pitch]]}])
and as stored in reference_path_preprocessed ([x, y, z, roll, yaw, pitch]).
The group advantage is the reward z-scored within the rollouts of one task.

D172 addition (user decision): the benchmark also judges the path's shape, and
WorldVLN rewards closeness to the expert path as well, so the final reward is
  reward = 0.5 * end_reward + 0.5 * nDTW
with nDTW computed exactly as UAV-Flow-Eval/metric.py does (positions / 100,
cosines of the three angles, sampling stride 2 for Turn/Move and 5 otherwise,
positions zeroed for Turn/Rotate, reference capped at 20 samples, eta = 1).
"""

import json
from pathlib import Path

import numpy as np

POS_SCALE_M, YAW_SCALE_DEG = 2.0, 10.0
SUCCESS_M, SUCCESS_DEG = 3.0, 10.0
DENSE_WEIGHT, SUCCESS_WEIGHT = 0.85, 0.15


def end_errors(flight, reference):
    """(position error in metres, yaw error in degrees) of a flight's last pose."""
    ref = np.asarray(reference[-1], dtype=float)
    if not flight:
        end, yaw = np.zeros(3), 0.0
    else:
        end = np.asarray(flight[-1]["state"][0], dtype=float)
        yaw = float(flight[-1]["state"][1][1])
    position = float(np.linalg.norm(end - ref[:3])) / 100.0
    heading = abs((yaw - ref[4] + 180.0) % 360.0 - 180.0)
    return position, heading


def _vectors(poses, step, zero_pos, cap=None):
    out = []
    for i, (position, angles) in enumerate(poses):
        if i % step == 0:
            p = np.zeros(3) if zero_pos else np.asarray(position, dtype=float) / 100
            out.append(np.concatenate([p, np.cos(np.deg2rad(np.asarray(angles, dtype=float)))]))
    return out[:cap] if cap else out


def ndtw(flight, reference, kind):
    """The official UAV-Flow-Eval nDTW (metric.py), 0 when undefined."""
    zero_pos = kind in ("Turn", "Rotate", "Turn/Rotate")
    step = 2 if kind in ("Turn", "Move", "Turn/Rotate") else 5
    model = _vectors([(s["state"][0], s["state"][1]) for s in flight], step, zero_pos)
    truth = _vectors([(r[:3], r[3:6]) for r in reference], step, zero_pos, cap=20)
    if not model or not truth:
        return 0.0
    cost = np.linalg.norm(np.asarray(truth)[:, None, :] - np.asarray(model)[None, :, :], axis=-1)
    dtw = np.full((len(truth) + 1, len(model) + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, len(truth) + 1):
        for j in range(1, len(model) + 1):
            dtw[i, j] = cost[i - 1, j - 1] + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    length = sum(np.linalg.norm(truth[i] - truth[i - 1]) for i in range(1, len(truth)))
    return float(np.exp(-dtw[-1, -1] / length)) if length > 0 else 0.0


def reward(flight, reference, kind=None):
    position, heading = end_errors(flight, reference)
    dense = 1.0 / (1.0 + (position / POS_SCALE_M) ** 2 + (heading / YAW_SCALE_DEG) ** 2)
    success = float(position <= SUCCESS_M and heading <= SUCCESS_DEG)
    end = DENSE_WEIGHT * dense + SUCCESS_WEIGHT * success
    path = ndtw(flight, reference, kind)
    return dict(
        reward=0.5 * end + 0.5 * path,
        end_reward=end,
        ndtw=path,
        dense=dense,
        success=success,
        position_error_m=position,
        yaw_error_deg=heading,
    )


def group_advantages(rewards, eps=1e-6):
    """Rewards z-scored within one task's group; all-equal groups carry no signal."""
    values = np.asarray(rewards, dtype=float)
    spread = values.std()
    if spread < eps:
        return np.zeros_like(values)
    return (values - values.mean()) / (spread + eps)


def score_flight_file(flight_path, task_path, kind=None):
    flight = json.loads(Path(flight_path).read_text(encoding="utf-8"))
    task = json.loads(Path(task_path).read_text(encoding="utf-8"))
    return reward(flight, task["reference_path_preprocessed"], kind or task.get("motion_kind"))
