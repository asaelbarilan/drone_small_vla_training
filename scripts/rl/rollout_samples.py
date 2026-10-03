"""D172 RL: turn rollouts into GRPO training samples.

A rollout batch is one or more evaluator passes over the same practice tasks:
each pass has flights/<task>.json (what the evaluator flew) and the server's
server_calls.jsonl (what the model answered, per call). This module

  1. splits the calls into episodes (the rollout server numbers them at /reset;
     older logs start a new episode at every all-zero state),
  2. matches every flight file to the episode that produced it: the flight's
     poses must be exactly the first poses of that episode (the evaluator may stop
     mid-chunk, and a crashed task can leave a partial episode that is re-flown),
  3. scores each flight with reward_uav_flow and z-scores the rewards within each
     task's group of flights (GRPO), and
  4. returns one sample per model call of a matched episode, carrying the
     episode's advantage, plus (D175, call_weight > 0) a per-move part: how much
     that call's moves brought the drone closer to the reference end, z-scored
     over all calls of the task's group (a potential-style shaping term).
"""

import json
import math
from pathlib import Path

import numpy as np
from reward_uav_flow import group_advantages, reward


def _key(pose):
    return (round(pose[0], 2), round(pose[1], 2), round(pose[2], 2), round(_yaw(pose[3]), 2))


def _yaw(degrees):
    return (degrees + 180.0) % 360.0 - 180.0


def episodes(calls_path):
    groups, current = [], None
    for line in Path(calls_path).read_text(encoding="utf-8").splitlines():
        call = json.loads(line)
        number = call.get("episode")
        starts = current is None or (
            number != current[0]["episode"] if number is not None else not any(call["proprio"])
        )
        if starts:
            current = []
            groups.append(current)
        current.append(call)
    return groups


def flown(episode):
    """The episode's poses as the evaluator logs them (start frame, yaw degrees)."""
    return [_key([*p[:3], math.degrees(p[3])]) for call in episode for p in call["poses"]]


def match(flights_dir, calls_path):
    """{task file name: episode calls} for every flight file with an exact match."""
    eps = [(flown(e), e) for e in episodes(calls_path)]
    found = {}
    for path in sorted(Path(flights_dir).glob("*.json")):
        states = json.loads(path.read_text(encoding="utf-8"))
        target = [_key([*s["state"][0], s["state"][1][1]]) for s in states]
        hits = [e for poses, e in eps if target and poses[: len(target)] == target]
        if hits:
            found[path.name] = hits[-1]
    return found


def call_gain(call, reference_end):
    """D175: closeness gained by one call's moves toward the reference end, in
    reward units (2 m or 10 degrees = 1), from the call's state to its last pose."""
    if not call["poses"]:
        return 0.0
    end = np.asarray(reference_end, dtype=float)

    def distance(position, yaw_deg):
        metres = float(np.linalg.norm(np.asarray(position, dtype=float) - end[:3])) / 100.0
        return metres / 2.0 + abs(_yaw(yaw_deg - end[4])) / 10.0

    before = distance(call["proprio"][:3], call["proprio"][3])
    last = call["poses"][-1]
    return before - distance(last[:3], math.degrees(last[3]))


def build(passes, tasks_dir, call_weight=0.0):
    """passes: [(flights_dir, calls_path), ...] over the same tasks.
    Returns (samples, per-flight summary rows)."""
    by_task = {}
    for number, (flights_dir, calls_path) in enumerate(passes):
        for name, episode in match(flights_dir, calls_path).items():
            flight = json.loads((Path(flights_dir) / name).read_text(encoding="utf-8"))
            task = json.loads((Path(tasks_dir) / name).read_text(encoding="utf-8"))
            score = reward(flight, task["reference_path_preprocessed"], task.get("motion_kind"))
            by_task.setdefault(name, []).append(dict(passage=number, episode=episode, **score))
    samples, rows = [], []
    for name, flights in sorted(by_task.items()):
        advantages = group_advantages([f["reward"] for f in flights])
        end = json.loads((Path(tasks_dir) / name).read_text(encoding="utf-8"))[
            "reference_path_preprocessed"
        ][-1]
        gains = [call_gain(call, end) for flight in flights for call in flight["episode"]]
        per_call = iter(group_advantages(gains) if call_weight else np.zeros(len(gains)))
        raw_gain = iter(gains)  # D185 PPO: the per-step reward itself
        for flight, advantage in zip(flights, advantages, strict=True):
            rows.append(
                dict(
                    task=name,
                    passage=flight["passage"],
                    reward=round(flight["reward"], 4),
                    success=flight["success"],
                    advantage=round(float(advantage), 4),
                    calls=len(flight["episode"]),
                )
            )
            for index, call in enumerate(flight["episode"]):
                samples.append(
                    dict(
                        task=name,
                        passage=flight["passage"],
                        advantage=float(advantage) + call_weight * float(next(per_call)),
                        flight_advantage=float(advantage),
                        call_gain=float(next(raw_gain)),
                        call_index=index,
                        last_call=index == len(flight["episode"]) - 1,
                        flight_reward=float(flight["reward"]),
                        **call,
                    )
                )
    return samples, rows


def ppo_advantages(samples, gamma=0.99, lam=0.95):
    """D185 PPO: GAE over each flight's calls, in place (s["gae"], s["return"]).
    Reward per call = its own progress gain (call_gain), plus the flight's end reward on
    its last call; V(state) = s["value_old"] (0 before the value head has learned).
    Returns {(task, passage): calls in order}."""
    flights = {}
    for s in samples:
        flights.setdefault((s["task"], s["passage"]), []).append(s)
    for calls in flights.values():
        calls.sort(key=lambda s: s["call_index"])
        advantage = 0.0
        for i in reversed(range(len(calls))):
            s = calls[i]
            last = s["last_call"] or i == len(calls) - 1
            reward = s["call_gain"] + (s["flight_reward"] if s["last_call"] else 0.0)
            after = 0.0 if last else calls[i + 1].get("value_old", 0.0)
            delta = reward + gamma * after - s.get("value_old", 0.0)
            advantage = delta + (0.0 if last else gamma * lam * advantage)
            s["gae"], s["return"] = advantage, advantage + s.get("value_old", 0.0)
    return flights


def summary(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r["task"], []).append(r)
    informative = sum(1 for g in groups.values() if any(r["advantage"] != 0 for r in g))
    return dict(
        tasks=len(groups),
        flights=len(rows),
        mean_reward=round(float(np.mean([r["reward"] for r in rows])), 4) if rows else None,
        success=round(float(np.mean([r["success"] for r in rows])), 4) if rows else None,
        groups_with_signal=informative,
    )
