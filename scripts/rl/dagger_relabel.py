"""D185 DAgger: relabel the states our own policy visited with the recorded expert flight.

Ross et al. 2011 (DAgger): fly the student, let an expert label the states the student
actually reached, train on them. Our practice tasks are recorded training flights, so the
expert is free and exact: for any logged pose we know
  - the progress line: the displacement from that pose to the flight's recorded end, in the
    drone's frame (metres; turn = the turn to the nearest path point plus the path's summed
    turn after it, degrees), and
  - the next 8 moves: follow the recorded path from its nearest point, with the pose's offset
    from the path removed linearly over the 8 steps, each move in the frame of the previous
    target pose (yaw in radians, like the training actions).
Input: the flights and server_calls.jsonl of a collection pass flown with the server's
--save-call-images (every call's photo saved); output: training rows in the trainer's
format (train_uav_flow_vla.py --extra-rows). Practice tasks are training flights only.
"""

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from rollout_samples import match  # noqa: E402

OFFICIAL_PROMPT = "Current State: {state}, What action should the uav take to {instruction}?"


def wrap(degrees):
    return (degrees + 180.0) % 360.0 - 180.0


def nearest_index(reference, pose):
    """Path index closest to a pose; 1 m of distance counts as much as 30 degrees of yaw."""
    best, index = math.inf, 0
    for i, r in enumerate(reference):
        d = math.dist(r[:3], pose[:3]) / 100.0 + abs(wrap(r[4] - pose[3])) / 30.0
        if d < best:
            best, index = d, i
    return index


def local(delta_xy, yaw_deg):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return c * delta_xy[0] + s * delta_xy[1], -s * delta_xy[0] + c * delta_xy[1]


def expert_labels(reference, pose, steps=8):
    """(progress line [m, m, m, deg], 8 moves [m, m, m, rad]) for a pose (cm, cm, cm, deg)."""
    j = nearest_index(reference, pose)
    end = reference[-1]
    forward, sideways = local(((end[0] - pose[0]) / 100.0, (end[1] - pose[1]) / 100.0), pose[3])
    turn = wrap(reference[j][4] - pose[3]) + sum(
        wrap(b[4] - a[4]) for a, b in zip(reference[j:-1], reference[j + 1 :])
    )
    progress = [forward, sideways, (end[2] - pose[2]) / 100.0, turn]

    offset = [pose[0] - reference[j][0], pose[1] - reference[j][1], pose[2] - reference[j][2],
              wrap(pose[3] - reference[j][4])]
    current = list(pose)
    moves = []
    for k in range(1, steps + 1):
        r = reference[min(j + k, len(reference) - 1)]
        keep = 1.0 - k / steps  # the offset from the path shrinks to zero over the chunk
        target = [r[0] + keep * offset[0], r[1] + keep * offset[1], r[2] + keep * offset[2],
                  r[4] + keep * offset[3]]
        dx, dy = local(((target[0] - current[0]) / 100.0, (target[1] - current[1]) / 100.0), current[3])
        moves.append([dx, dy, (target[2] - current[2]) / 100.0, math.radians(wrap(target[3] - current[3]))])
        current = target
    return progress, moves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flights", nargs="+", type=Path, required=True)
    parser.add_argument("--calls", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="rows .jsonl for --extra-rows")
    args = parser.parse_args()

    rows, flights = [], 0
    for flights_dir in args.flights:
        for task_name, calls in match(flights_dir, args.calls).items():
            task = json.loads((args.tasks / task_name).read_text(encoding="utf-8"))
            reference = task["reference_path_preprocessed"]
            flights += 1
            for call in calls:
                if not call.get("image"):
                    continue
                pose = call["proprio"]
                progress, moves = expert_labels(reference, pose)
                metres = [pose[0] / 100, pose[1] / 100, pose[2] / 100, pose[3]]
                state = ",".join(str(round(float(x), 1)) for x in metres)
                rows.append(dict(
                    format="official",
                    id=f"dagger:{task_name}:{call['episode']}:{call['call']}",
                    episode=task["source_flight"],
                    step=call["call"],
                    image=str((args.images / call["image"]).resolve()),
                    prompt=OFFICIAL_PROMPT.format(state=state, instruction=call["instr"]),
                    instruction=call["instr"],
                    chunk=moves,
                    chunk_len=len(moves),
                    progress=progress,
                ))
    args.out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(json.dumps(dict(flights=flights, rows=len(rows), out=str(args.out))))


if __name__ == "__main__":
    main()
