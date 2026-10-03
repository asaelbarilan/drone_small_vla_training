"""D182: how good is each "Left" line? Fresh model line vs goal-memory line vs truth.

In the goal-memory flight test (D180, s31000 --progress --goal-memory) the server logged, at
every call, the line the model wrote from the current photo (progress_said) AND the line it
actually used, recomputed from the first remembered goal (progress_text). Both refer to the
same drone pose, so they can be scored against the truth on identical states: the remaining
displacement from that pose to the benchmark's reference end, in the drone's frame (metres,
yaw in degrees, the trainer's remaining_to_end convention).

Prints per motion class and per call index: median position and turn error of the fresh line,
the memory line and simple combinations. Offline only; uses the logged flights and the
benchmark task files.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "rl"))
from rollout_samples import match  # noqa: E402
from train_uav_flow_vla import parse_progress  # noqa: E402

EVAL = Path("D:/drone_vla_pilot/simulators/uav_flow_repo/UAV-Flow-Eval")


def truth_left(proprio, ref_end):
    """Remaining displacement from a logged state (cm, yaw deg) to the reference end."""
    yaw = math.radians(proprio[3])
    dx, dy = (ref_end[0] - proprio[0]) / 100.0, (ref_end[1] - proprio[1]) / 100.0
    forward = math.cos(yaw) * dx + math.sin(yaw) * dy
    sideways = -math.sin(yaw) * dx + math.cos(yaw) * dy
    turn = (ref_end[4] - proprio[3] + 180.0) % 360.0 - 180.0
    return np.array([forward, sideways, (ref_end[2] - proprio[2]) / 100.0, turn])


def errors(line, truth):
    if line is None:
        return None
    line = np.asarray(line, dtype=float)
    position = float(np.linalg.norm(line[:3] - truth[:3]))
    turn = abs((line[3] - truth[3] + 180.0) % 360.0 - 180.0)
    return position, turn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("D:/drone_vla_pilot/runs/sim_eval_win/d178_s31000_goalmem"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    classes = json.loads((EVAL / "classified_instr.json").read_text(encoding="utf-8"))
    kind = {f: name for name, files in classes.items() for f in files}
    matched = match(args.run / "flights", args.run / "server_calls.jsonl")
    rows = []
    for task, calls in matched.items():
        ref = json.loads((EVAL / "test_jsons" / task).read_text(encoding="utf-8"))
        ref_end = ref["reference_path_preprocessed"][-1]
        for index, call in enumerate(calls):
            truth = truth_left(call["proprio"], ref_end)
            fresh = call.get("progress_parsed")
            memory = parse_progress(call.get("progress_text"))
            row = dict(task=task, cls=kind.get(task, "?"), call=index, truth=truth.tolist(),
                       fresh=fresh, memory=memory)
            for name, line in (("fresh", fresh), ("memory", memory)):
                e = errors(line, truth)
                row[name + "_pos"], row[name + "_turn"] = (e if e else (None, None))
            # Candidate rules, from the two logged lines only.
            if fresh and memory:
                mixed = [*memory[:3], fresh[3]]  # memory for position, fresh for turn
                e = errors(mixed, truth)
                row["mixed_pos"], row["mixed_turn"] = e
            rows.append(row)

    def med(values):
        values = [v for v in values if v is not None]
        return round(float(np.median(values)), 2) if values else None

    print(f"matched flights {len(matched)}, calls {len(rows)}")
    print("class            calls  pos err m: fresh / memory     turn err deg: fresh / memory / mixed")
    for name in list(classes) + ["ALL"]:
        mine = [r for r in rows if name == "ALL" or r["cls"] == name]
        print(f"{name:16s} {len(mine):5d}   {med([r['fresh_pos'] for r in mine])!s:>6} / {med([r['memory_pos'] for r in mine])!s:<6}"
              f"        {med([r['fresh_turn'] for r in mine])!s:>6} / {med([r['memory_turn'] for r in mine])!s:<6} / {med([r.get('mixed_turn') for r in mine])}")
    print("\nby call index (ALL): call  pos fresh/memory  turn fresh/memory")
    for index in range(0, 8):
        mine = [r for r in rows if r["call"] == index]
        if mine:
            print(f"  {index}: n={len(mine):3d}  {med([r['fresh_pos'] for r in mine])} / {med([r['memory_pos'] for r in mine])}"
                  f"   {med([r['fresh_turn'] for r in mine])} / {med([r['memory_turn'] for r in mine])}")
    if args.out:
        args.out.write_text(json.dumps(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
