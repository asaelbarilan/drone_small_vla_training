"""D172 RL: practice tasks for the UAV-Flow-Eval evaluator from simulator flights.

Each task file has the benchmark's format (test_jsons/*.json), so the unmodified
batch_run_act_all.py can fly it:

  instruction, instruction_unified   from the flight
  initial_pos                        raw_logs[0] in world centimetres/degrees
                                     (index 4 is the yaw the evaluator uses)
  reference_path_preprocessed        the flight's start-frame path, cm and deg,
                                     as [x, y, z, 0, yaw, 0] (the reward reads it)
  use_obj / obj_id / target_pos      the person and car are parked underground:
                                     simulator flight records carry no object
                                     placement, and without this a person left
                                     by the previous task would stay in view.

Only TRAINING flights are used: the 504 simulator validation flights are
excluded, and flights near test starts were never added to the store (D165).
Instructions that name a spawned object (person, dog, car, ...) are dropped,
because that object cannot be placed where it was when the flight was recorded.
Self-check: the reference end, rotated by the start yaw and added to the start,
must land on the flight's recorded world end point (the index's `end`).
"""

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_uav_flow_vla import motion_kind  # noqa: E402

SPAWNED = re.compile(
    r"\b(person|people|man|woman|individual|pedestrian|human|character|dog|pet|animal|car|vehicle)\b",
    re.IGNORECASE,
)
HIDDEN = [0.0, 0.0, -10000.0, 0.0, 0.0, 0.0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=Path("D:/drone_vla_pilot/data/rl/sim_paths.json"))
    parser.add_argument("--index", type=Path, default=Path("D:/drone_vla_pilot/data/uav_flow_sim_index.json"))
    parser.add_argument("--val", type=Path, default=Path("D:/drone_vla_pilot/data/rl/sim_val_flights.json"))
    parser.add_argument("--per-kind", type=int, default=25)
    parser.add_argument("--out", type=Path, default=Path("D:/drone_vla_pilot/data/rl/tasks"))
    args = parser.parse_args()

    paths = json.loads(args.paths.read_text(encoding="utf-8"))
    index = json.loads(args.index.read_text(encoding="utf-8"))
    val = set(json.loads(args.val.read_text(encoding="utf-8")))
    by_kind, dropped = {}, dict(validation=0, spawned_object=0, short=0, no_start=0)
    for episode, flight in sorted(paths.items()):
        source = episode.removeprefix("sim_")
        if episode in val or flight["split"] != "train":
            dropped["validation"] += 1
        elif SPAWNED.search(flight["instruction"]) or SPAWNED.search(flight["instruction_unified"]):
            dropped["spawned_object"] += 1
        elif len(flight["proprio"]) < 5:
            dropped["short"] += 1
        elif source not in index:
            dropped["no_start"] += 1
        else:
            by_kind.setdefault(motion_kind(flight["instruction_unified"]), []).append(episode)

    args.out.mkdir(parents=True, exist_ok=True)
    picker, worst, written = random.Random(23), 0.0, {}
    for kind, episodes in sorted(by_kind.items()):
        chosen = picker.sample(episodes, min(args.per_kind, len(episodes)))
        written[kind] = len(chosen)
        for episode in chosen:
            flight, start = paths[episode], index[episode.removeprefix("sim_")]
            reference = [[x * 100, y * 100, z * 100, 0.0, yaw, 0.0] for x, y, z, yaw in flight["proprio"]]
            # Self-check against the recorded world end point.
            theta = math.radians(start["start"][4])
            ex, ey = reference[-1][0], reference[-1][1]
            world = (
                start["start"][0] + ex * math.cos(theta) - ey * math.sin(theta),
                start["start"][1] + ex * math.sin(theta) + ey * math.cos(theta),
            )
            worst = max(worst, math.dist(world, start["end"][:2]))
            task = dict(
                instruction=flight["instruction"],
                instruction_unified=flight["instruction_unified"],
                initial_pos=start["start"],
                reference_path_preprocessed=reference,
                use_obj=1,
                obj_id=0,
                target_pos=HIDDEN,
                source_flight=episode,
                motion_kind=kind,
            )
            (args.out / f"{episode}.json").write_text(json.dumps(task), encoding="utf-8")
    summary = dict(
        available={k: len(v) for k, v in sorted(by_kind.items())},
        written=written,
        dropped=dropped,
        worst_end_mismatch_cm=round(worst, 2),
    )
    (args.out.parent / "tasks_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
