"""D187: score flights against their own task files (validation or practice tasks).

score_uav_flow_sim.py scores against the benchmark's test_jsons; rule 1 keeps those untouched,
so development decisions are scored here instead: every flights/<task>.json is matched to
tasks/<task>.json and gets the official-style nDTW (reward_uav_flow.ndtw: stride 2 for
Turn/Move, 5 otherwise, positions zeroed for Turn/Rotate, reference capped at 20 samples) and
success = end within 3 m and 10 degrees (WorldVLN's rule). Summary per motion type.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "rl"))
from reward_uav_flow import end_errors, ndtw  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flights", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rows = []
    for task_path in sorted(args.tasks.glob("*.json")):
        flight_path = args.flights / task_path.name
        if not flight_path.exists():
            continue
        task = json.loads(task_path.read_text(encoding="utf-8"))
        flight = json.loads(flight_path.read_text(encoding="utf-8"))
        kind = task.get("motion_kind", "other")
        position, heading = end_errors(flight, task["reference_path_preprocessed"])
        rows.append(dict(task=task_path.name, kind=kind, steps=len(flight),
                         ndtw=round(ndtw(flight, task["reference_path_preprocessed"], kind), 4),
                         end_m=round(position, 3), yaw_deg=round(heading, 1),
                         success=bool(position <= 3.0 and heading <= 10.0)))
    per = {}
    for r in rows:
        per.setdefault(r["kind"], []).append(r)
    summary = dict(flights=len(rows),
                   success=round(float(np.mean([r["success"] for r in rows])), 4) if rows else None,
                   mean_ndtw=round(float(np.mean([r["ndtw"] for r in rows])), 4) if rows else None,
                   per_kind={k: dict(n=len(v), success=f"{sum(r['success'] for r in v)}/{len(v)}",
                                     ndtw=round(float(np.mean([r["ndtw"] for r in v])), 4))
                             for k, v in sorted(per.items())})
    print(json.dumps(summary, indent=1))
    if args.out:
        args.out.write_text(json.dumps(dict(summary=summary, flights=rows), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
