"""D173 RL: choose the practice tasks of the next rollout round (RecoverFly-style).

RecoverFly (2608.09467) revisits unresolved failures instead of spending the
rollout budget on tasks the policy already solves. Given the per-flight rows of
earlier rounds (train_grpo_vla.py report.json "flights"), the next round takes
  - `failed_share` of its tasks from the tasks with the lowest mean reward so far
    that were not solved (success rate below 1), and
  - the rest as tasks not flown yet, spread over the motion types.
Writes the chosen task files into a zip for collect_rollouts.ps1.
"""

import argparse
import json
import random
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=Path("D:/drone_vla_pilot/data/rl/tasks"))
    parser.add_argument("--reports", nargs="*", type=Path, default=[])
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--failed-share", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--out", type=Path, required=True, help="zip to write")
    args = parser.parse_args()

    history = {}
    for report in args.reports:
        for row in json.loads(report.read_text(encoding="utf-8"))["flights"]:
            history.setdefault(row["task"], []).append(row)
    unsolved = sorted(
        (sum(r["reward"] for r in rows) / len(rows), name)
        for name, rows in history.items()
        if sum(r["success"] for r in rows) < len(rows)
    )
    retry = [name for _, name in unsolved[: int(args.count * args.failed_share)]]

    fresh_by_kind = {}
    for path in sorted(args.tasks.glob("*.json")):
        if path.name not in history:
            kind = json.loads(path.read_text(encoding="utf-8"))["motion_kind"]
            fresh_by_kind.setdefault(kind, []).append(path.name)
    picker = random.Random(args.seed)
    for names in fresh_by_kind.values():
        picker.shuffle(names)
    fresh, kinds = [], sorted(fresh_by_kind)
    while len(retry) + len(fresh) < args.count and any(fresh_by_kind.values()):
        for kind in kinds:
            if fresh_by_kind[kind] and len(retry) + len(fresh) < args.count:
                fresh.append(fresh_by_kind[kind].pop())

    chosen = retry + fresh
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in chosen:
            archive.write(args.tasks / name, name)
    print(json.dumps(dict(retried=len(retry), fresh=len(fresh), total=len(chosen), out=str(args.out))))


if __name__ == "__main__":
    main()
