"""D189: can Turn / Surround training flights become practice tasks? (local, no model)

UAV-Flow-Sim logs hold only the drone poses and the instruction; where the spawned person
or dog stood is not saved. This script places the object where the recorded flight implies
it was and renders the recorded poses, so the renders can be compared with the recorded
photos:
- Surround ("fly around the person"): the object is the centre of a circle fitted to the
  path (least squares, x-y).
- Turn ("turn to the direction of the dog"): the object lies on the final heading; the
  distance is not recorded, so several candidates are rendered.

Input: --flights JSON {flight_id: {"log": <log.json>, "photos": {frame_index: path}}}
(written by fetch_sim_flights.py). Output per flight: renders and a side-by-side sheet
(recorded photo | render) for the first and last frame. Runs in the UAV-Flow-Eval venv
from the UAV-Flow-Eval folder; the simulator is started like batch_run_act_all.py does.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import gym
import gym_unrealcv  # noqa: F401
import numpy as np
from gym_unrealcv.envs.wrappers import augmentation, configUE, time_dilation
from PIL import Image, ImageDraw

from batch_run_act_all import create_obj_if_needed, set_cam

ENV_ID = "UnrealTrack-DowntownWest-ContinuousColor-v0"


def circle_centre(xy):
    """Least-squares circle (Kasa): x^2 + y^2 + a x + b y + c = 0."""
    xy = np.asarray(xy, dtype=float)
    a = np.c_[xy[:, 0], xy[:, 1], np.ones(len(xy))]
    rhs = -(xy[:, 0] ** 2 + xy[:, 1] ** 2)
    (p, q, c), *_ = np.linalg.lstsq(a, rhs, rcond=None)
    cx, cy = -p / 2, -q / 2
    return cx, cy, math.sqrt(max(cx * cx + cy * cy - c, 0.0))


def placements(log, distances):
    """Candidate object positions [x, y, z] in world cm with a label each."""
    poses = log["raw_logs"]
    text = (log["instruction_unified"] + " " + log["instruction"]).lower()
    if any(w in text for w in ("around", "circle", "orbit", "surround")):
        cx, cy, r = circle_centre([p[:2] for p in poses])
        return [(f"centre_r{r / 100:.1f}m", [cx, cy, 0.0])]
    x, y, _, _, yaw, _ = poses[-1]
    out = []
    for d in distances:
        out.append((f"heading_{d:g}m", [x + d * 100 * math.cos(math.radians(yaw)),
                                        y + d * 100 * math.sin(math.radians(yaw)), 0.0]))
    return out


def object_kind(log):
    text = (log["instruction_unified"] + " " + log["instruction"]).lower()
    return 2 if "dog" in text else 1  # use_obj: 1 = person (BP_Character_21), 2 = dog (BP_Character_22)


def render(env, pose):
    x, y, z, _, yaw, _ = pose
    player = env.unwrapped.player_list[0]
    env.unwrapped.unrealcv.set_obj_location(player, [x, y, z])
    env.unwrapped.unrealcv.set_rotation(player, yaw - 180)
    set_cam(env)
    time.sleep(1.0)
    image = env.unwrapped.unrealcv.get_image(0, "lit")  # BGR, as the evaluator gets it
    return Image.fromarray(image[:, :, ::-1]).resize((224, 224))


def sheet(pairs, path):
    w = 224
    canvas = Image.new("RGB", (2 * w, len(pairs) * (w + 14)), "white")
    for row, (label, recorded, rendered) in enumerate(pairs):
        top = row * (w + 14)
        ImageDraw.Draw(canvas).text((2, top), label, fill="black")
        canvas.paste(recorded.resize((w, w)), (0, top + 14))
        canvas.paste(rendered, (w, top + 14))
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--distances", default="3,6,10")
    parser.add_argument("--person-id", type=int, default=0)
    args = parser.parse_args()
    distances = [float(d) for d in args.distances.split(",")]
    flights = json.loads(args.flights.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)

    env = gym.make(ENV_ID)
    env = time_dilation.TimeDilationWrapper(env, 10)
    env.unwrapped.agents_category = ["drone"]
    env = configUE.ConfigUEWrapper(env, resolution=(256, 256))
    env = augmentation.RandomPopulationWrapper(env, 2, 2, random_target=False)
    env.seed(0)
    env.reset()
    env.unwrapped.unrealcv.set_viewport(env.unwrapped.player_list[0])
    env.unwrapped.unrealcv.client.request(f"vrun t.MaxFPS {os.environ.get('UE_MAX_FPS', '10')}")
    env.unwrapped.unrealcv.set_phy(env.unwrapped.player_list[0], 0)
    time.sleep(1.0)
    env.unwrapped.unrealcv.new_obj("bp_character_C", "BP_Character_21", [0, 0, 0])
    env.unwrapped.unrealcv.set_appearance("BP_Character_21", 0)
    env.unwrapped.unrealcv.new_obj("BP_BaseCar_C", "BP_Character_22", [1000, 0, 0])
    env.unwrapped.unrealcv.set_appearance("BP_Character_22", 2)
    env.unwrapped.unrealcv.set_phy("BP_Character_22", 0)
    time.sleep(1.0)

    report = {}
    for fid, flight in flights.items():
        log = flight["log"]
        poses = log["raw_logs"]
        kind = object_kind(log)
        frames = sorted(int(k) for k in flight["photos"])
        pairs, tried = [], []
        for label, pos in placements(log, distances):
            create_obj_if_needed(env, dict(use_obj=kind, obj_id=args.person_id, obj_pos=pos, obj_rot=[0, 0, 0]))
            for i in frames:
                recorded = Image.open(flight["photos"][str(i)]).convert("RGB")
                rendered = render(env, poses[i])
                rendered.save(args.out / f"{fid}_{label}_f{i:03d}.png")
                pairs.append((f"{label} frame {i}: recorded | render", recorded, rendered))
            tried.append(dict(label=label, pos=[round(v, 1) for v in pos]))
        sheet(pairs, args.out / f"{fid}_sheet.png")
        report[fid] = dict(instruction=log["instruction_unified"], use_obj=kind, placements=tried)
        print(fid, report[fid], flush=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    env.close()


if __name__ == "__main__":
    main()
