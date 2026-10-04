"""D189: keep only Turn tasks whose placed object is really visible where the photo shows it.

The object is placed on z 0; where the ground is not level (plaza steps, rocks, kerbs) a dog
can end up buried or hidden (4 of 16 in the D189 spot check). For every Turn task this renders
the recorded last pose twice in the local simulator - without and with the object - and finds
the object as the largest changed region. The task is kept if that region exists (at least
--min-area of the image) and its centre column is within --max-col of the box centre Qwen found
in the recorded photo (0.08 of the width = about 9 degrees at 90 deg FOV). Writes
verify.json with every decision. Run from the UAV-Flow-Eval folder in its venv.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import gym
import gym_unrealcv  # noqa: F401
import numpy as np
from gym_unrealcv.envs.wrappers import augmentation, configUE, time_dilation
from PIL import Image
from scipy import ndimage

sys.path.insert(0, ".")
from batch_run_act_all import create_obj_if_needed, set_cam  # noqa: E402

HIDE = dict(use_obj=1, obj_id=1, obj_pos=[0, 0, -10000], obj_rot=[0, 0, 0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--min-area", type=float, default=0.002)
    parser.add_argument("--max-col", type=float, default=0.08)
    args = parser.parse_args()
    env = gym.make("UnrealTrack-DowntownWest-ContinuousColor-v0")
    env = time_dilation.TimeDilationWrapper(env, 10)
    env.unwrapped.agents_category = ["drone"]
    env = configUE.ConfigUEWrapper(env, resolution=(256, 256))
    env = augmentation.RandomPopulationWrapper(env, 2, 2, random_target=False)
    env.seed(0)
    env.reset()
    u, player = env.unwrapped.unrealcv, env.unwrapped.player_list[0]
    u.set_viewport(player)
    u.client.request("vrun t.MaxFPS 10")
    u.set_phy(player, 0)
    u.new_obj("bp_character_C", "BP_Character_21", [0, 0, -10000])
    u.new_obj("BP_BaseCar_C", "BP_Character_22", [1000, 0, -10000])
    time.sleep(1)

    def shot():
        time.sleep(1.2)
        return np.asarray(Image.fromarray(u.get_image(0, "lit")[:, :, ::-1]), dtype=float)

    out_path = args.tasks / "verify.json"
    result = json.loads(out_path.read_text()) if out_path.exists() else {}
    files = sorted(f for f in args.tasks.glob("sim_*.json"))
    for n, path in enumerate(files):
        if path.stem in result:
            continue
        t = json.loads(path.read_text())
        if t["motion_kind"] != "Turn":
            continue
        s, last = t["initial_pos"], t["reference_path_preprocessed"][-1]
        a = math.radians(s[4])
        x = s[0] + last[0] * math.cos(a) - last[1] * math.sin(a)
        y = s[1] + last[0] * math.sin(a) + last[1] * math.cos(a)
        u.set_obj_location(player, [x, y, s[2] + last[2]])
        u.set_rotation(player, s[4] + last[4] - 180)
        set_cam(env)
        create_obj_if_needed(env, HIDE)
        empty = shot()
        create_obj_if_needed(env, dict(use_obj=1, obj_id=t["obj_id"], obj_pos=t["target_pos"][:3], obj_rot=t["target_pos"][3:]))
        placed = shot()
        mask = ndimage.binary_opening(np.abs(placed - empty).sum(2) > 60, iterations=1)
        lab, k = ndimage.label(mask)
        box = t["object_from"]["box"]
        want = (box[0] + box[2]) / 2
        if k == 0:
            result[path.stem] = dict(keep=False, reason="not visible")
        else:
            sizes = ndimage.sum(mask, lab, range(1, k + 1))
            j = int(np.argmax(sizes)) + 1
            area = float(sizes[j - 1]) / mask.size
            col = float(np.median(np.nonzero(lab == j)[1])) / mask.shape[1]
            keep = area >= args.min_area and abs(col - want) <= args.max_col
            result[path.stem] = dict(keep=keep, area=round(area, 4), col=round(col, 3), want=round(want, 3),
                                     reason="ok" if keep else ("too small" if area < args.min_area else "wrong bearing"))
        if n % 20 == 0:
            out_path.write_text(json.dumps(result, indent=0))
            kept = sum(r["keep"] for r in result.values())
            print(f"{n + 1}/{len(files)} checked, kept {kept}/{len(result)}", flush=True)
    out_path.write_text(json.dumps(result, indent=0))
    kept = sum(r["keep"] for r in result.values())
    print(json.dumps(dict(checked=len(result), kept=kept)), flush=True)
    env.close()


if __name__ == "__main__":
    main()
