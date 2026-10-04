"""D189: render built Turn tasks next to their recorded last photo (local simulator, no model).

Run from the UAV-Flow-Eval folder in its venv. Places each task's object exactly as the
evaluator does (create_obj_if_needed) and renders the recorded last pose.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import gym
import gym_unrealcv  # noqa: F401
from gym_unrealcv.envs.wrappers import augmentation, configUE, time_dilation
from PIL import Image, ImageDraw

sys.path.insert(0, ".")
from batch_run_act_all import create_obj_if_needed, set_cam  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tasks = sorted(p for p in args.tasks.glob("sim_*.json"))
    tasks = [t for t in tasks if json.loads(t.read_text())["motion_kind"] == "Turn"]
    tasks = random.Random(1).sample(tasks, min(args.n, len(tasks)))

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
    u.new_obj("bp_character_C", "BP_Character_21", [0, 0, -1000])
    u.new_obj("BP_BaseCar_C", "BP_Character_22", [1000, 0, -1000])
    time.sleep(1)
    rows = []
    for path in tasks:
        t = json.loads(path.read_text())
        create_obj_if_needed(env, dict(use_obj=t["use_obj"], obj_id=t["obj_id"], obj_pos=t["target_pos"][:3], obj_rot=t["target_pos"][3:]))
        s, last = t["initial_pos"], t["reference_path_preprocessed"][-1]
        # the last pose in world cm (start frame -> world, as in build_rl_tasks.py)
        import math
        a = math.radians(s[4])
        x = s[0] + last[0] * math.cos(a) - last[1] * math.sin(a)
        y = s[1] + last[0] * math.sin(a) + last[1] * math.cos(a)
        u.set_obj_location(player, [x, y, s[2] + last[2]])
        u.set_rotation(player, s[4] + last[4] - 180)
        set_cam(env)
        time.sleep(1.5)
        render = Image.fromarray(u.get_image(0, "lit")[:, :, ::-1]).resize((200, 200))
        photo = Image.open(args.tasks / "photos" / f"{t['source_flight'].removeprefix('sim_')}_last.jpg").convert("RGB").resize((200, 200))
        rows.append((f"{path.stem[-8:]} {t['instruction_unified'][:30]} d={t['object_from']['distance_m']}m", photo, render))
    sheet = Image.new("RGB", (400 * 2, ((len(rows) + 1) // 2) * 214), "white")
    draw = ImageDraw.Draw(sheet)
    for k, (label, photo, render) in enumerate(rows):
        x0, y0 = (k % 2) * 400, (k // 2) * 214
        draw.text((x0 + 2, y0), label, fill="black")
        sheet.paste(photo, (x0, y0 + 14))
        sheet.paste(render, (x0 + 200, y0 + 14))
    sheet.save(args.out)
    env.close()


if __name__ == "__main__":
    main()
