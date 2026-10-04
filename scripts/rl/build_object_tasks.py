"""D189: Turn / Surround practice tasks with the person or dog placed (local, no simulator).

UAV-Flow-Sim logs do not record where the spawned person or dog stood, so build_rl_tasks.py
drops every flight that names one (D172). This builder reconstructs the object from the
recorded flight (checked against the recorded photos in D189):
- Surround ("Circle around the person ..."): least-squares circle through the path; the
  object stands at its centre (fitted radii equal the instructions' radii).
- Turn ("Turn to the direction of the dog"): Qwen3-VL-4B boxes the object in the flight's
  LAST photo; the box's centre column gives the bearing and its bottom row the distance on
  the ground plane (90 deg FOV, camera height = drone z, ground at z 0). The bearing is what
  the task scores (final yaw); the distance is about +-20 % (D189 calibration).
The object is BP_Character_21 (use_obj 1): appearance 1-19 = people, 20-35 = robot dogs
(0 is ignored by set_appearance when switching, D189). The person faces the drone.

--validation builds the same from the 504 held-out validation flights (never trained on);
the default uses training flights only. Task format = build_rl_tasks.py plus object fields.
"""

import argparse
import io
import json
import math
import random
import re
from pathlib import Path

import numpy as np

REPO = "datasets/wangxiangyu0814/UAV-Flow-Sim"
SURROUND = re.compile(r"circle|around|orbit|surround", re.I)
TURN_OBJ = re.compile(r"^turn .*\b(dog|person)\b", re.I)


def circle_centre(xy):
    xy = np.asarray(xy, dtype=float)
    a = np.c_[xy[:, 0], xy[:, 1], np.ones(len(xy))]
    (p, q, c), *_ = np.linalg.lstsq(a, -(xy[:, 0] ** 2 + xy[:, 1] ** 2), rcond=None)
    cx, cy = -p / 2, -q / 2
    return cx, cy, math.sqrt(max(cx * cx + cy * cy - c, 0.0))


def to_world(start, x, y):
    """Start-frame cm -> world cm (same rotation as build_rl_tasks.py's self-check)."""
    t = math.radians(start[4])
    return start[0] + x * math.cos(t) - y * math.sin(t), start[1] + x * math.sin(t) + y * math.cos(t)


def box_from_text(text):
    m = re.search(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", text)
    return [int(v) / 1000 for v in m.groups()] if m else None


def object_from_box(box, cam_x, cam_y, cam_z, yaw_deg):
    """Ground-plane position of the box's bottom centre seen from a level 90-degree camera."""
    x1, _, x2, y2 = box
    forward = cam_z * 0.5 / max(y2 - 0.5, 0.02)
    right = forward * ((x1 + x2) / 2 - 0.5) / 0.5
    t = math.radians(yaw_deg)
    return (cam_x + forward * math.cos(t) - right * math.sin(t),
            cam_y + forward * math.sin(t) + right * math.cos(t), math.hypot(forward, right))


def last_photos(ids, out_dir):
    """Last frame of each flight, reading only the parquet row groups that hold it."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    from PIL import Image
    fs = HfFileSystem()
    files = [pq.ParquetFile(fs.open(f"{REPO}/train-{s:05d}-of-00021.parquet", "rb")) for s in range(21)]
    groups = [(s, g, f.metadata.row_group(g).column(0).statistics)
              for s, f in enumerate(files) for g in range(f.metadata.num_row_groups)]
    need = {}
    for fid in ids:
        for s, g, st in groups:
            if st is not None and st.min <= fid <= st.max:
                need.setdefault((s, g), set()).add(fid)
    best, photos = {}, {}
    for n, ((s, g), fids) in enumerate(sorted(need.items())):
        t = files[s].read_row_group(g, columns=["id", "frame_idx", "image"])
        for fid, idx, img in zip(t.column("id").to_pylist(), t.column("frame_idx").to_pylist(), t.column("image").to_pylist()):
            if fid in fids and idx >= best.get(fid, -1):
                best[fid] = idx
                p = out_dir / f"{fid}_last.jpg"
                Image.open(io.BytesIO(img["bytes"])).convert("RGB").save(p)
                photos[fid] = (idx, p)
        print(f"row group {n + 1}/{len(need)}: {len(photos)} photos", flush=True)
    return photos


def qwen_boxes(items, snapshot):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(snapshot, device_map="cuda", quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"))
    proc = AutoProcessor.from_pretrained(snapshot)
    out = {}
    for n, (fid, path, what) in enumerate(items):
        msgs = [{"role": "user", "content": [{"type": "image", "image": Image.open(path).convert("RGB")}, {"type": "text", "text":
                 f"Locate the {what} in the image. Output its bounding box as JSON: {{\"bbox_2d\": [x1, y1, x2, y2]}}."}]}]
        inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to("cuda")
        ids = model.generate(**inp, max_new_tokens=60, do_sample=False)
        out[fid] = proc.batch_decode(ids[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        if n % 25 == 0:
            print(f"qwen {n + 1}/{len(items)}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=Path("D:/drone_vla_pilot/data/rl/sim_paths.json"))
    parser.add_argument("--index", type=Path, default=Path("D:/drone_vla_pilot/data/uav_flow_sim_index.json"))
    parser.add_argument("--val", type=Path, default=Path("D:/drone_vla_pilot/data/rl/sim_val_flights.json"))
    parser.add_argument("--validation", action="store_true", help="build from the held-out validation flights")
    parser.add_argument("--qwen", required=True, help="local Qwen3-VL-4B-Instruct snapshot folder")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="at most this many flights per kind (0 = all)")
    args = parser.parse_args()
    paths = json.loads(args.paths.read_text(encoding="utf-8"))
    index = json.loads(args.index.read_text(encoding="utf-8"))
    val = set(json.loads(args.val.read_text(encoding="utf-8")))
    (args.out / "photos").mkdir(parents=True, exist_ok=True)

    surround, turn = [], []
    for episode, flight in sorted(paths.items()):
        if (episode in val) != args.validation or flight["split"] != "train" or episode.removeprefix("sim_") not in index:
            continue
        text = flight["instruction_unified"]
        if SURROUND.search(text):
            surround.append(episode)
        elif TURN_OBJ.search(text):
            turn.append(episode)
    if args.limit:
        surround, turn = surround[: args.limit], turn[: args.limit]
    print("surround", len(surround), "turn", len(turn), flush=True)

    picker = random.Random(189)
    summary = dict(written={"Surround": 0, "Turn": 0}, dropped={})

    def write(episode, kind, obj_xy, is_dog, extra):
        flight, start = paths[episode], index[episode.removeprefix("sim_")]["start"]
        reference = [[x * 100, y * 100, z * 100, 0.0, yaw, 0.0] for x, y, z, yaw in flight["proprio"]]
        ex, ey = to_world(start, reference[-1][0], reference[-1][1])
        face = math.degrees(math.atan2(ey - obj_xy[1], ex - obj_xy[0]))  # person faces the drone's end point
        task = dict(instruction=flight["instruction"], instruction_unified=flight["instruction_unified"],
                    initial_pos=start, reference_path_preprocessed=reference, use_obj=1,
                    obj_id=picker.randint(20, 35) if is_dog else picker.randint(1, 19),
                    target_pos=[round(obj_xy[0], 1), round(obj_xy[1], 1), 0.0, 0.0, round(face, 1), 0.0],
                    source_flight=episode, motion_kind=kind, object_from=extra)
        (args.out / f"{episode}.json").write_text(json.dumps(task), encoding="utf-8")
        summary["written"][kind] += 1

    for episode in surround:
        start = index[episode.removeprefix("sim_")]["start"]
        cx, cy, r = circle_centre([[x * 100, y * 100] for x, y, _, _ in paths[episode]["proprio"]])
        m = re.search(r"radius of ([\d.]+)", paths[episode]["instruction_unified"])
        stated = float(m.group(1)) if m else None
        if stated is not None and abs(r / 100 - stated) > 0.25 * stated:
            summary["dropped"]["surround_radius_mismatch"] = summary["dropped"].get("surround_radius_mismatch", 0) + 1
            continue
        write(episode, "Surround", to_world(start, cx, cy), "dog" in paths[episode]["instruction_unified"].lower(),
              dict(method="circle_fit", radius_m=round(r / 100, 2), stated_radius_m=stated))

    photos = last_photos([e.removeprefix("sim_") for e in turn], args.out / "photos")
    items = [(e, photos[e.removeprefix("sim_")][1], "dog (a four-legged robot dog)" if "dog" in paths[e]["instruction_unified"].lower() else "person")
             for e in turn if e.removeprefix("sim_") in photos]
    texts = qwen_boxes(items, args.qwen)
    for episode, _, _ in items:
        box = box_from_text(texts[episode])
        if box is None:
            summary["dropped"]["no_box"] = summary["dropped"].get("no_box", 0) + 1
            continue
        start = index[episode.removeprefix("sim_")]["start"]
        idx = photos[episode.removeprefix("sim_")][0]
        x, y, z, yaw = paths[episode]["proprio"][min(idx, len(paths[episode]["proprio"]) - 1)]
        cx, cy = to_world(start, x * 100, y * 100)
        ox, oy, dist = object_from_box(box, cx, cy, start[2] + z * 100, start[4] + yaw)
        if not 0.8 <= dist / 100 <= 25:
            summary["dropped"]["distance_out_of_range"] = summary["dropped"].get("distance_out_of_range", 0) + 1
            continue
        write(episode, "Turn", (ox, oy), "dog" in paths[episode]["instruction_unified"].lower(),
              dict(method="qwen_box_last_photo", box=box, distance_m=round(dist / 100, 2), photo_frame=idx))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
