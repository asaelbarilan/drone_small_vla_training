"""D155: action-chunk UAV-Flow dataset with a genuinely unseen validation split.

Two changes over D154, both forced by the literature audit in section 12 of
docs/VLA_DRONE_TRAINING.md.

Chunks. Almost no drone VLA predicts a single step: UAV-Track VLA emits 25-step
chunks, FLIGHTVLA and Exp2VLA emit action chunks, LongFly and SpatialFly emit
waypoint sequences, and UAV-Flow's own paper describes pose trajectories with
look-ahead. Each example here carries the next K steps, so a training script can
take any prefix of them.

An unseen split. Seen-to-unseen collapse is the field's dominant failure - LongFly
36.39 to 11.27, SpatialFly 38.54 to 13.57. The D154 split shared 47 of 59
validation instructions with training, so it measured the seen case only. Two
labels are emitted per episode:

  split_seen   - random by episode, the D154 behaviour, kept for comparison
  split_unseen - validation holds out whole SITES, and every episode whose
                 instruction appears in that validation set is removed from
                 training entirely

`split_unseen` is the one the user asked for: no shared episode, no shared
instruction wording, no shared location. Episodes excluded by the instruction
sweep are marked `dropped` and belong to neither side.

Sites come from the absolute start coordinates in `raw_logs`, binned to a grid.
The shard spans tens of kilometres, so these are genuinely different places, not
different corners of one field.
"""

import argparse
import hashlib
import io
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/uav_flow_chunks_20260921"
STORE = Path("D:/drone_vla_pilot/data/uav_flow_chunks_20260921")
CHUNK = 8
SITE_METRES = 300
VAL_TARGET = 0.20
THUMB = 256
JPEG_QUALITY = 92
SEED = 1155


def seen_split(episode_id):
    digest = hashlib.sha256(episode_id.encode("utf-8")).digest()
    return "val" if int.from_bytes(digest[:4], "big") / 2**32 < VAL_TARGET else "train"


def collect(shard):
    handle = pq.ParquetFile(shard)
    frames, logs = defaultdict(dict), {}
    for group in range(handle.metadata.num_row_groups):
        table = handle.read_row_group(group, columns=["id", "frame_idx", "image", "log"])
        for episode_id, frame_idx, image, blob in zip(
            table.column("id").to_pylist(),
            table.column("frame_idx").to_pylist(),
            table.column("image").to_pylist(),
            table.column("log").to_pylist(),
            strict=True,
        ):
            frames[episode_id][frame_idx] = image["bytes"]
            if episode_id not in logs:
                logs[episode_id] = json.loads(blob)
    return frames, logs


def choose_val_sites(by_site, total, target=VAL_TARGET):
    """Take whole sites, in a fixed shuffled order, until the target share is met."""
    rng = random.Random(SEED)
    order = sorted(by_site)
    rng.shuffle(order)
    chosen, held = set(), 0
    for site in order:
        if held >= target * total:
            break
        chosen.add(site)
        held += len(by_site[site])
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shard",
        default="D:/drone_vla_pilot/data/uav_flow_20260921/train-00000-of-00054.parquet",
    )
    parser.add_argument("--chunk", type=int, default=CHUNK)
    parser.add_argument("--val-target", type=float, default=VAL_TARGET)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    picture_dir = STORE / "frames"
    picture_dir.mkdir(parents=True, exist_ok=True)

    frames, logs = collect(args.shard)

    episodes, by_site = {}, defaultdict(list)
    for episode_id, by_index in frames.items():
        log = logs[episode_id]
        track = log.get("preprocessed_logs")
        raw = log.get("raw_logs")
        if not track or not raw or len(track) < 2:
            continue
        order = sorted(by_index)
        usable = min(len(order), len(track))
        if usable < 2:
            continue
        site = (int(raw[0][0] // SITE_METRES), int(raw[0][1] // SITE_METRES))
        episodes[episode_id] = dict(
            order=order,
            usable=usable,
            pose=np.asarray(track[:usable], dtype=float),
            instruction=log["instruction_unified"].strip(),
            site=f"{site[0]}_{site[1]}",
        )
        by_site[f"{site[0]}_{site[1]}"].append(episode_id)

    val_sites = choose_val_sites(by_site, len(episodes), args.val_target)
    val_ids = {e for s in val_sites for e in by_site[s]}
    val_instructions = {episodes[e]["instruction"].lower() for e in val_ids}

    # Anything worded like a validation episode cannot stay in training.
    unseen = {}
    for episode_id, meta in episodes.items():
        if episode_id in val_ids:
            unseen[episode_id] = "val"
        elif meta["instruction"].lower() in val_instructions:
            unseen[episode_id] = "dropped"
        else:
            unseen[episode_id] = "train"

    rows, episode_rows = [], []
    for episode_id, meta in episodes.items():
        pose, order, usable = meta["pose"], meta["order"], meta["usable"]
        deltas = np.diff(pose, axis=0)
        folder = picture_dir / episode_id
        folder.mkdir(exist_ok=True)
        for k in range(usable - 1):
            path = folder / f"{order[k]:05d}.jpg"
            if not path.exists():
                picture = Image.open(io.BytesIO(frames[episode_id][order[k]])).convert("RGB")
                picture.thumbnail((THUMB, THUMB))
                picture.save(path, quality=JPEG_QUALITY)
            chunk = deltas[k : k + args.chunk]
            rows.append(
                dict(
                    id=f"{episode_id}:{order[k]:05d}",
                    episode=episode_id,
                    site=meta["site"],
                    split_seen=seen_split(episode_id),
                    split_unseen=unseen[episode_id],
                    step=k,
                    instruction=meta["instruction"],
                    image=str(path),
                    chunk=[[round(float(v), 3) for v in row] for row in chunk],
                    chunk_len=len(chunk),
                    steps_remaining=usable - 1 - k,
                )
            )
        episode_rows.append(
            dict(
                episode=episode_id,
                site=meta["site"],
                split_seen=seen_split(episode_id),
                split_unseen=unseen[episode_id],
                instruction=meta["instruction"],
                steps=usable - 1,
                endpoint_m=[round(float(v), 4) for v in pose[-1][:3]],
            )
        )

    steps_path = STORE / "steps.jsonl"
    steps_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    episodes_path = STORE / "episodes.jsonl"
    episodes_path.write_text("".join(json.dumps(e) + "\n" for e in episode_rows), encoding="utf-8")

    # Verification. These must hold or the split is not what it claims to be.
    def ids(label):
        return {e["episode"] for e in episode_rows if e["split_unseen"] == label}

    train_ids, held_ids = ids("train"), ids("val")
    train_instr = {episodes[e]["instruction"].lower() for e in train_ids}
    held_instr = {episodes[e]["instruction"].lower() for e in held_ids}
    train_sites = {episodes[e]["site"] for e in train_ids}
    held_sites = {episodes[e]["site"] for e in held_ids}
    checks = dict(
        shared_episodes=len(train_ids & held_ids),
        shared_instructions=len(train_instr & held_instr),
        shared_sites=len(train_sites & held_sites),
    )
    assert checks == dict(shared_episodes=0, shared_instructions=0, shared_sites=0), checks

    seen_train = {e["episode"] for e in episode_rows if e["split_seen"] == "train"}
    seen_val = {e["episode"] for e in episode_rows if e["split_seen"] == "val"}
    seen_overlap = len(
        {episodes[e]["instruction"].lower() for e in seen_val}
        & {episodes[e]["instruction"].lower() for e in seen_train}
    )

    manifest = dict(
        source_shard=str(args.shard),
        chunk_steps=args.chunk,
        site_grid_metres=SITE_METRES,
        val_target=args.val_target,
        episodes=len(episode_rows),
        steps=len(rows),
        sites=len(by_site),
        steps_sha256=hashlib.sha256(steps_path.read_bytes()).hexdigest(),
        episodes_sha256=hashlib.sha256(episodes_path.read_bytes()).hexdigest(),
        split_unseen=dict(
            train_episodes=len(train_ids),
            val_episodes=len(held_ids),
            dropped_episodes=sum(1 for e in episode_rows if e["split_unseen"] == "dropped"),
            val_sites=len(held_sites),
            train_steps=sum(1 for r in rows if r["split_unseen"] == "train"),
            val_steps=sum(1 for r in rows if r["split_unseen"] == "val"),
            leakage_checks=checks,
        ),
        split_seen=dict(
            train_episodes=len(seen_train),
            val_episodes=len(seen_val),
            val_instructions_also_in_train=seen_overlap,
            note="kept only to quantify our own seen-to-unseen gap; never report alone",
        ),
        task=f"current frame + instruction -> next {args.chunk} 6-DoF steps, episode start frame",
    )
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
