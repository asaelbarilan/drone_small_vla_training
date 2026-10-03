"""D170: add more real UAV-Flow shards to an existing official-format store.

The 10-shard store (D159) holds shards train-00000..00009 of 54. This appends
further shards without touching what is already there, so the current adapter,
its action statistics and its validation flights stay valid:

  split       the validation SITES of the existing store are kept fixed. A new
              flight whose start lies in one of those 300 m cells goes to "val";
              every other flight goes to "train". With --instruction-sweep, a
              train flight whose unified instruction also occurs in validation is
              marked "dropped", as prepare_uav_flow_official does (it drops ~44%
              of flights); without it, validation measures unseen places with
              seen wording, which is how the benchmark and deployment look.
  maths       process_episode from prepare_uav_flow_official (the official loop,
              verified in D158), unchanged.
  statistics  NOT recomputed: token meanings of the running adapter depend on
              them. The share of new steps outside the range is reported.

Shards are processed one at a time and may be deleted after, so the raw 4.8 GB
parquet files never have to fit on disk together. Flights already in the store
are skipped, so a shard can be re-run safely.
"""

import argparse
import io
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from prepare_uav_flow_chunks import SITE_METRES
from prepare_uav_flow_official import (
    JPEG_QUALITY,
    STORE,
    THUMB,
    process_episode,
    shard_images,
    shard_logs,
)


def existing_store(path):
    """Episode ids, validation sites and validation wordings of the store."""
    ids, val_sites, val_instr = set(), set(), set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ids.add(row["episode"])
            if row.get("split_unseen") == "val" and row.get("source") != "sim":
                val_sites.add(row["site"])
                val_instr.add(row["instruction_unified"].strip().lower())
    return ids, val_sites, val_instr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="store manifest (stats)")
    parser.add_argument("--instruction-sweep", action="store_true")
    parser.add_argument("--delete-shards", action="store_true", help="remove each parquet when done")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    stats = json.loads(args.manifest.read_text(encoding="utf-8"))["action_stats_train"]
    low, high = np.array(stats["q01"]), np.array(stats["q99"])
    episodes_path = STORE / "episodes.jsonl"
    backup = STORE / "episodes.before_more.jsonl"
    if not backup.exists():
        shutil.copy(episodes_path, backup)
    known, val_sites, val_instr = existing_store(episodes_path)
    assert val_sites, "no real validation sites found in the store"
    frames_dir = STORE / "frames"

    def write_frame(job):
        path, blob = job
        if not path.exists():
            picture = Image.open(io.BytesIO(blob)).convert("RGB")
            picture.thumbnail((THUMB, THUMB))
            picture.save(path, quality=JPEG_QUALITY)

    totals = dict(train=0, val=0, dropped=0, skipped=0, steps=0, clipped=0)
    for shard in args.shard:
        logs, counts = shard_logs(shard)
        keep = {}
        for episode_id, log in logs.items():
            track, raw = log.get("preprocessed_logs"), log.get("raw_logs")
            order = sorted(counts[episode_id])
            if episode_id in known or not track or not raw or min(len(order), len(track)) < 2:
                totals["skipped"] += 1
                continue
            keep[episode_id] = order
        with ThreadPoolExecutor(args.workers) as pool:
            jobs = []
            for episode_id, index, blob in shard_images(shard, set(keep)):
                if index not in keep[episode_id][: len(logs[episode_id]["preprocessed_logs"])]:
                    continue
                folder = frames_dir / episode_id
                folder.mkdir(exist_ok=True)
                jobs.append((folder / f"{index:05d}.jpg", blob))
                if len(jobs) >= 512:
                    list(pool.map(write_frame, jobs))
                    jobs = []
            list(pool.map(write_frame, jobs))
        rows = []
        for episode_id, order in sorted(keep.items()):
            log = logs[episode_id]
            raw = log["raw_logs"]
            site = f"{int(raw[0][0] // SITE_METRES)}_{int(raw[0][1] // SITE_METRES)}"
            unified = log["instruction_unified"].strip()
            if site in val_sites:
                split = "val"
            elif args.instruction_sweep and unified.lower() in val_instr:
                split = "dropped"
            else:
                split = "train"
            actions, proprio = process_episode(log)
            n = min(len(order), len(actions))
            moves = actions[: n - 1]
            totals["clipped"] += int(((moves < low) | (moves > high)).any(axis=1).sum())
            totals["steps"] += len(moves)
            totals[split] += 1
            folder = frames_dir / episode_id
            rows.append(
                dict(
                    episode=episode_id,
                    site=site,
                    split_unseen=split,
                    source="real_more",
                    instruction=log["instruction"].strip(),
                    instruction_unified=unified,
                    images=[str(folder / f"{order[i]:05d}.jpg") for i in range(n)],
                    actions=[[round(float(v), 6) for v in a] for a in actions[:n]],
                    proprio=[[round(float(v), 6) for v in p] for p in proprio[:n]],
                    endpoint_start_frame=[round(float(v), 4) for v in proprio[n - 1][:3]],
                )
            )
            known.add(episode_id)
        # Append per shard, so a crash loses at most the shard in progress.
        with open(episodes_path, "a", encoding="utf-8") as out:
            out.writelines(json.dumps(r) + "\n" for r in rows)
        if args.delete_shards:
            Path(shard).unlink()
        print(json.dumps(dict(shard=Path(shard).name, added=len(rows), **totals)), flush=True)

    report = dict(
        **totals,
        steps_outside_action_range=round(totals["clipped"] / max(totals["steps"], 1), 4),
        instruction_sweep=args.instruction_sweep,
        validation_sites=sorted(val_sites),
        note="validation sites fixed from the existing store; action stats not recomputed",
    )
    (STORE / "more_added.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
