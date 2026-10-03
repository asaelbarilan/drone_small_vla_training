"""D158: UAV-Flow samples in the official OpenVLA-UAV format.

Ports buaa-colalab/UAV-Flow OpenVLA-UAV/prismatic/vla/datasets/uav_dataset.py
exactly, because the D155 format drifted from it in four ways (section 13 of
docs/VLA_DRONE_TRAINING.md):

  action      4-D (dx, dy, dz, dyaw) in the drone's CURRENT local frame, from
              raw_logs columns [0,1,2,4] with yaw in degrees -> radians; the last
              frame's action is zero. D155 used 6-DoF deltas in the start frame.
  state       preprocessed_logs columns [0,1,2,4] (start frame, yaw in degrees),
              written into the prompt. D155 had no state.
  sampling    every frame including the last; first and last frames repeated 5
              extra times. D155 dropped the last frame and oversampled nothing.
  instruction the official code reads `instruction` (free wording); D155 used
              `instruction_unified` (fixed template). Both are kept here.

`_process_episode` and `_transform_to_local_frame` below are copied from the
official file with only the I/O changed; `verify()` re-runs a verbatim copy of
the official loop on every episode and asserts identical arrays.

The site-disjoint unseen split of D155 is reused unchanged (same sites, same
instruction sweep), so results stay comparable to the D155 baselines. Per-frame
actions are stored per episode; a trainer builds K-step chunks from them, which
is what the horizon experiment (K = 1, 8, 16) needs.
"""

import argparse
import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from prepare_uav_flow_chunks import (
    JPEG_QUALITY,
    SITE_METRES,
    THUMB,
    choose_val_sites,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/uav_flow_official_20260922"
STORE = Path(
    os.environ.get("UAV_FLOW_OFFICIAL_STORE", "D:/drone_vla_pilot/data/uav_flow_official_20260922")
)
REPEAT = 5  # official last_frame_repeat_count


def transform_to_local_frame(current_pose, next_pose):
    """Official _transform_to_local_frame, verbatim maths."""
    current_pos = current_pose[:3]
    current_yaw = current_pose[3]
    next_pos = next_pose[:3]
    next_yaw = next_pose[3]
    cos_yaw = np.cos(current_yaw)
    sin_yaw = np.sin(current_yaw)
    rotation = np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]])
    relative_pos = next_pos - current_pos
    local_pos = np.linalg.inv(rotation) @ relative_pos
    relative_yaw = next_yaw - current_yaw
    relative_yaw = (relative_yaw + np.pi) % (2 * np.pi) - np.pi
    return np.array([local_pos[0], local_pos[1], local_pos[2], relative_yaw])


def process_episode(log):
    """Official _process_episode: (actions, proprio), both (T, 4)."""
    trajectory_raw = np.array(log["raw_logs"])[:, [0, 1, 2, 4]].astype(float)
    trajectory = np.array(log["preprocessed_logs"])[:, [0, 1, 2, 4]].astype(float)
    proprio = trajectory
    trajectory_raw[:, 3] = np.deg2rad(trajectory_raw[:, 3])
    actions = np.zeros_like(trajectory)
    for i in range(len(trajectory) - 1):
        actions[i] = transform_to_local_frame(trajectory_raw[i], trajectory_raw[i + 1])
    actions[-1] = np.zeros(4)
    return actions, proprio


def official_samples(log, n_images):
    """Verbatim re-statement of the official __init__ sample loop, for verify()."""
    trajectory_raw = np.array(log["raw_logs"])
    trajectory = np.array(log["preprocessed_logs"])
    trajectory_raw = trajectory_raw[:, [0, 1, 2, 4]]
    trajectory = trajectory[:, [0, 1, 2, 4]]
    proprio = trajectory
    trajectory_raw[:, 3] = np.deg2rad(trajectory_raw[:, 3])
    actions = np.zeros_like(trajectory)
    for i in range(len(trajectory) - 1):
        cur, nxt = trajectory_raw[i], trajectory_raw[i + 1]
        c, s = np.cos(cur[3]), np.sin(cur[3])
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        local = np.linalg.inv(rot) @ (nxt[:3] - cur[:3])
        dyaw = (nxt[3] - cur[3] + np.pi) % (2 * np.pi) - np.pi
        actions[i] = np.array([local[0], local[1], local[2], dyaw])
    actions[-1] = np.zeros(4)
    samples = []
    for frame_idx in range(n_images):
        samples.append((frame_idx, actions[frame_idx], proprio[frame_idx]))
        if frame_idx == n_images - 1 or frame_idx == 0:
            for _ in range(REPEAT):
                samples.append((frame_idx, actions[frame_idx], proprio[frame_idx]))
    return samples


def shard_logs(shard):
    """Logs only: one pass over the shard without decoding any image."""
    handle = pq.ParquetFile(shard)
    logs, frame_counts = {}, {}
    for group in range(handle.metadata.num_row_groups):
        table = handle.read_row_group(group, columns=["id", "frame_idx", "log"])
        for episode, index, blob in zip(
            table.column("id").to_pylist(),
            table.column("frame_idx").to_pylist(),
            table.column("log").to_pylist(),
            strict=True,
        ):
            if episode not in logs:
                logs[episode] = json.loads(blob)
            frame_counts.setdefault(episode, []).append(index)
    return logs, frame_counts


def shard_images(shard, wanted):
    """Yield (episode, frame index, jpeg bytes) row group by row group, so a
    46 GB set of shards never has to fit in memory at once."""
    handle = pq.ParquetFile(shard)
    for group in range(handle.metadata.num_row_groups):
        table = handle.read_row_group(group, columns=["id", "frame_idx", "image"])
        for episode, index, image in zip(
            table.column("id").to_pylist(),
            table.column("frame_idx").to_pylist(),
            table.column("image").to_pylist(),
            strict=True,
        ):
            if episode in wanted:
                yield episode, index, image["bytes"]


def proprio_text(proprio):
    """Official: ','.join(str(round(x, 1)) for x in proprio)."""
    return ",".join(str(round(float(x), 1)) for x in proprio)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shard",
        nargs="+",
        default=["D:/drone_vla_pilot/data/uav_flow_20260921/train-00000-of-00054.parquet"],
        help="one or more parquet shards; sites and the split span all of them",
    )
    parser.add_argument("--val-target", type=float, default=0.12, help="D155 used 0.12")
    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=1.0,
        help="action range: 1.0 = the official q01/q99; 0.1 keeps fast turns (D158)",
    )
    parser.add_argument("--workers", type=int, default=8, help="threads for frame writing")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frames_dir = STORE / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: logs only, so sites and the split span every shard cheaply.
    episodes, by_site, shard_of = {}, {}, {}
    for shard in args.shard:
        logs, counts = shard_logs(shard)
        kept = 0
        for episode_id, log in logs.items():
            track, raw = log.get("preprocessed_logs"), log.get("raw_logs")
            if not track or not raw or len(track) < 2:
                continue
            order = sorted(counts[episode_id])
            if min(len(order), len(track)) < 2:
                continue
            site = f"{int(raw[0][0] // SITE_METRES)}_{int(raw[0][1] // SITE_METRES)}"
            episodes[episode_id] = dict(order=order, site=site, log=log)
            by_site.setdefault(site, []).append(episode_id)
            shard_of.setdefault(shard, []).append(episode_id)
            kept += 1
        print(f"read {shard}: {kept} flights", flush=True)

    val_sites = choose_val_sites(by_site, len(episodes), args.val_target)
    val_ids = {e for s in val_sites for e in by_site[s]}
    val_instr = {episodes[e]["log"]["instruction_unified"].strip().lower() for e in val_ids}

    # Pass 2: images, one shard at a time. JPEG decode and resize release the
    # GIL, so a thread pool uses every core instead of one.
    def write_frame(job):
        path, blob = job
        if path.exists():
            return
        picture = Image.open(io.BytesIO(blob)).convert("RGB")
        picture.thumbnail((THUMB, THUMB))
        picture.save(path, quality=JPEG_QUALITY)

    for shard in args.shard:
        wanted = set(shard_of.get(shard, []))
        written = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs = []
            for episode_id, index, blob in shard_images(shard, wanted):
                meta = episodes[episode_id]
                if index not in meta["order"][: len(meta["log"]["preprocessed_logs"])]:
                    continue
                folder = frames_dir / episode_id
                folder.mkdir(exist_ok=True)
                jobs.append((folder / f"{index:05d}.jpg", blob))
                written += 1
                if len(jobs) >= 512:
                    list(pool.map(write_frame, jobs))
                    jobs = []
            list(pool.map(write_frame, jobs))
        print(f"frames {shard}: {written}", flush=True)

    rows, mismatches, length_mismatch = [], 0, 0
    for episode_id, meta in sorted(episodes.items()):
        log, order = meta["log"], meta["order"]
        unified = log["instruction_unified"].strip()
        split = (
            "val"
            if episode_id in val_ids
            else "dropped"
            if unified.lower() in val_instr
            else "train"
        )
        actions, proprio = process_episode(log)
        # Official pairs image i with log row i; the counts must agree.
        n = min(len(order), len(actions))
        length_mismatch += int(len(order) != len(actions))

        reference = official_samples(log, n)
        ours = []
        for i in range(n):
            ours.append((i, actions[i], proprio[i]))
            if i in (0, n - 1):
                ours.extend([(i, actions[i], proprio[i])] * REPEAT)
        same = len(reference) == len(ours) and all(
            a[0] == b[0] and np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])
            for a, b in zip(reference, ours, strict=True)
        )
        mismatches += int(not same)

        folder = frames_dir / episode_id
        rows.append(
            dict(
                episode=episode_id,
                site=meta["site"],
                split_unseen=split,
                instruction=log["instruction"].strip(),
                instruction_unified=unified,
                images=[str(folder / f"{order[i]:05d}.jpg") for i in range(n)],
                actions=[[round(float(v), 6) for v in a] for a in actions[:n]],
                proprio=[[round(float(v), 6) for v in p] for p in proprio[:n]],
                endpoint_start_frame=[round(float(v), 4) for v in proprio[n - 1][:3]],
            )
        )

    assert mismatches == 0, f"{mismatches} episodes differ from the official loop"
    path = STORE / "episodes.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    train = [r for r in rows if r["split_unseen"] == "train"]
    stats_actions = np.concatenate([np.array(r["actions"]) for r in train])
    low, high = args.clip_percentile, 100 - args.clip_percentile
    per_split = {
        s: dict(
            episodes=sum(r["split_unseen"] == s for r in rows),
            frames=sum(len(r["actions"]) for r in rows if r["split_unseen"] == s),
            samples_with_oversampling=sum(
                len(r["actions"]) + 2 * REPEAT for r in rows if r["split_unseen"] == s
            ),
        )
        for s in ("train", "val", "dropped")
    }
    manifest = dict(
        source_shard=args.shard,
        clip_percentile=args.clip_percentile,
        format="official OpenVLA-UAV uav_dataset.py",
        verified_against_official_loop=dict(episodes=len(rows), mismatches=mismatches),
        image_log_length_mismatch_episodes=length_mismatch,
        splits=per_split,
        instructions_differ=sum(r["instruction"] != r["instruction_unified"] for r in rows),
        action_stats_train=dict(
            q01=np.percentile(stats_actions, low, axis=0).round(6).tolist(),
            q99=np.percentile(stats_actions, high, axis=0).round(6).tolist(),
            mean=stats_actions.mean(axis=0).round(6).tolist(),
            std=stats_actions.std(axis=0).round(6).tolist(),
            note=(
                f"percentiles {low}/{high} of all train actions incl. zero last actions, "
                "no oversampling; the official recipe uses 1/99"
            ),
        ),
        episodes_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        example_prompt=(
            f"Current State: {proprio_text(train[0]['proprio'][3])}, "
            f"What action should the uav take to {train[0]['instruction']}?"
        ),
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
