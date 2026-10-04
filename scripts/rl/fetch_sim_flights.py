"""D189: fetch a few UAV-Flow-Sim TRAINING flights (log + chosen photos) from Hugging Face.

Reads only the parquet row groups that hold the wanted flights (remote, no full shard
download): the shards are sorted by flight id and every row group's id min/max is in the
parquet metadata, so no column has to be scanned.
Picks --per-kind flights per pattern among training flights (the 504 validation flights
are skipped) and saves photos of the first, middle and last frame.
"""

import argparse
import io
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from PIL import Image

REPO = "datasets/wangxiangyu0814/UAV-Flow-Sim"
PATTERNS = {
    "turn_dog": r"^turn .*\bdog\b",
    "turn_person": r"^turn .*\bperson\b",
    "surround": r"around|circle|orbit|surround",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=Path("D:/drone_vla_pilot/data/rl/sim_paths.json"))
    parser.add_argument("--val", type=Path, default=Path("D:/drone_vla_pilot/data/rl/sim_val_flights.json"))
    parser.add_argument("--per-kind", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = json.loads(args.paths.read_text(encoding="utf-8"))
    val = set(json.loads(args.val.read_text(encoding="utf-8")))
    fs = HfFileSystem()
    files = [pq.ParquetFile(fs.open(f"{REPO}/train-{s:05d}-of-00021.parquet", "rb")) for s in range(21)]
    groups = [(s, g, md.row_group(g).column(0).statistics)
              for s, pf in enumerate(files) for md in [pf.metadata] for g in range(md.num_row_groups)]

    def where(fid):
        return [(s, g) for s, g, st in groups if st is not None and st.min <= fid <= st.max]

    picker, wanted = random.Random(189), {}
    for kind, pattern in PATTERNS.items():
        pool = sorted(e.removeprefix("sim_") for e, f in paths.items()
                      if e not in val and f["split"] == "train"
                      and re.search(pattern, f["instruction_unified"].lower()))
        for fid in picker.sample(pool, min(args.per_kind, len(pool))):
            wanted[fid] = kind
        print(kind, "training flights:", len(pool))

    out = {}
    args.out.mkdir(parents=True, exist_ok=True)
    for fid, kind in wanted.items():
        rows = [r for s, g in where(fid) for r in files[s].read_row_group(g).to_pylist() if r["id"] == fid]
        rows.sort(key=lambda r: r["frame_idx"])
        log = json.loads(rows[0]["log"])
        keep = sorted({0, len(rows) // 2, len(rows) - 1})
        photos = {}
        for r in rows:
            if r["frame_idx"] in keep:
                p = args.out / f"{fid}_f{r['frame_idx']:03d}.jpg"
                Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").save(p)
                photos[str(r["frame_idx"])] = str(p)
        out[fid] = dict(kind=kind, log=log, photos=photos)
        print(fid, kind, log["instruction_unified"], len(rows), "frames", flush=True)
    (args.out / "flights.json").write_text(json.dumps(out), encoding="utf-8")


if __name__ == "__main__":
    main()
