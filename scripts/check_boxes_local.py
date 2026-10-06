"""D198 local gate: run the written-box checks (box_eval.py) on an adapter, on held-out simulator
photos that have box labels (the 504 validation flights; never trained on).

Builds the rows like the trainer does (official prompt with the frame's state, the box label,
the box 8 frames later as next_box), fetches only the needed photos from Hugging Face (parquet
row groups), and prints the metrics. Usage (venv with peft, NF4 on an 8 GB GPU):
  python scripts/check_boxes_local.py --adapter D:/.../adapter_s4000 --boxes D:/.../boxes.jsonl --n 60
"""

import argparse
import io
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_uav_flow_vla as T  # noqa: E402
from box_eval import balanced_box_rows, evaluate_boxes  # noqa: E402

REPO = "datasets/wangxiangyu0814/UAV-Flow-Sim"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--boxes", type=Path, required=True)
    parser.add_argument("--paths", type=Path, default=Path("D:/drone_vla_pilot/data/rl/sim_paths.json"))
    parser.add_argument("--val", type=Path, default=Path("D:/drone_vla_pilot/data/rl/sim_val_flights.json"))
    parser.add_argument("--cache", type=Path, default=Path("D:/drone_vla_pilot/data/box_val"))
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--line", action="store_true", help="also generate the Left line (turn consistency)")
    args = parser.parse_args()
    paths = json.loads(args.paths.read_text(encoding="utf-8"))
    val = set(json.loads(args.val.read_text(encoding="utf-8")))
    labels = {}
    for line in args.boxes.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        parts = item["image"].replace("\\", "/").split("/")
        labels[(parts[-2], int(parts[-1].split(".")[0]))] = item["box"]
    rows = []
    for (episode, t), box in labels.items():
        if episode not in val:
            continue
        flight = paths[episode]
        state = ",".join(str(round(float(x), 1)) for x in flight["proprio"][min(t, len(flight["proprio"]) - 1)])
        later = labels.get((episode, min(t + 8, len(flight["proprio"]) - 1)))
        row = dict(id=f"{episode}:{t:05d}", episode=episode, step=t, instruction=flight["instruction"],
                   prompt=T.OFFICIAL_PROMPT.format(state=state, instruction=flight["instruction"]), box=box)
        if later is not None:
            row["next_box"] = later
        rows.append(row)
    rows = balanced_box_rows(rows, args.n)
    print(json.dumps(dict(labels=len(labels), picked=len(rows))), flush=True)

    # fetch the photos (cached)
    args.cache.mkdir(parents=True, exist_ok=True)
    missing = [r for r in rows if not (args.cache / f"{r['episode']}_{r['step']:05d}.jpg").exists()]
    if missing:
        fs = HfFileSystem()
        files = [pq.ParquetFile(fs.open(f"{REPO}/train-{s:05d}-of-00021.parquet", "rb")) for s in range(21)]
        groups = [(s, g, f.metadata.row_group(g).column(0).statistics)
                  for s, f in enumerate(files) for g in range(f.metadata.num_row_groups)]
        need = {}
        for r in missing:
            fid = r["episode"].removeprefix("sim_")
            for s, g, st in groups:
                if st is not None and st.min <= fid <= st.max:
                    need.setdefault((s, g), []).append(r)
        for (s, g), group_rows in need.items():
            table = files[s].read_row_group(g, columns=["id", "frame_idx", "image"])
            wanted = {(r["episode"].removeprefix("sim_"), r["step"]) for r in group_rows}
            for fid, idx, img in zip(table.column("id").to_pylist(), table.column("frame_idx").to_pylist(),
                                     table.column("image").to_pylist()):
                if (fid, idx) in wanted:
                    Image.open(io.BytesIO(img["bytes"])).convert("RGB").save(args.cache / f"sim_{fid}_{idx:05d}.jpg")
        print(json.dumps(dict(fetched_groups=len(need))), flush=True)

    def image_fn(row):
        picture = Image.open(args.cache / f"{row['episode']}_{row['step']:05d}.jpg").convert("RGB")
        picture.thumbnail((256, 256))
        return picture

    rows = [r for r in rows if (args.cache / f"{r['episode']}_{r['step']:05d}.jpg").exists()]
    from peft import PeftModel
    from uav_flow_action_tokenizer import ActionTokenizer
    api, processor, base, _ = T.setup("qwen", "nf4")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    tokenizer = ActionTokenizer(processor.tokenizer, T.official_stats())
    result = evaluate_boxes(model, processor, tokenizer, rows, 8, api.cuda, want_line=args.line, image_fn=image_fn)
    print(json.dumps(dict(adapter=str(args.adapter), **result)), flush=True)


if __name__ == "__main__":
    main()
