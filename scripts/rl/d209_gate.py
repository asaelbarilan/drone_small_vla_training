"""D209 fail-closed gate: split separation, matched rollouts and available supervision."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_uav_flow_vla as T
from rollout_samples import match

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--rollout", type=Path, required=True)
p.add_argument("--rows", nargs="+", type=Path, required=True)
a = p.parse_args()
blocked = set(json.loads((a.root / "held_out.json").read_text()))
manifest = json.loads((a.root / "practice_manifest.json").read_text())
tasks = {f.name: json.loads(f.read_text()) for f in (a.rollout / "tasks").glob("*.json")}
assert set(tasks) == {t["task"] for t in manifest["tasks"]}
assert not {t["source_flight"] for t in tasks.values()} & blocked
completed = {f.name.removesuffix("_2d.png") + ".json" for f in (a.rollout / "pass1").glob("*_2d.png")}
assert completed == set(tasks), "Incomplete practice flights"
matched = match(a.rollout / "pass1", a.rollout / "server_calls.jsonl")
assert set(matched) == set(tasks), "Some flights could not be matched to calls"
rows = []
for path in a.rows:
    part = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert part, f"Empty replay: {path}"
    assert not {r["episode"] for r in part} & blocked
    assert all(Path(r["image"]).is_file() for r in part), f"Missing images: {path}"
    assert all(r["chunk_len"] == 8 for r in part)
    rows += part
fresh = [json.loads(l) for l in a.rows[-1].read_text().splitlines() if l.strip()]
assert all(r.get("box_context") for r in fresh)
assert {r["episode"] for r in fresh} == {t["source_flight"] for t in tasks.values()}
episodes = [json.loads(l) for l in (a.root / "anchor/episodes.jsonl").read_text().splitlines()]
assert not {e["episode"] for e in episodes} & blocked
assert all(e["split_unseen"] == "train" for e in episodes)
labels = {T.frame_key(d["image"]): d["box"] for d in map(json.loads, (a.root / "boxes.jsonl").read_text().splitlines())}
labelled = sum(T.frame_key(im) in labels for e in episodes for im in e["images"])
assert labelled >= 900, f"Grounding replay coverage too small: {labelled}"
assert all((a.root / "anchor/frames" / T.frame_key(im)).is_file() for e in episodes for im in e["images"])
result = dict(status="passed", tasks=len(tasks), matched=len(matched), fresh_rows=len(fresh), replay_rows=len(rows), labelled_anchor_frames=labelled, held_out_overlap=0)
(a.root / "gate.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result))
