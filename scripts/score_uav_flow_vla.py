"""D155: roll out an action-token adapter and score it against every reference.

Four numbers, always together:

  representation floor - the true actions pushed through encode/decode and the
                         same rollout arithmetic. A perfect model cannot beat it,
                         so it says how much of any error is the format's fault.
  model on real frames
  model on flat gray frames
  model on swapped frames - every frame replaced by the same-index frame of a
                         different held-out flight (instruction unchanged)
  text-only baseline   - from baselines.json, never sees an image

`--subset train` scores episodes the model was trained on. That is the
memorisation check: if a model that has seen an episode still cannot reproduce it,
the fault is in the data, the labels or the rollout, not in generalisation.

Action tokens are read from the generated ids directly. Decoding them to text and
re-parsing loses 177 of 200 chunks to BPE re-merging.
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from train_uav_flow_vla import encode, setup
from uav_flow_action_tokenizer import ActionTokenizer, load_stats

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/uav_flow_chunks_20260921"
STORE = Path(os.environ.get("UAV_FLOW_STORE", "D:/drone_vla_pilot/data/uav_flow_chunks_20260921"))
api = None


def predict_chunk(model, processor, tokenizer, row, k, gray):
    batch = encode(processor, tokenizer, row, k, gray=gray, with_answer=False)
    with torch.inference_mode():
        out = model.generate(
            **api.cuda(batch), max_new_tokens=k * 6 + 2, do_sample=False, use_cache=True
        )
    ids = out[0, batch["input_ids"].shape[1] :].tolist()
    return tokenizer.decode(ids, k)


def swapped(episodes, by_episode):
    """Each flight's frames come from the next flight in the list (cyclic), matched
    by step index and clamped to that flight's length."""
    out = {}
    for i, episode in enumerate(episodes):
        own = by_episode[episode["episode"]]
        donor = by_episode[episodes[(i + 1) % len(episodes)]["episode"]]
        out[episode["episode"]] = [
            {**row, "image": donor[min(j, len(donor) - 1)]["image"]} for j, row in enumerate(own)
        ]
    return out


def rollout(model, processor, tokenizer, episodes, by_episode, k, gray, floor=False):
    errors, details, failed = [], [], 0
    for episode in episodes:
        steps = by_episode[episode["episode"]]
        total, ok = np.zeros(6), True
        for anchor in range(0, len(steps), k):
            row = steps[anchor]
            if floor:
                # the true chunk, pushed through the same quantiser the model uses
                chunk = tokenizer.decode(tokenizer.token_ids(row["chunk"][:k]), k)
            else:
                chunk = predict_chunk(model, processor, tokenizer, row, k, gray)
            if chunk is None:
                ok = False
                break
            total += chunk[: min(k, len(steps) - anchor)].sum(axis=0)
        if not ok:
            failed += 1
            continue
        truth = np.array(episode["endpoint_m"])
        errors.append(float(np.linalg.norm(total[:3] - truth)))
        details.append(
            dict(
                episode=episode["episode"],
                predicted=[round(float(v), 3) for v in total[:3]],
                truth=[round(float(v), 3) for v in truth],
                error_m=round(errors[-1], 3),
            )
        )
    ordered = sorted(errors)
    return (
        dict(
            episodes=len(episodes),
            scored=len(errors),
            unparsed_episodes=failed,
            distinct_endpoints=len({tuple(d["predicted"]) for d in details}),
            median_final_error_m=round(ordered[len(ordered) // 2], 3) if ordered else None,
            mean_final_error_m=round(sum(errors) / len(errors), 3) if errors else None,
        ),
        details,
    )


def sign_test(a_rows, b_rows):
    b = {r["episode"]: r["error_m"] for r in b_rows}
    pairs = [(r["error_m"], b[r["episode"]]) for r in a_rows if r["episode"] in b]
    wins = sum(x < y for x, y in pairs)
    n = sum(x != y for x, y in pairs)
    if not n:
        return dict(better=wins, comparable=0, p=1.0)
    k = min(wins, n - wins)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
    return dict(better=wins, comparable=n, p=round(p, 4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--split", default="split_unseen")
    parser.add_argument("--subset", default="val", choices=["val", "train"])
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model", default="smol256", choices=["smol256", "smol500", "qwen"])
    parser.add_argument("--precision", default="nf4", choices=["nf4", "bf16"])
    parser.add_argument("--out-dir", type=Path, default=REPORT)
    args = parser.parse_args()

    baselines = json.loads((REPORT / "baselines.json").read_text(encoding="utf-8"))[args.split]
    steps = [
        json.loads(s) for s in (STORE / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    episodes = [
        json.loads(s) for s in (STORE / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    by_episode = {}
    for row in steps:
        if row[args.split] == args.subset:
            by_episode.setdefault(row["episode"], []).append(row)
    for key in by_episode:
        by_episode[key].sort(key=lambda r: r["step"])
    chosen = sorted(
        (e for e in episodes if e[args.split] == args.subset), key=lambda e: e["episode"]
    )[: args.episodes]
    assert chosen, "no episodes in that subset"

    global api
    api, processor, base, _ = setup(args.model, args.precision)
    tokenizer = ActionTokenizer(processor.tokenizer, load_stats(args.split))
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    floor, _ = rollout(
        None, processor, tokenizer, chosen, by_episode, args.chunk, False, floor=True
    )
    real, real_rows = rollout(model, processor, tokenizer, chosen, by_episode, args.chunk, False)
    gray, gray_rows = rollout(model, processor, tokenizer, chosen, by_episode, args.chunk, True)
    swap, swap_rows = rollout(
        model, processor, tokenizer, chosen, swapped(chosen, by_episode), args.chunk, False
    )

    bar = baselines["target_to_beat_m"]
    summary = dict(
        adapter=args.adapter,
        split=args.split,
        subset=args.subset,
        chunk=args.chunk,
        caveat=(
            "rollout over recorded frames, not closed-loop flight; "
            "not comparable to published success rates"
        ),
        median_trajectory_m=baselines["median_trajectory_m"],
        representation_floor=floor,
        model=dict(real=real, gray=gray, swap=swap),
        baselines=dict(
            no_text_m=baselines["no_text"]["median_final_error_m"],
            text_only_m=baselines["text_nearest_neighbour"]["median_final_error_m"],
            bar_m=bar if args.subset == "val" else None,
        ),
        vision_sign_test=sign_test(real_rows, gray_rows),
        swap_sign_test=sign_test(real_rows, swap_rows),
        beats_text_only=args.subset == "val"
        and real["median_final_error_m"] is not None
        and real["median_final_error_m"] < bar,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"scored_{args.tag}.json").write_text(
        json.dumps(dict(summary=summary, real=real_rows, gray=gray_rows, swap=swap_rows), indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
