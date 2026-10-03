"""D158: score an official-format adapter (4-D drone-frame actions, K-step chunks).

Every held-out flight is replayed over its recorded frames. At frames 0, K, 2K...
the model sees the recorded frame, the recorded state and the instruction, and
its K actions are executed by dead reckoning in the drone frame. The endpoint in
the start frame is compared with the endpoint of the true actions integrated the
same way, so the representation floor is quantisation only.

Caveat, printed into every report: this is open loop. The state in the prompt is
the recorded one, so errors never feed back, and a small K is re-anchored to the
truth more often. The closed-loop simulator is the real test of the horizon; this
script ranks candidates cheaply and measures cost per call.

Reported per condition (real / gray / swapped frames): median endpoint error,
first-step action error per channel, and seconds and tokens per call. Text-only
baselines are recomputed here on the same endpoints, never seeing an image.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from baseline_uav_flow_chunks import NEIGHBOURS, build_vectors, cosine
from peft import PeftModel
from score_uav_flow_vla import sign_test
from train_uav_flow_vla import (
    OFFICIAL_REPORT,
    collate_left,
    encode,
    official_rows,
    official_stats,
    setup,
)
from uav_flow_action_tokenizer import ActionTokenizer

CAVEAT = (
    "open loop over recorded frames with the recorded state in the prompt; small K is "
    "re-anchored more often; not a closed-loop success rate"
)


def integrate(actions):
    """Start-frame (x, y, z) after executing drone-frame (dx, dy, dz, dyaw) steps."""
    pos, yaw = np.zeros(3), 0.0
    for dx, dy, dz, dyaw in actions:
        c, s = math.cos(yaw), math.sin(yaw)
        pos += (c * dx - s * dy, s * dx + c * dy, dz)
        yaw += dyaw
    return pos


def by_episode(rows):
    out = {}
    for row in rows:
        out.setdefault(row["episode"], []).append(row)
    for steps in out.values():
        steps.sort(key=lambda r: r["step"])
    return out


def truth_actions(steps):
    # step t's chunk starts with the true action at t; the last frame's is zero
    return np.array([s["chunk"][0] for s in steps])


def baselines(train_eps, val_eps):
    train_end = np.array([integrate(truth_actions(s)) for s in train_eps.values()])
    overall = train_end.mean(axis=0)
    train_vectors = build_vectors([s[0]["instruction"] for s in train_eps.values()])
    out = dict(no_text=[], text=[])
    for steps in val_eps.values():
        truth = integrate(truth_actions(steps))
        out["no_text"].append(float(np.linalg.norm(truth - overall)))
        vector = build_vectors([steps[0]["instruction"]])[0]
        sims = np.array([cosine(vector, t) for t in train_vectors])
        top = np.argsort(sims)[-NEIGHBOURS:]
        out["text"].append(float(np.linalg.norm(truth - train_end[top].mean(axis=0))))
    return {k: round(float(np.median(v)), 3) for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--chunk", type=int, required=True, help="the K the adapter was trained on")
    parser.add_argument("--instruction", default="instruction")
    parser.add_argument("--subset", default="val", choices=["val", "train"])
    parser.add_argument("--episodes", type=int, default=62)
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--precision", default="bf16", choices=["nf4", "bf16"])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--batch-size", type=int, default=8, help="predictions per forward pass")
    parser.add_argument("--out-dir", type=Path, default=OFFICIAL_REPORT)
    args = parser.parse_args()
    k = args.chunk

    train_eps = by_episode(official_rows("train", k, args.instruction))
    subset_eps = by_episode(official_rows(args.subset, k, args.instruction))
    chosen = sorted(subset_eps)[: args.episodes]
    donors = {e: chosen[(i + 1) % len(chosen)] for i, e in enumerate(chosen)}

    api, processor, base, _ = setup(args.model, args.precision)
    tokenizer = ActionTokenizer(processor.tokenizer, official_stats())
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    pad_id = processor.tokenizer.pad_token_id

    def predict_many(rows, gray=False):
        """Every prediction is independent - the frame and the state both come from
        the recording - so they all batch, whatever flight they belong to."""
        chunks, seconds, tokens = [], [], []
        for start in range(0, len(rows), args.batch_size):
            group = rows[start : start + args.batch_size]
            items = [
                encode(processor, tokenizer, row, k, gray=gray, with_answer=False) for row in group
            ]
            batch = collate_left(items, pad_id)
            width = batch["input_ids"].shape[1]
            tick = time.monotonic()
            with torch.inference_mode():
                out = model.generate(
                    **api.cuda(batch), max_new_tokens=4 * k + 2, do_sample=False, use_cache=True
                )
            elapsed = (time.monotonic() - tick) / len(group)
            for row_index in range(len(group)):
                ids = out[row_index, width:].tolist()
                chunks.append(tokenizer.decode(ids, k))
                seconds.append(elapsed)
                tokens.append(len(ids))
        return chunks, seconds, tokens

    def anchors_of(episode, condition):
        steps = subset_eps[episode]
        n_actions = len(steps) - 1  # the last frame's zero action is not a move
        rows = []
        for anchor in range(0, n_actions, k):
            row = steps[anchor]
            if condition == "swap":
                donor = subset_eps[donors[episode]]
                row = {**row, "image": donor[min(anchor, len(donor) - 1)]["image"]}
            rows.append(row)
        return rows, n_actions

    def rollout(condition):
        errors, first_err, seconds, tokens, failed, rows_out = [], [], [], [], 0, []
        plan = {e: anchors_of(e, condition) for e in chosen}
        flat = [row for e in chosen for row in plan[e][0]]
        if condition == "floor":
            predictions = [tokenizer.decode(tokenizer.token_ids(r["chunk"]), k) for r in flat]
        else:
            predictions, seconds, tokens = predict_many(flat, gray=condition == "gray")
        cursor = 0
        for episode in chosen:
            rows, n_actions = plan[episode]
            executed, ok = [], True
            for index, row in enumerate(rows):
                chunk = predictions[cursor + index]
                if chunk is None:
                    ok = False
                    break
                first_err.append(np.abs(chunk[0] - np.array(row["chunk"][0])))
                executed.extend(chunk[: min(k, n_actions - index * k)])
            cursor += len(rows)
            steps = subset_eps[episode]
            if not ok:
                failed += 1
                continue
            truth = integrate(truth_actions(steps)[:n_actions])
            predicted = integrate(executed)
            errors.append(float(np.linalg.norm(predicted - truth)))
            rows_out.append(
                dict(
                    episode=episode,
                    error_m=round(errors[-1], 3),
                    predicted=np.round(predicted, 3).tolist(),
                    truth=np.round(truth, 3).tolist(),
                )
            )
        summary = dict(
            scored=len(errors),
            unparsed_episodes=failed,
            median_final_error_m=round(float(np.median(errors)), 3) if errors else None,
            first_step_mae=np.round(np.mean(first_err, axis=0), 5).tolist() if first_err else None,
        )
        if seconds:
            summary.update(
                calls=len(seconds),
                seconds_per_call=round(float(np.median(seconds)), 3),
                batch_size=args.batch_size,
                tokens_per_call=int(np.median(tokens)),
                calls_per_flight_second=round(5 / k, 3),
            )
        return summary, rows_out

    results, rows = {}, {}
    for condition in ("floor", "real", "gray", "swap"):
        results[condition], rows[condition] = rollout(condition)
        print(json.dumps({condition: results[condition]}), flush=True)

    base_scores = baselines(train_eps, {e: subset_eps[e] for e in chosen})
    summary = dict(
        adapter=args.adapter,
        chunk=k,
        subset=args.subset,
        episodes=len(chosen),
        caveat=CAVEAT,
        results=results,
        baselines_m=base_scores,
        beats_text_only=args.subset == "val"
        and results["real"]["median_final_error_m"] is not None
        and results["real"]["median_final_error_m"] < base_scores["text"],
        gray_sign_test=sign_test(rows["real"], rows["gray"]),
        swap_sign_test=sign_test(rows["real"], rows["swap"]),
        donor_rule="each flight's frames come from the next flight in sorted order",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"scored_{args.tag}.json").write_text(
        json.dumps(dict(summary=summary, rows=rows), indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
