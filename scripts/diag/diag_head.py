"""D203 diagnosis: run the step-2 action head on held-out real VAL rows exactly as in training
(trainer encode, ground-truth progress line, dataset photo), NF4 locally. Measures the yaw bias
and whether the head outputs "still" when the true chunk is still."""
import json, math, random, sys
import numpy as np
import torch

sys.path.insert(0, "scripts")
import train_uav_flow_vla as T
from action_head import ActionHead, slot_id, slot_states
from peft import PeftModel
from uav_flow_action_tokenizer import ActionTokenizer

ADAPTER = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60

api, processor, base, _ = T.setup("qwen", "nf4")
model = PeftModel.from_pretrained(base, ADAPTER).eval()
tok = ActionTokenizer(processor.tokenizer, T.official_stats())
head = ActionHead.load(f"{ADAPTER}/action_head.pt", "cuda").eval()
T.ACTION_HEAD.update(on=True, slot=slot_id(tok))

rows = T.official_rows("val", 8, instruction="both", progress=True)
rows = [r for r in rows if not r.get("mirror")]
random.seed(0)
still = [r for r in rows if all(abs(v) < 1e-9 for s in r["chunk"] for v in s)]
straight = [r for r in rows if r not in still and max(abs(s[3]) for s in r["chunk"]) < 0.01]
turning = [r for r in rows if max(abs(s[3]) for s in r["chunk"]) >= 0.05]
pick = {"still": random.sample(still, min(N // 3, len(still))),
        "straight": random.sample(straight, min(N // 3, len(straight))),
        "turning": random.sample(turning, min(N // 3, len(turning)))}
print(json.dumps({k: len(v) for k, v in pick.items()} | dict(all_still=len(still), all_straight=len(straight))), flush=True)

for name, group in pick.items():
    errs, preds, trues = [], [], []
    for row in group:
        batch = T.encode(processor, tok, row, 8)
        mask, target = batch.pop("slot_mask"), batch.pop("action_target")
        batch.pop("has_action"); batch.pop("labels")
        with torch.inference_mode():
            out = model(**api.cuda(batch), output_hidden_states=True)
            norm = head(slot_states(out, mask.cuda(), 8))[0].float().cpu().numpy()
        pred = tok.denormalise(norm)
        true = np.array(row["chunk"])
        preds.append(pred); trues.append(true)
    P, Y = np.array(preds), np.array(trues)
    d = P - Y
    print(f"== {name} n={len(group)}")
    print("  mean error (pred-true) dx dy dz [cm], dyaw [deg]:",
          np.round([d[..., 0].mean() * 100, d[..., 1].mean() * 100, d[..., 2].mean() * 100, math.degrees(d[..., 3].mean())], 2))
    print("  mean |error|                                     :",
          np.round([abs(d[..., 0]).mean() * 100, abs(d[..., 1]).mean() * 100, abs(d[..., 2]).mean() * 100, math.degrees(abs(d[..., 3]).mean())], 2))
    print("  pred step size cm / yaw deg (median):", round(float(np.median(np.linalg.norm(P[..., :3], axis=-1)) * 100), 2),
          round(float(np.degrees(np.median(P[..., 3]))), 3),
          " true:", round(float(np.median(np.linalg.norm(Y[..., :3], axis=-1)) * 100), 2), round(float(np.degrees(np.median(Y[..., 3]))), 3))
    if name == "still":
        ok = [(np.linalg.norm(p[:, :3] * 100, axis=1) < 3).all() and (np.degrees(np.abs(p[:, 3])) < 1).all() for p in P]
        print("  passes the evaluator's still rule (all 8 steps <3 cm, <1 deg):", sum(ok), "/", len(ok))
        print("  example pred (cm,cm,cm,deg):", np.round(np.c_[P[0][:, :3] * 100, np.degrees(P[0][:, 3])], 2)[:3].tolist(), "line", row.get("progress"))
