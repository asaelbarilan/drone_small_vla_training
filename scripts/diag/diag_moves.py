"""D204 early-warning check (free, local GPU, NF4): does an adapter's MOVE output behave like it
can fly? D203 showed training accuracy can hide a model that never stops, so this checks what the
evaluator needs, the way the server answers (no teacher forcing):

  still rows    (true chunk all zero): does the answer pass the evaluator's still rule
                (every step < 3 cm and < 1 degree)?
  straight rows (no turn): yaw bias per step (should be ~0)
  turning rows  : predicted vs true yaw per step

Held-out real VAL rows of the local store, ground-truth Left line as context (as goal memory
hands it over), no box line (unlabelled frames). Token adapters: greedy over the 256 action
tokens (server D191); adapters with action_head.pt: the head (D201). Same rows for every
adapter (seed 0), so checkpoints compare directly.

  python scripts/diag/diag_moves.py <adapter dir> [n rows, default 60]
"""
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train_uav_flow_vla as T  # noqa: E402
from uav_flow_action_tokenizer import ActionTokenizer  # noqa: E402

ADAPTER = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
K = 8

from peft import PeftModel  # noqa: E402
from transformers import LogitsProcessor, LogitsProcessorList  # noqa: E402

api, processor, base, _ = T.setup("qwen", "nf4")
model = PeftModel.from_pretrained(base, ADAPTER).eval()
tok = ActionTokenizer(processor.tokenizer, T.official_stats())
low, high = tok.vocab_size - tok.n_bins, tok.vocab_size
head = None
if (Path(ADAPTER) / "action_head.pt").exists():
    from action_head import ActionHead, slot_id, slot_states
    head = ActionHead.load(str(Path(ADAPTER) / "action_head.pt"), "cuda").eval()


class ActionTokensOnly(LogitsProcessor):
    def __call__(self, input_ids, scores):
        masked = torch.full_like(scores, float("-inf"))
        masked[:, low:high] = scores[:, low:high]
        return masked


def answer(row):
    """The adapter's 8 moves (metres, radians) for this row, as the server would answer."""
    ctx = {k: v for k, v in row.items() if k not in ("box", "next_box")}
    ctx["progress_text"] = T.progress_text(row["progress"])
    batch = T.encode(processor, tok, ctx, K, with_answer=False)
    with torch.inference_mode():
        if head is not None:
            batch = T.append_ids(batch, [slot_id(tok)] * K)
            mask = torch.zeros_like(batch["input_ids"])
            mask[:, -K:] = 1
            out = model(**api.cuda(batch), output_hidden_states=True)
            return tok.denormalise(head(slot_states(out, mask.cuda(), K))[0].float().cpu().numpy())
        ids = model.generate(
            **api.cuda(batch), max_new_tokens=4 * K, min_new_tokens=4 * K, do_sample=False, use_cache=True,
            logits_processor=LogitsProcessorList([ActionTokensOnly()]),
            pad_token_id=processor.tokenizer.pad_token_id,
        )[0, batch["input_ids"].shape[1]:].tolist()
    steps = tok.decode(ids, K)
    return np.array(steps if steps is not None else [[0.0] * 4] * K, dtype=float)


rows = [r for r in T.official_rows("val", K, instruction="both", progress=True) if not r.get("mirror")]
random.seed(0)
still = [r for r in rows if all(abs(v) < 1e-9 for s in r["chunk"] for v in s)]
straight = [r for r in rows if r not in still and max(abs(s[3]) for s in r["chunk"]) < 0.01]
turning = [r for r in rows if max(abs(s[3]) for s in r["chunk"]) >= 0.05]
groups = {"still": random.sample(still, min(N // 3, len(still))),
          "straight": random.sample(straight, min(N // 3, len(straight))),
          "turning": random.sample(turning, min(N // 3, len(turning)))}

report = dict(adapter=str(ADAPTER), kind="head" if head is not None else "tokens")
for name, group in groups.items():
    P = np.array([answer(r) for r in group])
    Y = np.array([r["chunk"] for r in group], dtype=float)
    entry = dict(n=len(group),
                 step_cm=round(float(np.median(np.linalg.norm(P[..., :3], axis=-1)) * 100), 2),
                 true_step_cm=round(float(np.median(np.linalg.norm(Y[..., :3], axis=-1)) * 100), 2),
                 yaw_deg=round(float(np.degrees(P[..., 3].mean())), 3),
                 true_yaw_deg=round(float(np.degrees(Y[..., 3].mean())), 3),
                 abs_err_cm=round(float(np.abs(P[..., :3] - Y[..., :3]).mean() * 100), 2),
                 abs_err_yaw_deg=round(float(np.degrees(np.abs(P[..., 3] - Y[..., 3]).mean())), 3))
    if name == "still":
        entry["pass_still_rule"] = int(sum(
            (np.linalg.norm(p[:, :3] * 100, axis=1) < 3).all() and (np.degrees(np.abs(p[:, 3])) < 1).all() for p in P))
    report[name] = entry
print(json.dumps(report), flush=True)
