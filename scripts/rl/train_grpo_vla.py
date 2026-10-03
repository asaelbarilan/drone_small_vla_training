"""D172 RL: GRPO update of the Qwen3-VL LoRA policy from simulator rollouts.

Follows WorldVLN's action-aware GRPO in the parts that transfer to a token VLA
(github.com/EmbodiedCity/WorldVLN.code, Worldmodel/runtime/tools/GRPO/):
rollouts of the current policy on TRAINING tasks, a reward on the flight's end
(reward_uav_flow.py), advantages z-scored within each task's group of rollouts,
and an SFT anchor from expert flights (their 12 GRPO : 1 SFT replay mix).

Policy, per model call: 4*K action tokens sampled from
    pi(token) = softmax(logits[256 action tokens] / temperature)
exactly as the rollout server sampled them (uav_flow_eval_server.py
--rollout-temperature). Loss per token (PPO clip, as in GRPO):
    -min(r * A, clip(r, 1 - eps, 1 + eps) * A),  r = exp(logp_new - logp_old)
with A the flight's advantage. The "old" log-probabilities are recomputed here
with the rollout adapter before any update (so rollout and training machines
need not agree to the last bit); the recorded ones are kept as a check.

Samples whose group has no reward spread (A = 0) carry no gradient and are
skipped, as RIPT-VLA's dynamic sampling does.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from rollout_samples import build, ppo_advantages, summary  # noqa: E402
from train_uav_flow_vla import (  # noqa: E402
    collate,
    encode,
    official_rows,
    official_stats,
    setup,
)
from uav_flow_action_tokenizer import ActionTokenizer  # noqa: E402


def encoded(processor, tokenizer, sample, chunk, rollout_dirs):
    """Prompt + saved rollout image + the sampled answer tokens (no EOS)."""
    image = Image.open(rollout_dirs[sample["passage"]] / sample["image"]).convert("RGB")
    row = dict(prompt=sample["prompt"])
    if "progress_text" in sample:  # D175: the server's progress line is context
        row["progress_text"] = sample["progress_text"]
    batch = encode(processor, tokenizer, row, chunk, with_answer=False, image=image)
    prompt_length = batch["input_ids"].shape[1]
    answer = torch.tensor([sample["ids"]], dtype=batch["input_ids"].dtype)
    batch["input_ids"] = torch.cat([batch["input_ids"], answer], dim=1)
    batch["attention_mask"] = torch.cat([batch["attention_mask"], torch.ones_like(answer)], dim=1)
    for key in ("mm_token_type_ids", "token_type_ids"):
        if key in batch:
            batch[key] = torch.cat([batch[key], torch.zeros_like(answer)], dim=1)
    return batch, prompt_length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flights", nargs="+", type=Path, required=True,
                        help="one evaluator flights/ folder per rollout pass over the same tasks")
    parser.add_argument("--calls", type=Path, required=True, help="the rollout server's server_calls.jsonl")
    parser.add_argument("--images", type=Path, required=True, help="the rollout server's rollout_images/")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--init-adapter", type=Path, required=True, help="the rollout policy")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accum", type=int, default=8)
    # D175: WorldVLN uses 0.02 at lr 8e-7; 0.2 (the LLM default) is 10x looser.
    parser.add_argument("--clip", type=float, default=0.05)
    parser.add_argument(
        "--call-weight",
        type=float,
        default=0.5,
        help="D175 per-move reward: weight of each call's own progress advantage "
        "added to its flight's advantage (0 = flight reward only)",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="D175: the policy writes progress lines; the expert anchor gets them too",
    )
    parser.add_argument("--sft-weight", type=float, default=0.1,
                        help="weight of the expert (SFT) loss on the tasks' own flights")
    parser.add_argument("--objective", default="grpo", choices=["grpo", "dpo", "ppo"],
                        help="dpo: trajectory preference pairs, best vs worst flight per task "
                        "(D173, AeroDPO/GRAPE style); the rollout adapter is the reference. "
                        "ppo (D185): per-call advantages from a value head (GAE) instead of "
                        "group z-scores; RL4VLA found PPO > GRPO > DPO for VLAs")
    parser.add_argument("--gamma", type=float, default=0.99, help="ppo: discount per call")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="ppo: GAE lambda")
    parser.add_argument("--value-coef", type=float, default=0.5, help="ppo: value loss weight")
    parser.add_argument("--value-lr", type=float, default=1e-4, help="ppo: value head learning rate")
    parser.add_argument("--value-head", type=Path,
                        help="ppo: value head from the previous round (value_head.pt); zero-initialised if absent")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    parser.add_argument("--dpo-margin", type=float, default=0.1,
                        help="minimum reward gap between a pair's two flights")
    parser.add_argument("--precision", default="bf16", choices=["nf4", "bf16"])
    parser.add_argument("--wandb-project")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()

    # One server serves every pass: episodes are numbered across passes and each
    # flight is matched to its own episode by its exact poses.
    passes = [(folder, args.calls) for folder in args.flights]
    rollout_dirs = [args.images] * len(passes)
    samples, rows = build(passes, args.tasks, args.call_weight)
    stats = summary(rows)
    # GRPO/DPO: samples whose group has no spread carry no signal. PPO keeps every call:
    # its advantage comes from the value head, not from the group.
    samples = [s for s in samples if s.get("ids") and (args.objective == "ppo" or s["advantage"] != 0)]
    temperature = samples[0]["temperature"]
    assert all(s["temperature"] == temperature for s in samples), "mixed rollout temperatures"
    print(json.dumps(dict(rollouts=stats, samples_with_signal=len(samples))), flush=True)

    api, processor, model, _ = setup("qwen", args.precision)
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(args.init_adapter), is_trainable=True)
    tokenizer = ActionTokenizer(processor.tokenizer, official_stats())
    low, high = tokenizer.vocab_size - tokenizer.n_bins, tokenizer.vocab_size
    pad_id = processor.tokenizer.pad_token_id
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [dict(params=params, lr=args.lr)]
    value_head = None
    if args.objective == "ppo":
        # D185: V(state) read from the last hidden state at the end of the prompt (before
        # the answer); zero-initialised so the first round starts from V = 0 (Monte Carlo).
        config = model.config
        hidden = getattr(getattr(config, "text_config", config), "hidden_size")
        value_head = torch.nn.Linear(hidden, 1).cuda()
        torch.nn.init.zeros_(value_head.weight)
        torch.nn.init.zeros_(value_head.bias)
        if args.value_head and args.value_head.exists():
            value_head.load_state_dict(torch.load(args.value_head, map_location="cuda"))
        groups.append(dict(params=list(value_head.parameters()), lr=args.value_lr))
    optimiser = torch.optim.AdamW(groups, lr=args.lr)

    def token_logps(items, with_values=False):
        """Per-token log-probabilities of each item's answer under the current
        policy: softmax over the 256 action tokens only (the output layer is wider
        than the tokenizer's vocabulary), at the rollout temperature. With
        with_values (PPO), also V(state) from the hidden state ending the prompt."""
        batch = api.cuda(collate([b for b, _ in items], pad_id))
        output = model(**batch, output_hidden_states=with_values)
        logits = output.logits
        out, values = [], []
        for row, (_, prompt_length) in enumerate(items):
            answer = batch["input_ids"][row, prompt_length : prompt_length + 4 * args.chunk]
            positions = logits[row, prompt_length - 1 : prompt_length - 1 + answer.shape[0], low:high]
            scaled = positions.float() / temperature
            out.append(scaled.log_softmax(-1).gather(1, (answer - low)[:, None])[:, 0])
            if with_values:
                values.append(value_head(output.hidden_states[-1][row, prompt_length - 1].float())[0])
        if with_values:
            return torch.stack(out), torch.stack(values)
        return torch.stack(out)

    # Old log-probabilities under the rollout adapter, before any update.
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(samples), args.batch_size):
            part = samples[i : i + args.batch_size]
            items = [encoded(processor, tokenizer, s, args.chunk, rollout_dirs) for s in part]
            if value_head is not None:
                logps, values = token_logps(items, with_values=True)
                for s, v in zip(part, values.tolist(), strict=True):
                    s["value_old"] = v
            else:
                logps = token_logps(items)
            for s, lp in zip(part, logps, strict=True):
                s["old"] = lp.tolist()
    drift = [abs(a - b) for s in samples for a, b in zip(s["old"], s["logps"], strict=True)]
    print(json.dumps(dict(old_logp_vs_rollout_max=round(max(drift), 4),
                          mean=round(sum(drift) / len(drift), 5))), flush=True)

    if args.objective == "ppo":
        # D185 GAE over each flight's calls: reward = the call's own progress gain
        # (rollout_samples.call_gain: closeness gained, 2 m or 10 deg = 1), plus the
        # flight's end reward on its last call.
        flights_ppo = ppo_advantages(samples, args.gamma, args.gae_lambda)
        values_all = [s["gae"] for s in samples]
        mean = sum(values_all) / len(values_all)
        std = (sum((v - mean) ** 2 for v in values_all) / len(values_all)) ** 0.5 + 1e-6
        for s in samples:
            s["advantage"] = (s["gae"] - mean) / std
        print(json.dumps(dict(ppo_flights=len(flights_ppo), adv_mean=round(mean, 4), adv_std=round(std, 4),
                              return_mean=round(sum(s["return"] for s in samples) / len(samples), 4))), flush=True)

    # Expert anchor: SFT examples from the tasks' own recorded flights.
    sources = {json.loads((args.tasks / s["task"]).read_text(encoding="utf-8"))["source_flight"] for s in samples}
    expert = [
        r
        for r in official_rows("train", args.chunk, "both", progress=args.progress)
        if r["episode"] in sources
    ]
    print(json.dumps(dict(expert_examples=len(expert), expert_flights=len(sources))), flush=True)

    run = None
    if args.wandb_project:
        import wandb

        run = wandb.init(project=args.wandb_project, entity="asael", name=args.out.name, config=vars(args) | dict(temperature=temperature))
    report = dict(args={k: str(v) for k, v in vars(args).items()}, rollouts=stats, flights=rows,
                  samples=len(samples), temperature=temperature, updates=[])
    model.train()
    rng = random.Random(7)
    step = 0

    def sft_term(scale=1.0):
        if not (args.sft_weight and expert):
            return 0.0
        rows_sft = rng.sample(expert, min(args.batch_size, len(expert)))
        batch = collate([encode(processor, tokenizer, r, args.chunk) for r in rows_sft], pad_id)
        loss_sft = model(**api.cuda(batch)).loss
        (scale * args.sft_weight * loss_sft).backward()
        return float(loss_sft)

    if args.objective == "dpo":
        # Trajectory log-probability = sum over its calls of the answer-token
        # log-probs. Loss per pair: -log sigmoid(beta * ((w - w_ref) - (l - l_ref))).
        # Its gradient is -beta * sigmoid(-z) * (grad w - grad l), so z is computed
        # without gradients first and each call is then back-propagated with that
        # fixed weight: exact, and it never holds a whole flight in memory.
        flights = {}
        for s in samples:
            flights.setdefault((s["task"], s["passage"]), []).append(s)
        by_task = {}
        for (task, _), calls in flights.items():
            by_task.setdefault(task, []).append(calls)
        pairs = []
        for task, group in by_task.items():
            group.sort(key=lambda calls: calls[0]["flight_reward"])
            if group[-1][0]["flight_reward"] - group[0][0]["flight_reward"] >= args.dpo_margin:
                pairs.append((group[-1], group[0]))
        report["dpo_pairs"] = len(pairs)
        print(json.dumps(dict(dpo_pairs=len(pairs))), flush=True)

        def trajectory(calls, weight=None):
            total = 0.0
            for i in range(0, len(calls), args.batch_size):
                part = calls[i : i + args.batch_size]
                items = [encoded(processor, tokenizer, s, args.chunk, rollout_dirs) for s in part]
                logps = token_logps(items).sum()
                if weight is not None:
                    (weight * logps).backward()
                total += float(logps)
            return total

        for epoch in range(args.epochs):
            rng.shuffle(pairs)
            for u in range(0, len(pairs), args.accum):
                optimiser.zero_grad(set_to_none=True)
                logs = dict(dpo=0.0, z=0.0, pair_acc=0.0, sft=0.0)
                chunk_pairs = pairs[u : u + args.accum]
                for win, lose in chunk_pairs:
                    with torch.no_grad():
                        margin = (trajectory(win) - sum(sum(s["old"]) for s in win)) - (
                            trajectory(lose) - sum(sum(s["old"]) for s in lose)
                        )
                    z = args.beta * margin
                    weight = args.beta * torch.sigmoid(torch.tensor(-z)).item() / len(chunk_pairs)
                    trajectory(win, -weight)
                    trajectory(lose, weight)
                    logs["dpo"] += float(-torch.nn.functional.logsigmoid(torch.tensor(z))) / len(chunk_pairs)
                    logs["z"] += z / len(chunk_pairs)
                    logs["pair_acc"] += float(z > 0) / len(chunk_pairs)
                logs["sft"] = sft_term()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimiser.step()
                step += 1
                report["updates"].append(dict(step=step, epoch=epoch, **{k: round(v, 5) for k, v in logs.items()}))
                print(json.dumps(report["updates"][-1]), flush=True)
                if run:
                    run.log(logs, step=step)
    for epoch in range(args.epochs if args.objective in ("grpo", "ppo") else 0):
        order = list(range(len(samples)))
        rng.shuffle(order)
        per_update = args.batch_size * args.accum
        for u in range(0, len(order) - per_update + 1, per_update):
            optimiser.zero_grad(set_to_none=True)
            logs = dict(pg=0.0, ratio=0.0, clipped=0.0, sft=0.0, value=0.0)
            for a in range(args.accum):
                part = [samples[j] for j in order[u + a * args.batch_size : u + (a + 1) * args.batch_size]]
                items = [encoded(processor, tokenizer, s, args.chunk, rollout_dirs) for s in part]
                if value_head is not None:
                    new, values = token_logps(items, with_values=True)
                    returns = torch.tensor([s["return"] for s in part], device=values.device)
                    loss_value = ((values - returns) ** 2).mean()
                    extra = args.value_coef * loss_value  # one backward with the policy loss
                    logs["value"] += float(loss_value) / args.accum
                else:
                    new = token_logps(items)
                    extra = 0.0
                old = torch.tensor([s["old"] for s in part], device=new.device)
                adv = torch.tensor([s["advantage"] for s in part], device=new.device)[:, None]
                ratio = (new - old).exp()
                loss_pg = -torch.minimum(ratio * adv, ratio.clamp(1 - args.clip, 1 + args.clip) * adv).mean()
                ((loss_pg + extra) / args.accum).backward()
                logs["sft"] += sft_term(1.0 / args.accum) / args.accum
                logs["pg"] += float(loss_pg) / args.accum
                logs["ratio"] += float(ratio.mean()) / args.accum
                logs["clipped"] += float(((ratio - 1).abs() > args.clip).float().mean()) / args.accum
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimiser.step()
            step += 1
            report["updates"].append(dict(step=step, epoch=epoch, **{k: round(v, 5) for k, v in logs.items()}))
            print(json.dumps(report["updates"][-1]), flush=True)
            if run:
                run.log(logs, step=step)
    model.save_pretrained(args.out / f"adapter_{args.objective}")
    if value_head is not None:
        torch.save(value_head.state_dict(), args.out / "value_head.pt")
    report["elapsed_s"] = round(time.monotonic() - start, 1)
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if run:
        run.finish()


if __name__ == "__main__":
    main()
