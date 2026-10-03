# Small drone VLA on UAV-Flow (Qwen3-VL-4B + LoRA)

A vision-language-action model for drones that is small enough to run next to a VLM on one
8 GB GPU: a LoRA adapter (rank 32) on Qwen3-VL-4B-Instruct that turns one camera photo, the
drone's state and a language instruction into 8 drone-frame moves (dx, dy, dz, dyaw) as 32
action tokens. Before its moves it writes a **progress line** - what is left to the end of the
flight - and at inference a small server rule (**goal memory + arrival deadband**) keeps that
line consistent when the target leaves the camera view.

## Results (UAV-Flow-Sim, official evaluator, 100 tasks = first 10 of each class)

| Model | Params | Success (3 m, 10 deg) | nDTW |
|---|---|---|---|
| **This model: progress line + goal memory + deadband** | 4B | **69%** | **0.449** |
| This model, progress line only | 4B | 61% | 0.383 |
| Same run before progress tokens (update 18,000) | 4B | 52% | 0.367 |
| OpenVLA-UAV (released checkpoint, same harness) | 7B | 67% | 0.395 |

Model card, training phases, curves and per-class results: [`release/qwen3vl4b_uavflow_vla/`](release/qwen3vl4b_uavflow_vla/).
Full report: [`reports/uav_flow_closed_loop_20260925/REPORT.md`](reports/uav_flow_closed_loop_20260925/REPORT.md).

## Layout

| Path | What |
|---|---|
| `scripts/prepare_uav_flow_official.py`, `prepare_uav_flow_more.py` | UAV-Flow parquet shards -> training store (official OpenVLA-UAV format, site split, simulator flights) |
| `scripts/train_uav_flow_vla.py` | LoRA training (bf16 or NF4, K-step chunks, mirror copies, progress tokens, cosine or warmup-stable-decay schedule, validation per motion type) |
| `scripts/qwen_backend.py`, `uav_flow_action_tokenizer.py` | model loading; 256-bin action tokens on the vocabulary tail |
| `scripts/uav_flow_eval_server.py` | server for the official UAV-Flow-Eval client (`--progress`, `--goal-memory`, `--memory-deadband`, sampling for RL) |
| `scripts/score_uav_flow_sim.py` | official nDTW + success rate per class |
| `scripts/rl/` | post-training: GRPO, PPO (value head + GAE), DAgger (recorded flights as teacher), DPO; reward; the Windows loop `rl_loop.ps1 -Method grpo|ppo|dagger` |
| `scripts/aws/`, `scripts/win_eval/` | the exact launch scripts used (Linux GPU training, Windows simulator tests) |
| `scripts/analyze_progress_lines.py` | goal-memory error analysis |
| `docs/` | research notes, decision log (`LOG.md`), plan (`PLAN.md`), training write-up |
| `reports/` | data manifests (action ranges), evaluation reports |

## Reproduce

1. **Data:** download UAV-Flow (real flights, parquet shards) and UAV-Flow-Sim; run
   `prepare_uav_flow_official.py` (first shards, fixes the validation sites) then
   `prepare_uav_flow_more.py` for the rest. Action ranges: `reports/uav_flow_official_10shard/manifest.json`.
2. **Training:** six phases on one 24 GB GPU (A10G), batch 32, LoRA r32 all-linear; exact
   arguments in `release/qwen3vl4b_uavflow_vla/phases.json` and `scripts/aws/*.sh`
   (`train_long_g5_d171.sh`, then `switch_d176.sh`, `switch_d177.sh`, `decay_d179.sh`).
3. **Evaluation:** install UAV-Flow-Eval (UnrealZoo, Windows), start
   `uav_flow_eval_server.py --model qwen --path <adapter> --chunk 8 --precision bf16 --progress --goal-memory --memory-deadband 1.0,5`,
   run the official `batch_run_act_all.py`, score with `score_uav_flow_sim.py`
   (`scripts/win_eval/eval_variants.ps1` automates this).
4. **Post-training (optional):** `scripts/rl/rl_loop.ps1 -Method dagger|ppo|grpo` on the simulator machine.

Environment variables: `QWEN_MODEL` (local snapshot of Qwen/Qwen3-VL-4B-Instruct),
`UAV_FLOW_OFFICIAL_STORE` (prepared store), `PYTHONPATH=scripts`.

## Notes

- Only the Qwen path is supported here (the SmolVLM experiments were dropped).
- UAV-Flow / UAV-Flow-Sim declare no dataset licence; this work is for research use.
- Credits: UAV-Flow (arXiv 2505.15725), OpenVLA-UAV, WorldVLN (success rule, arXiv 2605.15964),
  Qwen3-VL (Apache-2.0).
