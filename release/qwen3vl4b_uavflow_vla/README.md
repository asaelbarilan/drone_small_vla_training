---
license: apache-2.0
base_model: Qwen/Qwen3-VL-4B-Instruct
library_name: peft
pipeline_tag: robotics
tags: [vision-language-action, uav, drone, uav-flow, lora, qwen3-vl]
---

# Qwen3-VL-4B UAV-Flow VLA with progress tokens (LoRA)

A drone vision-language-action model: one forward camera photo, the drone's state and a
language instruction in; 8 drone-frame moves (dx, dy, dz, dyaw) out, as 32 action tokens.
It is a LoRA adapter (rank 32, about 322 MB) on Qwen3-VL-4B-Instruct, so it shares its weights
with the base VLM and fits next to it on one 8 GB GPU (llama.cpp, adapter switched per request).

Before its moves the model writes one **progress line** - what is left to the end of the
flight, in its own frame - and at inference a small server rule (**goal memory + arrival
deadband**) keeps that line consistent when the target leaves the camera view.

## Results (UAV-Flow-Sim, official evaluator, 100 tasks = first 10 of each class)

Success = the flight ends within 3 m and 10 degrees of the reference end (WorldVLN's automatic
rule); nDTW = the official metric.py. Same harness, same tasks for every row.

| Model | Params | Success | nDTW |
|---|---|---|---|
| **This adapter, progress line + goal memory + deadband** | 4B | **69%** | **0.449** |
| This adapter, progress line only | 4B | 61% | 0.383 |
| This adapter, progress line + goal memory (no deadband) | 4B | 61% | 0.409 |
| Same run at update 18,000 (before progress tokens) | 4B | 52% | 0.367 |
| Earlier adapter (real + sim, 4,500 updates) | 4B | 50% | 0.336 |
| OpenVLA-UAV (released checkpoint) | 7B | 67% | 0.395 |

Per class (best row): Rotate 10/10, Move 10, Retreat 10, Shift 9, Ascend/Descend 9, Approach 8,
Land 5, Pass 4, Turn 3, Surround 1. Weighted by the benchmark's task counts, as the papers
report, about 68% (published: WorldVLN 79.1%, ImagineUAV 70.9%, OpenVLA-UAV 65.6%).
100 tasks give about +-5 points of noise. Full per-flight scores: `eval/*/score.json`,
summary `eval/results.json`, flights and server logs `eval/*/flights_and_server_log.zip`.

## Input / output format

- Prompt: `Current State: {x},{y},{z},{yaw_deg}, What action should the uav take to {instruction}?`
  with one 224x224 RGB photo; state in metres and degrees relative to the flight start.
- Answer: `Left +FF.F,+SS.S,+UU.U,+YYY\n` (21 tokens: metres forward, sideways, up; degrees of
  turn, summed) then 32 action tokens = 8 steps x (dx, dy, dz, dyaw) in the drone frame at that
  step. Each value is one of 256 bins on the last 256 vocabulary ids; the bin ranges are the
  0.1/99.9 percentiles in `data/action_stats_manifest_10shard.json` (`action_stats_train`).
- Server (code: `scripts/uav_flow_eval_server.py`):
  `--model qwen --path adapter_s31000 --chunk 8 --precision bf16 --progress --goal-memory --memory-deadband 1.0,5`.
  Goal memory: the first progress line fixes a goal in the start frame; every later call is handed
  the line recomputed from the drone's pose. Deadband: within 1 m and 5 degrees the line becomes 0
  (the model stops); without it, small overshoots ("-2 deg") kept the model moving.

## Training (details: `training/phases.json`, curves: `training/curves.png`)

LoRA r32, alpha 32, dropout 0, all linear layers (vision tower, projector, language model),
bf16 base, batch 32 (8 x 4 accumulation), one NVIDIA A10G, about 38,700 updates in six phases:

| Phase | Data | Schedule | Updates |
|---|---|---|---|
| 0 | 10 real UAV-Flow shards (1,987 train flights), mirror copies, both instruction wordings | cosine, peak 5e-4 | 2,500 |
| 1 | + UAV-Flow-Sim (9,941 flights, near-test flights excluded) | cosine, peak 2e-4 | 2,000 |
| 2 | same; 504 simulator flights held out for validation | cosine, 5e-5, early stop | about 2,700 |
| 3 | all 54 real shards + simulator | cosine, peak 5e-4, 3% warmup | 18,000 |
| 4 | + progress line | steady 1.142e-4 until real and simulator accuracy flatten | 10,000 |
| 5 | same | cosine decay to 0 | 3,000 |

Held-out move-token accuracy over phases 3-5: real flights 51.6% -> 54.5%, simulator
91.1% -> 91.7%. `training/report_long_run.json` holds every update's loss and every validation
(real, simulator, per motion class); logs of all phases are in `training/logs/`.
Total cloud cost about $125 (AWS).

## Files

| Path | What |
|---|---|
| `adapter_s31000/` | the final adapter (use this) |
| `adapter_s18000/` | the same run before progress tokens (ablation) |
| `training/` | phases.json, report_long_run.json, curves.png, logs |
| `data/` | action ranges, held-out simulator flights, excluded near-test flights, added shards, RL practice tasks |
| `eval/` | 100-task list, per-flight scores and flights for every variant and for OpenVLA-UAV |
| `rl_trial/` | GRPO trial round (8 tasks x 2 flights; pipeline check, not a result) |
| `code/` | scripts snapshot and git commit |

## Limitations

- Weak on Turn (face a person/dog: 3/10), Pass, Land and Surround (orbit).
- Simulator benchmark only (UAV-Flow-Sim, teleporting evaluator, no flight physics); no real flight.
- The progress line counts down to the *recorded* end of a demonstration; instructions without a
  stated distance (Pass) are ambiguous by construction.
- UAV-Flow and UAV-Flow-Sim declare no dataset licence; this adapter is for research use.

## Credits

Data and evaluator: UAV-Flow (Wang et al., arXiv 2505.15725). Baseline: OpenVLA-UAV. Success rule:
WorldVLN (arXiv 2605.15964). Base model: Qwen3-VL-4B-Instruct (Apache-2.0).
