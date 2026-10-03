# Qwen3-VL-4B drone VLA on UAV-Flow: training and closed-loop evaluation

Date: 2026-09-25. Branch `codex/vla-aws-pilot-20260916`, decisions D158–D164 in
`CHANGES.md`. This file is the self-contained record of the first model we
trained end to end and flew in the official UAV-Flow closed-loop simulator.

> **Status 2026-10-03 (sections 10-21 are the later updates).** Best flight-tested
> model: the final long-run adapter (4B) with progress tokens, goal memory and the arrival
> deadband: **success 69% / nDTW 0.449** on the 100 tasks, above the released OpenVLA-UAV
> (7B) at 67% / 0.395 in the same harness (section 21). WorldVLN reports 79.1%. Progress
> tokens and goal memory: section 19; flight tests: section 20; the memory analysis and fix:
> section 21. RL pipeline works end to end (trial, section 20). Decisions: section 17.

## 1. Headline

On 100 closed-loop UAV-Flow-Sim tasks (10 per motion class), our LoRA adapter on
Qwen3-VL-4B reaches **mean nDTW 0.129**. The released OpenVLA-UAV checkpoint,
run through the identical harness, reaches **0.395** and wins 77 of 100 paired
flights. Our model is better only on Retreat. It was trained on 1,987 **real**
flights and never saw simulator data; the released model's action statistics are
the simulator (`sim`) ones.

| Motion class | OpenVLA-UAV (released, 7B) | Ours (Qwen3-VL-4B + LoRA) |
|---|---|---|
| Turn | 0.176 | 0.154 |
| Move | 0.121 | 0.044 |
| Shift | 0.675 | 0.106 |
| Rotate | 0.349 | 0.110 |
| Surround | 0.753 | 0.001 |
| Ascend/Descend | 0.775 | 0.057 |
| Approach | 0.389 | 0.320 |
| Retreat | 0.293 | **0.355** |
| Pass | 0.252 | 0.134 |
| Land | 0.167 | 0.004 |
| **Mean nDTW (100 tasks)** | **0.395** | **0.129** |
| Median steps flown | 50 | 26 |
| Median end distance to reference end | 0.61 m | 2.62 m |

nDTW is the official automatic metric (`UAV-Flow-Eval/metric.py`, unchanged
functions). The papers' headline number is a human-judged success rate
(OpenVLA-UAV 65.6%), which is a different metric and is not reported here.

## 2. The model

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen3-VL-4B-Instruct` (Hugging Face), loaded in bf16 (no quantisation) |
| Adapter | LoRA, rank 32, alpha 32, dropout 0, bias none |
| LoRA targets | all linear layers of language model, vision tower and merger (`q/k/v/o_proj, gate/up/down_proj, qkv, attn.proj, linear_fc1/2`) |
| Trainable parameters | LoRA only; base frozen |
| Action head | none added: actions are 256-bin tokens on the last 256 vocabulary ids (OpenVLA scheme), spliced by id |
| Output per call | K = 8 future steps x 4 values = 32 action tokens + EOS |
| Adapter files | `D:\drone_vla_pilot\release\qwen3vl4b_uavflow_k8_10shard_s2500\` (also on the AWS disk, `~/runs/official_k8_10shard/adapter_s2500`) |

## 3. Data

| Item | Value |
|---|---|
| Source | UAV-Flow real-world set, `wangxiangyu0814/UAV-Flow`, shards `train-00000` … `train-00009` of 54 |
| Flights read | 5,000 |
| Split | site-disjoint unseen split: validation holds out whole 300 m sites; any training flight whose instruction wording also appears in validation is dropped |
| Train / validation / dropped | 1,987 / 791 / 2,222 flights |
| Train frames | 133,637 (153,507 with the official first/last-frame x5 oversampling) |
| Train examples after mirroring | 307,014 |
| Format | official OpenVLA-UAV `uav_dataset.py`, ported and verified: 0 mismatches over 5,000 flights |
| Action | (dx, dy, dz, dyaw) in the drone's current local frame, from `raw_logs` [0,1,2,4], yaw deg→rad, last frame's action zero |
| State in prompt | `preprocessed_logs` [0,1,2,4] (start frame, metres, yaw in degrees), rounded to 0.1 |
| Prompt | `Current State: {x,y,z,yaw}, What action should the uav take to {instruction}?` |
| Instruction wording | both `instruction` (free) and `instruction_unified` (template), alternating by frame |
| Horizon | K = 8 steps (1.6 s at 5 Hz); padded with zero actions past the end of a flight |
| Action range | per channel 0.1/99.9 percentiles of training actions (official is 1/99): q_low [-0.544, -0.442, -0.133, -0.457], q_high [1.188, 0.442, 0.193, 0.276]; 256 uniform bins on [-1, 1] |
| Augmentation | left-right mirror copy of every example: image flipped, dy and dyaw negated, side words swapped (left↔right, clockwise↔counterclockwise); verified that a mirrored chunk integrates to the mirrored path |
| Images | frames stored as 256 px thumbnails (JPEG q92) |
| Data manifest | `data_manifest_10shard.json` (sha256 of the episode file included) |

## 4. Training

| Item | Value |
|---|---|
| Hardware | 1 x NVIDIA A10G 24 GB (AWS g5.2xlarge), peak 12.7 GiB |
| Batch | 8 per step x 4 accumulation = 32 examples per update |
| Updates | 2,500 (80,000 examples ≈ 0.26 epoch of the mirrored set) |
| Optimiser | AdamW, lr 5e-4, cosine schedule with 3% linear warm-up, gradient clip 1.0 |
| Gradient checkpointing | on (non-reentrant) |
| Wall time | ≈ 4.2 h (≈ 6.0 s per update), about $5 |
| Tracking | Weights & Biases, run `asael/vla training/official_k8_10shard` |
| Code | `scripts/train_uav_flow_vla.py --format official --chunk 8 --precision bf16 --instruction both --mirror --batch-size 8 --accum 4 --workers 6 --updates 2500 --schedule cosine` |

Loss (token cross-entropy on the 33 answer tokens): train 10.57 → 2.55.
Held-out loss (128 fixed validation examples), every 250 updates: 11.02, 2.94,
2.80, 2.72, 2.70, 2.64, 2.61, 2.60, 2.58, 2.57, 2.566 — still falling at the end,
no overfitting (a 1-shard run overfit after about 1.3 epochs).

## 5. Offline evaluation (before the simulator)

150 unseen-site validation flights, open loop (recorded frames and states),
endpoint error after integrating the predicted chunks:

| Condition | Median endpoint error |
|---|---|
| Representation floor (true actions through the tokeniser) | 0.072 m |
| Real photos | 3.067 m |
| Flat gray photos | 3.115 m |
| Photos from a different flight | 3.121 m |
| Text-only nearest-neighbour baseline | 4.533 m |
| No-text mean-path baseline | 4.208 m |

The adapter beats both text baselines, but the camera contributes only about
5 cm (gray sign test p = 0.29, swap p = 0.085). A mirror probe (60 flights x 3
frames; does the action change when the photo is mirrored?) rose over training:
0% at update 250, 11% at 500, 19% at 1,000, 14% at 1,500, 12% at 2,000, 26% at
2,500. The released OpenVLA-UAV changes its action in 16 of 20 cases.

## 6. Closed-loop evaluation setup

| Item | Value |
|---|---|
| Benchmark | UAV-Flow-Eval (github.com/buaa-colalab/UAV-Flow), DowntownWest map |
| Simulator | UnrealZoo `Collection_WinNoEditor_0424_25` (official Windows build) |
| Machine | AWS g6.xlarge, NVIDIA L4 24 GB, Windows Server 2022, GRID driver 596.86 |
| Tasks | 100 of the 273 test tasks: the first 10 (sorted by file name) of each of the 10 classes |
| Episode limit | 100 executed steps, or 10 consecutive near-still steps (official rule) |
| Our server | `scripts/uav_flow_eval_server.py` (official HTTP protocol). Model outputs metres; the simulator uses centimetres, so the state is divided by 100 into the prompt and each step multiplied by 100. The 8 predicted steps are integrated in the drone frame and returned as 8 poses; a new image is taken only after each chunk |
| OpenVLA-UAV server | same file, official maths (one step per call, `unnorm_key="sim"`), bf16 + eager attention (official uses flash-attention) |
| Local changes to the evaluator | render frame rate capped (`t.MaxFPS 10`); offscreen rendering (the instance has no desktop) |
| Known quirk kept as-is | the official evaluator passes UnrealCV's BGR image to PIL as RGB (orange sky) — both models see the same images |
| Model calls | OpenVLA-UAV 5,463; ours 404 |
| Scoring | `scripts/score_uav_flow_sim.py` (official nDTW functions + end distance / yaw) |

Invalid attempts, kept for the record and not used: the Linux UnrealZoo build
(`Collection_v4`) never moved the camera with the drone (every task's first frame
identical); a laptop run froze the 8 GB GPU; our first Windows attempt had a
missing library and made no model calls.

## 7. Caveats for a publication

1. **Domain gap.** Trained on real flights, evaluated in simulation. The released
   baseline carries simulator action statistics. This is the most likely cause
   of the gap and must be stated, not hidden.
2. **Subset.** 100 of 273 tasks, first 10 per class — not the full benchmark.
3. **Metric.** nDTW only; no success rate (needs human judgement or a validated
   automatic rule).
4. **Units.** The metre-to-centimetre conversion is inferred from the released
   model's action statistics (forward q99 ≈ 48.7 per step), not documented by
   the authors.
5. **Undertrained.** 0.26 epoch of 10 of 54 shards; held-out loss still falling.
6. **Single seed**, single checkpoint (update 2,500).
7. **Evaluator changes** (frame-rate cap, offscreen) affect rendering speed, not
   the images; the released baseline ran under the same changes.

## 8. Files

| File | Content |
|---|---|
| `REPORT.md` | this description |
| `closed_loop_ours.json` | per-flight and per-class scores, our adapter |
| `closed_loop_openvla_uav.json` | the same for OpenVLA-UAV |
| `data_manifest_10shard.json` | data split, counts, action statistics, checksums |
| `adapter_config.json` | exact LoRA configuration |
| Flight logs and plots | `D:\drone_vla_pilot\runs\sim_eval_win\{qwen,openvla}\flights\` and `s3://<bucket>/results/` |


## 9. Update D165 (2026-09-25): adding simulator flights

The domain-gap explanation was tested directly. The adapter above was trained
for 2,000 more updates on real + UAV-Flow-Sim flights and flown on the same 100
tasks.

**Simulator data.** `wangxiangyu0814/UAV-Flow-Sim`, 21 shards, 10,109 flights.
Leak check against the 273 test tasks: only 1 test task had a simulator flight
with the same start (<0.5 m) and the same instruction; 54 had a simulator flight
starting within 0.5 m. Every simulator flight starting within 0.5 m of any test
start was removed (168), leaving 9,941 flights / 307,834 frames. The simulator
data is recorded in the same town (DowntownWest) as the tests, so this is a
same-environment, disjoint-trajectory evaluation. Simulator positions are in
centimetres and were divided by 100 to match the real data; the action range
(section 3) was kept, and 5.6% of simulator steps fall outside it.

**Training.** Initialised from the section-2 adapter (update 2,500), fresh AdamW,
lr 2e-4 cosine with 3% warm-up, 2,000 updates x 32 examples, same K = 8, mirror,
both wordings; 1,121,502 training examples (real + simulator, mirrored), about
3.4 h on one A10G. Train loss 2.23 -> 1.19; held-out loss on the REAL unseen
split 2.565 -> 2.582 (unchanged). Adapter:
`D:\drone_vla_pilot\release\qwen3vl4b_uavflow_k8_realsim_s2000\`.

**Closed loop, same 100 tasks (mean nDTW; one empty flight counted as 0):**

| Motion class | OpenVLA-UAV | Ours, real only | Ours, real + sim |
|---|---|---|---|
| Turn | 0.176 | 0.154 | 0.120 |
| Move | 0.121 | 0.044 | 0.020 |
| Shift | 0.675 | 0.106 | 0.664 |
| Rotate | 0.349 | 0.110 | 0.259 |
| Surround | 0.753 | 0.001 | 0.612 |
| Ascend/Descend | 0.775 | 0.057 | 0.721 |
| Approach | 0.389 | 0.320 | 0.343 |
| Retreat | 0.293 | 0.355 | **0.467** |
| Pass | 0.252 | 0.134 | 0.142 |
| Land | 0.167 | 0.004 | 0.051 |
| **Mean** | **0.395** | **0.128** | **0.333** |
| Median steps / end distance | 50 / 0.61 m | 26 / 2.62 m | 34 / 1.29 m |

Simulator data lifts the adapter from 0.128 to 0.333 (better than the real-only
adapter on 69 of 100 flights), which confirms the domain gap as the main cause.
It is now within 0.06 of the released 7B model (theirs better on 62 of 100) with
a 4B model that shares its weights with the testbed's VLM, and it beats it on
Retreat. Still weak: Land, Move, Turn, Pass. Files:
`closed_loop_ours_real_sim.json`, `training_report_real_sim.json`,
`sim_data_added.json`.

## 10. Success rate, comparable with the papers (D172)

The papers report a success rate; UAV-Flow's is human-judged. WorldVLN's code
(github.com/EmbodiedCity/WorldVLN.code, `reward_uavflow.py`) defines success
automatically: the flight ends within **3 m** of the reference end **and within
10 degrees** of its heading. Applied to the same 100 flights:

| Model | Success (3 m + 10 deg) | 3 m only | Mean nDTW |
|---|---|---|---|
| OpenVLA-UAV (released, 7B) | **67%** | 77% | 0.395 |
| Ours, real + sim (section 9) | 50% | 64% | 0.333 |
| Ours, real only (section 1) | 38% | 55% | 0.128 |

The rule gives OpenVLA-UAV 67%, close to its paper's 65.6%, so these numbers are
comparable with published ones (WorldVLN 79.1%, ImagineUAV 70.9%). Per class it
is less reliable (it scores orbit tasks by end point only).

## 11. Checking the test harness itself (D167)

Before interpreting failures, the harness was audited from the run logs:

- The state the simulator sends back equals the pose our 8th move reached, in
  567 of 567 calls (largest difference 3e-14 cm).
- The evaluator places the drone exactly on our moves, in order, in 99 of 100
  flights (the 100th got one unparseable reply).
- Left/right conventions agree between the real training data and the simulator
  (left = negative sideways move and negative yaw in both).
- The simulator teleports the drone to each pose; there is no flight physics.
- Benchmark flaw found: 8 of the 273 test files start with a tilted pose
  (non-zero roll/pitch in `initial_pos`), which bends their reference path (in
  the worst one "move 9 m left" becomes a 7.8 m dive) while the camera stays
  level. 2 of our 100 tasks are affected; both models score about 0 on the worst
  one. Re-scoring against a level reference changes the means only slightly
  (ours 0.336 -> 0.345, OpenVLA-UAV 0.395 -> 0.401). We report the official numbers.

## 12. Why our flights end early (D167, D168)

- 89 of 100 flights end by the evaluator's rule "10 consecutive moves below
  3 cm and 1 degree": the model returns a whole chunk of "stay" moves.
- On target-defined tasks the drone stops at a roughly fixed distance whatever
  the correct one is: Pass stops at 5.8 m (correct 14.0 m, correlation -0.05),
  Approach at 3.6 m (correct 5.9 m), Land at 3.3 m and never descends (0 of 10
  flights). When the distance is written in the instruction it stops correctly
  (Shift 4.8 m vs 5.0 m, Ascend/Descend 6.2 m vs 6.0 m, correlation 1.0).
- Mechanism: with 8 moves per answer, one "stop" answer is 8 still moves, so a
  single wrong stop ends the flight; a one-step model must answer "stop" about
  10 times in a row. In training, "stop" targets are 15-27% of the examples of
  short simulator flights (last frame repeated 5 extra times).
- Test ("stop-confirm", no retraining): when the model answers "stop", execute
  one still move and ask again. Result **0.316** vs 0.333 (10 flights better, 17
  worse; Land unchanged at 0.05). The model keeps answering "stop": it believes
  it has arrived. Not adopted; the fix belongs in training.
- Land training data checked: 98% of simulator Land flights descend (median
  -2.26 m), 81% of real ones. The data is fine; the descent comes at the end of
  a Land flight, which our drone never reaches.

## 13. Validation redesign and the learning rate (D169)

- The simulator set ships no split. 504 simulator flights (56 per motion type,
  5% of 10,109) are now held out for validation; real validation grew from 128 to
  1,024 examples; loss is reported per motion type. 5% rather than 20-30%: the
  real test is the separate benchmark, and the simulator flights are the data
  that lifted nDTW from 0.128 to 0.333.
- Continuing the section-9 adapter at lr 2e-4 raised the simulator validation
  loss at update 500 (0.737 -> 0.767), the known rise after re-warming the
  learning rate. At lr 5e-5 it was flat, then fell slowly: 0.737 (start) ->
  0.743 (500) -> 0.738 (1,000) -> 0.737 (1,500) -> 0.732 (2,000) -> **0.732
  (2,500, best)**; real validation stayed about 2.62. Stopped at about update
  2,700 to save budget; `d169/d169_adapters.tgz` holds updates 1,000, 2,000 and
  the best (2,500). Not flight-tested.
- Research (`docs/research/VLA_PLATEAU_20260929.md`): (1) the validation loss is
  a poor gauge of flying; in section 9 it stayed flat (2.565 -> 2.582) while
  nDTW went 0.128 -> 0.333, and OpenVLA trained until its training action-token
  accuracy passed 95%; (2) we were far under-trained: OpenVLA-UAV used 200,000
  updates x 32 at lr 5e-4 with LoRA rank 32 (UAV-Flow paper, appendix C), we had
  about 6,000; (3) the best LoRA learning rate is about 10x a full fine-tune's
  (Thinking Machines, "LoRA Without Regret"), so 5e-5 was too low; (4) the loss
  bump after re-warming is the known "stability gap" and recovers.

## 14. All 54 real shards (D170)

- The 44 unused real shards (train-00010..00053) were added: **21,795 flights**,
  18,430 train + 3,365 validation, 1.43 M new frames. The real validation sites
  of the 10-shard split stay fixed (four 300 m cells), so the earlier numbers
  stay valid. Action statistics were not recomputed (0.8% of new steps fall
  outside the range). Store: `s3://<bucket>/d170/store.tar`
  (75 GB).
- Decision: the old "instruction sweep" (dropping training flights whose wording
  also appears in validation) is not applied to the new shards: it would drop
  about 44% of them, and the benchmark and real use repeat the same wordings.
- Validation now also reports teacher-forced action-token accuracy (exact and
  within one bin). First values (section-9 adapter, small smoke sample): real
  0.50 exact / 0.68 within one bin; simulator 0.90 / 0.91 (Pass 0.52, Approach
  0.82, Land 0.92).

## 15. The long run (D171, training now)

All 54 real shards + 9,437 simulator training flights, from the D169b best
adapter; lr 5e-4 (the recipe), one 3% warm-up and a cosine decay over **26,000
updates** x 32 examples (about 0.83 M examples, one A10G, about 45 h, about $55);
validation every 1,000 updates (1,024 real + 1,024 simulator examples, loss and
token accuracy, per motion type); adapters every 2,000 updates to
`s3://.../d171/run/`. No early stop on loss; a guard stops the run if simulator
token accuracy stays more than 2 points below its best for 3 checks. Started
2026-09-30 15:29 UTC (wandb `asael/vla training/official_k8_d171`). Results
pending.

## 16. RL and post-training options: built, not yet run (D172, D173)

Following WorldVLN's action-aware GRPO where it transfers to a token VLA:

- **Practice tasks**: 200 tasks from simulator TRAINING flights in the
  benchmark's file format (25 per motion type), validation flights and flights
  near test starts excluded. Tasks whose instruction names a spawned object
  (person, dog, car; 2,874 flights) are excluded because the flight records carry
  no object placement; hence no orbit (Surround) tasks. Self-check: reference
  end vs recorded world end, worst 32 cm.
- **Reward**: 0.5 x end reward (0.85 x 1/(1 + (distance/2 m)^2 + (yaw/10 deg)^2)
  + 0.15 x success) + 0.5 x nDTW. The nDTW part matches the benchmark's to 5e-5
  on 300 flights; the success part reproduces 67/50/38%.
- **Rollouts**: the unmodified evaluator flies each task 8 times; the server
  samples only among the 256 action tokens at a set temperature and records each
  call's photo, prompt, tokens and log-probabilities.
- **Update**: GRPO (PPO-clipped per token, advantages z-scored within each task's
  8 flights, old log-probabilities recomputed) with an expert anchor (SFT on the
  task's recorded flight: WorldVLN's 12:1 mix, EG-GRPO's expert-in-group idea);
  or, on the same rollouts, trajectory preference pairs (best vs worst flight per
  task, DPO, as in AeroDPO and GRAPE).
- **Next rounds**: half the worst unsolved tasks again, half new (RecoverFly).
- **Inference options**: re-ask after N of the 8 moves; contrastive decoding
  against a gray photo (PCD/VCD) to amplify what the camera contributes.
- Skipped: learned world models / imagination (WorldVLN uses an 8B video model,
  ImagineUAV a 1.3B one; neither fits next to the VLM on 8 GB).

## 17. Decisions

| Date | Decision | Why |
|---|---|---|
| 09-22 | 4-D local-frame actions, state in the prompt, official sampling (OpenVLA-UAV format) | Verified 0 mismatches against the official loop |
| 09-23 | 8-move chunks (K = 8), both instruction wordings, mirror copies, wider action range (0.1/99.9%) | Fewer calls; free data; the official 1/99% range costs a perfect model 4 m+ on 10% of flights |
| 09-24 | Evaluate on the official Windows simulator, 100 tasks (10 per class) | The Linux build's camera does not follow the drone |
| 09-25 | Add simulator flights (leak-checked) | Domain gap; nDTW 0.128 -> 0.333 |
| 09-28 | Stop-confirm not adopted | 0.316 vs 0.333; the model believes it has arrived |
| 09-28 | No imagination / world model | Video models too large for the 8 GB target |
| 09-28 | Simulator validation 5% (504 flights) with per-type loss | The test is a separate benchmark; keep the training data |
| 09-29 | Recipe learning rate 5e-4, one warm-up, much longer training | Research: 3% of the recipe's updates, LR 10x too low |
| 09-29 | All 54 real shards, no instruction sweep | Under-trained rather than short of repeats; the sweep drops 44% |
| 09-30 | No early stop on loss; guard on token accuracy | Loss did not track flying (section 13) |
| 09-30 | Train on the single-GPU g5 instead of waiting for the p4d quota | Same cost per example; quota pending 8 days |
| 09-30 | RL reward = 0.5 end + 0.5 nDTW | Optimise what the benchmark measures |
| 09-30 | Add all post-SFT options (EG-GRPO anchor, RecoverFly, DPO, PCD, re-ask) | User decision; each tested separately |

## 18. What was trained and flight-tested, and what was not

| Model / option | Data | Training | Flight-tested |
|---|---|---|---|
| Real only, K = 8 (section 1) | 10 real shards | 2,500 updates, lr 5e-4 | Yes: nDTW 0.128, success 38% |
| Real + sim (section 9) | + 9,941 sim flights | +2,000 updates, lr 2e-4 | Yes: nDTW 0.333, success 50% |
| Stop-confirm (inference) | - | - | Yes: 0.316, not adopted |
| D169, lr 2e-4 | real + sim, sim validation | stopped at ~560 updates | No |
| D169b, lr 5e-5 | same | ~2,700 updates, best at 2,500 | No |
| D171 long run | all 54 real shards + sim | 26,000 updates planned, running | Pending |
| Re-ask after 4 moves; contrastive decoding | - | - | Built, not tested |
| RL: GRPO and DPO, RecoverFly rounds | 200 practice tasks | Built, not run | - |
| Imagination / world model | - | Not built (decision) | - |

Costs so far: September AWS usage $56, fully covered by credits; the D171 run
adds about $55.

## 19. Progress tokens and goal memory (D175-D179)

- **Problem (section 12, research note docs/research/VLA_BEYOND_WORLDVLN_20260930.md):** the
  model does not know when to stop. It has its heading in the prompt but turns for a fixed
  number of steps whatever angle is asked; WorldVLN's four worst classes (Pass, Rotate,
  Surround, Turn) are the same "how far / when to stop" tasks.
- **Progress tokens:** before its 8 moves the model writes one fixed-format line with what is
  left to the end of the flight, in its own frame: `Left +12.3,-00.4,+00.0,-090` (metres
  forward, sideways, up; degrees of turn; summed turn, so a full circle reads 360). 21 tokens,
  labels free from the recorded end pose (checked against the re-integrated actions: 1 mm
  median). At test time the server lets the model write the line, then the moves.
- **Goal memory (server only, no training):** the first line fixes a goal in the start frame;
  every later call is handed the line recomputed from the drone's actual pose.
- **Training (D171 -> D179):** the long run went on to 18,000 updates without the line, then
  continued with it at a steady learning rate until accuracy flattened, and ended with a
  3,000-update decay at 31,000. Teacher-forced move-token accuracy 51.6% -> 54.5% (real),
  91.1% -> 91.7% (simulator).

## 20. Flight test of the long run (D180, 2026-10-03)

Same 100 tasks, success = end within 3 m and 10 degrees (WorldVLN's rule):

| Model | Success | nDTW | Rotate | Land | Pass | Turn |
|---|---|---|---|---|---|---|
| Section 9 adapter | 50% | 0.336 | 3 | 1 | 0 | 4 |
| Long run at 18,000 (no line) | 52% | 0.367 | 5 | 2 | 1 | 5 |
| Final (31,000), progress line | 61% | 0.383 | **8** | **6** | 1 | 4 |
| Final, progress line + goal memory | 61% | **0.409** | 6 | 4 | **6** | 3 |
| OpenVLA-UAV (released, 7B) | 67% | 0.395 | 1 | 5 | 4 | 10 |

- The final model gains on exactly the "when to stop" classes; Rotate and Land beat the 7B
  model. The 18,000 -> 31,000 step mixes the line with 13,000 more updates and the decay.
- Goal memory gives the best path shape of any model in this harness and fixes Pass (the
  object leaves the camera view), but breaks some numeric tasks (section 21).
- The line is always well-formed (0 unparsed of 615 calls).
- **RL trial (same day):** 8 tasks x 2 flights, one GRPO update on the Windows box, about
  $0.40, 22 min. Every step works (flights matched to calls, expert anchor 171 examples,
  adapter saved, machine off). Finding: 4 of 8 tasks gave two identical flights at
  temperature 1.0 - the policy is very peaked, so RL needs more exploration.

## 21. Why goal memory broke Rotate, Shift and Land, and the fix (D182)

On the 615 logged calls of the memory flight test, both lines were scored against the true
remainder to the reference end on the same drone states:

| | Model's own line | Memory line |
|---|---|---|
| Median position error | 0.43 m | 0.08 m |
| Median turn error | 0.86 deg | 0.32 deg |

Memory is the more accurate line. It fails at arrival: the drone flies all 8 moves of a call,
often ends slightly past the goal, and memory then says "-0.2 m" or "-2 deg". Training lines
only ever count down to exactly 0, so the model keeps moving ("Turn 90 deg" drifted to 22 deg;
"Move 5 m left at 50 deg" flew 10.6 m past and timed out). Land fails differently: the first
estimate from the start photo is short (6.8 m vs 9.4 m) and memory freezes it.
Fix: `--memory-deadband 1.0,5` - within 1 m and 5 deg the line becomes 0, which the model
reads as "stop". Offline, "the line says stop exactly when the drone has arrived" rises from
74% to 97% (Shift), 79% to 95% (Rotate), 60% to 99% (Ascend/Descend).

**Flight test (D182):** progress line + goal memory + deadband, same 100 tasks:
**success 69%, nDTW 0.449**, against OpenVLA-UAV 67% / 0.395 - better on both, with a 4B
model. Per class: Rotate 10/10, Move 10, Retreat 10, Shift 9, Ascend/Descend 9, Approach 8,
Land 5, Pass 4, Turn 3, Surround 1. Weighted by the benchmark's task counts (as the papers
report) this is about 68%, against WorldVLN's 79.1%. Remaining gaps: Turn (face a person or
dog), Pass, Land, Surround. With 100 tasks a difference of a few points is within noise.
