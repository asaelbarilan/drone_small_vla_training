# CURRENT VLA PLAN (updated 2026-10-03, D182) - the live list; everything below it is older history

> **RULE 1 (user, 2026-10-03): the 273 UAV-Flow-Eval benchmark test tasks are NOT flown again,
> for any purpose (tuning, checks, ablations), until the user decides otherwise.** The 100 of them
> flown so far (first 10 per class) were used for development decisions (goal memory, deadband);
> the paper must say so. Decisions from now on use validation tasks built from the 504 held-out
> UAV-Flow-Sim flights (D169, `release/qwen3vl4b_uavflow_vla/sim_val_flights.json`).

> **Decision (user, 2026-10-04): every run after D190 uses the D191 fixed server** (greedy
> decoding restricted to action tokens). D190 itself keeps the old server so its before / after
> stay comparable; the next run re-measures the D190 models with the fixed server first.

### D196 plan - staged retrain so the model finds the target itself (user, 2026-10-06; ST4VLA recipe)

Why: D193 co-training taught the box FORMAT, not grounding (same box every photo; the Left line
ignores even a correct box). ST4VLA (2602.10109) shows the order matters: grounding first, then
actions on top - plain co-training after action training only partly works.

| Stage | What is trained | Data | Updates / time / cost | Must show before going on |
|---|---|---|---|---|
| 0. Build + measure | trainer: grounding-only mode, balanced box sampling, box-token loss weight, lower learning rate for the vision part, and a validation that lets the model WRITE the box (centre error, IoU) | - | code, local checks, $0 | the new metrics reproduce D193's failure on the D193 model (same box every time) |
| 1. Grounding first | fresh LoRA on the base Qwen3-VL-4B: answer = target box + box after the moves, no moves | our 81k labelled frames (simulator + real), balanced so off-centre targets are as common as centred ones; validation = the 504 held-out simulator flights | about 2,000-3,000 updates, about 3 h on one L4, about $5 | written boxes close to the labels (IoU near the base model's own), and NOT one average box |
| 2. Actions on top | continue the stage-1 LoRA: answer = box + next box + Left line + 8 moves (box kept wherever labelled), DAgger rows, mirror, photo augmentation; vision part at a lower learning rate so the action training cannot erase the grounding (ST4VLA damps that gradient x0.5) | full real + simulator store | about 15,000-20,000 updates, 40-50 h on one L4, about $40-50 | every 1,000 updates: move accuracy climbing toward the old 54 % real / 91.7 % sim, AND box metrics not falling |
| 3. Validation flights | Windows box, fixed server (D191) with --box | the 100-task validation check set | about 2 h, about $3 | Turn / Pass better than 4/12 and 9/15 without losing other kinds (rule 1: no test tasks) |

Total about $50-60. Stop rules: stage 1 grounding not clearly better than D193 -> stop and rethink
before stage 2; stage 2 box metrics falling for 2 checks in a row -> lower the action weight / vision
learning rate before continuing.

### D193 checklist - what every run must keep (user, 2026-10-05: "insert what we have learned so we don't forget")

| # | Lesson | Where it lives |
|---|---|---|
| 1 | Progress line ("Left ...") before the moves | trainer `--progress`, server `--progress` (D175) |
| 2 | Goal memory + arrival deadband 1 m / 5 deg | server `--goal-memory --memory-deadband 1.0,5` (D175, D182) |
| 3 | Greedy moves restricted to the 256 action tokens (orbit bug) | server default since D191 |
| 4 | DAgger corrections of rounds 1 and 2 stay in the training mix | trainer `--extra-rows` (D188, D190) |
| 5 | Mirror copies, with the box mirrored too | trainer `--mirror`, `mirror_box` (D193) |
| 6 | Photos as the evaluator sends them: BGR, 224 px | trainer `--photo-aug` (D192, D193) |
| 7 | The model must use the photo: box of the target before the Left line | trainer `--boxes`, server `--box` (D192, D193) |
| 8 | Decisions on validation only; the 273 test tasks untouched | rule 1, `d190/check_tasks.zip` (D187) |
| 9 | Judge by validation flights per kind, not by loss | D169, D190 |
| 10 | Turn / Surround practice and validation tasks place the person / dog (appearance 1-35, never 0) | D189 |
| 11 | Machines turn themselves off; bucket from `$VLA_BUCKET`; no HF upload yet | D181, repo rules |

### Data sets

| Set | Made from | Used for |
|---|---|---|
| **Train** | UAV-Flow real + simulator training flights (and the 200 practice tasks built from them, `scripts/rl/build_rl_tasks.py`) | learning only: supervised training, DAgger, RL |
| **Validation** | the 504 held-out UAV-Flow-Sim flights (D169, `release/qwen3vl4b_uavflow_vla/sim_val_flights.json`) -> 307 flyable tasks (`build_rl_tasks.py --validation`) | every development decision and error analysis |
| **Test** | the 273 UAV-Flow-Eval benchmark tasks | untouched; flown once at the end for the paper. 100 of them were used during development before 2026-10-03 (disclosed) |


Goal: beat WorldVLN (79.1% success on UAV-Flow) with a 4B VLA that fits 8 GB.
Where we are: 50% success on our 100-task test (OpenVLA-UAV 67%). Biggest losses: Land, Pass,
Rotate, Turn = "knowing when to stop" (docs/research/VLA_BEYOND_WORLDVLN_20260930.md).
Rules: ask before every AWS spend; machines turn themselves off; results go into
reports/uav_flow_closed_loop_20260925/REPORT.md after each step.

| # | Step | What it answers | Cost | Status |
|---|---|---|---|---|
| 1 | D171 long run, progress tokens from 18,000, decay to 31,000 | More training + "when to stop" | about $75 | DONE (D179) |
| 2 | Flight test 18,000 / final + line / final + line + memory | Does it fly better? | about $4 | DONE (D180): 52% / 61% / 61% (nDTW 0.409 > OpenVLA-UAV 0.395) |
| 3 | RL trial (8 tasks x 2 flights, 1 GRPO update) | Does the RL pipeline work? | about $0.40 | DONE: works end to end; policy too peaked at T=1.0 |
| 4 | Memory + snap to 0 (--memory-deadband 1.0,5) flight test | Recover Rotate/Shift/A-D, keep Pass | about $2 | DONE (D182): **69% / nDTW 0.449, above OpenVLA-UAV 67% / 0.395** |
| 5 | Decide: RL rounds (24 tasks x 8 flights about $4, 48 tasks about $8, T=1.3) | Land/Turn and remaining stop errors | per round | waits for 4 and the credit balance |
| 6 | Re-ask after 4 moves (--execute-steps 4) | Overshoot safety net | about $3 | only if 4 does not fix overshoot |

Decision rule: each step starts from the best model so far; a step that does not beat the
previous best in the flight test is written down and dropped.

Before step 3: upload d175/vla_code.zip (scripts, src, 10-shard manifest; .sh files with LF endings).
Before step 5: step 3 exports d175/anchor_store.tar (the expert flights the RL loop needs).

Parked: sample-and-verify at test time; snapshot + delete the Windows disk (about $15/month);
check remaining AWS credits. Not doing: video imagination, hand-parsing numbers.

## All ideas from our research, one table (status as of 2026-09-30)

Sources: A = docs/VLA_DRONE_TRAINING.md section 15 (E0-E10), B = docs/research/VLA_AFTER_SFT_20260925.md,
C = docs/research/VLA_PLATEAU_20260929.md, D = docs/research/VLA_BEYOND_WORLDVLN_20260930.md.
Status: DONE (in the model), BUILT (code ready, not run), PLANNED, IDEA (not built), TESTED-NO, REJECTED.

| Idea | From | Evidence behind it | Status |
|---|---|---|---|
| Official OpenVLA-UAV format (4-D local actions, state in prompt) | A E0 | 0 mismatches vs official loop | DONE |
| Wider action range (0.1/99.9%) | A E2 | official cap costs 4 m+ on 10% of flights | DONE |
| 8-move chunks (K = 8) | A E3 | fewer calls; K = 16 untested | DONE |
| Both instruction wordings + mirror copies | A E4 | free data | DONE |
| Add simulator flights | A E6 | nDTW 0.128 -> 0.333 | DONE |
| Recipe LR 5e-4, one warm-up, long training | C | 3% of recipe updates, LR 10x low | DONE (D171 running) |
| All 54 real shards | A E10 | under-trained | DONE (D171 running) |
| Token accuracy in validation; judge by flights | C | loss flat while nDTW tripled | DONE |
| Stop-confirm (re-ask after a "stop" answer) | REPORT 12 | 0.316 vs 0.333 | TESTED-NO |
| Re-ask after 2-4 of the 8 moves | B #1 | BID, DreamFly | BUILT (--execute-steps) |
| Contrastive decoding vs gray photo | B #2 | PCD: OpenVLA up to +50% | BUILT (--contrast-alpha) |
| Mirror-contrastive decoding (mirror photo + swap left/right) | B paper idea 2 | new, no segmenter needed | IDEA |
| **Progress tokens** (remaining displacement before the moves) | D #1 | ECoT +28%; CognitiveDrone 59.6 -> 77.2% | PLANNED (build now) |
| **Goal memory** in the prompt | D #2 | CronusVLA, DreamFly memory | PLANNED (build now) |
| History frames (first + current) | A E5 | Pi-0-UAV style | IDEA (goal memory is the cheap form) |
| Lite world-action loss (predict next-frame embedding) | B #7 | WorldVLN Approach 45 -> 98%, Land 46 -> 93% | IDEA |
| Self-grounding labels from the base VLM (target left/right/near) | B #6 | ECoT-style; Qwen3-VL 24/24 on target probe | IDEA |
| "Blind means unsure" loss | B #5 | none (new) | IDEA |
| Sim RL: GRPO, reward end + nDTW | A E9, B #4 | WorldVLN +10 pts after SFT | BUILT |
| Expert flight in each RL group (EG-GRPO) | B #4 | 18.7 -> 43.1% on UAV-Flow data | BUILT (--sft-weight) |
| Skip all-equal groups (dynamic sampling) | B | RIPT-VLA, SimpleVLA-RL | BUILT |
| Retry failed tasks (RecoverFly) | B | +3-8 pts TravelUAV | BUILT (select_tasks.py) |
| Preference pairs (DPO) | B | AeroDPO; RL4VLA: DPO weakest | BUILT (--objective dpo) |
| RL fixes: multi-round driver, per-move reward, clip 0.05 | D section 5 | WorldVLN code: 3,000 iterations, clip 0.02 | PLANNED (build now) |
| Rejection-sampling fine-tune (SFT on best rollouts) | B #3 | weaker cousin of RL | IDEA (reuses RL rollouts, cheap) |
| Sample-and-verify at test time | D #4 | RoboMonkey +9% | IDEA (later) |
| FAST-style compressed action tokens | A E7 | - | IDEA (parked) |
| Offline RL on recorded frames | A E8 | - | REJECTED (sim RL instead) |
| LoRA rank 64 / bigger base | C #4 | slower learning, not a floor | IDEA (only if the rest fails) |
| Video imagination (WorldVLN, ImagineUAV) | B, D | does not fit 8 GB | REJECTED |
| World-model RL | B | imagined rollouts get exploited | REJECTED |
| Flow-matching action head | B | breaks llama.cpp weight sharing | REJECTED |
| Hand-parse numbers from the instruction | D | not a VLA result | REJECTED |

Paper angles collected so far (B, D): camera-reliance audit of aerial VLAs; progress tokens as
"coordinate-space imagination" instead of video; mirror-contrastive decoding; one backbone for
VLM + VLA; cheap RL in a teleport simulator; benchmark flaws (tilted starts, Pass has no distance).

