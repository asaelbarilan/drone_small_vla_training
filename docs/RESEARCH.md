# RESEARCH - documentation of our research (the data the paper will be written from)

**This is NOT the paper.** It is the documentation of our research: every number, run, analysis and decision, with its source, so the paper can be written from it later.

## Required contents (this file must hold all of these)

1. Problem and claim - the task, the one-sentence claim, the contributions list, why it matters.
2. Related work - table of prior models (size, data, open weights, published scores), their
   exact numbers with sources, methods others tried and dropped.
3. Benchmark - what it is, test-set size and split per category, exact metrics and success
   rule, how scores are averaged (per task or per class).
4. Evaluation protocol - train / validation / test split and what each is used for; any
   test-set use during development (disclosed); a harness correctness check.
5. Benchmark flaws - broken or ambiguous test cases, metric blind spots.
6. Data - sources, sizes, split method, filtering and exclusions, preprocessing, augmentation,
   licences.
7. Model - base model, size, fine-tuning method and config, input / output format, action
   (output) representation.
8. Inference - decoding settings, any run-time rules, latency and memory.
9. Training - every phase (start point, data, schedule, learning rate, warmup, steps, batch),
   why each schedule was chosen, curves (loss, accuracy, learning rate), final checkpoint choice.
10. Compute - hardware, time per step, total GPU hours, total cost.
11. Main results - ours vs baselines on the same harness, plus published numbers.
12. Per-category results for every variant.
13. Statistics - confidence intervals, paired significance tests.
14. Ablations - remove each new component one at a time, data mixes, controls for size or
    step count.
15. Sanity controls - does the model use each input (blank or swapped image, text only).
16. Error analysis - failure modes per category, with numbers.
17. Qualitative examples - plots, frames, model outputs over an episode.
18. Post-training (RL, DAgger, ...) - method, why chosen, settings, before / after on
    validation, cost, what broke and how it was fixed.
19. Deployment - target hardware, memory, latency, quantisation.
20. Limitations - simulator-only, weak categories, ambiguities.
21. Ethics and licences - model, data, use restrictions.
22. Reproducibility - code links, exact commits, seeds, data lists, configs, where every
    artefact is stored.
23. Release - model card, weights, inference example, requirements.
24. Decision log - every decision with date, reason, evidence (`docs/LOG.md`, D-numbers).

Below: Part 1 is the same list made specific to this project. Part 2 (below it, to be filled) holds the data
itself, one section per item, each with its number, its source file and the D-entry in
`docs/LOG.md`. Rule 1 applies: decisions on validation, the test set flown once at the end.

## Research history (the story so far, 2026-08 to 2026-10-04)

Sources: decisions D-1..D151 in the old combined repo
(`drone_architectures_paper/uav_vla_aws_pilot_20260916/docs/RESEARCH_LOG.md`,
`docs/VLA_DRONE_TRAINING.md`), D152 onward in `docs/LOG.md`.

0. **Origin - the architecture paper.** We compared drone architectures with Gemma as the
   VLM planner. Gemma could find and reach targets but needed about 1.7 s per decision while
   the drone kept flying; no Gemma configuration completed a mission, and every architecture
   number came from hand-written policies, not a learned network (D-27, D-40). That is why we
   decided to train a VLA: a model that outputs the drone's moves itself.
1. **First Qwen steps (D-115 to D-134).** Qwen3-VL-4B passed an image gate (6/6 object
   localisation on new frames). First QLoRA training on the local RTX 4060 (8 GB) on 16
   examples from our own simulator; the action convention was fixed (forward-right-down,
   clockwise yaw). It still missed 13/16 exact training answers, and the data had a hidden
   cheat (the goal bearing was computed from the simulator's goal position).
2. **SmolVLM vs Qwen (D135-D139).** At the user's request, SmolVLM 256M and 500M were trained
   on the same data and settings: capacity vs training length, then more balanced data. The
   small models often failed even to produce a valid action.
3. **OpenFly (D140-D151).** Smol-256, Smol-500 and Qwen trained on official OpenFly data and
   compared with the released OpenFly model. The metric failed: the released model was
   charged for unparsed output, the panel could not separate models, and a no-vision oracle
   beat every model. Our normalisation was also wrong (D148). The OpenFly simulator would not
   render on our hardware (D151), so OpenFly was stopped.
4. **UAV-Flow (D152-D157).** Switched to UAV-Flow (real flights + a simulator benchmark);
   read the four UAV-Flow papers in full; goal set to better results on less data. SmolVLM-256M
   endpoint pilot (D153): first model whose score depended on the camera (better than blinded
   on 62/83), but it lost to a trivial instruction lookup. Three harness bugs fixed in our
   scoring code. Rebuilt as one example per step (D154, 34,018 steps); Qwen shard runs.
5. **Official format (D158-D161).** Ported the official OpenVLA-UAV format (4-D moves, K = 8,
   256 bins, 0.1/99.9 ranges). The 10-shard adapter was the first to beat the text-only
   baselines (3.07 m vs 4.21-4.53 m), but gray or swapped photos changed almost nothing:
   the gain came from state + instruction, not the image.
6. **Closed loop in the simulator (D162-D166).** The local run failed and the Linux AWS run
   was invalid; a Windows AWS GPU box worked. OpenVLA-UAV nDTW 0.395; ours 0.128 (real only),
   0.336 / 50% success after adding simulator data. Harness audit: state chaining exact,
   evaluator flies our poses, sign conventions agree.
7. **Stopping early (D167-D169).** Land and Pass flights stopped about halfway (stop chunk).
   A stop-confirm rule was tried; the Land training data was checked and fine. Training with
   504 held-out simulator flights for validation and early stopping.
8. **The big run to 18,000 (D170-D171).** All 54 real shards + simulator, one A10G:
   adapter_s18000, 52% success / nDTW 0.367.
9. **Progress tokens to 31,000 (D172-D179).** Research on what the papers do not solve (goal:
   beat WorldVLN 79.1%). Progress line added ("Left" = what remains to the flight end);
   steady lr then decay to 31,000: 61% / 0.383.
10. **Goal memory + deadband (D180-D182).** Goal memory keeps the first line's goal; it broke
    Rotate/Shift (small overshoots kept the drone moving), fixed by the deadband (1 m, 5 deg):
    69% / 0.449, above OpenVLA-UAV (67% / 0.395) at about half the size.
11. **Release and repo split (D183).** Package for paper, GitHub and Hugging Face (not uploaded);
    training machine closed; VLA and architecture work split into two repositories.
12. **Post-training methods (D184-D185).** Research on methods cheaper than GRPO; GRPO, PPO and
    DAgger built and switchable; PPO value loss blew up (3.5 -> 211) and was fixed by
    layer-normalising the value head's input.
13. **Rule 1 (D187).** The 273 test tasks are frozen until the final run; the 100 already used
    are disclosed. Decisions now use 307 validation tasks from the 504 held-out flights.
14. **DAgger round 1 (D188).** 50 practice (train) flights relabelled, 300 updates. On 73
    Land + Pass validation tasks: 61.6% -> 74.0% (Land 26 -> 36 of 41; 11 fixed, 2 broken,
    p = 0.02). Pass did not move: the first distance guess is always about 10 m, so long
    Pass tasks stop short. Turn is the biggest gap to OpenVLA-UAV (3/10 vs 10/10).

Next: fix Pass; check Turn on validation; more DAgger; camera controls; one final run on the
273 test tasks; measure the 8 GB claim.

## Part 1 - the list

### A. Problem and claim
1. The task: language-instructed drone flight from one forward camera (UAV-Flow).
2. The claim in one sentence (small 4B LoRA VLA, runs next to a VLM on 8 GB, beats a 7B baseline).
3. Contributions list (progress line, goal memory + deadband, training recipe, RL/DAgger result).
4. Why small matters (8 GB, shared base with the VLM, adapter switched per request).

### B. Related work and baselines
5. Table of drone VLAs: name, size, data, open weights, published UAV-Flow success.
6. Published numbers: WorldVLN 79.1%, ImagineUAV 70.9%, OpenVLA-UAV 65.6% (+ sources).
7. WorldVLN per-class table (paper) for the per-class comparison.
8. What other papers use for action space, chunk length, number of updates, RL method.
9. Methods tried and dropped by the field (and by us), with reasons.

### C. Benchmark and evaluation protocol
10. UAV-Flow-Eval description: simulator, teleporting evaluator, no physics, 10 classes.
11. Test set: 273 tasks, tasks per class (Land 54, Shift 49, Approach 42, Pass 40, ...).
12. Metrics: success rule (3 m and 10 deg, WorldVLN's) and nDTW (official metric.py), exact code.
13. Task-weighted vs class-averaged score, and which one each paper reports.
14. Harness audit: state chaining, evaluator flies our poses, sign conventions.
15. Benchmark flaws: 8/273 tilted starts, Pass has no distance, Surround scored by end point.
16. Evaluation rule (rule 1): train / validation / test table, and the disclosure that 100 test
    tasks were used during development.

### D. Data
17. UAV-Flow real: shards, flights, train/val split method (by site), instruction sweep.
18. UAV-Flow-Sim: flights, the 168 near-test flights excluded, the 504 held out for validation.
19. Validation task set: 307 tasks from the 504 held-out flights, per class counts, how built.
20. Practice (train) tasks for RL/DAgger: 200 tasks, how built.
21. Preprocessing: frames, 224x224, state format, mirror augmentation, both instruction wordings.
22. Action representation: 4-D drone frame, K = 8, 256 bins, 0.1/99.9 percentile ranges.
23. Dataset licence status (UAV-Flow declares none: research use).

### E. Model
24. Base model (Qwen3-VL-4B-Instruct), precision, LoRA config (r, alpha, dropout, targets).
25. Prompt format and answer format (progress line 21 tokens + 32 action tokens).
26. Progress line definition (remaining displacement to the flight end, drone frame, yaw summed).
27. Inference rules: goal memory, deadband (1 m, 5 deg), execute steps, stop rule.
28. Parameter count, adapter size, memory footprint, latency per call.

### F. Training
29. Every phase: init, data, schedule, peak lr, warmup, updates, batch (phases.json).
30. Schedule decisions and why (cosine, warmup-stable-decay, plateau rule, decay).
31. Training curves: loss, held-out accuracy (real, sim, per class), lr.
32. Held-out accuracy at the end of each phase.
33. Hardware, time per update, total GPU hours, total cost.
34. Checkpoints kept and which one is final.

### G. Main results
35. Main table: ours vs OpenVLA-UAV (and published numbers), success + nDTW.
36. Per-class table for every variant.
37. Final run on all 273 test tasks (once, at the end).
38. Confidence intervals (bootstrap over tasks) and paired significance tests.
39. Steps / path length per flight, failure modes per class.

### H. Ablations
40. No progress line vs progress line (18,000 vs 31,000; plus a matched-updates ablation).
41. Progress line alone vs + goal memory vs + deadband.
42. Camera controls on the final model: gray photo, swapped photo, text only.
43. Chunk length / execute steps, if tested.
44. Real only vs real + sim data.

### I. Analyses
45. Goal-memory error analysis (fresh line vs memory line accuracy, arrival overshoot).
46. Why flights stop early (stop chunk, per class flown/ref length).
47. Pass analysis: first distance guess always ~10 m, no correlation with true length.
48. Does the model use the camera (open-loop gray/swap results, mirror probe).
49. Qualitative examples: flight plots, first frames, progress lines along a flight.

### J. Post-training (RL / DAgger)
50. Methods compared and chosen (GRPO, PPO, DAgger) and why (cost per gain).
51. DAgger: tasks, rows, updates, lr, before/after on validation, paired test.
52. PPO: value-head design, the layer-norm fix, stability numbers.
53. GRPO trial: what it showed (pipeline check, flat groups at temperature 1.0).
54. Cost of every RL run.

### K. Deployment
55. 8 GB measurement: llama.cpp / GGUF, memory and latency next to the VLM.
56. Real-drone path: how the progress line and goal memory map to real odometry.

### L. Limitations and ethics
57. Simulator only, no physics, no real flight.
58. Weak classes (Turn, Pass, Surround) and the ambiguity of Pass.
59. Test-set use during development (disclosure).
60. Licences (base Apache-2.0, data research-only).

### M. Release and reproducibility
61. Hugging Face: adapter, model card, inference example, action ranges.
62. GitHub: code, README "reproduce in 5 steps", requirements, scripts per phase.
63. Exact commits, seeds, data lists (held-out, excluded, task lists).
64. Where every artefact is stored (S3 prefixes, local folders), bucket kept out of the repo.

## Part 2 - the data

Status as of 2026-10-04. Item numbers refer to Part 1. **MISSING** = not measured yet.
D-numbers refer to `docs/LOG.md` (D158+) or the old `RESEARCH_LOG.md` (up to D151).

### A. Problem and claim

**1. Task.** Language-instructed drone flight on UAV-Flow. Input: one forward RGB photo
(224x224), the drone's state relative to the flight start (x, y, z in metres, yaw in
degrees) and an instruction. Output: 8 drone-frame moves (dx, dy, dz, dyaw).

**2. Claim (current numbers).** A 4B LoRA VLA (Qwen3-VL-4B, rank 32, adapter about 322 MB)
with a progress line, goal memory and a deadband gets 69% success and nDTW 0.449 on 100
UAV-Flow-Eval tasks. The 7B OpenVLA-UAV gets 67% and 0.395 on the same harness (D182). The
100 tasks are a development subset, see item 16. Not yet claimed: full 273-task result, 8 GB
deployment.

**3. Contributions (draft).**
- Progress line: the model writes what remains to the flight's end before its moves (D175).
- Goal memory + arrival deadband at inference (D175, D182).
- Training recipe: real + simulator data, a long single-GPU run, warmup-stable-decay with a
  plateau rule on token accuracy (D169-D179).
- Cheap post-training: DAgger with geometric relabelling from recorded flights. On validation
  Land + Pass it goes from 61.6% to 74.0% (D188).
- Analyses: harness audit, why flights stop early, why goal memory breaks numeric tasks,
  and why Pass is ambiguous.

**4. Why small.** The adapter shares the base weights with the VLM and is switched per
request. Target: one 8 GB GPU with llama.cpp. MISSING: measurement (item 55).

### B. Related work and baselines

**5. Drone VLAs** (`docs/research/SMALL_DRONE_VLA_WEIGHTS_20261002.md`):

| Model | Size | Open weights | UAV-Flow result |
|---|---|---|---|
| WorldVLN | 8B backbone + action decoder | yes (73 GB) | 79.1% (paper) |
| ImagineUAV | 1.3B | - | 70.9% (paper) |
| OpenVLA-UAV | 7B | yes | 65.6% (paper); 67% / nDTW 0.395 on our 100 tasks |
| LaZeAsh qwen2.5-vl-uav-flow, smolvla, pi05 | 0.45-3B | yes | none published (community) |
| Exp2VLA-Pi05 | ~3B | yes | own Isaac Lab benchmark only |
| AeroDPO 2B, LiteVLA-H 256M, GRaD-Nav++, VLA-AN | small | no | - |
| OpenFly agent 7B | 7B | yes | OpenFly benchmark: 34.3% seen / 22.6% unseen (v7) |

**6. Published numbers.** WorldVLN 79.1% (arXiv 2605.15964). ImagineUAV 70.9% (2606.01205).
OpenVLA-UAV 65.6% (UAV-Flow, 2505.15725). All are task-weighted over the 273 tasks.

**7. WorldVLN per class** (paper Table 1; reproduces 79.1% = 216/273 when weighted):
Land 92.6%, Shift 85.7%, Approach 97.6%, Pass 40.0%, Ascend/Descend 94.7%, Turn 60.0%,
Move 100%, Rotate 46.7%, Retreat 91.7%, Surround 58.3%.

**8. What others use.**
- OpenVLA-UAV: 4-D moves, K = 1, 256 bins, trained to more than 95% train token
  accuracy, about 200k steps.
- WorldVLN: RL with an expert replay ratio of 12:1, clip 0.02-0.05.
- ImagineUAV: predicts the future view (video diffusion); without it, success falls from
  70.9% to 40.2%.
- Sources: `docs/research/VLA_AFTER_SFT_20260925.md` and
  `docs/research/VLA_BEYOND_WORLDVLN_20260930.md`.

**9. Tried and dropped.**

| What | Where | Why dropped |
|---|---|---|
| Gemma as VLM planner | architecture paper | no configuration finished a mission; 1.7 s per decision |
| Own-simulator data | D-126 | hidden goal-bearing cheat |
| SmolVLM 256M / 500M | D135-D139, D153 | invalid actions; lost to an instruction lookup |
| OpenFly | D140-D151 | broken metric; no-vision oracle won; simulator would not render |
| Stop-confirm rule | D168 | nDTW 0.333 -> 0.316; the model really believes it is done |
| Loss as a stop gauge | D169 | replaced by token accuracy |
| GRPO as main RL | D180, D184 | half the groups flat at temperature 1.0; DAgger and PPO cheaper |

### C. Benchmark and evaluation protocol

**10. UAV-Flow-Eval.**
- Unreal simulator (DowntownWest town), official `batch_run_act_all.py`, unmodified.
- The evaluator teleports the drone to each returned pose: no flight physics.
- A flight ends after 10 near-still moves or 100 moves.
- Runs on Windows (AWS g6.xlarge L4), driven by SSM; the Linux build's camera was frozen
  (D163).
- Colours are BGR shown as RGB in the official evaluator too; left as is.

**11. Test set.** 273 tasks: Land 54, Shift 49, Approach 42, Pass 40, Ascend/Descend 19,
Turn 15, Move 15, Rotate 15, Retreat 12, Surround 12.

**12. Metrics.**
- Success: the flight ends within 3 m and 10 deg of the reference end (WorldVLN's rule).
- nDTW: the official metric.py, with stride 2 for Turn/Move and 5 otherwise, positions
  zeroed for Turn/Rotate, and the reference capped at 20 samples.
- Code: `scripts/score_uav_flow_sim.py` for the test tasks, `scripts/score_tasks.py` for
  validation, `scripts/rl/reward_uav_flow.py`.
- Our rule reproduces OpenVLA-UAV at 67% on 100 tasks, against 65.6% published.

**13. Averaging.** The papers average over tasks (task-weighted), not over classes. Our 100
tasks are 10 per class. Task-weighted estimate from our per-class rates: about 68% (D182).

**14. Harness audit** (D167):
- state chaining is exact (567/567 calls);
- the evaluator flies exactly our poses in 99/100 flights;
- left/right conventions agree between real and simulator data;
- the camera shows the targets, and first frames differ per task (D163).

**15. Benchmark flaws.**
- 8/273 tasks have a tilted `initial_pos`, which bends the reference path (2 are in our 100).
- Pass instructions give no distance: the end is 7.1-14.7 m past the object.
- Surround (orbit) is scored by end point only, so every model gets about 0 success.
- 238/273 test instructions occur verbatim in the simulator training data (templated
  wording).
- 54 test tasks have a simulator flight starting within 0.5 m; those 168 flights are
  excluded from training (D165).
- The training data cannot be replayed live for object tasks. Each UAV-Flow-Sim flight
  has only photos and `log.json` (drone poses, `raw_logs` / `preprocessed_logs`,
  `instruction`, `instruction_unified`). Where the spawned person, dog or car stood is not
  saved, although the test tasks have it (`target_pos`, `obj_id`). So 733 Turn ("turn to
  the dog") and 165 Surround training flights cannot become practice tasks, and DAgger or
  RL cannot practise those classes from the training data. It is fair (same for every
  team) but incomplete for closed-loop post-training (checked 2026-10-04).
  Fix (D189): the object is reconstructed from the recording - Surround from a circle fit
  (radii equal the instructions), Turn from a Qwen3-VL box on the last photo - and checked
  by rendering; 492 Turn + 165 Surround training tasks and 23 + 55 validation tasks kept.

**16. Evaluation rule (rule 1, D187).**

| Set | Made from | Used for |
|---|---|---|
| Train | UAV-Flow real + simulator training flights; 200 practice tasks built from them | learning only (supervised, DAgger, RL) |
| Validation | 504 held-out UAV-Flow-Sim flights -> 307 flyable tasks | every decision and error analysis since D187 |
| Test | the 273 UAV-Flow-Eval tasks | flown once at the end |

Disclosure: 100 test tasks (first 10 per class) were used for development decisions before
2026-10-03 (D163-D182).

### D. Data

**17. UAV-Flow real.** 54 shards.
- First 10 shards: 1,987 train flights, with a site split and an instruction sweep (D160).
- Shards 10-53: +21,795 flights (18,430 train, 3,365 val). The validation sites stay fixed
  (11031_1227, 11160_717, 11207_2618, 11207_2619), and these new shards have no
  instruction sweep (D170).

**18. UAV-Flow-Sim.**
- 10,109 flights, of which 168 near-test flights are excluded, leaving 9,941 (312k frames,
  median 25 frames / 5 m).
- Of these, 504 are held out for validation (56 per motion type, D169).
- The simulator data shares the town with the test set: same environment, different
  trajectories.

**19. Validation tasks.** 307 tasks from the 504 held-out flights (`build_rl_tasks.py
--validation`), at `d187/val_tasks.zip`. Per kind: Shift 58, Land 41, Ascend/Descend 36,
Pass 32, Approach/Move 25, Turn/Rotate 20, Retreat 3, other 92.

**20. Practice tasks.**
- 200 tasks, 25 per kind, built from simulator training flights (D172).
- Instructions naming spawned objects are dropped, and no Surround task is included.
- Self-check: worst end mismatch is 32 cm.

**21. Preprocessing.**
- Official OpenVLA-UAV format: every frame, the first and last frames repeated 5 times,
  and the state in the prompt (D158).
- Photos are 224x224, with mirror copies (sideways and yaw negated).
- Both instruction wordings are used (`instruction` and `instruction_unified`).
- Simulator centimetres are converted to metres (D165).
- 1,121,502 training examples including mirrors (D165, 10 shards + simulator).

**22. Actions.**
- 4-D drone-frame moves, K = 8 per answer, 256 bins per value on the last 256
  vocabulary ids.
- Range: the 0.1/99.9 percentiles (`action_stats_manifest_10shard.json`).
- The official q01/q99 range clipped fast motion: 24.5% of steps had a clipped channel
  (D158).
- 29% of simulator Land dz steps are still below the range (D168).

**23. Licence.** UAV-Flow and UAV-Flow-Sim declare no licence, so this is research use only.

### E. Model

**24. Model.**
- Base: Qwen/Qwen3-VL-4B-Instruct, bf16, unquantised.
- LoRA: r 32, alpha 32, dropout 0, on all linear layers (vision tower, projector and
  language model).

**25. Formats.**
- Prompt: `Current State: {x},{y},{z},{yaw}, What action should the uav take to
  {instruction}?` plus one photo.
- Answer: the progress line (21 tokens), then 32 action tokens.

**26. Progress line.**
- Format: `Left +FF.F,+SS.S,+UU.U,+YYY`, the remaining forward, sideways and up metres
  and the summed turn in degrees, in the drone frame, clipped to ±360.
- Labels come from the stored per-frame state.
- Check on 300 flights: the line matches the re-integrated actions to a 1 mm median in
  xy, 0.00 deg in yaw (D175).

**27. Inference rules.**
- Goal memory: the first line fixes a goal in the start frame, and later calls get the
  line recomputed from the drone's pose.
- Deadband `--memory-deadband 1.0,5`: within 1 m and 5 deg the line becomes 0.
- Server: `scripts/uav_flow_eval_server.py --chunk 8 --precision bf16 --progress
  --goal-memory --memory-deadband 1.0,5`, greedy decoding.

**28. Size and speed.**
- The adapter is about 322 MB.
- Flight test: about 2 h per 100 tasks with the progress line (71 min without it) on an L4.
- Open loop, batched: 0.35 s per call (D160).
- MISSING: latency and memory on 8 GB.

### F. Training

**29. Phases** (`release/qwen3vl4b_uavflow_vla/phases.json`):

| Phase | Start | Data | Schedule | Updates | Result |
|---|---|---|---|---|---|
| 0 | base | 10 real shards, mirror, both wordings | cosine, peak 5e-4, 3% warmup | 2,500 | nDTW 0.128 / 38% |
| 1 | phase 0 | + simulator (9,941) | cosine, 2e-4 | 2,000 | nDTW 0.336 / 50% |
| 2 | phase 1 | 504 sim flights held out | cosine, 5e-5, early stop | about 2,700 (best 2,500) | not flown |
| 3 | phase 2 best | all 54 real shards + simulator | cosine, peak 5e-4, 3% of 26,000 warmup | 18,000 | nDTW 0.367 / 52% |
| 4 | checkpoint 18,000 | + progress line | steady 1.142e-4, plateau rule | 18,001-28,000 | - |
| 5 | checkpoint 28,000 | same | cosine 1.142e-4 -> 0 | 28,001-31,000 | final: 61%; with memory + deadband 69% / 0.449 |

All phases use batch 32 (8 x 4 accumulation). Total: about 38,700 updates.

**29b. All training so far, in words (and what is planned).**

The model is always the same: Qwen3-VL-4B-Instruct with one LoRA adapter (rank 32, on every
linear layer, vision included). Input: one photo, the drone's state and the instruction.
Everything it learns is written as text tokens after the input.

What the answer looks like has grown over time:
- phases 0-3: 8 moves only (32 action tokens);
- phases 4-5 and DAgger: the progress line `Left +FF.F,+SS.S,+UU.U,+YYY` (what is left to the
  flight's end), then the 8 moves;
- D193: the target's box, then the box after the moves, then the line, then the moves.

The trained runs:
1. **Phase 0 (D159-D160).** 10 real shards, mirror copies, both instruction wordings.
   2,500 updates, learning rate 5e-4. Moves only.
2. **Phase 1 (D165).** Adds 9,941 simulator flights. 2,000 updates, learning rate 2e-4.
3. **Phase 2 (D169).** 504 simulator flights held out for validation. About 2,700 updates at
   5e-5, early stop.
4. **Phase 3 (D171).** All 54 real shards plus the simulator flights. 18,000 updates,
   learning rate 5e-4, cosine. Result: adapter s18000, 52 % on test.
5. **Phase 4 (D176-D177).** The progress line is added before the moves. Steady learning
   rate 1.142e-4 to update 28,000, stopped when accuracy went flat.
6. **Phase 5 (D179).** Decay to 0, ending at update 31,000. Result: adapter s31000, the main
   model (61 % on test).
7. **DAgger round 1 (D188).** The model flew 50 Land/Pass practice tasks. Every state it
   visited was relabelled from the recorded flight (true line + 8 moves back onto the path):
   394 examples. 300 updates at 2e-5, mixed with the expert flights of 200 practice tasks.
   Validation Land + Pass 61.6 -> 74.0 %.
8. **DAgger round 2 (D190).** 150 new practice tasks: 50 Land, 75 Turn and 25 Surround (with
   the dog / person placed, D189). 934 new examples plus round 1's. 300 updates. Validation
   67 -> 69 %, not significant; Turn unchanged.
9. **Box-first + imagination (D193-D194).** Starts from round 2.
   - Answer: target box, then the box after the 8 moves ("imagination": where the target will
     be), then the line, then the moves.
   - Boxes labelled by the base Qwen on 81k photos; boxed rows repeated 3x.
   - Photo augmentation (red/blue swap, 224 px), so photos look like what the evaluator sends.
   - DAgger examples repeated 20x.
   - 4,000 updates at 1e-4, cosine.
   - Result: moves kept, but the box is the same "average" box on every photo, and the line
     ignores it (D195).

Not trained, used only when flying (server rules):
- **Goal memory (D175):** the first line fixes a goal; later calls get the line recomputed
  from the drone's position.
- **Deadband (D182):** within 1 m and 5 deg the line becomes 0, so the drone stops.
- **Action-token-only greedy decoding (D191).**
- **Box phase (D193, `--box`):** the model writes its boxes first, then the line and moves.

Planned (D196, ST4VLA order: grounding first, then actions on top):
- **Stage 1, about 2,500 updates.** A fresh LoRA from the base model learns only the boxes:
  the target box and the imagined box after the moves. Labels are balanced so left, centre and
  right targets are equally common.
- **Stage 2, about 15,000-20,000 updates.** The same LoRA learns the full answer: box,
  imagined box, line, moves. It keeps mirror copies, both wordings, photo augmentation and the
  DAgger examples. The vision part trains about 10x slower, so the flying cannot erase the
  grounding.
- **When flying:** goal memory and the deadband stay as they are. The difference is that the
  first line, which goal memory locks in, should now come from a real box.

What is new compared with what we have done: the order (boxes alone first), the balanced boxes,
the slower vision part, and a check that lets the model write boxes during validation.
Everything else is reused.

**30. Schedule decisions.**
- Loss is a poor gauge, so stopping uses token accuracy (D169, D170).
- The run did not stop at a fixed count while accuracy still rose (D176).
- Progress tokens were merged into the run instead of a separate fine-tune (D176). This
  gave up the clean with/without comparison.
- The plateau rule also counts simulator accuracy (D177).
- The run ended with a decay (D179).
- Learning rate: the first high-rate bumps (D165, D169) led to 5e-5 in phase 2, then back
  to the recipe's 5e-4 for the long run.

**31. Curves.**
- `release/qwen3vl4b_uavflow_vla/curves.png`.
- `report_long_run.json`: every update's loss and every validation.
- wandb project "vla training": runs official_k8_10shard, official_k8_realsim,
  official_k8_d169b, official_k8_d171.

**32. Held-out move-token accuracy** (D179):

| Step | Real exact | Real within 1 bin | Sim exact | Sim within 1 bin | Pass | Land | Approach |
|---|---|---|---|---|---|---|---|
| 1 | 51.6% | 71.4% | 91.1% | 93.1% | 62.3% | 77.1% | 81.9% |
| 18,000 | 53.3% | 74.1% | 91.1% | 93.3% | 62.8% | 76.8% | 82.3% |
| 28,000 | 53.8% | 74.9% | 91.6% | 93.8% | 64.0% | 77.7% | 83.2% |
| 31,000 | 54.5% | 75.5% | 91.7% | 94.1% | 64.9% | 78.0% | 83.4% |

**33. Compute.**
- One NVIDIA A10G 24 GB (g5.2xlarge), about 6.2-6.8 s per update of 32 examples.
- The long run is 31,000 updates, about 56 GPU hours (computed from that speed).
- Total training cost about $125 (AWS). September usage was $56.03, covered by credits.
- Flight tests and RL ran on a g6.xlarge (L4), about $1 per hour.
- MISSING: exact total GPU hours from the bills.

**34. Checkpoints.**
- `d171/run/adapter_s2000 ... s31000` and adapter_best on S3.
- The checkpoint at 31,000 with optimizer state is in `release/lineage/`.
- Final model: adapter_s31000. Ablation: adapter_s18000. DAgger: `d188/dagger_r1_adapter/`.

### G. Main results

**35-36. 100 test tasks (development, pre-rule-1), success per class out of 10:**

| Model | Turn | Move | Shift | Rotate | Surround | A/D | Approach | Retreat | Pass | Land | Success | nDTW |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Real + sim, 2,000 updates (D166) | 4 | 7 | 9 | 3 | 0 | 9 | 7 | 10 | 0 | 1 | 50% | 0.336 |
| s18000 (D180) | 5 | 7 | 9 | 5 | 1 | 9 | 5 | 8 | 1 | 2 | 52% | 0.367 |
| s31000 + line | 4 | 8 | 9 | 8 | 1 | 9 | 6 | 9 | 1 | 6 | 61% | 0.383 |
| + goal memory | 3 | 10 | 7 | 6 | 1 | 7 | 7 | 10 | 6 | 4 | 61% | 0.409 |
| + deadband (D182) | 3 | 10 | 9 | 10 | 1 | 9 | 8 | 10 | 4 | 5 | **69%** | **0.449** |
| OpenVLA-UAV 7B | 10 | 10 | 9 | 1 | 0 | 10 | 9 | 9 | 4 | 5 | 67% | 0.395 |

Source: `release/qwen3vl4b_uavflow_vla/results.json`.
Earlier: the real-only adapter had nDTW 0.129 (D164); OpenVLA-UAV beat it on 77/100 paired
flights.

**37. All 273 test tasks.** MISSING, by design: flown once at the end (rule 1).

**38. Statistics.**
- 100 tasks give about ±5 points of noise.
- D188 paired sign test on validation: p = 0.02.
- MISSING: bootstrap confidence intervals for every row.

**39. Flight length.**
- Median flight was 34 steps (D166) and 41 with the line (D180).
- Flown vs reference length (D166 adapter): Land 3.3/10.2 m, Pass 7.9/15.7, Approach
  3.6/6.0.
- 89/100 flights ended by an all-zero chunk ("stop") (D167).

### H. Ablations

**40. Progress line.** s18000 52% / 0.367 vs s31000 + line 61% / 0.383. Confounded with
13,000 more updates and the decay. MISSING: a matched ablation with the same updates and
no line.

**41. Inference rules.** Line only 61% / 0.383; + memory 61% / 0.409; + deadband 69% /
0.449. The deadband recovered Rotate (6 -> 10), Shift (7 -> 9) and A/D (7 -> 9), and Pass
fell from 6 to 4.

**42. Camera controls.**
- Open loop, earlier adapters only. D157: gray and swapped photos gave identical results
  (p = 0.75 / 1.0).
- D160: real 3.067 m vs gray 3.115 m vs swapped 3.121 m (gray p = 0.29). Mirror probe:
  23% of answers changed.
- D153 SmolVLM: real 3.21 m vs blinded 6.47 m (p = 7.5e-6).
- MISSING: closed-loop camera controls for the final model, on validation.

**43. Chunk / re-ask.** Stop-confirm (D168): nDTW 0.333 -> 0.316, not adopted.
`--execute-steps` was built but not flown.

**44. Data.** Real only nDTW 0.128 vs real + sim 0.333 (D164, D166): the domain gap is the
main cause.

**44b. Action head vs move tokens** (D202-D204, same data, same stage-1 start): an OpenVLA-OFT
style continuous head (4-layer MLP on 8 slot tokens, L1) flew 12 % (never stopped: it cannot
output an exact zero; its near-zero bias grew with training). The same recipe with 32 bin tokens
matched the old model's move accuracy after 2,000 updates and holds still 20/20. For this task,
where stopping exactly is required, bin tokens beat the continuous head - opposite to OFT's
LIBERO result. Possible fixes not tried: a zero class / deadzone, more updates of the head alone.

### I. Analyses

**45. Goal-memory error analysis** (D182, 615 calls):
- The memory line is more accurate than the fresh line: median 0.08 m vs 0.43 m, and
  0.32 vs 0.86 deg.
- Failures happen at arrival. Overshoot lines ("-2 deg") were never in training, so the
  drone kept moving.
- The deadband raised offline stop agreement: Shift 74 -> 97%, Rotate 79 -> 95%,
  A/D 60 -> 99%.

**46. Early stop** (D167, D168):
- The model returns an all-zero chunk, and when re-asked it keeps saying stop: it believes
  it is done.
- The Land training data is fine (98% of simulator Land flights descend), but the descent
  comes at the end.

**47. Pass** (D188, validation):
- All 14 failures stop short, 2.9-10 m before the end. Successes are 7.8-13 m long,
  failures mostly 13-20 m.
- The first progress line is about 10 m on every task, with a correlation of 0.04 with the
  true length.

**48. Hard classes** (D174, from the task files):
- Rotate is pure numeric yaw of 15-180 deg.
- Turn ends exactly at the bearing to the object.
- Pass ends 7.1-14.7 m past the object.
- The early model ran every Turn for 34 steps whatever the angle.

**49. Qualitative examples.** Flight plots (`*_2d.png`) and first frames for every flown
task are on S3 and in `D:/drone_vla_pilot/runs/`. MISSING: chosen figures.

### J. Post-training

**50. Method choice** (D184, `docs/research/VLA_CHEAPER_THAN_GRPO_20261003.md`):
- DAgger gives an exact label per call, which is cheapest.
- PPO gives per-call advantages from a value head.
- GRPO gives one reward per flight.
- All three are switchable in `rl_loop.ps1 -Method`.

**51. DAgger round 1** (D188):
- 50 practice tasks (25 Land, 25 Pass) flown greedily, giving 394 relabelled rows.
- 300 updates at lr 2e-5 from adapter_s31000; loss 0.74 -> 0.51.
- Validation, 73 tasks: 61.6% -> 74.0% (Land 26 -> 36/41, Pass 19 -> 18/32), nDTW
  0.287 -> 0.305.
- 11 tasks fixed, 2 broken, p = 0.02.

**51b. DAgger round 2 plan (D190, user-approved 2026-10-04, about $10):**

| Step | What happens | Tasks | From which set |
|---|---|---|---|
| 1. Before | measure the current (round-1) model, no learning | 100 | validation |
| 2. DAgger round 2 | the model flies, gets corrected, learns | 150 (50 Land, 75 Turn, 25 Surround) | training |
| 3. After | measure the new model, no learning | the same 100 | validation |

- The 100 validation tasks cover every kind: Land 15, Pass 15, Shift 15, A/D 10, Approach/Move
  10, Rotate 10, Surround 10, Turn 12, Retreat 3. A round is rejected if any kind gets worse.
- Teacher = the recorded expert flight (dagger_relabel.py). The update uses round 2's rows plus
  round 1's (DAgger aggregates) and the expert anchor flights of every kind, 300 updates at lr 2e-5.
- Checked: no flight is in both sets; every check-set flight is a held-out validation flight.

**51c. DAgger round 2 result (D190):** 100 validation tasks, 67% -> 69% (3 fixed, 1 broken, p = 0.63):
Land 13 -> 15/15, Pass 8 -> 9/15, Rotate 10 -> 9/10, Turn 4/12 and Surround 2/10 unchanged, others equal.
No clear gain; Turn did not learn from 75 Turn practice tasks.

**51d. Staged retrain plan (D196-D198, current).** D193 taught the box FORMAT but not grounding:
on 60 held-out simulator photos its written box has x correlation -0.02 with the target and a
centre error of 265, worse than an "always the centre" box (251). ST4VLA (2602.10109) shows
action-only training erases grounding and co-training afterwards only partly restores it; the fix
is grounding first, then actions. Our base Qwen3-VL-4B already finds our targets, so stage 1 is
short and uses our 81k labels (no extra labelling).

| Step | What | Data / mix | Size / cost | Gate |
|---|---|---|---|---|
| 0 | code + checks (box_eval.py; trainer --grounding-only, --grounding-mix, --sim-share, --vision-lr-mult, --box-val) | - | done, $0 | the check flags D193 as failing - passed |
| 1 | fresh LoRA on the base model, answer = box now + box after the moves, no moves | 81k labelled photos, balanced left / centre / right / empty (30/30/30/10) | ~1,000 updates, ~2 h incl. setup, ~$3 | written boxes follow the target: x correlation clearly > 0 and centre error below the always-centre baseline |
| 2 | continue that LoRA: box + next box + Left line + 8 moves | 1 box-only example per 10 flight examples; flight examples half simulator / half real; DAgger rows; mirror; colour + 224 px augmentation; vision LoRA at 0.1x learning rate | ~8,000 updates, ~20 h, ~$20 | every 1,000 updates: move accuracy toward 54 % real / 91.7 % sim and box checks not falling |
| 3 | validation flights, fixed server (D191) with --box, goal memory + deadband | 100-task validation check set | ~2 h, ~$3 | Turn / Pass up, other kinds not down |
| 4 | DAgger rounds with the new model | practice tasks incl. Turn / Surround | ~$6 per round | validation again |

**51e. Staged retrain results (D199-D205, 2026-10-06 to 2026-10-08).**

Step 1 (D199, boxes first, fresh LoRA, 1,000 updates, ~$3) - written-box checks on 120 balanced
held-out simulator photos (always-centre baseline: error 265):

| Update | box centre error | IoU | x correlation | next-box error |
|---|---|---|---|---|
| 1 (base Qwen) | 117 | 0.29 | 0.85 | - |
| 250 | 61 | 0.61 | 0.86 | 118 |
| 1000 (final) | 55 | 0.66 | 0.86 | 127 |

Grounding restored (D193 had x correlation -0.02, error 265).

Step 2, first try (D202, continuous action head, 8,000 updates, ~$25): offline it looked fine
(sim 76 % exact / 90 % within one bin), but it flew **12 %** on the 100 validation tasks (D203)
vs 69 % for D190 - every flight ran the full 104 steps. Diagnosis (D203, free, local): the head
itself could not output "still" or "no turn" - 4 cm and +0.6 deg per step when the truth was 0 -
so goal memory never reached the deadband and the drone never stopped. The bias grew with
training (still rows passing the evaluator's still rule: 13/20 at update 2,000, 0/20 from 4,000
on), so it was not under-training. Bin tokens have an exact zero bin. See 44b.

Step 2, second try (D204, move tokens, otherwise identical): a free local check of each 500-update
copy, the way the server answers (scripts/diag/diag_moves.py, 60 held-out real rows):

| Update | still rule | straight yaw bias | turning yaw (true -4.6) | real / sim move acc | box error / x corr |
|---|---|---|---|---|---|
| 500 | 20/20 | -0.16 deg | -0.4 deg | - | - |
| 1000 | 20/20 | -0.09 deg | -3.2 deg | 52.0 / 89.0 % | 65 / 0.83 |
| 2000 | 20/20 | +0.03 deg | -4.1 deg | 52.7 / 89.8 % | 66 / 0.83 |
| 5000 | - | - | - | 54.1 / 91.3 % | 62 / 0.81 |
| old D188 flyer | 20/20 | -0.02 deg | -4.0 deg | (s31000: 54.5 / 91.7) | - |

Move accuracy reached the old model's level after ~2,000 updates (the old one took 31,000); the
box correlation slowly fell (0.86 -> 0.81). Stopped early at ~5,100 updates (user, saves ~$9);
the update-5,000 checkpoint with optimizer state is on S3.

Step 3, validation flights of D204 update 2,000 (D205), PARTIAL - 89 of 100 flown:

| Kind | D204 s2000 (boxes first) | D190 (round-2 DAgger) |
|---|---|---|
| Turn (find the person) | **10/12** | 4/12 |
| Surround | **5/10** | 2/10 |
| Ascend/Descend | 10/10 | 9/10 |
| Shift | 13/15 | 13/15 |
| Approach/Move | 5/8 | 5/8 |
| Pass | 6/9 | 7/9 |
| Retreat | 1/1 | 1/1 |
| Land | 9/14 | 14/14 |
| Turn/Rotate (spin in place) | 3/10 | 9/10 |
| all (89 flown) | 62 | 64 |

The tasks that need the target in the photo improved sharply (Turn 4 -> 10 of 12, Surround 2 -> 5);
Land and spin-in-place got worse - the kinds D190's extra DAgger rounds had polished. Next: the
full 100, update 5,000 (D206), and why Land / Rotate dropped.

D205/D206 live check (2026-10-08, about 15:12 UTC; D206 PARTIAL, 85/100 completed):
D205 s2000 is complete: 67/100 success, mean nDTW 0.3345. D190 remains 69/100, 0.3795.
On exactly the same 85 completed D206 tasks: s5000 70/85, nDTW 0.3697; s2000 59/85,
0.3263; D190 61/85, 0.3953. Per-kind s5000 / s2000 / D190 successes:
Approach/Move 6/7 / 4/7 / 5/7; A/D 9/10 / 10/10 / 9/10; Land 9/12 / 8/12 / 12/12;
Pass 7/9 / 6/9 / 7/9; Shift 15/15 / 13/15 / 13/15; Surround 3/10 / 5/10 / 2/10;
Turn 12/12 / 10/12 / 4/12; Turn/Rotate 9/10 / 3/10 / 9/10. Retreat not yet in this subset.
Source: read-only AWS SSM command ed74eb49-40f3-4f3a-8224-a1a65610a554, scoring the
existing Windows flight JSONs against C:/runs/d206/tasks via reward_uav_flow.end_errors
and ndtw. Only s5000 tasks with completed *_2d.png were included; no learning or new
flights launched. This is an interim snapshot, not the final 100-task result. D190 used
the pre-D191 server, so differences are a model-plus-inference comparison, not an isolated
training effect. s5000 leads success but does not dominate nDTW or every class.
D206 remains configured to upload its results and shut down; no subsequent run was queued.

### D206 final verification (2026-10-08)

Completed all 100 validation tasks at 15:38:55 UTC (18:38:55 Israel time). Results and
transcript uploaded as d206/s5000.zip and d206/eval.log. EC2 Windows instance verified stopped
at the completion check after 16:03 UTC. No training resumed or additional jobs queued.

| Model | Success | Mean nDTW |
|---|---|---|
| D205, D204 step 2000 | 67/100 | 0.3345 |
| D206, D204 step 5000 | 81/100 | 0.3792 |
| D190, earlier DAgger round 2 | 69/100 | 0.3795 |

| Kind | s2000 | s5000 | D190 |
|---|---|---|---|
| Approach/Move | 5/10 | 8/10 | 5/10 |
| Ascend/Descend | 10/10 | 9/10 | 9/10 |
| Land | 10/15 | 12/15 | 15/15 |
| Pass | 8/15 | 10/15 | 9/15 |
| Retreat | 3/3 | 3/3 | 3/3 |
| Shift | 13/15 | 15/15 | 13/15 |
| Surround | 5/10 | 3/10 | 2/10 |
| Turn | 10/12 | 12/12 | 4/12 |
| Turn/Rotate | 3/10 | 9/10 | 9/10 |

Source: reports/d206_completion_20261008/score.json, computed locally with the unchanged
scripts/score_tasks.py and scripts/rl/reward_uav_flow.py against d190/check_tasks.zip and the
completed d206/s5000.zip archive. D205 full scores were verified by SSM in the preceding
snapshot; D190 source is D:/drone_vla_pilot/runs/d190/after_score.json.
Step 5000 leads validation success but is not uniformly better: Land trails D190 and Surround
trails step 2000. nDTW is essentially tied with D190. D190 used the older server; comparison
with it includes inference changes. These are validation results, not benchmark test results,
and 81% must not be presented as beating WorldVLN's published test score. This final result
supersedes the partial D206 snapshots above; historical observations are preserved.
Further supervised continuation or DAgger/PPO/GRPO remains a separate decision; no launch.

### D207 - authorized continuation from step 5000 to 8000 (2026-10-08)

User requested the remaining 3,000 updates, then machine shutdown; DAgger to be considered
later. Explicit choice: keep grounding-mix 0.1 (one box-only example per ten flight examples).
All other D204 data and schedule settings retained, including vision-lr-mult 0.1, sim-share
0.5, box-repeat 3, progress, next-box, mirror, photo augmentation, and prior DAgger rows.
Script: scripts/aws/resume_d207.sh; bash -n passed; staged at d207/resume_d207.sh in S3.
Resume restores step-5000 adapter and optimizer, retaining total --updates 8000, cosine lr
2e-4 / warmup 0.03. It does not restart at update zero or run the old smoke training.
Outputs go to d207/stage2; D204 checkpoints and the 81/100 evaluated model remain intact.
12-hour host shutdown deadline, 10-hour trainer wall limit, final S3 sync and shutdown on
success/failure, with explicit verification that the completed checkpoint reached 8000.
EC2 guest shutdown behavior verified stop. Startup pending: two StartInstances attempts
failed with AWS InsufficientInstanceCapacity; training instance is still stopped.
A 15-minute bounded retry is configured for the same existing instance through 18:20 UTC
2026-10-08. No different hardware or new instance is authorized. Monitor verifies launch,
completion and shutdown; no DAgger/PPO/GRPO or Windows flight evaluation is queued.
This entry describes authorized/prepared work, NOT a successful training restart.

D207 launch update (2026-10-08 16:51 UTC): AWS capacity retry succeeded; existing Linux
instance started and SSM became Online. Read-only preflight found no running trainer or
D207 service. Launched the staged script once as ubuntu under vla-d207.service; systemd
confirmed active/running (SSM command de98be14-988e-48c0-b2a6-829ff969665c).
Backup shutdown confirmed at 2026-10-09 04:51 UTC; wrapper also schedules its own 12-hour
limit and shuts down after final upload. Setup/data restoration is in progress; advancing
training past step 5000 is not yet verified. No Windows evaluation or DAgger/RL launched.

D207 training verified (2026-10-08 17:05 UTC): setup completed, checkpoint step 5000 verified,
optimizer restored with --resume, and actual update 5010 logged (loss 0.885, 8.036 s/update).
A10G utilization was 100%, memory 20,903 MiB; vla-d207.service active/running. The 1:10
box-only mix and total --updates 8000 remain unchanged. Source: SSM read-only invocation
f74b6a95-fc23-4ab9-b5fd-ddf02bdafd44; log /home/ubuntu/resume_d207.log; wandb run stage2_d207.
Backup shutdown remains 2026-10-09 04:51 UTC. User notified that updates are advancing;
continue quiet monitoring until completion/failure. No new validation flight or RL launch.

### D207 completion verified (2026-10-09)

Completed the authorized 3,000 additional updates from checkpoint 5000 to total 8000,
with optimizer resume, original cosine schedule, and grounding_mix=0.1 unchanged.
The wrapper logged D207_COMPLETE step=8000 and D207_EXIT=0 at 00:36:13 UTC;
FINAL_UPLOAD_EXIT=0 at 00:36:20 UTC (03:36 Israel). EC2 DescribeInstances
subsequently confirmed i-05f21d6200f5c1045 stopped. No restart or further job queued.

S3 verification: d207/stage2/report.json status=complete, updates=8000,
resumed_at=[5000], best_step=8000 (selected by simulator validation loss).
adapter_s8000/adapter_model.safetensors and checkpoint/adapter_model.safetensors
are each 322,279,296 bytes; checkpoint/state.pt is 645,101,597 bytes.
The wrapper validated checkpoint step 8000 before its successful final upload.
Sources: reports/d207_completion_20261009/report.json and resume_d207.log,
mirroring d207/stage2/report.json and d207/resume_d207.log.

| Held-out metric | Step 5000 | Step 6000 | Step 7000 | Step 8000 |
|---|---:|---:|---:|---:|
| Real action token accuracy | 0.5411 | 0.5414 | 0.5423 | 0.5459 |
| Simulator action token accuracy | 0.9125 | 0.9136 | 0.9142 | 0.9150 |
| Simulator validation loss | 0.50112 | 0.49076 | 0.48326 | 0.48202 |
| Box centre error (lower better) | 62.3324 | 69.1948 | 58.8017 | 57.7059 |
| Box IoU | 0.6369 | 0.6373 | 0.6441 | 0.6493 |
| Box x correlation | 0.807 | 0.747 | 0.819 | 0.819 |
| Next-box centre error (lower better) | 106.0408 | 116.8430 | 114.7708 | 113.6781 |

Current-box localization and action metrics improved at 8000 versus 5000;
next-box centre error worsened. These are held-out training metrics, not flight
success rates or evidence of statistically significant gains. The last measured
100-task flight validation result remains D206 step5000: 81/100, nDTW 0.3792.
Step8000 has not been flown. No Windows evaluation, DAgger, PPO or GRPO launched.
Completion monitor is retired; further evaluation/training requires a separate decision.


### D208 - step8000 flight validation authorized and launched (2026-10-09)

User requested flight evaluation before deciding on DAgger. Started the existing Windows
instance i-03a3c8632314bf5fc; SSM preflight found no Python/simulator jobs. Launched once
via C:/runs/d208/launch.ps1 (PID 3296; SSM e0e8ea8e-7f28-4755-85f3-1057a5973264).
Source wrapper: scripts/win_eval/run_d208.ps1, PowerShell parser check passed.
The model is d207/stage2/adapter_s8000; exact D206 code d204/vla_code.zip and
100 validation tasks d190/check_tasks.zip; flags --progress --goal-memory
--memory-deadband 1.0,5 --box, chunk 8, bf16. No benchmark test tasks or training.
Outputs: d208/s8000.zip, d208/eval.log, d208/completion.json (or failure.txt).
Backup shutdown set at about 07:39 UTC for seven hours; wrapper uploads and shuts down
on exit. The driver reports an already-scheduled timer and retains the earlier deadline.
At 07:41 UTC the launcher transcript was active and setup was still in progress;
no flight completion claimed yet. Completion monitor checks progress, scores all100 with
the existing scorer, compares D206/D205/D190, and verifies stopped state. No DAgger/RL queued.


D208 live verification 2026-10-09 07:43 UTC: server ready on port5008 with the
step8000 adapter, 100 tasks selected, evaluator attempt1 started and Collection
simulator PID1760 running. First flight not yet complete at this check.
Source: SSM 68ef6b5b-3320-4d79-a37e-c004585d5aab. Completion monitor active.

### D208 final validation (2026-10-09)

Completed 100/100 validation flights at 10:29:03 UTC (13:29 Israel); S3 completion.json
confirms complete and adapter d207/stage2/adapter_s8000. Verified 100 flight plots
and scored all 100 task/flight pairs with unchanged scripts/score_tasks.py and
scripts/rl/reward_uav_flow.py. EC2 i-03a3c8632314bf5fc verified stopped.
S3 d208/s8000.zip is 31,161,714 bytes; eval.log and completion.json also uploaded.
Sources: reports/d208_completion_20261009/score.json, completion.json, s8000.zip,
tasks.zip and extracted results; D206 comparison reports/d206_completion_20261008/score.json.

| Model | Success | Mean nDTW |
|---|---:|---:|
| D205 step2000 | 67/100 | 0.3345 |
| D206 step5000 | 81/100 | 0.3792 |
| D208 step8000 | 82/100 | 0.4255 |
| D190 older DAgger | 69/100 | 0.3795 |

| Kind | s2000 | s5000 | s8000 | D190 | s5000 nDTW | s8000 nDTW |
|---|---|---|---|---|---:|---:|
| Approach/Move | 5/10 | 8/10 | 8/10 | 5/10 | 0.3901 | 0.3897 |
| Ascend/Descend | 10/10 | 9/10 | 9/10 | 9/10 | 0.6113 | 0.6117 |
| Land | 10/15 | 12/15 | 15/15 | 15/15 | 0.3087 | 0.4005 |
| Pass | 8/15 | 10/15 | 9/15 | 9/15 | 0.4222 | 0.4498 |
| Retreat | 3/3 | 3/3 | 3/3 | 3/3 | 0.4828 | 0.5703 |
| Shift | 13/15 | 15/15 | 15/15 | 13/15 | 0.6638 | 0.6764 |
| Surround | 5/10 | 3/10 | 3/10 | 2/10 | 0.2358 | 0.2732 |
| Turn | 10/12 | 12/12 | 10/12 | 4/12 | 0.1356 | 0.2210 |
| Turn/Rotate | 3/10 | 9/10 | 10/10 | 9/10 | 0.1548 | 0.2543 |

Step8000 has the highest observed success and nDTW on this validation set.
The one-task success gain is small: paired comparison gives six newly successful
tasks and five regressions. No statistical significance claim. Land and Rotate
improve; Pass and Turn regress; Surround remains weak. D190 used an older server,
so comparison to D190 includes inference changes. This is validation, not benchmark
test performance. No DAgger/PPO/GRPO started. Completion monitor retired; next
training decision remains with the user.


### D209 - grounded DAgger round authorized and prepared (2026-10-09)

User approved the proposed failure audit and one 300-update DAgger round from D207
step8000, followed by the fixed 100-task validation. No PPO/GRPO or benchmark test.
Audit of D208's18 failures: all7 failed Surround tasks reached104 steps without a stop;
all6 failed Pass tasks stopped with heading under1.2deg and position error3.36-6.05m.
Both failed Turn tasks stopped with11.2/22.7deg error. These are observed failure
signatures, not proof of a single underlying cause. Source: failure_audit.json below.

Practice100, seed209: Surround25, Pass25, Turn20; Land/Rotate/Shift/Ascend-Descend/
Approach-Move/Retreat5 each. All source flights excluded from the504 held-out flights;
no validation trajectories or corrections enter training. Anchor replay:200 training
flights,967 box-labelled frames, zero held-out overlap. Prior D188/D190 rows retained.
The reference-path relabeler is an approximate off-path oracle, not a human expert.

Implementation: dagger_relabel.py --box-context carries student-predicted box prefixes;
train_uav_flow_vla.encode masks those tokens from loss, supervising expert progress/actions.
Grounding-only replay uses independent existing D193 base-model box labels (pseudo-labels,
not human ground truth), never student box_context rows. Grounding-mix0.1 now counts the
added DAgger flight rows too. Portable frame-key matching handles Linux/Windows stores.
Six CPU encoding tests passed; Python and PowerShell syntax checks passed. Original D204
trainer/relabeler/server match repository HEAD before changes; server code is unchanged.

Recipe: init step8000, fresh optimizer,300 updates, lr2e-5 cosine/warmup0.05,
batch4 x accum8, extra-repeat4, box-repeat3, next-box, grounding-mix0.1, vision-lr-mult0.1.
The compact anchor's training-sample fit curve is not held-out validation. Independent
before/after box checks use the same60 cached held-out images; final flight evaluation
uses d190/check_tasks.zip and unchanged D206/D208 server flags.

Sources: scripts/win_eval/run_d209.ps1, scripts/rl/d209_gate.py, d209_box_check.py,
tests/test_dagger_grounding.py; reports/d209_preparation/{failure_audit.json,
practice_manifest.json,practice_tasks.zip,payload.zip,vla_code.zip}.
Staged S3 d209/vla_code.zip and d209/payload.zip. Windows startup requested; launch
and advancing collection/training are not yet verified. A12h shutdown deadline and
per-process time limits bound the workflow; wrapper saves outputs and stops on failure.


D209 launch verified 2026-10-09 12:37 UTC: Windows preflight found no existing model
or simulator jobs; both prior DAgger row files and step8000 weights are present.
Launcher PID5060 is active, started.json and run.log created. SSM launch command
3bb6eac7-6092-4971-8b80-555e85773070; verification751334c8-76d9-442a-bfc4-20d7b5d8c6c1.
Backup shutdown was scheduled before bootstrap (12h, about2026-10-10 00:36UTC).
Setup/prechecks are in progress; collection and optimizer updates not yet verified.
Completion heartbeat d209-dagger-and-validation-completion is active every15minutes.

D209 preflight attempt1 stopped safely before collection or training. All six encoding
tests passed on Windows, but Start-Process/WaitForExit exposed a null ExitCode and the
wrapper treated it as failure (S3 d209/failure.txt, encoding_tests.err). Fixed RunPy to
record the native Python LASTEXITCODE in a child-worker exit file. Verified locally
with real exit0 accepted and exit7 rejected. RetryPreflight permits only this setup
retry: it refuses if any rollout directory or training output exists, and preserves
the first failure record. No optimizer updates or practice flights occurred in attempt1.

D209 preflight retry dispatched after EC2 reached stopped, then restarted and SSM Online
(2026-10-09 about12:46UTC). Command bff77798-aa45-446e-8d37-f935f18899b7 first refuses any
existing job/rollout/train output/retry marker, sets a fresh12h backup stop, downloads
the repaired bundle, creates retry_started.json and launches -RetryPreflight once.
The active heartbeat distinguishes historical attempt1 failure from any new failure;
no further automatic retry is authorized after this setup retry.

D209 retry verified at2026-10-09 12:49UTC: encoding_tests.exit=0 and all6 tests passed.
No current failure.txt. GPU process2472 is loading step8000 for d209_box_check.py;
box_before.json is the target. Collection and optimizer updates have not started.
Source: SSM5e05539a-defc-4fd7-b0e3-8d2a0c762b60. Earlier error audit is trace-based;
no frame-by-frame camera review or causal attribution of visual errors is claimed.

D209 optimizer updates verified 2026-10-09 16:07 UTC. All 100 practice flights
completed; ROLLOUTS_DONE at 15:57:52 UTC. Data gate passed: 100 tasks matched,
611 fresh rows, 1,939 replay rows, 967 labelled anchor frames, zero held-out overlap.
Training reached update 40/300 (loss 0.6183; about 10.2 seconds/update). No current
failure marker. Final box check and 100 validation flights are still pending.
Evidence: read-only SSM b973f7f3-9511-4f08-9e89-0ce74712f0b7, remote
C:/runs/d209/{train.out,gate.json,run.log}. Training-start notification sent;
continue monitoring without any additional restart or round.

### D209 completed: grounded DAgger round (2026-10-09)

Completed at 19:59:38 UTC: 300 updates and all 100 validation flights. Local rescoring with scripts/score_tasks.py (reward_uav_flow metrics) exactly matches the uploaded score, with 100 flight plots verified. EC2 verified stopped on the final check; no additional run launched.

| Run | Success | nDTW |
|---|---:|---:|
| D206 step5000 | 81/100 | 0.3792 |
| D208 step8000 | 82/100 | 0.4255 |
| D209 DAgger300 | 83/100 | 0.4350 |

| Kind | D206 success / nDTW | D208 success / nDTW | D209 success / nDTW |
|---|---|---|---|
| Approach/Move | 8/10 / 0.3901 | 8/10 / 0.3897 | 9/10 / 0.4520 |
| Ascend/Descend | 9/10 / 0.6113 | 9/10 / 0.6117 | 10/10 / 0.6621 |
| Land | 12/15 / 0.3087 | 15/15 / 0.4005 | 15/15 / 0.4030 |
| Pass | 10/15 / 0.4222 | 9/15 / 0.4498 | 10/15 / 0.4254 |
| Retreat | 3/3 / 0.4828 | 3/3 / 0.5703 | 2/3 / 0.5033 |
| Shift | 15/15 / 0.6638 | 15/15 / 0.6764 | 15/15 / 0.6773 |
| Surround | 3/10 / 0.2358 | 3/10 / 0.2732 | 0/10 / 0.2767 |
| Turn | 12/12 / 0.1356 | 10/12 / 0.2210 | 12/12 / 0.2162 |
| Turn/Rotate | 9/10 / 0.1548 | 10/10 / 0.2543 | 10/10 / 0.2903 |

Paired versus D208: 5 gains and 4 regressions. One additional success is not evidence of a reliable general improvement. All ten Surround flights hit 104 steps and fail; Surround fell from 3/10 to 0/10. This is trace evidence, not a visual causal diagnosis.

Same 60 held-out pseudo-labelled frames: centre error 65.8494 -> 70.9322 (worse), IoU 0.5932 -> 0.5857 (worse), x correlation 0.837 -> 0.810 (worse), empty accuracy 0.9 -> 0.9, next-centre error 122.6374 -> 114.6506 (better).

Recommendation: retain D208 as the balanced baseline pending investigation of Surround and grounding regressions; do not automatically promote D209 or launch another round. This recommendation does not change the saved adapters. No PPO/GRPO or benchmark test was run.

Evidence: reports/d209_completion_20261010/{score.json,score_remote.json,report.json,completion.json,box_before.json,box_after.json,s300.zip,results/flights,ALL_TASKS.md}. Uploaded artifacts: d209/update/adapter_s300, d209/update/checkpoint, d209/update/report.json, d209_eval/s300.zip. Historical d209/failure.txt is from the first preflight at 12:38 UTC, superseded by completion.json.

### D210 - Offline DAgger / Surround audit (2026-10-10)

The earlier interpretation that D209 destroyed three successful Surround behaviors is incorrect. D208's three endpoint successes stopped after 11 steps and travelled only 0.0303 m each. The reference loops end 0.13-0.17 m from the start, with heading within four degrees; standing still satisfies the endpoint success rule. D209 correctly predicts a signed 360-degree initial goal on these three tasks and starts orbiting, but times out. This does not establish that D209 completes circles or improves grounding. It changes the interpretation of the 3/10 -> 0/10 count.

#### Evidence and controls

- Matched all 100 D208 evaluation episodes, 100 D209 evaluation episodes and 100 D209 practice episodes to logged poses. Reconstructed all 611 fresh corrective rows using the actual relabeler; this is offline analysis, not new training or evaluation.
- The three apparent regressions are sim_2025-03-27_16-26-34, 17-33-19 and 17-36-45. D208 initially emits zero remaining progress and stops. D209 emits +360,+360,-360 respectively, travels 9.94,9.91,11.98 m and reaches 104 steps. Two trajectories closely follow the reference arc; the third visibly drifts outward. See surround_comparison.png. These are trajectory plots, not camera-image causal analysis.
- Evaluation and collection scripts explicitly pass `-m 100`. The evaluator checks the limit after a returned action chunk, giving 104 executed steps for 8-step chunks. All ten validation Surround references contain 127-347 moves; all 25 practice Surround references contain 127-378 moves. Matching one action to one reference increment cannot finish them within this horizon. Faster actions could in principle finish; this audit does not claim a physical impossibility.
- Offline closed-loop replay of the actual relabeler's ideal actions fails endpoint success on all ten tasks at 104 steps, both before and after action-token quantization. First all-still chunks occur at 128-352 steps with a longer replay horizon. This is an oracle replay control, not evidence that the learned model succeeds with a longer limit.
- Practice contains 237 Surround corrective rows and zero all-still expert chunks; validation also never reaches the reference end. Thus this fresh round provides no successful orbit-completion/stop examples. Sixteen premature-stop student calls are relabelled as movement, mostly appropriately correcting false stopping. Prior replay/anchors may still contain stopping supervision; this count concerns fresh Surround rows only.
- One concrete phase-alias defect exists in practice: sim_2025-03-27_16-43-03 call1 at accumulated yaw -0.452 degrees jumps from reference index0 to220/222. The nearest-pose rule labels only -1.330 degrees remaining instead of approximately -361.330. It lacks history and confuses the end with the beginning. Only 1/237 fresh Surround calls has an error over180 degrees; it is not supported as the dominant explanation. No Surround corrective-action components exceed the saved tokenizer bounds.
- The PPO/GRPO reward is unsafe to adopt unchanged for this task: a synthetic stationary trajectory gets endpoint success on all10 circles and combined reward0.4454-0.4888, versus0.3823-0.3882 for the correct first104 reference moves. This direct counterexample demonstrates an incentive to stop early. D209 was supervised DAgger and did not optimize this reward, so this is a risk for proposed PPO, not the cause of D209's behavior.
- The existing nDTW caps the reference at20 stride5 samples, i.e. indices0..95 for Surround, while leaving the model trace uncapped. It measures against a prefix of these long circles; it must not be interpreted as full-orbit fidelity. Preserve historical scores for comparability and add a separately labelled full-path diagnostic before longer-horizon comparisons.
- Outside Surround, endpoint success increases from79/90 to83/90. The genuine endpoint regression is one Retreat task; overall changes are5 gains/4 losses, three losses being the stationary Surround cases. This single validation set does not establish statistical significance.
- The 60-frame box regression remains real relative to the fixed pseudo-labels: centre65.8494 ->70.9322, IoU0.5932 ->0.5857, correlation0.837 ->0.810. Next-centre improves122.6374 ->114.6506. Shared-parameter interference is a hypothesis; these artifacts do not isolate its cause. No new visual-grounding ablation was performed.

#### Decision and recommended next experiment

Do not proceed directly to PPO with the current reward. Retain both adapters; withdraw the claim that the Surround endpoint count alone makes D208 the better orbit policy. First establish loop completion and stopping under a diagnostic horizon (e.g.500 steps) on a fixed small practice subset for both adapters, with full-path coverage/winding diagnostics and stationary-policy negative controls. Keep the historical100-step scores separate. If preserving the100-step operational budget is mandatory, resample expert paths/actions to a feasible horizon and validate those targets before further training. Make path matching phase-aware, add terminal replay, and audit the first-progress/goal-memory mismatch. Any new AWS run or PPO training remains a separate user decision; none was launched during D210.

Sources: scripts/rl/dagger_relabel.py:35-72; scripts/uav_flow_eval_server.py:196-223,337-351; scripts/win_eval/eval_variants.ps1:78; scripts/rl/collect_rollouts.ps1:67; scripts/rl/reward_uav_flow.py:57-83; local UAV-Flow-Eval/batch_run_act_all.py:351,454. Reproducible scripts and evidence under reports/d210_dagger_audit/: audit.py, path_audit.py, oracle_replay.py, call_audit.json, summary.json, surround_traces.json, surround_path_audit.json, oracle_replay.json, surround_comparison.png; archived practice.zip and extracted practice/. Uses saved artifacts only. No benchmark test flights, model changes or cloud compute launches.

### D211 - DAgger repair requirements and scope correction (2026-10-10)

User requested a clear repair specification before more work, emphasizing that recording/photo frequency, reference-path point frequency, commanded action spacing and observation/replanning cadence need not coincide. Current corrections choose eight consecutive recorded points and inherit their approximately0.1m spacing in the audited circles. This is not a units-conversion error. Original supervised target generation has not been comprehensively audited; do not assert the coupling is exclusive to DAgger.

The teacher is our geometric script, not a human or independently validated policy. Reference data came from existing simulation trajectories. It chooses a nearby position/heading, follows eight successors and linearly removes off-path displacement; it does not reason over the image or check collision clearance. Claims in the earlier D184 note that this teacher was free/exact or guaranteed correct are withdrawn. D210 checked selected numerical properties of all611 fresh rows across nine kinds and deeply analyzed237 Surround rows. It did not certify every row, inspect every image, or validate the1939 old replay rows/anchor labels. One detected loop-phase error does not establish that only one error exists. Non-Surround action clipping also needs auditing.

Repair sequence if DAgger is resumed:
1. Define task semantics, independent full-path completion/stopping checks, action-distance/rate contract, and feasible rollout limits. Keep original benchmark scores unchanged and label new diagnostics separately. Circles require ordered coverage and direction; stationary policy must fail.
2. Represent the reference as geometry plus route progress, not one action per recording. Select targets by arc length with adaptive spacing for curvature and recovery; interpolate position and unwrapped heading, validate action bounds and quantization, and choose observation/chunk cadence independently. Do not blindly multiply deltas or select every nth photo. Derive spacing from training-only controls and deployment needs.
3. Track phase/history to distinguish loop start/end and avoid backward jumps. Specify behavior off-path, past the endpoint, reverse direction and ambiguous self-intersections. Handle terminal corrections explicitly. Obstacle-blind interpolation cannot be called a safe expert.
4. Validate the teacher BEFORE learning: all practice tasks under chosen horizon, raw and tokenized action integration, displaced-state recovery, wrap/units/frame invariants, terminal/stop tests, clipped-action distributions, simulator/collision checks and image/trajectory review stratified across all kinds and all flagged cases. Use checks independently formulated from generator internals. Human review of representative examples complements automated checks; neither certifies unseen states.
5. Audit fresh and historical replay separately; quarantine unvalidated records with provenance and reason. No validation/test trajectories enter targets. Preserve original data and historical metrics; never silently overwrite labels.
6. Only after teacher acceptance, a bounded supervised trial with grounding retention and paired heldout flight evaluation. Missing fresh completed-circle examples is a coverage observation, not proven cause of failure. Box regression cause remains unestablished.

Alternative: defer teacher repair and pursue RL after independently validating reward, horizon and policy-likelihood plumbing. RL removes action-label generation, not the need to validate task semantics and training correctness. No new run is authorized or launched by this document. Detailed research/plan: docs/research/PPO_PREFLIGHT_20261010/.

### D212 - Evidence-first PPO design and pre-validation plan (2026-10-10)

**D212 documentation clarification:** At the user's request, [the concise working plan](research/PPO_PREFLIGHT_20261010/WORKING_PLAN.md) is now the primary execution guide: validate scoring → repair/test PPO → check simulator signal and cost → bounded pilot → matched validation. The long plan remains a technical reference; its alternatives are not all required work. Implementation starts with offline checks, not a training launch. No model or run was changed by this reorganization.

User requested a detailed script/training design with failure prevention and pre-validation before further paid compute. Research inspected 31 candidates and retained 27 scoped sources (25 papers, one author technical article, one official documentation source), including competing PPO and group-relative VLA approaches, foundations, benchmark protocols and negative evidence. This is a decision-focused review, not reproduction or a guarantee of local improvement. Resource and license unknowns are explicit. The evidence schema/coverage validator passes; this does not validate the proposed trainer.

**Decision: do not launch the current PPO flag unchanged.** Local code review found that action replay omits the sampled box prefix; final rollout calls are treated as terminal without proper timeout bootstrapping; the current surrogate clips token ratios rather than an explicitly chosen joint chunk ratio; remainder batches can be omitted; gradient clipping/checkpoint state need repair. D210 already demonstrates that stationary circles can outscore correct partial circles under the existing reward. That reward was not used for D209 supervised training and is not its causal explanation.

The proposed first PPO experiment adapts the existing pipeline rather than migrating frameworks: frozen D208 box/progress prefix generation, separately trainable action adapter initialized from D208, initially frozen vision, exact multimodal likelihood replay, a small normalized value head, explicit joint-chunk objective and complete rollout/checkpoint provenance. This composite-policy choice isolates action learning; it must be evaluated as deployed and is not a proven literature recipe. PPO and GRPO both have successful VLA precedents under different settings; no universal superiority is inferred. GRPO avoids a critic but still needs correct rewards/probabilities and informative repeated flights.

Before optimization: specify task completion independently of endpoint proximity, repair full-path diagnostics without overwriting historical metrics, separate recording/waypoint/action/replanning frequencies, and test stationary/spinning/partial/wrong-direction/wrong-object controls. Surround requires ordered positional coverage around the intended target, not yaw accumulation alone. Ambiguous metadata is quarantined. Human review of concrete trajectory/camera examples complements automated invariants; neither constitutes certification of every unseen state. The 273 test tasks remain untouched.

Ordered gates: (A) offline semantics/action/split controls; (B) CPU PPO/GAE, sample accounting, failure-injection and checkpoint tests; (C) real-model prefix/probability parity, one-update and memory smoke tests; (D) small practice-only simulator controls and measured reward variance/throughput; (E) at most two budgeted learning rounds with hard episode/call/update/wall/cost ceilings and retention canaries; (F) matched heldout comparison with per-task/per-kind evidence and uncertainty. Numeric thresholds are proposed engineering guardrails to calibrate and freeze, not published guarantees. No new cloud spend is authorized by the planning request.

Current readiness is RED: the plan does not implement or pass these gates. No training, flight evaluation, instance launch or model change occurred in this documentation task. DAgger teacher repair can be deferred while PPO correctness is established; task semantics and action feasibility remain prerequisites for both. No day-long run should be launched simply because a process starts.

Detailed deliverables: [PPO design and gates](research/PPO_PREFLIGHT_20261010/PLAN.md), [evidence synthesis/options](research/PPO_PREFLIGHT_20261010/SYNTHESIS.md), [readable evidence](research/PPO_PREFLIGHT_20261010/EVIDENCE.md), [resource/license inventory](research/PPO_PREFLIGHT_20261010/RESOURCES.md), [protocol](research/PPO_PREFLIGHT_20261010/PROTOCOL.md), [search log](research/PPO_PREFLIGHT_20261010/SEARCH_LOG.md), evidence.json, candidates.json and validation.json. Local evidence: scripts/rl/{train_grpo_vla.py,rollout_samples.py,reward_uav_flow.py}, scripts/uav_flow_eval_server.py and reports/d210_dagger_audit/.

### D213 - Offline PPO scoring controls and first correctness repairs (2026-10-10)

User authorized starting the concise working plan. Implemented an offline first pass; no AWS calls, model training, simulator flights or benchmark test execution. The end-to-end PPO launch gate remains blocked, not completed.

**Scoring evidence:** `scripts/rl/reward_v2.py` adds an explicitly diagnostic orbit check (positional sweep/direction, radial and altitude corridors, segment shortcuts/jumps, closure and stopping) and whole-path similarity with both endpoints and signed angle representation. It does not replace historical reward_uav_flow.py and cannot certify object identity or task meaning. A circle fitted from the reference is labelled reference agreement, not independent expert truth. All25 practice reference circles pass the geometric control; stationary and first104-move controls fail. Reproduced all10 saved D210 stationary-over-partial reward counterexamples with unchanged historical values.

**Metadata audit:** inspected all100 practice task records.55 use absent/sentinel target geometry;35 of those are object-relative Approach/Move, Pass or Retreat tasks requiring additional target evidence. Missing object metadata alone is not a defect for explicit displacement/rotation tasks.45 tasks contain derived object metadata (20 Turn,25 Surround). No new semantic reward specifications are approved. Example needing resolution: original Land instruction says “Descend to4.0 meters altitude”, whereas unified text says “Land4.0 meters”; destination altitude, displacement and contact must not be conflated. Practice IDs/source flights have zero overlap with the saved100-validation tasks and saved heldout/exclusion manifests; current benchmark-manifest identity still needs verification.

**Implemented PPO repairs:** server sampling now logs raw box/progress context and exact action-context token IDs; trainer preserves box context and rejects token mismatch. PPO rejects incomplete/unapproved old rollout records before model loading, including missing terminal/truncation information and timeout bootstrap. GAE uses explicit reward_v2 and actual-final-state values; no endpoint-gain fallback. PPO clips the joint action-chunk ratio, checks behavior/replay drift, preserves recorded behavior log probabilities, disables dropout for PPO gradient evaluation, detaches critic features, includes critic parameters in gradient clipping and processes weighted remainder batches. Empty/no-update runs fail explicitly. Existing GRPO/DPO objectives remain separate. Core math and production encoding replay are tested on CPU, not a real VLM.

**Checkpoint groundwork:** atomic checkpoint helper saves adapter, critic, optimizer, optional scheduler/scaler, RNG states, shuffle RNG and metadata with checksums and a committed pointer. Tests verify deterministic next-update/RNG restoration, tamper rejection and preservation of the previous commit on a failed save. This helper is not yet wired into an end-to-end restart/orchestration workflow; the historical trainer export alone still does not satisfy recovery requirements.

**Validation:**30 CPU tests passed (24 new preflight tests plus6 existing grounding tests), including analytic/finite-difference policy gradients, joint versus token ratios, terminal versus truncated returns, reset boundaries,0/1/17/31/32/33-sample handling, probability drift, orbit negative controls and full-path tails/yaw direction. Changed scripts compile. Nine saved practice examples (one per kind, deterministic selection) rendered as first/last-call camera frames and XY traces; the last-call image is not necessarily the final camera state. Inspected the review pages; they support diagnosis, not certification of every label or flight.

**Remaining before a paid pilot:** independently resolve task semantics/geometry for the pilot subset; complete frame/action and metadata provenance checks; implement and verify frozen-prefix/action adapter switching, value warmup, checkpoint integration and process-failure/deadline handling; then real-model likelihood/memory parity and simulator signal/cost checks. No claim that all Step1/Step2 gates passed or that PPO is ready to launch.

Evidence and reproduction: [offline report](../reports/d213_offline_preflight/REPORT.md), [camera/trajectory review](../reports/d213_offline_preflight/REVIEW.md), summary.json, task_review.json, orbit_controls.json, historical_controls.json, input_hashes.json and tests.txt in that report directory. Commands: `python scripts/rl/preflight_offline.py`; `python scripts/rl/preflight_review_pack.py`; `python -m unittest discover -s tests -v`. Code: scripts/rl/{reward_v2.py,ppo_math.py,checkpoint_state.py,preflight_offline.py,preflight_review_pack.py,rollout_samples.py,train_grpo_vla.py}, scripts/uav_flow_eval_server.py and tests/test_ppo_preflight.py.

### D214 - Geometry-first PPO reward candidate (2026-10-10)

User selected whole-path geometric agreement as the initial PPO reward direction, rather than first/last-image comparison. Implemented `scripts/rl/path_reward.py`: continuous reference-polyline projection with history-constrained phase; signed normalized progress; distance-weighted deviation and reverse-motion penalties; one final completion bonus after ordered coverage, endpoint, corridor and stopping checks. Positions use metres and unwrapped yaw uses a declared1 m/radian scale; roll/pitch are ignored to match the xyz/yaw action interface. Reference/photo point frequency does not prescribe action frequency. Historical reward_uav_flow.py is unchanged.

This supersedes requiring independently reconstructed object geometry for the first candidate reward: the explicit objective is following the demonstrated route. Reference quality and physical feasibility remain assumptions to check; the reward does not certify object identity or collision clearance. Completion is not inferred from first/last camera images. Full-path similarity remains an additional diagnostic.

Offline results:100/100 saved practice references pass self-controls. On all25 practice circles, reward orders complete reference above its first104 moves above stationary. Ten new tests cover this ordering, reverse circles, yaw-only spinning, repeated arcs, overshoot, reference/action sampling density on a fixed line, teleport shortcuts, yaw-only tasks, revisited segments and stopping bonus. During development one Land reference exposed a phase tie on a revisited segment; fixed continuity-aware tie selection and added a separate synthetic regression. This was an implementation bug, not bad reference movement spacing. Full CPU suite:40 tests pass.

Pictured Pass task sim_2025-04-30_15-37-22: reference reward2.000; saved flight0.240; stationary0.000. Saved flight reaches100% projected progress but ends2.72 metric units beyond the reference endpoint and receives no completion bonus. Positive partial credit and overshoot penalties are now explicit rather than adjudicating the final photograph. Values use candidate weights, not calibrated benchmark scores.

Status: offline-tested candidate, not connected to PPO or launch-approved. Still calibrate practice-only corridor/movement/yaw weights, discounted return incentives and detour recovery; connect executed transitions/stop status and complete trainer integration. No training or AWS execution. Evidence: [candidate report](../reports/d214_path_reward/REPORT.md), scores.json and tests.txt; reproduce with `python scripts/rl/audit_path_reward.py` and `python -m unittest discover -s tests -v`.

### D215 - Geometry PPO integration and bounded AWS preflight (2026-10-10)

User requested sustained autonomous completion, then explicitly approved the complete preflight on the existing AWS Windows machine. Scope is saved-observation sampling, probability replay, one tiny synthetic-reward PPO update, frozen-adapter checks, checkpoint reload and resource measurement. No simulator flights, benchmark tasks or full training are authorized by this preflight.

Local evidence:59 tests pass in the installed Qwen/PEFT runtime. Integrated separate frozen prefix/trainable action adapters, detached-feature critic warmup, joint-action PPO, atomic optimizer/RNG checkpoints, interruption recovery, bounded single-owner subprocess stages, exact within-chunk execution limits, independent simulator pose readback and observation-only bootstrap transport. The simulator integration is mock-tested, not flight-validated. Historical evaluator logs contain commanded poses; they do not independently prove measured execution.

A real-model NF4 check with the older locally cached D204 step2000 adapter exposed likelihood mismatch between cached sampling and full-sequence replay: seeded probe maximum token difference0.024895 and joint difference0.069590; same-cache forced generation matched exactly. Cached gradient replay now uses the same autoregressive path. Full eight-action local run committed only step0; no final report or completed update is available, and the process is no longer running. GPU occupancy was about7.94GB on the8GB laptop. A separate one-action control completed one update in53.422 seconds total, peak allocated6.138GiB/reserved6.537GiB; this is not evidence that the full eight-action pilot passes, nor did that earlier control verify a nonzero actor parameter delta or reload. The AWS harness explicitly adds those requirements.

AWS preparation: isolated source bundle SHA25678ea6a87b4f7348e8d64fe483d05857d48425703572d3329fa0e6137da42946b. Existing Windows g6.xlarge i-03a3c8632314bf5fc was stopped and has been started for this authorized check. Planned45-minute independent shutdown,30-minute bounded runner including reserve, no automatic failed-run restart. Use D208 step8000, BF16, chunk8, two saved-context samples, one optimizer update and exact reload comparison. Candidate cannot be promoted: rewards are synthetic and no flight improvement is tested.

Price List API checked for us-east-1: standard on-demand Windows g6.xlarge $0.9888/hour,45minutes $0.7416 compute, excluding storage/transfer/tax; pricing effective2026-10-01. Linux machine remains unused. Upload destination ownership was verified against the connected account; only project code/tests and three known practice fixtures were packaged.

Evidence: reports/d215_ppo_integration/{tests.txt,parity_probe/report.json,one_action_control/model/report.json,aws_bundle_manifest.json}; AWS first attempt: all59 CPU tests passed on the Windows runtime, but the GPU stage exited1. The failure marker uploaded and automatic shutdown began. Detailed GPU evidence recovery is pending because the initial uploader did not complete. EC2 stopped was verified. Automatic approval review rejected a new billable instance start for log recovery, so recovery requires explicit approval. Proposed recovery: same instance, at most5 running minutes, read existing model/report/error logs only, then stop; $0.0824 estimated compute. The uploader was fixed locally to prioritize results and continue after individual file failures; this change has not been redeployed or tested on AWS. No failed trainer restart. No claim of training readiness until full preflight and live simulator collection gates pass.



D215 recovery update: user explicitly approved a brief start solely to recover logs. Started14:21:12UTC; recovered five diagnostic files through SSM; issued StopInstances after recovery, with an independent180-second guest shutdown also scheduled. No trainer restart. EC2 subsequently confirmed stopped after recovery. Recovered report proves the failure was `PackageNotFoundError: bitsandbytes` in runtime-identity metadata collection, after BF16 model loading but BEFORE sampling or any optimizer update. Elapsed90.50s, peak allocated9.186GiB/reserved9.211GiB are loading-stage measurements, not backward memory. The failed artifact uploader encountered a locked launcher.err file and aborted before uploading model reports.

Local repairs: record absent optional bitsandbytes as null while still requiring torch/transformers/PEFT; add absence/required-dependency regressions; extract cached token replay and verify its nonzero gradients against a full float32 reference on a tiny model. All64 local tests pass, including a Windows regression with an actually locked launcher stream and an injected individual-file upload failure. Uploader now excludes live launcher streams, prioritizes diagnostics and catches errors per file. These fixes have not yet passed a corrected AWS run. Recovered evidence: reports/d215_ppo_integration/aws_results/{results/model/report.json,results/01.log,results/failure.json,upload_failure.txt,launcher.err}; tests_after_recovery.txt. Full D208/BF16 preflight remains incomplete; no flight or learning improvement claim.



D215 corrected rerun authorized by user (2026-10-10): isolated C:/runs/d215_preflight_r2, code SHA256 3d45ec00cc965f2699125373b1e999f867b0f7d9e9b59fa7cdd853fdef008d20, 69 files, 262950 bytes; all64 local tests passed. Same Windows instance was stopped before start; independent45-minute shutdown accepted at14:37:11UTC; no Python/Collection process existed and L4 memory used34MiB. Dispatched once through SSM command6f292c88-25ef-4563-96b8-c73f072dcc19. Scope unchanged: saved observations, D208 BF16 chunk8, one synthetic PPO update plus exact reload; no simulator flights. Outcome pending. Evidence: reports/d215_ppo_integration/aws_bundle_r2_manifest.json and tests_r2.txt.



D215 corrected AWS rerun result (2026-10-10): all64 AWS CPU tests passed (2.327s). D208 BF16 chunk8 sampled two saved observations, passed old-policy replay parity and completed one optimizer update in70.828s. Action adapter changed; frozen prefix/vision adapters were bitwise unchanged. Pre-update KL0, post-update KL0.0136212 (limit0.05). Model stage200.296s; peak allocated17.0753GiB/reserved18.0117GiB, fitting the L4. This uses synthetic rewards and establishes no flight improvement.

The preflight correctly FAILED its final exact reload gate: all32 token log probabilities differed, maximum absolute difference0.04768324. Optimizer/critic recovery returned step1, but output identity was not preserved. No automatic AWS retry or flight launch. Diagnostics plus step0/step1 adapters and optimizer checkpoints uploaded (21 objects observed); unlike attempt1, the repaired uploader saved the detailed report. EC2 automatic shutdown completed; DescribeInstances confirmed stopped at approximately14:51UTC.

Local diagnosis reproduced loss of small FP32 LoRA updates when PEFT creates adapter modules on a BF16 base: weight difference1.37016e-5 after ordinary reload, exactly0 after direct restoration from the saved safetensors. This supports a checkpoint precision-loss cause; it does not yet prove the entire AWS discrepancy is resolved. Added exact action tensor restore after module allocation, with key/dtype/shape and bitwise-weight checks, in both smoke reload and trainer resume. A real-PEFT tiny BF16 model regression verifies exact weights and logits after reload. All65 local tests pass (6.242s). The fix is local only; corrected full D208/BF16 reload still needs validation before a learning pilot.

Evidence: reports/d215_ppo_integration/aws_r2_results/results/{model/report.json,00.log,01.log,failure.json,model/checkpoints/CURRENT.json}; tests_after_r2.txt; scripts/rl/composite_policy.py and tests/test_composite_policy.py. No bucket name is stored in repository evidence.



### D216 - Continuous pre-training test session (2026-10-10)

User explicitly requested finishing the repaired reload check and practice checks in one AWS session, keeping the instance running between tests and stopping once testing finishes, before learning training. Same Windows instance, no replacement hardware. Wrapper now supports KeepRunning; session owns final stop and a3-hour backup shutdown. Source bundle SHA25694a404028b781a4d7b49d0486e71b8740b0b42199c449aa34de65ca08bbf128c; practice source9d9bf82c5260a5a42adfc0baa7687a3ef211a8dbb3fe5051761ef2f27cb3ef71. Full-model test dispatched14:59:47UTC;65 AWS CPU tests passed2.814s. Test outputs are non-promotable.

Practice protocol prepared: first sorted practice task per Approach/Move, Turn, Pass, Surround, two stochastic attempts each with unchanged D208 composite policy, temperature1.0, chunk8. Max128 executed moves for other kinds,320 for Surround. Reuse official object/reset/camera setup; first test measured translation/yaw and return-to-origin, then collect independent pose readback after every action, exact inside-chunk truncation, final-observation bootstrap, candidate whole-path rewards. Policy images resized224x224 as official evaluator. No benchmark tasks or learning training. Report variation and limitations, not an improvement claim.



D216 full-model gate PASSED: D208 BF16 chunk8, one PPO update, exact action weights and token log probabilities after checkpoint reload, frozen adapters unchanged. Model-stage200.797s; update71.094s; peak allocated17.0753GiB/reserved18.0117GiB. Post-update KL0.0136212. Evidence reports/d216_pretraining/model_report.json. Continued on the same running instance to the practice-only checks; no learning run started.



D216 simulator integration corrections: first practice startup failed before collection because ConfigUEWrapper defaulted offscreen=False despite the environment flag; Unreal logged D3D swap-chain error887A0022. Explicit offscreen=True fixed startup. Live movement calibration then found translation correct but yaw wrong (requested+10deg, measured-68.76deg at early readback). A simulator-only probe measured approximately180.233deg offset between blueprint command and final actor/camera yaw, plus temporal interpolation: a10deg change was still short at0.1s and settled by0.6-1.6s. The new collector had incorrectly reused measured actor yaw as command yaw.

Corrected reset calibrates the command offset to the task heading; execution uses initial_command_yaw plus desired relative yaw and waits for measured heading within0.005deg (3s bounded wait). Calibration now passes translation,+/-10deg and return-to-origin. Explicit pose readback remains the source of rewards. These are changes to the new pre-training collector; results are not directly comparable to historical D208/D209 scores under a different execution protocol. First two approach attempts each stopped after18 moves (~35% path progress), receiving partial reward~0.22 and no completion bonus. No invalid transitions. Other practice checks ongoing. All66 local tests pass, including command-offset and exact cold checkpoint reload regressions.



D216 final testing outcome (2026-10-10): one continuous Windows AWS session, latest launch14:57:34UTC. Final StopInstances requested16:26:17UTC after tests and verified uploads; EC2 confirmed stopped at 16:34:21 UTC on 2026-10-10 (DescribeInstances). No restart between testing stages, no learning pilot or benchmark flight.

Eight practice flights verified:533 executed moves,72 full sampled chunks. Approach:18/18 moves,35.28%/35.51% progress,reward0.223413/0.221447,0/2 complete. Turn:26/26 moves,100%/99.10%,1.990786/1.970890,2/2. Pass:42/42 moves,100%/100%,0.024580/0.407265,0/2; spatial endpoint errors2.0896/2.1334m. Surround:183/178 moves,100%/100%,0.690925/1.779035,1/2; spatial endpoint errors0.87365/0.26404m. Two stochastic attempts on one practice task per kind under a new measured execution protocol are not directly comparable with legacy D208/D209 validation. The successful178-move trajectory would be truncated at100; this is not a randomized horizon ablation or learning-improvement claim.

Transport follow-up: practice3 was interrupted after152 moves and remains quarantined. practice4 calibration failed because feedback chased residual interpolation. Final small correction waits for stable readback (at least1.5s and five successive readings within0.001deg; five-second maximum wait); a12-heading simulator-only probe passed. practice5 stopped normally at183 moves, but its post-flight0.01deg gate rejected a maximum0.01316394deg discrepancy; maximum position discrepancy0.0013492cm. Declared executed-heading tolerance is now0.05deg (0.87mm at1m/radian yaw scale). Recovery independently proves the first ten-quiet-move stop occurs exactly at183, verifies every call boundary/actual pose, and preserves the source SHA256/original failure. Interrupted transport runs cannot be recovered as completed flights. practice6 completed the other repeat normally at178 moves. Rewards continue to use measured poses.

Full no-update replay audit PASSED:73 chunks (72 practice plus one deliberate timeout probe;2336 token probabilities), maximum token/joint error0.0. The probe executed2 of8 proposed moves and replayed real final-observation bootstrap features [1,2560]; it is excluded from learning. Audit402.687s including load; median replay5.312s,mean5.35015s. Image SHA256, independent legacy reward reproduction, S3/local sizes and ZIP CRC passed; final archive4709661bytes, report15236bytes. No Python/Collection process remained before stop.

Reward prevalidation caught a negative control: old combined endpoint tolerance accepted45deg for a22deg target (reward1.860109). Candidate v2 adds an independent5deg endpoint-heading gate; the control becomes incomplete (0.860109). All100 reference self-controls and25 circle full>partial104>stationary controls pass. Original v1 evidence is retained; v2 rescoring changes no score/completion among these eight flights. This engineering threshold does not independently certify physical task success. Final local suite72 tests passed2.692s. The later reward change is offline-tested; earlier65 AWS CPU tests/full-model warm reload are separate evidence. The cold-load exact_restore marker has a tiny real-PEFT/BF16 regression, not a new full-GPU cold-load result.

Decision: prepare72 v2 action chunks with unique image namespaces but keep launch/admission flags false. Exclude interrupted runs and timeout probe. Proposed next pilot: one round, at most4 updates (reduced from8 after timing replay), actor lr1e-6,critic lr1e-4,frozen D208 prefix/vision,gamma1,clip/hard KL0.05,90min training and3h instance ceiling ($2.9664 compute at checked rate; excludes storage/transfer/tax). First verify fixed-seed before/after checks on four disjoint practice tasks. Existing engine performs three initial full-batch passes plus one full-batch KL pass per update, so optimizer timing alone understates cost. No training launched or queued.

Evidence: reports/d216_pretraining/{README.md,model_report.json,live_replay_report.json,reward_v2_controls.json,tests_final.txt,routes.png,final_source_manifest.json,pilot_proposal.json}; verified/{summary.json,manifest.json,ppo_batch/rollouts.json}; original initial/,second/,final/ archives and recovery proof. New test/verification scripts are under scripts/rl/{practice_preflight.py,practice_preflight_driver.py,simulator_transport.py,audit_live_replay.py,recover_completed_practice.py,summarize_practice_preflight.py,plot_practice_preflight.py}; reward changes in path_reward.py/executed_rollouts.py.


D216 prepared-batch backup verified: d216/prepared/ppo_batch_v2.zip, 6523699 bytes, SHA256 bfbf74ce7f598ac18820380cc53649dd263b574f0a323396a79a04ccd0b95961. It contains72 action chunks, reward v2, unique checked images and closed launch gates; no instance restart or training dispatch.



### D217 - Authorized bounded PPO learning pilot (2026-10-10)

User explicitly authorized starting the PPO experiment after D216 testing and shutdown. One session on the same Windows g6.xlarge, no duplicate jobs or automatic retry. Reuse the 72 exactly replayed D216 chunks (eight practice flights), reward v2, D208 step8000 frozen prefix/vision and trainable action adapter. This is the first flight-reward learning pilot, not the synthetic preflight.

Bounds: one round, at most four PPO updates, actor lr1e-6, critic lr1e-4, batch1/accum8, gamma1, clip0.05, hardKL0.05. Training stage has a90-minute process ceiling; full EC2 session three hours from start, with an independent guest shutdown and upload reserve. No benchmark tasks. No automatic extension or failed-trainer restart.

Added seed-aware server reset, collector seed recording and disjoint task selection. All75 local tests passed2.130s, including same-seed/different-seed and legacy reset regressions. A full-GPU same-input/same-seed sampling gate must pass remotely before admission. Before/after development: seed7, second sorted practice task per Approach/Move, Turn, Pass, Surround;128/320 executed-move limits. Simulator bitwise determinism is not claimed. New training verification requires action change, frozen prefix/vision invariant and exact saved-weight/probability reload.

Development tasks: sim_2025-03-14_17-23-11 (Approach), sim_2025-03-27_15-33-11 (Turn), sim_2025-04-30_15-45-14 (Pass), sim_2025-03-27_15-34-01 (Surround). All are disjoint from the four D216 training tasks. Four matched tasks offer a development signal only, not statistical proof of improvement. Existing batch stays closed; runtime admission creates a separate approved copy only after baseline and GPU seed gates pass.

Prepared source manifest: reports/d217_ppo_pilot/source_manifest.json; tests.txt. Launch and outcome pending.




D217 launch: same stopped Windows instance started16:41:37UTC; SSM online16:41:53UTC. Guest shutdown deadline19:41:37UTC successfully established before setup; no Python/Collection process or existing D217 directory. Source S3 upload/head size6973801bytes verified; immutable SHA2568fbe48c166e0266af1caec54500f312c21a0b7e47c6cfc943b2e109d4bb9b1ae. Setup/dispatch SSM174d8794-1a8d-4926-a4c7-1457d128e172 pending confirmation. Completion heartbeat d217-ppo-pilot-completion active every10minutes; no additional launch/retry authorized. Evidence reports/d217_ppo_pilot/launch.json.



D217 dispatch verified: pilot_driver.py running from isolated C:/runs/d217_ppo/code with results/status.json stage1 seed_sampling_gate. All75 AWS CPU tests passed2.880s; GPU model loading for seeded sampling gate. No optimizer update yet. Evidence SSM6d96531a-7fc3-4144-a1ef-cdfed4936005; launch.json updated.



D217 final failure (2026-10-10): after75 CPU checks passed2.880s, GPU seed stage failed before sampling because source.zip omitted reports/uav_flow_official_10shard/manifest.json, which official_stats() requires for action normalization. Base model loaded but no policy sample, baseline/candidate flight or optimizer update occurred. Failure after109.359s; therefore no PPO efficacy result. This was a deployment packaging error.

Diagnostics uploaded16:46:42–45UTC and recovered locally; evidence.zip4424bytes, ZIP CRC passed. SSMa2379ad1-ec22-401e-8f27-ae142cdebe6b confirmed no Python/Collection processes. Immediate Stop-Computer failed because backup shutdown was already scheduled; monitor requested EC2 StopInstances16:55:51UTC. EC2 subsequently confirmed stopped17:02:42UTC.

Local corrections only: include normalization manifest; add a pre-model assets gate verifying its hash against all72 collected samples and100-task practice inventory; cancel scheduled backup at final outcome then schedule forced shutdown10s later. Repaired archive extracted into a separate temporary directory passed assets gate (action_stats SHA2565284fb8dda90b4bb1c95ccb038d93fd69f30d98d203ff3117cdce375095dddb4). All75 local tests passed3.575s; shutdown fix syntax-checked but not remotely executed. Original archive preserved. No cloud upload of repair, no automatic restart/retry. Evidence reports/d217_ppo_pilot/{README.md,outcome.json,evidence/,tests_after_failure.txt,repaired_assets_gate.json,source_repaired_not_launched.zip}.



D217 shutdown confirmed: DescribeInstances returned stopped at17:02:42UTC (20:02 Israel) on2026-10-10. No restart. Completion monitor removed after final failure notification. Extracted-archive negative control also passed: withholding the manifest fails the new assets gate before model loading.



D217 retry2 authorized (2026-10-10): user explicitly approved restart and requested verification that it runs properly. Fresh C:/runs/d217_ppo_r2 and d217/r2 outputs preserve attempt1. Exact archive extracted locally; assets hash gate and75 tests passed2.157s. Source SHA2562b9c3a3bb01788f34762badb2d65d80914a062514e6db3c10be4457204339960,6975105bytes; includes normalization manifest. Same stopped instance started17:20:03UTC; session deadline20:20:03UTC (23:20 Israel). Same4-update/90min training cap, no automatic retry or extension. Stage order now assets,CPU,seed,baseline,admit,training,candidate,comparison. Verify actual gates and optimizer progress, not merely launch.



D217r2 dispatch verified: assets gate passed remotely (all72 action-stat identities match;100 practice tasks). CPU stage running. Backup deadline established; no preexisting Python/Collection or retry directory. Actual launch evidence SSM0dae4e6d-8a87-4fae-b3c8-c1a1a4faf505. Monitor d217-retry-verification active. No GPU/optimizer pass claimed yet.



D217r2 GPU seed gate PASSED: two identical saved-image calls atseed7 reproduced actionids,tokenlogps,contextids,boxtext andprogresstext exactly. All75 remoteCPU tests passed2.972s. Pipeline advanced to baseline_development; optimizer not yetstarted. Evidence SSMf46781f3-19d0-4a45-9eae-104c61026d73 andremote results/seed_gate.json.



D217r2 live execution verified: simulator calibration saved and passed; four action chunks recorded, HTTP predict200 responses advancing through17:28:09UTC. Baseline_development remains active; no completed-flight or optimizer-update claim yet. Evidence SSM4f019570-8231-412e-a946-cb65e2fc02ed. Continuous monitor will notify only first committed update and final outcome/actionrequired; fixed20:20:03UTC shutdown remains.



D217r2 first real PPO update verified18:02UTC (21:02 Israel): PPO_COMMITTED step1 matches CURRENT.json and committed manifest step1. Post-update KL0.006393057 below0.05, gradnorm32.2165 before clipping. Checkpoint step_000001_069c2634a4d24fdb9b44d6d4438229c1. Allfour baseline flights completed and admission passed; no candidate comparison yet, no improvement claim. Training active, fixed4-update cap/deadline unchanged. Evidence reports/d217_ppo_pilot/r2/first_update.json, SSMaaada61a-b6c2-4289-b9dd-ddd29f888266. First-update notification delivered; monitor now notifies final outcome/actionrequired only.



D217r2 early stop (2026-10-10): three PPO updates committed, full-batch postKL0.006393057,0.014743725,0.025864186, all below0.05. During fourth update preparation, ppo_engine.py116 raised the per-microbatch KL guard before optimizer.step; the fourth update was not applied. This shared guard rejects nonfinite KL OR KL>0.05; exact failing value was not logged, so do not claim the numerical cause is proven or conflate sample-level guard with full-batch average.

Wrapper marked training stage failed after3232.125s totalpipeline and skipped final frozen-weight/reload verification and candidate flights. Four baseline runs completed: Approach34moves/reward0.358493/progress0.781104; Turn15/0.924297/0.931755; Pass64/0.064481/0.923893; Surround10/-0.0000124335/0. All0/4 geometric completions. No before/after comparison or improvement claim.

Artifacts uploaded18:17:35–46UTC: evidence.zip2330029bytes and step3 checkpoint step_000003_4d10df9c2265439b810fce11a825becf (adapter322279296bytes,state529137772bytes). Archive CRC and CURRENT-to-manifest SHA verified. EC2 subsequently confirmed stopped18:25:26UTC. No restart or extra training. Evidence reports/d217_ppo_pilot/r2/{README.md,outcome.json,evidence/,checkpoint_pointer.json,checkpoint_manifest.json}. Nextproposal: offline guard diagnosis, then explicit early-stop checkpoint verification and its four matched candidate flights; do not weaken guard or infer benefit.



D217r2 final verification: EC2 stopped18:25:26UTC (21:25 Israel). Downloaded step3 adapter322279296bytes and optimizerstate529137772bytes both SHA256-match committed manifest. No further learning/restart. Monitor deleted after final notification.



### D218 - Authorized saved PPO candidate diagnosis and evaluation (2026-10-10)

User approved diagnosing the D217r2 guard stop, verifying the saved step3 checkpoint, and flying the four matched candidate practice tasks. No training or optimizer update authorized. Offline trusted checkpoint inspection confirms step3/cursor3 and next sample order30,31,68,20,29,44,71,11. A no-update GPU replay will distinguish excessive from nonfinite sampled KL, reproduce the full-batch recordedKL, compare saved visual weights with step0, verify frozen-prefix identity and exact weight/probability reload. Only after checks pass, fly the same second-sorted practice task per kind atseed7,128/320moves; compare against existing D217r2 baseline, without reflights or benchmark tasks.

One75-minute session on existing Windows instance, no automatic restart/extension, upload then shutdown. Local75tests passed2.808s; extracted-archive comparison self-control gives zero delta for allfour baseline copies. Code SHA256663d1001f709204a05fee39c8692efa410598e04c439c2ebcc0c7b07ed7f1f9e,6980164bytes. Evidence reports/d218_saved_ppo_eval/{offline_checkpoint.json,comparison_self_test.json,source_manifest.json,tests.txt}. Launch pending.



D218 launch: instance started18:44:02UTC,SSMonline18:44:18UTC; independent shutdown19:59:02UTC confirmed. No existing Python/Collection orD218directory. D217r2 checkpoint and baseline present. Dispatch SSMd7a8699a-337f-407d-a41d-fb0587ec5c15; monitor d218-saved-ppo-evaluation active. No optimizer/training in stagecommands; first actual verification result pending.



D218 diagnosis verified19:06UTC: all72sample replay reproduced full-batchKL0.025864185682722664. Original update4 guard reproduced at sample31,Surround practice task sim_2025-03-27_15-28-37.json,passage0,zero-basedcall5: finiteKL0.0915347338>0.05,logratio0.3994098902. Thus a finite per-sample policy-change limit, not nonfinite arithmetic, stopped learning.208 visual tensors unchanged againststep0; prefix/runtime identity and exactreloadprobabilities passed. Three candidate flights completed,Surround pending; no final comparison claim yet. Evidence reports/d218_saved_ppo_eval/diagnosis_observed.json,SSMf2723527-24f1-4eee-b162-4f4e460613f8. No extra training.



D218 final outcome and trajectory maps (2026-10-11): saved D217r2 step3 evaluated on four matched practice tasks at seed7, no additional training. Completion 0/4 -> 1/4. Approach progress 78.11% -> 68.55%, reward 0.358493 -> 0.344056; Turn 93.18% -> 100%, reward 0.924297 -> 1.988998; Pass 92.39% -> 95.56%, reward 0.064481 -> 0.594134; Surround 0% -> 100%, reward -0.000012 -> 0.717035. All episodes ended model_stop. Surround now traces an almost closed loop but differs from the reference geometry; maximum combined spatial/yaw deviation 0.840779 exceeds 0.75. Turn crosses the progress criterion but overshoots final heading; passing is not equivalent to better final heading accuracy. Approach stops shorter; Pass follows more of the detour but has a visible reversal and remains incomplete. These are four one-seed practice cases, not a benchmark or general improvement claim.

Sources: reports/d218_saved_ppo_eval/evidence/{comparison.json,completion.json,verification.json,after/episodes/}; baseline reports/d217_ppo_pilot/r2/evidence/before/episodes/. Before/after task identities and seed7 matched locally. Full trajectory map reports/d218_saved_ppo_eval/before_after_maps.png and PDF, reproducible with plot_before_after.py; x/y in meters from stored centimeters, Turn uses unwrapped heading against executed moves (no photo-frequency alignment). Spatial maps omit altitude; pass labels use the existing full scorer. EC2 confirmed stopped 2026-10-10 21:44:52 UTC; monitor deleted. No AWS work launched for plotting.



### D219 - Reward literature review (2026-10-11)

User asked what research says about drone/VLA PPO rewards. Methods-focused review retained 20 primary works and screened 10 additional candidates. Source-linked taxonomy and evidence: reports/d219_reward_literature/README.md and evidence.json; schema/count validator passed, but full resource/compute/reproduction audit remains incomplete. Direct precedents include RecoverFly (UAV-VLA PPO with progress and event rewards), EG-GRPO (UAV-Flow initialization with instruction-conditioned trajectory scoring), and CosFly-VLA (tracking geometry, grounding and terminal feedback). SimpleVLA-RL supplies the sparse binary alternative; VLA-RL supplies learned process rewards. UAV-Flow itself separates semantic success from reference-path similarity.

Inference: keep geometry as useful feedback but audit task-specific success and stopping separately from path similarity. D218 Turn highlights scorer sensitivity; it does not prove a causal reward exploit. Candidate next step is offline trajectory-ranking and resampling tests before any authorized paid pilot. No reward implementation, AWS work, benchmark flights, or further training performed.



### D220 - Local movement reward correction and counterexample audit (2026-10-11)

User authorized local validation/fixes across movements and examples before further training. Implemented scripts/rl/movement_reward.py, a versioned opt-in candidate preserving historical path_reward scores and default callers. Continuous terminal quality replaces the 98%-progress one-point bonus cliff. Turn uses symmetric heading error (10% requested angle, capped at5deg), penalizes excursions/reversal, and distinguishes metres from degrees. Measured-segment integration aggregates back to actual actions; original jumps/start-frame mismatches are rejected. Land completion requires explicit contact evidence; collisions cannot receive positive reward. Settings remain provisional, training_ready=False and samples unapproved; no admitted training data rewritten.

Evidence reports/d220_reward_audit/{README.md,audit.json,all_movements.png,turn_reward_fix.png,tests.txt}; generators scripts/rl/{audit_movement_reward.py,render_movement_audit.py}. 974/974 local audit checks passed: nine synthetic movement categories,100 practice references with early-stop/wrong-direction/perpendicular-detour controls, eight saved D217/D218 flights, resampling/duplicate/stopping/collision controls. All100 reference geometric self-controls passed; this is internal consistency, not independent semantic validation. Repository unittest suite:82 run,80 passed,2 skipped (Qwen/PEFT environment unavailable),6.898s. Tests include toy optimizer unit fixtures; no VLA training, AWS calls or new flights.

Saved scores before->after PPO, old=>corrected: Approach0.358493->0.344056 =>0.395388->0.346220; Turn0.924297->1.988998 =>1.663823->1.074717; Pass0.064481->0.594134 =>0.040179->0.601994; Surround-0.000012->0.717035 =>-0.000012->0.999233. Turn now prefers the smaller endpoint error0.617deg over1.416deg. Diagnostic completion becomes before-pass/after-fail; not model improvement or a benchmark result. Others remain incomplete.

Failures found during audit: initial Turn drift potential refunded an out-and-back excursion; fixed with integrated drift and excursion quality. Two apparent ascent failures were test-fixture mistakes: changing altitude along a steep route is not necessarily off-path; replaced with perpendicular detour. Final controls pass.

Semantic blockers: Land contains altitude-only instructions and lacks contact telemetry. Ascend9m at30deg reference rises4.5m (9m slope length), while an instruction explicitly saying altitude4m rises0.349m. Task intent must be reconciled rather than silently changing labels. Object identity/side/clearance remain unverified by geometry. No Takeoff category in the100 practice inventory. Candidate is not approved for all-movement training. A learned reward network/hybrid is possible (D219), but requires validated labels/preferences and these same counterexamples. No DAgger label repair, horizon change, KL-guard change or paid run performed.




### D221 - Explicit altitude practice repairs and landing evidence gate (2026-10-11)

User requested fixing altitude/landing and identifying pre-training checks. Reviewed normalized instructions and references plus active UnrealReadback transport. Created six separate practice copies: five commands explicitly describing distance along a slope, and one numeric Land4m reclassified as relative vertical descent. Full reference lines agree with normalized geometry within2cm. These are new, explicit practice instructions, not a claim about original author intent; original tasks remain untouched. Four object-relative actual landing tasks remain blocked on landing-site/contact evidence.

Implemented scripts/rl/vertical_tasks.py and audit_vertical_tasks.py: task repair, reference/instruction integrity checks, opt-in scorer, sustained contact validator requiring declared surface/fresh measurements/stability. Eight unit tests pass0.020s; six repaired reference controls plus24 bad variants pass54 assertions. Evidence reports/d221_vertical_landing/{README.md,manifest.json,tasks/,tests.txt}. Landing negative controls cover missing/brief contact, wrong surface, hover clearance, speed, collision and stale/duplicate timestamps.

Active simulator transport sets actor poses and reads pose/image only; no touchdown/contact provider exists there. Local validator is not live telemetry. Cannot certify actual landing offline; no inferred boolean or fabricated surface height. Next gates: semantic task/split review, simulator contact calibration, no-learning altitude/frame/stop preflight, fresh on-policy reward-versioned data/probability checks, then bounded PPO with unchanged KL protections/frozen-weight/reload/matched-flight checks. No AWS or training; no change to DAgger labels, historical evaluation, admitted PPO batches or runtime defaults.



### D222 - Local scripted altitude simulator preflight and KL explanation (2026-10-11)

User approved proceeding with pre-training checks and asked about KL. Wired explicit reward_kind opt-in through flight_collector, preserving legacy defaults. Added collector regression for horizon/bootstrap/reward totals/unapproved samples;11 tests passed0.182s. Verified6 corrected practice source flights disjoint from saved validation/exclusion manifests.

Executed scripts/rl/vertical_sim_preflight.py in installed local Unreal with scripted reference commands, no inference/training. Six of six tasks passed in213.5s; max measured endpoint error0.000637973cm; all stop detection and per-action reward sums correct. Evidence reports/d222_vertical_preflight/{README.md,simulator/completion.json,simulator/*,split_gate.json,source_hashes.json,collector_tests.txt,probe.out,probe.err}. This is kinematic plumbing verification, not policy performance, task understanding or physical landing. Simulator closed; no AWS or benchmark flight.

Landing site/contact telemetry remains absent. Real-policy baseline and fresh on-policy/replay checks are still required before any non-landing pilot; no PPO batch admitted or optimizer update performed. KL explained from actual code: sampled joint-action estimate r-1-log(r), microbatch and post-update guards, not path distance or an added loss term. D217/D218 sample0.0915347 vs limit0.05 and batch0.0258642; threshold unchanged.

**52. PPO.**
- Value head on the layer-normalised last hidden state; GAE with gamma 0.99, lambda
  0.95; reward = call_gain plus the flight reward on the last call; clip 0.05.
- Fix: value loss went 3.5 -> 211 without the layer norm (D185). With it, 3.54 -> 2.51,
  ratio about 1.0, 1-3% clipped (D188).

**53. GRPO trial** (D180): 8 tasks x 2 flights; the pipeline works; 4/8 tasks gave
identical flights at temperature 1.0, so there was no signal.

**54. Costs.**

| Run | Cost |
|---|---|
| D180 (3 flight tests + trial) | about $5 |
| D182 | about $2 |
| D185 check | about $0.60 |
| D188 | about $6.30 |

### K. Deployment

**55. 8 GB.** MISSING. GGUF conversion was shown to work (D157).

**56. Real drone.** The line and goal memory need only odometry (pose relative to the
start). MISSING: real flight.

### L. Limitations and ethics

**57.** Simulator only, teleporting evaluator, no physics, no real flight.

**58.** Weak on Turn (3/10), Surround (1/10), Pass (4/10) and Land (5/10) on test; Pass is
ambiguous by construction.

**59.** 100 test tasks were used during development (item 16).

**60.** The base model is Apache-2.0; the data is research-use only.

### M. Release and reproducibility

**61. Hugging Face.**
- Staged in `D:/drone_vla_pilot/release/hf_upload/`: adapter, config, action_stats,
  results, phases, curves, `example_inference.py`. Not uploaded (user decision).
- MISSING: final model card.

**62. GitHub.**
- github.com/asaelbarilan/drone_small_vla_training: trainer, data prep, server, scorers,
  RL loop, AWS and Windows scripts, README with reproduce steps.
- `requirements.txt`, `.env.example`.

**63. Reproducibility.**
- Phases and arguments are in `phases.json`.
- Data lists: `sim_val_flights.json`, `uav_flow_sim_excluded.json`, `tasks100.json`, and
  the validation and practice task zips.
- Base model revision pinned (D-130). MISSING: seeds listed per phase.

**64. Storage.**
- S3 bucket (name in `$env:VLA_BUCKET`, not in the repo):
  - `d170/store.tar`: 75 GB training store;
  - `d171/run/`: adapters;
  - `release/`: package tar, `g5_logs.tgz` and `lineage/`;
  - `d187/val_tasks.zip`, `d172/rl_tasks.zip`, `d188*/`.
- Local:
  - `D:/drone_vla_pilot/release/`;
  - `D:/drone_vla_pilot/runs/`;
  - `D:/drone_vla_pilot/data/rl/`.

### D223 - Authorized non-landing real-policy preflight (2026-10-11)

User approved no-learning policy preflight excluding landing, before further training. Selected13 practice tasks: one each Approach/Move, Retreat, Shift, Turn, Turn/Rotate, Pass, Surround, plus all6 D221 explicit altitude copies. Selection is disjoint from772 held-out/excluded source IDs. Baseline D208 step8000 bf16, seed7,128moves or320Surround; no optimizer updates or admission approval. Separate two-move horizon probe exercises actual-final-observation bootstrap. Every sampled call is replayed without gradients against its saved image/context and baseline identity; reward totals, movement counts and stopping boundaries are checked.

Local38 tests passed (19 reward,8 vertical,11 collector); new scripts compile. Dedicated scripts/rl/nonlanding_preflight.py, nonlanding_preflight_driver.py, nonlearning_job.py, nonlanding_check.py; audit_live_replay.py adds configurable expected episode count preserving default8. Wrapper scripts/win_eval/run_d223.ps1 uploads evidence then shuts down. Source overlay SHA256 f9907b89abcfa111fa6d132736d5bdbd13c8be8ecbc20519c7d44f754d92cc6f,92522bytes. Evidence reports/d223_nonlanding_preflight/{selection_gate.json,tasks/,overlay.json}.

Existing Windows instance started around15:02UTC, SSM ready15:02:38UTC. Backup shutdown18:02UTC confirmed before setup. No existing workers or D223 directory. Dispatched15:06:20UTC; launch SSM reported child-process timeout but independent15:07:13UTC read confirms job alive and model loading, not a failed trainer or successful flight claim. No duplicate launch. Heartbeat d223-no-learning-preflight-completion monitors final evidence and stop. No training, benchmark, landing, KL changes or automatic retry. Outcome pending. Geometry-only reward cannot verify object identity, passing side or clearance; these remain separate readiness limitations.



D223 final outcome (2026-10-11): no-learning preflight completed in2527.485s. All13 practice flights plus2move horizon probe recorded;76 sampled chunks replay exactly (token/joint error0); finite actual-final bootstrap1x2560. Independent local reward sums agree within2.22e-16; no invalid transitions. All13 main episodes ended stationary-action model_stop before caps. Geometric completion3/13: Turn,Turn/Rotate,and1/6altitude. Not comparable to D20882/100 validation (different selected tasks/scorer). Remaining kinds incomplete. Altitude:40deg3m descent endpoint0.017m; three tasks stop0.77–1.00m short;85deg9m ascent barely moves;75deg5m descent goes4.55m sideways with almost no descent,reward-1.4594. Thus move limits did not cause these endings.

Artifacts uploaded15:49:45UTC; evidence.zip9944566bytes, CRC/ETag verified,SHA2568790adb5546107d86202e2a5c7fdc72146f4282877ed1ffd476b63777af64d17. EC2 stopped verified around15:57UTC. Evidence reports/d223_nonlanding_preflight/{README.md,local_verification.json,evidence/}. No optimizer updates,benchmark,landing or restart. Requested technical gate passed; no automatic training approval. Next candidate is a bounded non-landing geometry-reward PPO pilot with explicit task mix/budget/matched checks and unchanged KL/frozen/reload protections. Object identity,passing side/clearance remain semantic limitations; landing contact still blocked. Completion heartbeat removed after verified stop and final notification.

### D224 - Reward-gated small non-landing PPO experiment (2026-10-11)

User authorized a small PPO experiment, landing excluded, conditional on verifying that better attempts receive better rewards before training. Scope deliberately narrowed to four numeric practice tasks:60-degree right turn,4m relative vertical descent,9m85-degree ascent,5m75-degree descent. These are a subset of the D223 split-verified practice tasks. Compare saved baseline seed7 with two fresh attempts seed8/9 per task. Only fresh seed8/9 samples may train. After learning, repeat samefour tasks seed7; this is same-task diagnostic evaluation, not held-out-task generalization.

Predeclared gate: endpoint improvement >0.10m (or>0.5deg for rotation), with lateral excursion no worse by>0.03m and final heading no worse by>0.5deg (rotation reversal tolerance0.1deg). At least one meaningfully better comparable pair per task; every such pair must have strictly higher reward. Stop without learning on ranking contradiction or missing meaningful variation. These conservative engineering thresholds are not statistical certification. Controlled straight-route fractions0/10/25/50/75/90/100percent and wrong-direction controls all rank correctly on four selected tasks. Prior13-task progressive-path probe also found no inversions. Full regression91 tests:89pass,2environment-dependent skips in4.446s.

Stages: controlled rank tests;8fresh practice flights; independent endpoint/drift/heading rank gate; exact no-gradient replay plus D223 horizon probe; admission; at most2actor PPO updates (lr1e-6,critic1e-4,clip/KL0.05 unchanged,frozenD208prefix/vision); exact reload;4matched candidate flights; comparison. Baseline D208bf16,chunk8,128moves. Trainer cap3500s plusprocesscap3600s; admission requires5700s remaining for learning/evaluation/upload. One3hour session, no automatic retry/extension/promotion. On rank gate block, preserve attempts and stop. No landing or benchmark.

Files scripts/rl/{reward_ranking_gate.py,reward_gated_flight_driver.py,reward_gated_job.py,reward_gated_outcome.py},scripts/win_eval/run_d224.ps1. Evidence reports/d224_reward_gated_ppo/{tasks/,controls.json,tests.txt,source_manifest.json,overlay.zip}. Overlay11910bytes SHA256c9f3e1a1873734b2a1dfedb91db0f534ffea0d20c6348ad7044e07123c5968f5. ExistingWindowsinstance start accepted17:31UTC,SSMonline17:31:51UTC; independentdeadline20:31:30UTC(23:31:30Israel). Setup underway; no training or collection claimed yet.


D224 dispatched17:34:08UTC on the existing Windows instance. Independent status read confirms controlled_reward_order passed and eight_fresh_attempts stage loading baseline. SSM dispatch9eb68c32-cc09-4da1-b1be-8fc611f8f3c7; status evidence0c676e85-d6ce-40ef-a9a8-d24762b79232. Harddeadline20:31:30UTC. Monitor d224-reward-gated-ppo-completion active. No optimizer update yet; conditional training gates remain mandatory.



D224 outcome: intentionally blocked before learning at independent_ranking_and_variation (exit2),1156.859s pipeline. Eight fresh seed8/9 flights completed. Two tasks lacked a meaningful comparable pair:60degturn (~0.907deg error,identical actiontoken sequences all3seeds) and4mverticaldescent (~0.823–0.828m error;seed8/9 tokens identical). Four comparable pairs across steep tasks all correctly ranked; no inversion detected. Steepdescentseed9 improves endpoint6.6249->0.9904m,lateralexcursion4.7379->1.7594m,reward-1.45944->-0.002236,still incomplete. Steepascentstationaryseeds7/9 (~9merror,nearzeroreward) correctly outrank seed8 wrong excursion(12.55merror,reward-2.87737); no useful upward attempt observed.

Zero optimizer/critic updates; replay,admission,training,reload,candidate evaluation never reached. This is failure to meet the conservative every-task variation gate, not proof rewards are wrong or PPO cannot learn. No bypass/retry. Possible next investigation is saved-distribution concentration/exploration or a separately approved targeted pilot on the variable steepdescent task. Evidence reports/d224_reward_gated_ppo/{README.md,local_verification.json,evidence/}. Independent reward sums agree4.44e-16,archiveCRC/ETag verified;5336485bytes SHA256a6d89a7a61aeefb1618c28b16e171facb4e0fbd831e1f816ebe7683c5414675f. Uploaded17:54:40UTC; EC2 shutdown in progress at first verification.

D224 shutdown verification: AWS DescribeInstances confirmed Windows EC2 stopped at approximately 2026-10-11 18:02 UTC. Artifacts were uploaded before shutdown. Zero optimizer updates; no restart or gate bypass.


### D225 - Local repeated-action and early-stop diagnosis (2026-10-11)

User authorized local diagnosis before more AWS spending. Audited all12 D223/D224 selected-task episodes (seeds7/8/9),56 calls;124 assertions verify exact goal-memory formatting, decoded/integrated proposals (<1e-8 stored-unit error) and measured stop boundaries. Evidence reports/d225_local_sampling_diagnosis/{README.md,diagnosis.json,analyze.py}, including source/input hashes. No AWS calls, inference, training, simulator flight, or runtime change.

Main finding: D224 memory-deadband1m/5deg zeros residual before the current reward's0.5m endpoint tolerance. In4m descent seed8, at measured-3.172m the model still says-0.8m remaining, but runtime supplies zero. Allseeds stop near0.82m short. This is a context/scoring mismatch; improvement after fixing it remains untested. Turn already completes (~0.9deg error), so its low variation is not analogous to failed descent.

Separate upstream error: initial remembered slope goals put vertical motion on sideways axis.85deg ascent seeds7/9 remember(0,0,0),seed8(0.8,9,0),instead of reference(0.784,0,8.966).75deg descent seeds7/8 remember(1.3,-4.8,0),seed9 correctly(1.3,0,-4.8),reference(1.294,0,-4.830). Goal memory preserves the first valid prediction. Planned PPO freezes prefix generation; action-only updates do not directly repair this generator. All initial images differ acrossseeds; prefix is greedy, so these are not isolated action-sampling comparisons.

Sampling enabled atT1,topk0,topp1,seeding per episode. Turn87.5percent selectedtokens havep>.99; firstseed8 chunk probability.99897, laterchunks.87880/.54946. Identical token sequences, not decoder collapse, explain repeated turn commands. Full logits not saved: entropy, alternate actions, total stop probability and temperature counterfactual unavailable. No discrete STOP token:10consecutive<3cm peraxis/<1deg measured moves. All12 end at10-63moves,not128cap; all saved unexecuted final tails are small. Quantizedzero decodes(+.270cm,+.013cm,-.055cm,-.056deg),below stop threshold.

Decision/recommendation: address numeric instruction-to-goal validity and compatible residual-zeroing tolerance before exploration or paid PPO. Prepare/test explicit numeric parsing and threshold/overshoot controls locally, then same-input no-learning context counterfactual with richer distribution logging; no reference-path oracle in normal inference. Historical deadband had overshoot rationale, so do not globally remove without checks. No fix, new paid run, gate bypass, training approval or efficacy claim. Earlier exploration-first suggestion was incomplete; upstream context has stronger concrete evidence.

### D226 - Local numeric-goal/arrival fix and discovered sampling override (2026-10-11)

User authorized local fixes and actual action checks before further AWS compute. Added explicit opt-in numeric_goals.py / server --numeric-goals: strict instruction-only slope/relative-height/turn parsing, correct axes, signed residual through overshoot, at-most0.10m/1deg arrival, canonical simultaneous axis wording for action generation. Box prefix retains original instruction. Unsupported/ambiguous commands and landing fail closed; no reference-path oracle or historical-mode replacement.

Found another material D225 correction: Transformers4.57.3 auto-replaced requested T1/top-p1 with model defaults T0.7/top-p0.8; saved D223/D224 server.log line13 confirms this. Recorded logps were untruncated T1, so replay equality did not certify actual sampling probabilities. Server now opts out of model defaults, verifies actual score/logp parity every sampled call, records action-categorical-v2; train_path_ppo rejects legacy/mismatched contract before model loading. Existing old sampled batches are not validated PPO data and must not be reused. No KL/gate weakening.

Downloaded existing D208 adapter to laptop; digest75d55cf748d084c463ccb26842c4266abecbc7e038b29960ee94fea887d1a711 matches. Local RTX4060/NF4 saved-image paired checks:6/6 coarse direction/stop cases pass; actual sampler logp error0 in12generations. At0.83m descent shortfall old context produces still chunk, fixed proposes0.531m descent, endpoint error0.309m. Ascent now1.539m upward instead of1.582m sideways. Steepdescent now1.057m downward with~0.001m sideways; turninitial24.59deg remains unchanged; synthetic at-goal pose still stops. Numeric-goal-only initial patch still overshot forward4.37m on descent; three-input wording ablation justified canonicalization. Fixed slope chunks are almost vertical; complete route proportions, corridor and stopping remain unverified.

54 distinct tests pass (11numeric,11collector,1observation,3reset,4sampler,24PPOpreflight). Evidence reports/d226_numeric_goal_fix/{README.md,model_sampling_fixed/actions.json,wording.json,source_manifest.json,test logs}. Earlier exploratory model/model_final files predate corrected sampler and are superseded. No VLA optimizer update, EC2 start, simulator flight, benchmark or reward change. Local NF4 differs from BF16; no full-flight/efficacy claim. Next gate is local closed-loop practice route/endpoint/stop validation, then fresh correctly sampled runtime-matched data before any paid training. Landing excluded.

### D227 - Interactive local reward inspection: 4 m descent (2026-10-11)

User requested playable, editable reward examples before more flights/training. Built a synthetic 4 m descent dashboard with depth 0-6 m in 0.1 m increments, sinusoidal sideways detour 0-2 m in 0.25 m increments, stopping toggle, playback, additive component display, and preview-only terminal-credit multiplier. All 549 paths scored directly with movement_reward.py / path_reward.py; both stopped/unstopped variants checked, additive terminal identity verified. Default current rewards: ideal straight 4 m = 2.000000; straight 3.2 m = 1.022430; 4 m with 1 m lateral detour = 0.946651; straight 5 m overshoot = 0.972305. These illustrate geometry scoring and are not model flight results or semantic validation. Weight control changes dashboard arithmetic only, not reward source or training settings. Browser checks passed for playback/pause, scrub, presets, stopping credit, preview weighting, and 360px layout. Evidence: reports/d227_reward_dashboard/summary.json. No AWS calls, simulator flight, model inference, or training. Local closed-loop D226 validation remains outstanding.


### D228 - Local repeated numeric flights: variation confirmed, reward gate not cleared (2026-10-12)

User authorized local no-learning repetitions and reward ordering checks before training. Eight completed practice episodes (four D224 numeric tasks, seeds7/8, scene seed7, max128 moves, local RTX4060 NF4 exact D208 adapter) use D226 numeric goals and verified categorical sampling. All four token sequences differ, though initial turn/vertical/ascent chunks remain identical; initial rendered images differ, so seed causality is not isolated. 89 calls/689 measured moves pass sampler-logp parity (max0), image hashes, valid movement and exact reward attachment; saved adapter digest unchanged. Three geometric completions out of eight, not a benchmark: turn both (rewards1.653/1.658); vertical one (0.747 horizon vs1.376 stopped); ascent neither (-4.928/-0.043, both horizon); steep descent neither (-2.281/+0.002, both stopped). Model overshoot/near-goal settling remains poor despite correct signed residuals; ascent can ignore negative remaining height.

Unchanged independent gate finds zero qualifying dominance pairs (turn difference too small, vertical endings differ, slopes trade endpoint accuracy against lateral/heading errors). No clean reward inversion established and no all-task ranking clearance. Material trade-off: steep descent reaches within0.068m after3.61m maximum route deviation for-2.281, while near-inaction ends4.988m short for+0.002. Current route penalties prefer inaction to a severe detour; review that preference explicitly before changing rewards or training. No gates/weights weakened; zero optimizer updates, AWS off.

Fixed newly exposed JSON serialization of NumPy goal metadata by native-float normalization; all12 numeric-goal tests pass. Initial failed attempt retained. A30-minute bound interrupted the first steep-descent flight after six completed episodes; only missing pair rerun under20-minute cap, completed451.7s. Interrupted data excluded. Own local simulator and policy server closed. Evidence reports/d228_local_numeric_flights/{README.md,summary.json,retry1/analysis.json,retry1/flight_paths.png,retry1/episodes,finish/episodes}; matching source hashes and post-run adapter hash saved. Local NF4 does not establish BF16, physical collision/contact, semantic success, training-runtime replay or generalization. Decision: training remains blocked pending deliberate route-vs-inaction preference validation and local correction/stopping work.


### D229 - Arrival-priority altitude reward, local recorded-flight regression (2026-10-12)

User requested clean arrival > arrival after detour > near-inaction before training. Implemented movement-reward-candidate-v4 in scripts/rl/movement_reward.py, changing Ascend/Descend only. Preserve v3 at reports/d229_arrival_reward/movement_reward_v3.py. Other categories, contact/collision/invalid-motion guards and geometric completion thresholds unchanged. Historical D224 admission remains pinned to v3 and cannot silently admit v4 samples; no training gate bypass or launch authorization.

For altitude, combine original integrated deviation and backtracking costs C, charge B(C)=0.5*(1-exp(-C/0.5)) via successive cumulative differences. Net progress remains signed/telescoping. Stopped arrival credit uses progress times final endpoint/heading Gaussian quality, without the historical worst-departure multiplier. Thus costs remain nonnegative and monotonic but saturate below0.5; clean paths retain the highest return2.0. This deliberately prioritizes arrival over severe route inefficiency. Saturation weakens discrimination among very large detours; it is not an obstacle-clearance/safety reward. Collision telemetry is still required for physical safety. No positive reward from returning to start after loops. Geometric corridor success remains separate from shaping reward.

Rescored the two exact D228 steep-descent episodes (input SHA256s in comparison.json): seed7 ending0.067804m from target changes -2.281488 -> +1.486883 (arrival credit0.986178); seed8 ending4.988103m short stays +0.002413256. Synthetic clean same reference scores2.0. Both recorded episodes retain geometric_complete=false. Per-transition totals conserved; v3 replay matches saved rewards to1e-9. Old dashboard remains explicitly v3 historical evidence, not the new scorer.

35 local tests passed:22 reward tests,5 executed-rollout tests,8 vertical/contact tests. New tests/test_arrival_reward.py checks exact recorded preference, synthetic ascent/descent clean/detour/idle/wrong-direction/overshoot/reversal controls, loop non-profit, duplicate-observation and resampling stability, collision/jump rejection and unchanged other categories. Evidence reports/d229_arrival_reward/{comparison.json,movement_reward_v3.py,source_hashes.json,tests.txt}. No AWS, simulator, model inference or optimizer updates. This validates the requested local reward ordering, not PPO improvement or full training readiness; fresh correctly versioned rewards/data and existing preflight gates remain necessary.


### D230 - Local PPO first gate blocked by overshoot-credit counterexample (2026-10-12)

User authorized local rescoring of eight D228 episodes, exact probability replay, and only if checks pass at most two PPO updates and matched flights. Rescoring all eight with D229 v4 exposed an unintended preference: ascent seed7 ends9.694m from target (initial distance approximately9m), horizon without stopping, yet scores+0.500004. Its projected route progress is saturated at1 while bounded cost cannot exceed0.5. The better ascent seed8 ends0.371m short but scores only+0.528335. Corrected steep descent preference remains+1.486883 vs+0.002413. Turn rewards unchanged; vertical stopped completion+1.538475 vs horizon+0.797725.

Independent synthetic9m vertical no-stop controls reproduce the failure: stationary0, clean1, double-distance overshoot positive despite returning to the initial endpoint-error magnitude. Evidence reports/d230_local_ppo/{rescoring.json,overshoot_controls.json,rescore.py}. First gate BLOCKED; zero optimizer updates, no inference/replay, no simulator, no AWS. Do not treat successful two-descent regression as all-altitude readiness. Next correction must remove endpoint overshoot credit while preserving clean > detour arrival > inaction and loop/resampling/collision protections, then rerun all eight and controls before probability replay or learning.

Prepared backwards-compatible --precision/--adapter parameters in scripts/rl/audit_live_replay.py for a later local NF4 audit; defaults preserve historical BF16 behavior. Audit was not launched because the earlier reward gate failed. No admitted rollout artifact or trained checkpoint was created.


### D231 - Overshoot correction and conditional local PPO (2026-10-12)

User authorized fixing overshoot, repeating local checks, then a two-update local PPO experiment only if gates pass. movement-reward-candidate-v5 changes altitude shaping potential to min(ordered route fraction, 1-current endpoint distance/initial endpoint distance). Its signed differences telescope; overshoot loses credit and recovery restores it. Retains D229 bounded route cost and arrival bonus, original geometric completion/collision/invalid-movement checks; other kinds unchanged. D229/v4 source preserved under reports/d231_overshoot_ppo/movement_reward_v4.py. Recorded-flight test output now writes D231 without overwriting historical D229 comparison.

36 local tests passed (23 reward,5 execution,8 vertical). Includes ascent/descent overshoot, wrong direction, recovery, looping, duplicate observations and resampling. Eight actual D228 flights rescored: turns1.653/1.658; vertical0.795/1.498; ascent-0.577/0.526; steep descent1.473/0.002394. Synthetic clean descent2.0. No invalid movement. These establish explicit local preference controls, not all-task or semantic certification. Source samples remain unchanged; old reward versions cannot be mixed into the candidate batch.

Started all89-call no-gradient replay locally with D208 NF4 and exact saved images/contexts. Pending replay and real final-observation bootstrap before admission. scripts/rl/admit_d231.py requires completed replay,89calls/8episodes, image hashes, sample contract and actual-move reward sums. Intended conditional pilot: at most2updates,lr1e-6,critic1e-4,clip/KL0.05,frozen prefix/vision,exact checkpoint reload, followed by4matched seed7 practice flights at128moves. No landing,AWS,benchmark,automatic retry or promotion. Outcome pending.


D231 local outcome: all89 sampled chunks replayed exactly (zero token/joint error), three real final-observation bootstraps finite,493.672s. Admitted89calls/8episodes with v5 reward. Local trainer completed critic warmup and saved step0; no committed actor update after approximately40minutes. User requested AWS handoff. Stopped only recorded trainer process tree; wrapper exit1 at2417.672s. Preserve initial checkpoint/logs; no local candidate flights or performance claim. Heavy local memory use (~7.9GBGPU and20GBprocess private memory) observed, not a proved root cause.

### D232 - Authorized AWS continuation of bounded PPO diagnostic (2026-10-12)

User requested moving the experiment to AWS while leaving. Existing Windows g6.xlarge started07:40:37UTC; fixed shutdown10:40:37UTC (13:40Israel), instance shutdown behavior stop verified, SSM online. Local NF4 samples are not reused for BF16 optimization. Fresh BF16 eight flights on four numeric practice tasks,seeds7/8,128moves,explicit numeric goals,landing excluded; fresh two-move horizon probe; allcall replay; v5 reward attachment, sampling/image checks; at most2actorupdates lr1e-6 critic1e-4 clip/KL0.05 frozenprefix/vision exactreload; four seed7 matched candidate flights. No test benchmark, auto retry or promotion.

Gate scope is explicitly a bounded diagnostic using D231 synthetic/recorded preferences, rejecting observed independent dominance inversions, no action variation or no reward variation. It does not claim D224's old every-task dominance gate passed; user authorized the revised reward and conditional pilot. Whole job deadline3h includes setup and upload; wrapper stops after final outcome, no shutdown between stages. Package260457bytes SHA2567c712e7e5ef30a39daaf2970e84e4e826535ae31dcb7fc38907c1a3408d4bcb5. Scripts d232_{job,gate,flights,outcome}.py and run_d232.ps1. Evidence reports/d232_aws_ppo. Dispatch pending verification.

D232 dispatch verified: SSM launch295a600e-d8d7-4d8e-ad64-0cbc6d9437cd returned childPID2720 then a transport error; independent read5561b7b6-b3c8-43ec-b314-d562d8baa21d confirms real job stage2 eight_fresh_attempts running, all23 reward regressions and controlled-order stage passed, model loading. No duplicate dispatch. Hard shutdown10:40:37UTC remains active. Completion heartbeat d232-aws-ppo-completion created, first committed-update and final-result notifications only. Training has not started yet. Local trainer process tree stopped; no committed local update.

### D232 completion - two PPO updates, no meaningful flight change (2026-10-12)

All stages completed in 5329.907s; four matched seed7 candidate flights verified against fresh BF16 baseline. Two committed actor updates, post-update KL0.01483305/0.02559541 below0.05, action_changed=true, frozen_adapters_unchanged=true, reload_logps_exact=true. Evidence archive CRC verified locally; matching task IDs/seeds and pose counts checked. Uploaded final checkpoint adapter/state/manifest and evidence; EC2 independently verified stopped at09:22UTC, before deadline. No restart or promotion.

Before -> after: right turn reward1.839558 ->1.839558, endpoint0.077602 ->0.077602m, heading0.579 ->0.579deg,30moves; vertical descent1.975640 ->1.975655, endpoint0.029727 ->0.029727m, heading0.141 ->0.139deg,44moves; steep ascent1.481637 ->1.480794, endpoint0.076237 ->0.076315m, heading0.986 ->1.007deg,108moves; steep descent1.704983 ->1.704970, endpoint0.024143 ->0.024140m, heading0.033 ->0.045deg,61moves. All stopped via model_stop both times. Progress unchanged:0.99035,0.99317,1,1. Geometric completions remain2/4; slope detours remain. Tiny differences do not establish improvement; paths nearly overlap. Four training-task comparisons are not generalization evidence. No claim that further updates would solve route errors.

Sources: reports/d232_aws_ppo/evidence/{comparison.json,completion.json,training_report.json,attempts/episodes,after/episodes}; verification_local.json; before_after.png (positions converted from centimetres to metres; turn shown as yaw over pose index). Reward fixes passed local controls but this pilot shows no meaningful behavioral benefit. Decision: retain baseline; review evidence before any further paid experiment.

### D233 - Local saved-action audit and earlier Surround confound (2026-10-12)

User requested local diagnosis before more AWS. Audited D232 saved episodes and source code against exact deployed package SHA256. All five inspected source files match package. All1024 sampled action tokens across32calls/four seed7 pairs are identical before/after; move counts and stop decisions also unchanged. New action adapter digest c79577c6035955272262b7815a66e14ab614797d940785b0deb63942f3fbf7ac is recorded in candidate calls, versus baseline75d55cf748d084c463ccb26842c4266abecbc7e038b29960ee94fea887d1a711; prefix unchanged. Code selects action adapter immediately before generation, consumes sampled tokens directly, and resets RNG once per episode. No output-cache or wrong-path bug identified by this audit; runtime activation is not independently proved by a file hash alone.

Selected-token median probability before0.9999318;75.390625% exceed0.99. Same seed plus concentrated probabilities is compatible with unchanged choices, not proof of the complete cause. Two updates cover16of75chunks under deterministic seeded order (reconstructed:1turn,2vertical,8ascent,5descent); no full epoch. All32image hashes differ, only17text contexts match: chosen-logp differences cannot isolate weight effect. Full logits/entropy/alternatives were not saved. Do not state bugs are ruled out or more updates would fix behavior.

User recalled D218 Surround changed substantially. Direct historical audit found the critical first-call confound: D217r2 baseline context progress_text='Left +00.0,+00.0,+00.0,+000'; D218 candidate='Left +00.0,-00.1,+00.0,+360'. First32action tokens differ at16positions; baseline stop=true,candidate stop=false. Baseline2calls/10moves vs candidate26calls/201moves. Both initial images and contexts differ. Thus the earlier visible gain is not an isolated action-PPO comparison: the upstream generated goal says done in one case and360degrees remaining in the other. This does not establish why the frozen-prefix outputs differed (images/runtime/sensitivity need controlled replay). D232 instead uses instruction-derived numeric goals. Correct earlier explanation: old Surround change does not prove present unchanged flights are a bug, nor prove PPO alone caused that older gain.

Evidence: reports/d233_saved_action_audit/{audit.py,audit.json}; historical reports/d217_ppo_pilot/r2/evidence/before/episodes/sim_2025-03-27_15-34-01_r0.json and reports/d218_saved_ppo_eval/evidence/after/episodes/sim_2025-03-27_15-34-01_r0.json. No cloud operations, inference or optimizer updates in this diagnosis. Next discriminating test: exact same saved images and fixed goal context through old/new action adapters, assert active adapter at generation, compare full token distributions and sampled/decoded actions with identical RNG. Do not launch further paid training on an unverified attribution.

### D234 - Persistent full PPO training objective (2026-10-12)

User requested carrying the task through substantive AWS PPO training and evaluation without repeated routine confirmations. Goal active. New total instance-hour cap requested (6/12/24 hours); answer pending, so no longer paid run dispatched. Existing pilot caps are not extended. Landing and273testbenchmark remain excluded. Independent local preparation continues.

Immediate controlled test: reports/d233_saved_action_audit/same_input.py loads old/new/old adapters on identical saved D232 images and token contexts. It asserts active action adapter, compares full teacher-forced token distributions and fresh seed7 samples, and requires exact old-repeat identity. Saved D232 adapter/config/restore marker downloaded; manifest digest matches committed step2 and each downloaded file hash verified. No instance start. Local NF4 test is a loading/sensitivity diagnostic, not BF16 equivalence. Process bounded to1200seconds; result pending at this entry.

Requirements for substantive training rather than another smoke test: choose balanced authorized nonlanding practice data and held-out development tasks; no benchmark or held-out data in gradients; fresh on-policy data per round; exact runtime/sampler identity; full intended batch coverage explicitly reported with unprocessed samples accounted for; checkpoint/frozen/reload assertions; unchanged numerical/reward validity guards; fixed full-session shutdown including evaluation/upload reserve. A KL stop is recorded as early stopping, not silently bypassed or called a completed schedule. Include other nonlanding movements rather than redefine full training as only the four numeric examples. Numeric-goal parsing currently supports only turn/vertical/slope: broader task goal handling must be checked before admission, and cannot use reference trajectories as inference targets.

Before dispatch finalize a budget-compatible schedule with explicit episode/update/round counts, measured time estimate and held-out evaluation reserve. Compare paths, endpoint/heading/stopping and per-kind rewards, retain baseline unless evidence supports promotion. A saved checkpoint alone is not goal completion; neither is a test suite. No guarantee of improvement or instruction to spend indefinitely. Same-input diagnosis must be resolved first.

D234 preparation: added committed actor-batch coverage to ppo_engine report: scheduled/committed update windows, total/unique actor sample visits, unvisited indices, per-task visits and schedule_exhausted vs max_updates. Read-only forward/critic passes do not count as actor coverage. Four engine tests passed, including partial cap, remainder batch, no-op resume coverage and interrupted-resume equivalence. No loss, optimizer, sampler, threshold or training defaults changed. This corrects reporting, not the unchanged-action behavior. Controlled same-input test remains running under1200s limit; session14433, local process36456. Longer cloud budget remains unanswered; no dispatch.

D234 inventory and operational plan saved under reports/d234_full_ppo_preparation. Audited95nonlanding practice tasks across8kinds; zero validation/exclusion source overlap; all reference self-score checks pass. Only7 instructions supported by numeric parser;88 require generated-goal audit. This is a preparation inventory, not a semantic or launch gate. Longer schedule must include all intended movement types and explicit full-batch coverage, fresh on-policy rounds, development checks and validation/upload reserve; final counts await measured diagnosis and user runtime cap.


### D233 controlled same-input result (2026-10-12)

Completed bounded local NF4 old/new/old-repeat diagnostic in671.094s, exit0. Eight fixed saved D232 image/token contexts (first and penultimate call of each of four tasks), active action adapter asserted, verified checkpoint hashes. All eight full distributions changed: maximum log-probability changes0.3369-0.4375; mean full categorical per-token KL ranges0.000003045-0.00106799. One teacher-forced argmax position changed out of256. Fresh seed7 generation nevertheless produced identical256/256 sampled tokens across old/new. Reloading old reproduced every log-probability and sampled token exactly (zero maximum repeat error in every case).

This establishes that the candidate adapter affects computation in this local controlled runtime, while identical sampled actions can persist despite changed distributions. It rules out an entirely inactive candidate adapter for these eight local cases, not every possible cloud/runtime issue or all inputs. NF4 diagnostic is not BF16 equivalence and its full categorical per-token KL is not the trainer's sampled joint-action KL estimator. No training, simulator flight or instance start. Evidence reports/d233_saved_action_audit/{same_input.py,same_input.json,same_input_logits.npz,same_input.log}. Repeat fixed-input activation/distribution check in the intended BF16 runtime at the next authorized session before training. Do not infer that more updates necessarily improve flights. Total longer AWS runtime cap still pending; no cloud launch.

D234 historical goal screening: reports/d234_full_ppo_preparation/{screen_goals.py,historical_goal_screen.json} reads21saved episodes (D22313,D217baseline4,D218candidate4). D223 ascent sim_2025-03-27_17-35-01 begins with all-zero progress and immediate stop, ending9.00337m away. D217 Surround also begins with zero progress/stop; D218 corresponding first context contains360degrees. These reproduce known failure contexts, not new controlled causal evidence. Other nonzero initial goals cannot be certified correct merely from this screen. Older summary schemas leave endpoint/completion null explicitly rather than assuming success. Source inspection confirms generated-goal mode locks the first parsed goal and applies deadband; explicit numeric mode bypasses that inference. Full nonlanding training must preserve explicit numeric repairs where supported and audit generated contexts for other kinds. This screen is not a launch gate or rationale to force every zero goal to move. No model, cloud or training launched.

D234 prepared scripts/rl/audit_d232_fixed_inputs.py for the intended BF16 runtime, derived from completed D233 diagnostic without altering historical script/results. Required explicit evidence/baseline/checkpoint/output/precision; pinned committed manifest and adapter digests; required config/weights/exact-restore hashes; exactly8contexts; finite distributions, exact old-repeat and observed candidate influence required. New output directory prevents historical overwrite; records runtime identity. Syntax and CLI help verified locally only. GPU BF16 gate remains unexecuted and cannot be claimed passed. No cloud start, training or new inference. Paid session still requires total runtime cap.

D234 partitions pinned by reports/d234_full_ppo_preparation/partition.py:87training practice sources,8development sources (one per nonlanding kind),85nonlanding validation sources from existing100task validation set. Train/dev/validation source sets disjoint; actual selected task bytes hashed including existing altitude repairs. Selection SHA256 ordering is independent of outcome. Training kind counts:Approach4,Altitude4,Pass24,Retreat4,Shift4,Surround24,Turn19,Rotate4; balance rounds rather than silently overweight large kinds. Development and validation have historical exposure and are not an untouched test benchmark. partitions.json SHA2566c7c346d6a6dacf1f69b76e5ee022234d72aaba197b2389da71ee08bec5ed0bc. This pins eligible pools, not a claim all87will fit a runtime cap or a launch gate.

Timing basis saved separately: D232 total5329.907s,training stage2164.468s,two committed updates,16of75actor samples visited. A single75sample pass needs10accumulation windows. Training-stage duration includes warmup/setup/fullbatchKL/checkpoints, so dividing by two does not yield reliable marginal update timing. Final round counts and evaluation reserve still require the approved cap and measured broader-task throughput. No AWS or training launched.

D234 goal waiting on required total AWS runtime cap: unanswered across repeated continuation turns despite completed independent local checkpoint diagnostic, coverage reporting/tests, task inventory/partitions and prepared BF16 audit. EC2 DescribeInstances independently confirms the existing g6.xlarge is stopped. No launch or cost-cap extension performed. Goal is incomplete: broader generated-goal audit, budget-specific round schedule/runner, BF16 checks, substantive training and final evaluation remain. The next committed cloud schedule cannot be selected or dispatched without the requested cap. Preserve prepared artifacts and full objective; do not substitute another short pilot or claim training complete. Await user total instance-hours (requested6/12/24, or explicit alternative).

### D234 approved12-hour full-pass PPO session (2026-10-12)

User explicitly approved up to12instance-hours total including setup/training/evaluation/recovery. Existing Windows g6.xlarge start request succeeded16:41:37UTC; deadline2026-10-13 04:41:37UTC (07:41Israel), no extension. Starting from D208/D207step8000, not D232. Prepared up to2balanced on-policy rounds,8kinds x1task x2seeds=16episodes per round; full admitted sample pass with ceil(N/8)updates,lr2.5e-7,critic1e-4,clip/KL0.05 unchanged, frozen prefix/vision. Each new round recollects fresh data and has independent replay plus2movehorizon probe. Fixed development8 and matched baseline/candidate85nonlandingvalidation. Evaluation reserve13500s before updates; second round only if budget remains. No guarantee2roundsfit or improvement; no auto promotion, landing or273testbenchmark.

Mixed instruction-only goal routing preserves numeric repairs where supported, generated goal memory otherwise. Validation-only collector preserves original instructions and is explicitly ineligible for training admission. One original validation altitude task sim_2025-03-27_17-17-02 has normalized/reference mismatch (reference endpoint150.6,6,-131.4cm); preserve original task, flag it, do not rewrite benchmark to pass a strict training-label check. Training altitude checks remain unchanged. No prepared repaired-validation variants are used.

Local checks:12numeric-goal tests,23reward regressions,4trainercoverage/recovery tests,4preparation tests passed. Preparation tests verify95clean>inaction controls, pinned task bytes, real comparison identity/count and validation-only rejection; not semantic certification. Python and PowerShell syntax checked. Package590341bytes SHA256c9557823774b3754858738544c2c18e01a99bcaabaf5058f45a26a52e68c99e8 uploaded. BF16 fixed-input gate precedes new flights. Single-dispatch wrapper preserves final artifacts/committed checkpoints and shuts down on success/failure; readiness deadline15min. EC2 shutdown behavior stop independently verified. Dispatch pending readiness; no actor update yet.

D234 user steering: user explicitly rejected spending the12hourcap without intermediate evidence. Revised protocol pauses after every4committed actor updates and flies8matched development tasks before continuation. Long85pairvalidation moved after demonstrated development gain. Continue only when sampled actions change, initial goal contexts are comparable, completion does not regress, and completion improves or both mean reward and endpoint error improve meaningfully. Unchanged/regressed/confounded cases stop for review; no claim a small dev gain proves general improvement. Planned optimizer schedule/config stays fixed across the pause; new exact pause/resume equivalence test passed. Five preparation tests now include real D232 unchanged-action rejection and synthetic gain/regression/goal-confound checks.

Original D234 setup FAILED before any fresh flights or PPOupdates: missing reports/uav_flow_official_10shard/manifest.json during BF16diagnostic. Evidence uploaded automatically at17:01UTC and downloaded reports/d234_full_ppo_completion/evidence. Wrapper already initiated shutdown before attempted sequence-change command; that SSM command timed out. EC2 independently verified stopped. This was a deployment packaging error, not evidence of PPO behavior. Corrected package includes normalization manifest and validates its hash against all saved D232diagnostic calls before GPUloading. No training-label/reward/KL gate weakened.

Corrected preflight recovery prepared under C:/runs/d234_early_review, same existinginstance, same absolute deadline2026-10-13 04:41:37UTC. Total12instancehourapproval includes recovery; wall deadline is not extended for stopped time. Package593591bytes SHA2565742e65620d8fb8d3c80948b87c4fa5fe1d4adfc22e17cda2a3b4c2e82ab9ef6. Five engine tests, five preparation tests, prior12numeric/23reward tests passed. Normalization SHA2565284fb8dda90b4bb1c95ccb038d93fd69f30d98d203ff3117cdce375095dddb4. Recovery start requested; verify actual state and single dispatch before claiming execution.

D234 early-review corrected dispatch verified independently at17:13:42UTC: bootstrap17:12:45UTC PID4956; Python driver and BF16 audit child live. deployment_assets passed (180pinnedtaskfiles; normalization hash matched), reward regressions passed, bf16_fixed_inputs running. No actorupdates yet. Deadline remains04:41:37UTC. DispatchSSM db9a96c6-0ee8-4697-ad10-b50024b01ad9; independent verificationSSM f4f8338e-afb8-4017-a1ad-897e8fa54e61. Completion monitor d234-full-ppo-completion updated to currentroot and required EVERY4update intermediate reports. Original setupfailure remains historical and must not be confused with this run. Notify each intermediate result; unchanged/regression/goalconfound gate blocks furthertraining and longvalidation.

### D234 early-review recovery stopped at BF16 repeatability gate (2026-10-12)

Corrected run passed deployment_assets and reward regressions, then failed bf16_fixed_inputs after303.843seconds. All24 forward/generation cases completed (8old,8new,8old_repeat); assertion requiring zero old-repeat log-probability error AND identical sampled tokens failed. This is a repeatability-gate failure, not evidence of reward quality or training efficacy. Zero fresh flights, zero actor updates, zero batch coverage; frozen/reload training checks and intermediate comparisons were not reached. No new trained checkpoint.

Diagnostic limitation: audit_d232_fixed_inputs.py asserts before saving summary/raw logits, so the uploaded evidence does not contain mismatch magnitudes or identify whether probabilities, tokens, or both differed. Do not infer numerical drift, adapter corruption or a successful candidate effect from the assertion alone. Next local preparation should preserve failed-case summaries and weight/dtype/runtime identities before assertions, then isolate initial adapter loading versus reloading and deterministic execution; retain the gate until explained. No automatic cloud retry, threshold relaxation, or deadline extension.

Evidence uploaded17:18UTC and downloaded to reports/d234_early_review/evidence.zip and evidence/ (failure.json and stages/bf16_fixed_inputs/00.log). AWS independently confirmed the existing instance stopped during this monitoring turn. Completion heartbeat deleted after failure notification and stop verification. Full PPO objective remains incomplete; no behavioral-improvement claim and no model promotion.


### D235 - Local BF16 reload diagnosis and evidence-preserving diagnostic (2026-10-12)

The local reproduction identifies a precision-loss mechanism in PEFT 0.17.1. Initial PeftModel.from_pretrained preserves saved FP32 LoRA values. Deleting the action adapter and calling load_adapter on the same BF16 base instead produces values exactly equal to FP32 -> BF16 -> FP32 rounding. The final FP32 dtype alone hides the lost precision.

We reproduced this on both CPU and the local RTX 4060 GPU using a tiny Llama and slices of actual D207 baseline q_proj LoRA A/B tensors. Reload alone (without even loading the candidate) changed both tensors: maximum weight error 0.0001204684. Output-logit error reached 0.00390625 and one argmax token changed among 256 positions. Repeating without reload gave zero error. Restoring the initial effective action snapshot gave identical weights and zero output error. Prefix weights were unchanged. An FP32-base control showed zero weight/output difference across reload.

This is direct evidence of a loader precision bug in the local reproduction, and a strong explanation for D234. It is not a measurement of the missing full 4B AWS output differences. Full BF16 VLA verification remains pending; the laptop has 8 GiB VRAM. The earlier NF4 repeat test passed and does not certify BF16 behavior. This finding does not establish that D232's unchanged flight paths were caused by this reload issue.


Diagnostic now saves failures, raw probabilities, sampled tokens, tensor hashes/dtypes and partial cases before assertions. Added no-reload control and uncorrected reload measurement; final repeat restores the initial effective action snapshot exactly, with strict weight/output/prefix checks. No gate relaxation or production loader change. CPU BF16, local GPU BF16, FP32 negative control, and diagnostic failure-persistence checks passed. Sources: reports/d235_reload_diagnosis/{README.md,reproduce.py,tiny_bf16_reproduction.json,tiny_cuda_bf16_reproduction.json,tiny_fp32_control.json,diagnostic_checks.json}; scripts/rl/{audit_d232_fixed_inputs.py,reload_audit.py}. No AWS operations/spending, new flights or training. Full BF16 VLA check remains pending; no cloud restart authorized by this local task.

### D236 - Authorized bounded full-model verification; AWS capacity blocker (2026-10-12)

User authorized one bounded AWS full-model reload verification, explicitly no training. Prepared verification-only d236_verify.py and run_d236.ps1: at most30minutes instance runtime including setup/upload, at most1200seconds diagnostic subprocess, fixed deadline, upload on exit and shutdown. Five passes on eight pinned contexts: initial, no-reload repeat, saved candidate, uncorrected baseline reload, exact initial-snapshot restoration. Requires exact restored weights/log-probabilities/sampled tokens, unchanged prefix and finite candidate influence. No flights or optimizer updates, no full-training driver invocation.

Package reports/d236_aws_reload_verification/code.zip:597201bytes SHA25629f41e576a4e0c2184211752c1c00cc8f85340b77fe4c800912a9300f373ddda. Python AST and PowerShell parser checks passed; includes prior normalization manifest plus repaired diagnostic/helpers. Uploaded prepared package only. The existing Windows instance was stopped and shutdown behavior stop verified. StartInstances failed with InsufficientInstanceCapacity (AWS connector reports SDK max retries3). No verification job dispatched, no inference/training/flight results, and no claim of passing full-model verification. No alternate instance launched. Preserve package for a future authorized available-capacity attempt; no recurring monitor or unbounded retry created.

D236 capacity-retry authorization: user explicitly requested retrying the SAME existing instance when capacity becomes available, retaining30minute total runtime cap and verification-only scope; no training. Created heartbeat d236-reload-verification every5minutes to check state and retry only capacity failures before any successful dispatch. After successful start persist immutable start+30minute deadline, set backup shutdown before setup, allow at most10minutes for SSM readiness, never restart a failed diagnostic or extend limits. Notify completion/pass/failure; remove monitor after independently verified stop. Immediate retry around19:21UTC again failed InsufficientInstanceCapacity (connector SDK max retries3). Verification has not run and no pass claimed. Existing package remains unchanged; no alternative machine/hardware authorized.

D236 capacity retry succeeded2026-10-12 19:31:39UTC. Immutable verification-only deadline20:01:39UTC (23:01:39Israel),includingstartup/setup/upload. SSMreadinessdeadline19:41:39UTC. No job dispatched yet; no training,flights,restarts,deadlineextensions. Monitor persisted fixed deadline immediately.


D236 verification dispatched19:35:07UTC after package hash verification. Independent SSM0e5deb10-86a7-4ada-9b33-e4b7c3622b1b confirmed wrapperPID2896,driver4936,audit1732 live and dispatch markers. Backup shutdown20:01:39UTC scheduled successfully before setup; no training/flight process launched. Original dispatch SSM c2a59ef7-3c14-46e6-99dd-b6240deabdc6 returned transport Failed invalidargument after printing PID; independent process evidence confirms actual verification running. Do not duplicate. Running notice delivered; full-model pass still pending.


### D236 - Full 4B BF16 exact reload verification PASSED (2026-10-12)

Verification-only job completed in461.515seconds (five-pass audit349.78seconds) with zero training updates and no simulator flights. All8fixed D232 contexts passed exact old/no-reload and old/restored-repeat log-probability equality and sampled-token equality. All712 original action tensors restored exactly; prefix weights unchanged across all5passes. Downloaded evidence.zip CRC checked; independent local comparison of all40raw arrays confirmed finite values and bit-exact repeat arrays. Checkpoint/image/context identity checks passed in the diagnostic. Results uploaded and wrapper initiated shutdown; observed stopping19:44:28UTC, final stopped verification pending below.

Uncorrected baseline reload changed712action tensors, with maximum per-context log-probability errors0.312496-0.480463; sampled tokens happened to remain unchanged for all8contexts. Restoring the initial effective weights produced zero probability error and identical256sampled tokens. This reproduces full-model loading drift and validates the diagnostic snapshot restoration. No threshold relaxed. Candidate D232weights changed504tensors and influenced all8distributions;3of256 fresh sampled tokens changed in one fixed context. That is adapter sensitivity, not a new flight improvement or PPO-efficacy result. It does not contradict the historical D232 paired flights with identical saved actions, whose images/contexts differed.

Evidence: reports/d236_aws_reload_verification/evidence/{completion.json,fixed_inputs/same_input.json,fixed_inputs/same_input_logits.npz,audit.log}; verification_local.json. ArchiveSHA256eeee1f2caba48bc2c78a38cc9d0995b4e1bc2f96f19b56289f9dc78e3f201542. User authorized verification only; no further training launched or approved by this result. Production training/deployment loader changes were outside this diagnostic fix. Full PPO training objective remains incomplete.

D236 shutdown independently verified: EC2 DescribeInstances returned stopped at2026-10-12 19:52:17UTC (22:52:17Israel), before20:01:39UTC deadline. Eight-case full-model exact reload verification passed; zero training/flight activity. Final completion notification delivered and shutdown monitor removed; no further work launched.


### D237 - Four-update PPO review authorized (2026-10-13)

User authorized applying the verified reload fix and PPO with comparison after4updates before any continuation; approved up to4instance-hours total. Shared CompositePolicy.load now always exact-restores saved action tensors, including unmarked baselines; reusable reload_action_exact used by trainer checkpoint verification. Existing frozen prefix loading semantics retained.5composite BF16 integration tests,5engine/coverage/resume tests,5preparation tests,2new orchestration review-boundary tests passed. New test demonstrates favorable behavior cannot trigger continuation and step3cannot be represented asstep4.

D237 pipeline: assets/reward/BF16gate,8development baseline flights,16fresh practice flights(8nonlandingkinds xseeds11/12),fresh2moveprobe,all-callreplay/admission,4committedupdates(lr2.5e-7,critic1e-4,clip/KL.05),exactreload/frozenchecks,8matchedcandidateflights then unconditional userreview andshutdown. No furtherround/longvalidation/landing/testbenchmark/promotion. Plannedfull-passconfig retained for explicit future resume;onlyfirst4windowsauthorized. Localunit-test updates are not cloud training.

Package602975bytes SHA256c7d9166e6b07226fc6471033d33d408fe8150a6760bd8d815fd9e383de75681f. Same existingWindowsinstance start succeeded2026-10-13 10:53:08UTC;immutableharddeadline14:53:08UTC(17:53Israel),includingallsetup/upload. No extension/restart. Monitor d237-four-update-comparison created. Actualdispatch andfirstactorupdate stillpending.

D237 backup shutdown14:53:08UTC verified bySSMc6c512b9-f5e1-4281-aa40-df1ae8026f59; noexistingworkers/markers,123GBfree,diagnosticassetsavailable. Single dispatch submittedSSMecb66d49-1bbd-424d-abac-db14e144954d; actualworkerstatuspending. No duplicate launch permitted.

D237 independent launch verification: SSM 60b01a4a-c74f-4733-a1f6-713da8ea9109 confirmed bootstrap/dispatched markers at 2026-10-13 10:56:53 UTC, live driver and BF16 audit. Status reports deployment_assets and regressions passed, bf16_fixed_inputs running. No cloud actor update verified yet. Four-update review boundary and 14:53:08 UTC hard shutdown remain unchanged.

### D238 - User-approved two-task replacement within D237 deadline
User approved switching now to numeric turn/altitude pilot, retaining 2026-10-13 14:53:08 UTC (17:53 Israel) hard deadline and mandatory review after four updates. D237 was superseded during baseline flights with zero actor updates; its wrapper/process tree was terminated and evidence preserved. Instance remains on; original backup shutdown remains.
D238 comparison: 23.471-degree right turn and 9 m ascent at 30 degrees, seed7. Training: separate-source 25.968-degree right turn and 3 m descent at40degrees, seeds11/12. Turn instructions explicitly rewritten from reference yaw: synthetic numeric diagnostic, not object-grounding validation. Source partitions remain disjoint. Numeric parser and clean-versus-inaction controls passed for all four tasks; two orchestration tests verify unconditional review and rejection of fewer than four reported updates. Existing passed D237 BF16 exact-repeat gate reused without model-code change. Replay, admission, frozen weights and exact checkpoint reload remain required. Accumulation8, actorlr2.5e-7, KL/clip.05 unchanged. Insufficient samples or failed checks still block training.
Baseline/candidate stage caps25min; four collection flights45min; replay40min;35min comparison/upload reserve. Four updates then two matched candidate flights and unconditional stop. Compare saved action tokens and decoded actions as well as paths/rewards; a change alone is not improvement.
Overlay14201bytes SHA2566ac50d9687201e95fad0fbe70f95219fa3e6d3a368a2ea3e3477cb914935b824. Single replacement dispatch SSM246abad1-b9fb-422b-8987-054f97ce9858; actual launch pending independent verification. Monitor updated to D238 scope; no duplicate/retry/extension authorized.
D238 actual launch independently verified by SSM f7bb6225-7170-48a5-b810-a7c9fd35ba3c: dispatched 2026-10-13 11:37:30 UTC; deployment_assets and regressions passed; before_dev running with live driver/flight processes. Zero actor updates at this check. Original 14:53:08 UTC deadline unchanged.

D238 final training outcome (2026-10-13 12:19 UTC): failed before second actor update at ppo_engine.py117 pre-step KL guard. Exactly one PPO_COMMITTED log: step1 post_kl0.009965519255729305, grad_norm36.19400405883789. Saved CURRENT points step_000001_2b8a2aad9e554233a501b979d7781c4f; adapter322279296bytes and state529036908bytes uploaded with manifest and uploaded marker. This guard also rejects nonfinite KL; log does not disclose offending sample/value, so no numerical cause beyond the guard is established. First update alone does not establish flight/action improvement. Four-update target not achieved; no candidate flights, comparison, final frozen/reload report.
Both development baselines, all4 training episodes, horizonprobe, replay and admission completed. Admission32calls/4episodes, observed-dominance comparisons empty (no qualifying pair; not reward correctness certification). Evidence downloaded reports/d238_two_task/evidence; archive6496710bytes SHA256fbfb51d7aae97037a58a56a1e47c67eedb2bfbd18eed9171c726ddf242533d5d. Wrapper initiated shutdown; EC2 independently stopping after upload. Stop confirmation pending. No retry or additional flights/updates authorized. Monitor restricted to independent stop confirmation and cleanup.
D238 shutdown independently confirmed by EC2 DescribeInstances at 2026-10-13 12:29:47 UTC (15:29:47 Israel): i-03a3c8632314bf5fc stopped. One committed update preserved; no candidate comparison ran. No restart performed. Completion monitor closed after final shutdown verification.

### D239 - Local saved-checkpoint KL diagnosis and logging repair (2026-10-13)

User requested local reproduction and logging repair, with no more AWS compute. Downloaded D238 step-one checkpoint and verified manifest hashes. Replayed the complete next-update order [0,14,8,15,23,24,21,13] with exact saved images/contexts/action tokens and BF16 inference. All values finite; sample15 (descent task sim_2025-03-27_16-41-45.json, call15) is the fourth encountered and only window sample above0.05: local sampled k3=0.07520359754562378, joint log ratio=-0.41459739208221436, ratio=0.6606062054634094. Earlier samples0/14/8 pass. No optimizer steps or flights performed.

Root mechanism reproduced: post-update acceptance uses the full32-sample mean (saved AWS step1 mean0.009965519255729305), whereas the next pre-step guard applies the same0.05 limit per microbatch, here one sample. Accumulation8 does not average this guard. Finite outliers can therefore stop the next update after a successful average check. Changed recorded-action probabilities do not establish different or better flights. Original AWS offending value/sample was not logged; this is local reproduction, not a bit-exact historical measurement. Full32 mean was not independently recomputed; all8 next-window samples plus one auxiliary sample were replayed. Local configuration hash differs only in model path (verified by reproducing saved hash with only that field changed). Versions Torch/Transformers/PEFT match; BF16 uses no bitsandbytes. Laptop CUDA with CPU weight offload differs from cloud hardware.

Logging repaired in scripts/rl/ppo_engine.py: atomic kl_failure.json contains phase/aggregation, actual value, threshold, finite/nonfinite reason, sample provenance and old/new token probabilities, sequence ratios, and committed checkpoint pointer. Successful update logs include max sample KL, worst index, and count over threshold. No limit or acceptance rule changed. Three logging tests and five existing engine/resume/coverage tests passed, including a low-average/high-outlier regression that commits once and then rejects the outlier without another optimizer step.

Evidence and detailed limitations: reports/d239_kl_diagnosis/REPORT.md, diagnosis.json, local_replay/replay.json and kl_failure.json, replay.log, logging_tests.log, engine_tests.log. Forward-only reproduction utility defaults to the next update window. Decision: keep AWS off; review a consistent batch-level versus individual-sample guard policy before another training run. Do not silently weaken the threshold. No new training authorized or performed.

### D240 - Consistent KL rule and three-update resume authorized (2026-10-13)
User authorized local KL rule repair/tests, then resume D238 for three additional updates and show matched before/after before continuing; reuse existing baseline flights. Full-batch sampled KL mean now controls pre-update admission and post-update acceptance with unchanged0.05 limit. Finite microbatch outliers are diagnostic; nonfinite microbatch outputs still abort immediately. Each new checkpoint records kl_policy=full_batch_mean_v2. Existing optimizer, RNG, samples, advantages and shuffle order restore without resetting; saved config/identity equality remains required. Old checkpoint config remains unchanged, with an outer stage budget and independent shutdown enforcing the shorter new runtime.
Local checks:5KL diagnostics/resume tests,5existing engine tests and3baseline/review-boundary tests passed. Tests cover low-mean/high-outlier acceptance, high-mean rejection with prior checkpoint retained, nonfinite pre-admission/gradient rejection, exact1-to4resume with precisely3optimizer steps matching uninterrupted training, baseline seed/hash mismatch rejection and mandatory4-update/frozen/exact-reload boundary. Python compilation and PowerShell parser passed. No cloud training claimed from toy tests.
D240 workflow reuses D238 two seed7 development baselines and all32 admitted training calls; no baseline/collection reflights. Verify37baseline/task file hashes,11inference-code hashes, original model provenance and savedstep1 checkpoint manifest before GPU work. Copy committedstep1 into a new output root; resume up to total4; fly only2candidate tasks, seed7, same driver/reward/limits; save exact tokens, decoded actions, paths and reward/completion/stopping comparison. Unconditional user review and shutdown; no automatic continuation even if favorable. Proposed bounded execution uses same instance, fresh2hour total cap including setup/upload, no retry or extension. Instance independently stopped before preparation. Launch/start/deadline pending; no AWS compute started yet.
D240 first StartInstances attempt after successful local checks failed InsufficientInstanceCapacity (AWS SDK exhausted its internal retries). No instance start, dispatch, optimizer update or flight occurred. Prepared overlay12066bytes SHA25692eb9f0e342049d4056c6765ff3a6915fd7f29d50702f95e754e514dadb490ba uploaded. Capacity-only retries may target the same stopped instance before any successful start; do not retry a dispatched verification/training job. At successful start persist a new immutable start+2hour deadline, with SSM readiness cap10min and independent Windows shutdown before setup. No alternate hardware, job restart or deadline extension. Baseline/collection reuse remains mandatory; halt after comparison for user review.

### D240 - Four total PPO updates and two matched comparisons completed (2026-10-13)
Successful start15:45:14UTC, immutable deadline17:45:14UTC; reuse gate passed. Exactly3new updates resumed savedstep1 to total4, all32samples visited once across4windows. Final meanKL0.00742611584<0.05; frozen adapters unchanged and exact reload logps passed. Baselines and collection reused, zero reflights. Savedstep4/checkpointstate/manifest and evidence uploaded, archive downloaded/CRC checked and action counts independently verified locally.
Both saved action streams changed: ascent101/384common tokenpositions, turn22/96; these are trajectory-aligned proposed-token differences, not same-context attribution. First turncommand changes0.0536443rad to0.0248919rad; ascentfirstdecodedchange atmove25. Ascent reward1.453934->1.474038,moves124->91,endpoint0.095819->0.101164m,heading0.934->0.174deg,maxrouteerror3.794895->4.392329m,geometriccompletionfalse->false. Turn reward1.910751->1.843995,moves18->29,endpoint0.042557->0.046739m,heading0.223->0.722deg,completiontrue->true. Allstop=model_stop. Same tasks/seed7; noinitialgoalconfounds detected, simulatorimagesmaydiffer. Completion1/2unchanged; meanrewarddelta-0.023326. Real updates and changed behavior, no overall improvement/generalization claim. Unconditional user review; no continuation/promotion.
Evidence reports/d240_resume_review/{REPORT.md,before_after.png,verification_local.json,evidence/}. Job1525.125sec. Instance initiatedshutdownafteruploads around16:15UTC; independentlystopping16:20:09UTC, finalstopconfirmationpending. Monitoronlyshutdown; neverrestart.

D240 shutdown independently verified by EC2 DescribeInstances at 2026-10-13 16:22:22 UTC (19:22:22 Israel): instance stopped. Four total updates and two comparisons saved; no further training or flights. Monitor closed after verification; awaiting user review.

### D241 - Local reward/advantage accounting after D240 (2026-10-13)
CPU-only saved-data audit; no AWS calls/training/flights/production changes. Recomputed all4development reward totals and32stored GAE/normalized advantages within1e-9. Ascent delta+0.020104 decomposes terminalcredit+0.021977, potential-0.000594, extra routepenalty-0.001279. Rawroutecost2.105779->2.200509 but boundedpenalty0.492588->0.493868 near0.5cap. No direct reward for fewer moves; prior suggestion of such a bonus is corrected. Arrival-priority tradeoff masks differences between large detours. Turn lower reward agrees with worse endpoint/heading.
All3successful terminal chunks have positive rawGAE but negative centeredadvantage (descent-0.107300,turn-0.283472/-0.425135); pooledmeanGAE0.198968. Arithmetic correct, but small mixed-task credit assignment needs controlled local testing; not proof normalization is inherently invalid or caused dev regression. Joint32-token likelihood includes unexecuted terminal tails (7/5/4moves), a variance concern rather than a proven bug. Trainingtasks differ from developmenttasks. Report and complete evidence reports/d241_reward_advantage_audit/{REPORT.md,audit.py,audit.json,advantages.csv}. Recommendation: locally test non-saturating distinctions among bad detours while retaining arrival-over-inaction controls, plus advantage/critic/task-balance and executed-prefix tests before another paid round. No automatic reward or PPO change authorized/performed by this audit.

### D242 - Local reward and stopping-signal candidate tests (2026-10-14)
User authorized local tests/fixes/examples before more AWS. Added opt-in rational altitude route-cost curve b*C/(b+C),b0.5, legacy default unchanged; slower saturation but still bounded, and absolute penalties smaller. Equal-endpoint synthetic clean/small/medium/large detour/inaction rank2.000/1.8493/1.6627/1.5745/0; prior actual steep-descent arrival1.538729 beats near-inaction0.002394. Real D240 ascent remains before1.542463<after1.560481: route-versus-terminal-heading tradeoff unresolved, not claimed fixed.
Added opt-in scale_only advantage normalization and CLI option; preserves raw GAE signs. Three saved successful terminal signals change -0.1073/-0.2835/-0.4251 to +0.3865/+0.2103/+0.0686. CPU gradient tests verify direction; cannot establish learned behavior or critic accuracy. Default centered mode retains legacy metadata; alternative configuration is pinned and incompatible resume rejected. Full chunk likelihood unchanged; unexecuted-tail variance concern not proven correctness bug.23 tests passed (6candidate,5KL,5engine,7movement), exact candidate pause/resume verified; saved-data controls also passed. No AWS calls/new VLA training/flights. Candidates not enabled in paid workflow. Evidence reports/d242_local_candidates/REPORT.md and examples.json/logs. Review objective tradeoff before more spending; do not blindly resume old stored advantages under a new reward/policy.


### D243 - Local altitude terminal-credit correction (2026-10-14)
User authorized local goal, reward correction/tests/examples before more AWS. Added opt-in rational_tolerant_heading v7: retain D242 rational cumulative cost; altitude terminal heading gets equal credit inside existing5degree tolerance and smooth excess penalty outside. Endpoint arrival replaces projected-progress multiplier, normalized against initial endpoint utility so idle credit is exactly0 even for short targets. Ordered route potential/cost and geometric completion unchanged; pure Turn/other-kind scores and default v5 preserved. Initial heading-only candidate still misranked D240 due terminal projected-progress multiplier; removed double counting via endpoint-based arrival, without tuning weights to this pair.
Saved ascent now before1.567099>after1.561074; potential0.989353/0.988759,route penalty0.404059/0.407425,terminal0.981805/0.979740. Both remain geometric failures. Controlled clean(1degree heading)/small/medium/large detour/idle scores1.994249/1.849343/1.662684/1.574543/0. Eight new test methods plus6D242 and7legacy movement tests passed(21total), including120 route/heading comparisons, endpoint/overshoot/loop/observation/safety controls and sign-preserving toy resume. Saved score components independently sum within1e-9. No AWS/new flights/VLA training; objective choice remains provisional and rational cost bounded. Candidate not enabled in paid workflow. Evidence reports/d243_altitude_tradeoff/{REPORT.md,examples.json,examples.png,examples.py,tests.log}; code scripts/rl/movement_reward.py and tests/test_d243_altitude_tradeoff.py. Before any authorized paid run, consistently pin reward/config and recompute on-policy returns/advantages; never blind-resume centered D240 stored advantages under the new objective. Four-update user review remains required. No learned-improvement claim.


### D244 - Training-procedure reassessment before further spending (2026-10-14)

User challenged repeated heuristic reward changes and expensive small PPO pilots, requested research and a focused reading guide, and required stopping if the next attempt does not work. Reviewed 30 candidates (including seven explicitly inherited D219 screening records), retaining 20 primary methods/results inspections: three direct UAV solutions and three aerial benchmarks. Structural evidence validator passed; no method was reproduced and replacement-stack resource/compatibility audit remains incomplete. Evidence: reports/d244_training_reassessment/{PROTOCOL.md,SEARCH_LOG.md,evidence.json,EVIDENCE.md,READ_THIS_FIRST.md,REASSESSMENT.md,validation.json}.

Recommendation: pause paid continuation; separate implementation validity, objective validity and actual learning. D240 changed actions after four total updates over 32 calls from four episodes but showed no overall improvement; altitude training was descent while development was ascent. D243 rescoring is development evidence, not independent reward validation. D242 sign-preserving normalization is an unproven candidate; centered successful-step advantages are not inherently a bug. No new numerical outcome or trained improvement claimed.

Before a separately approved pilot: freeze independent completion/safety/route/stopping criteria; use blinded development and held-out trajectory rankings; audit exploration, critic/GAE and executed-action accounting; isolate one learning hypothesis; profile collection/update/comparison cost. Four-update review remains an early diagnostic, not a convergence promise. Failed or inconclusive next pilot stops this procedure for reassessment; no automatic reward patch, retry, deadline extension or continuation. Learned rewards, GRPO, expert-assisted learning and hierarchy are alternatives with prerequisites, not selected replacements. No AWS calls, training, flights or production-code changes in D244. User reading priorities: reward-design pitfalls, SimpleVLA-RL exploration, and direct aerial EG-GRPO; optional human-preference rewards and statistical evaluation. User reviews desired behavior, not implementation correctness.


### D245 - Actual training-attempt quality and PPO signal audit (2026-10-14)
User requested local check for better behavior before AWS. All four D238 training episodes reused by D240 and all32 checkpoint samples matched by tokens/context/image identity/execution count; v5 rewards and centered GAE independently recomputed on CPU. No AWS, flights, model forwards or optimizer updates.
Descent seed12 completes in25moves versus seed11 horizon128: reward1.953157 versus0.782991, endpoint0.061115 versus0.077934m, heading0.249 versus7.187deg. Independent straight-segment spatial deviation0.060734 versus0.393375m. Useful better behavior exists and original reward already ranks it correctly. Turn seed12 takes28 versus59moves, reverse rotation0.504 versus36.368deg, reward1.755560 versus1.497395; both complete, seed12 has slightly worse final precision, so smoother rather than strict all-metric dominance. Goals/start poses/provenance match; images/contexts differ. Empty old admission comparisons resulted from same-end-reason and >0.1m endpoint-improvement predicates, not absence of useful variation.
Failed descent has15/16negative advantages; successful descent2/4positive, mean+0.809834 versus-0.514377. Half the32calls come from long failed descent. All3successful terminal rawGAEs positive but centered advantages negative, reproduced; not proof normalization bug or causal regression. Critic lambda-return error is not independent calibration. D240 ratio-outside-clip fractions0/.75/.5/.25 recorded, not equivalent to zero-gradient fractions. Recommendation: PPO remains plausible given actual informative behavior; no paid continuation yet. Focus next local diagnostic on arrival/stop credit and effective weighting, then separately authorized same-movement learning evaluation rather than descent-to-ascent transfer. No reward/engine production changes made. Evidence reports/d245_training_signal_audit/{REPORT.md,audit.py,audit.json,training_attempts.csv,training_attempts.png}. User-facing chart explicitly labels both colors as old pre-PPO training attempts.

### D246 - Stopping-credit controls (2026-10-14)
User requested local target-stop versus early-stop validation before AWS. Forty synthetic controls (ascent/descent/left/right, v5 and local candidate reward, 0/25/50/75/100% route) passed remaining-return ordering and exact-value two-action categorical PPO gradient direction. Original ascent at75%: stop remainingreturn0.147684 versus finish1.25; atgoal stop1.0 versus depart/backtrack-0.597656. No new reward patch warranted by these controls. Existing D242 opt-in scale_only retains positive rawGAE for all3saved successful terminalchunks: normalized+0.386454/+0.210282/+0.068619 versus centered-0.107300/-0.283472/-0.425135. Not proof centering caused regression or learned behavior improved. Bad-critic counterexample produces positive individual early-stop advantage+0.181436; normalization cannot fix misestimated values. No actual premature-stop episode exists in this4flight batch, so empirical early-stop credit remains unvalidated. No direct timing incentive undergamma1; quiet waiting then stopping ties earlier stopping. Termination is10quietreadbacks, notserverstopflag. Productionreward/defaultnormalization/checkpoints unchanged; do not reinterpret old savedadvantages undernewmode. Added audit_stopping_credit.py/test_d246_stopping_credit.py;17tests passed including existing candidate/reward/toyresume checks. NoAWS, flights orVLAupdates. Evidence reports/d246_stopping_credit/{REPORT.md,audit.json}. Recommendation: retain candidate for controlled evaluation; no unconditional training-readiness claim.

### D247 - Actual behavior-critic stopping audit (2026-10-14)
User authorized local critic checks/fix if needed, noAWS until signal validated. Matched32savedcheckpoint samples to4D238episodes and verifiedmanifest/payload; recomputedGAE/lambda returns1e-9. Usesstoredbehaviorvalues afterwarmup, notfinalstep4critic inference. Ideal stationaryholds atrealcallboundaries give negativeMC-style stopcredit at19/20incomplete states andpositiveat11/12complete states. Incompletepositive exception descentseed11call10:endpoint0.137488m,yaw5.379deg(limit5),V0.390760,holdreturn0.539842,surplus+0.149082. Complete negative exception turnseed11call6:V0.922656,holdreturn0.774986,surplus-0.147670;actualcontinuationreturns1.022823 andimprovesprecision, so notautomaticcriticbug. Statebaseline cannotreverse pairreturnordering. Holdactioncounterfactual spansup totwochunks; exactlambda.95GAEunknownwithoutintermediatecontext/value. Noactualearlyterminalstop existsinbatch. Sixteenhorizon calls explicitlycensored;remaining16terminalcalls realizedreturnRMSE0.628332 notindependentcalibration. NoGAE/criticimplementationbugestablished,noweight/reward/normalizationchangejustified;readinessfalse. Added audit_critic_stopping.py,test_d247_critic_stopping.py;6CPUtests passed includingcensoring,multi-callGAE,lambda1identities,savedaudit. Evidence reports/d247_critic_stopping_audit/{REPORT.md,audit.json,states.csv}. NoAWS,flights,modelinference,optimizerupdates. Needactualsampledbranchedoutcomes/intermediatevalues before claimingcriticcalibrated; do notinsertconstructedholds intoon-policyPPO orforcevaluefits todesiredsigns. D246scale_only remainscandidate,notvalidatedrepair.


### D248 - Accepted-goal stopping objective and local execution controls (2026-10-14)
User explicitly specified continue until position/heading acceptable, then stop without extra precision; noAWS. Implemented opt-in accepted_goal stopping for numericturn/altitude: preserveprearrivalshaping,freezeatfirstacceptedreadback,chargeadditional normalizedposition/yawmotion,terminal+1accepted/-1premature,0horizon. Existingroute/corridor/collision/invalidgatesretained;10quietmoveexecutionstillrequired. Historicaldefaultcontinuous/v5andpaidconfigsunchanged.9branchpairs pass via deterministicCPUtransport through actualcollector/rewardattachment,36uniquescriptedexecutions;notVLA/physicsflights. Two branches startfromD247savedprefixes. Savedheading5.379degoutside5limit:oldstop/continue0.539842/0.698541,new-1/0.991976. Savedacceptableturn:old0.774986/1.061052,new1/0.929760,soextrarefinementnowloses.44tests pass includingbidirectionalturn/altitude,reentry,duplicates,horizon,safety,accounting,compatibility;updatedobsoletecollectortestv3expectationtocurrentv5. Evidence reports/d248_accepted_goal_stopping/{REPORT.md,examples.json,examples.png,validation.json};sourcehashespinned. New accepted_goal_reward.py and audit/test;opt-inthroughmovement_reward/executed_rollouts/flight_collector. NoAWS/modelinference/optimizerupdates. Binaryboundaryisintentionalobjectivechange,notcriticrepair. Oldcritic/advantagetargetsnotreusable;scriptedactionsunapprovedforPPO. Learning/criticcalibrationremainunverified,training_readyfalse. Prefixscanquadraticandfirstacceptanceexecutionresolutiondependent;no perphototimepenalty. No furtherpaidworkauthorized.


### D249 - Real recorded action stopping credit (2026-10-14)
User approved local model-action check, noAWS. Audited12realrecordedepisodes/121sampledchunks:4D238BF16trainingflights plus8D228NF4localflights keptseparate.6acceptedstops,2failedstops,4timeouts. FoundactualD228prematurestop4.988mshort (prior noearlystopfindingwaslimitedtoD238batch). VerifiedD238checkpointidentityandD228taskmanifesthashes;rescoredexecutedposesviaattach_path_rewards(v8),samplesremainunapproved. Terminalchunkrewardsaccepted+0.974168..+0.999027,failed-1.003282/-0.999531. ActualjointPPOfunctionderivativeatratio1onrecorded32tokenlogpshascorrectsignfor4baselinecontrols(-.5,0,.5,.9);sufficientbaselineinterval(-.999531,.974168),notcalibratedcriticorclampingrule. NoVLAparametergradient/weightupdate. D238staleOLDcritic sensitivity recomputedGAE:all3terminalsignalspositive underraw/centered/scale_only,notnewcriticvalidationorresumeauthorization. Wrongcriticcanstillreversesign;timeoutsnotcalledterminalexamples. Added audit_real_action_stopping.py,test_d249_real_action_stopping.py;4testsandfullsaveddataassertionspass. Evidence reports/d249_real_action_stopping/{REPORT.md,audit.json};hashespinned. NoAWS,flights,VLAforwards,optimizerupdates,productionchanges. Reward/lossdirectionconditionallypasses;newcriticcalibrationandactualweightupdateuntested,paid_training_readyfalse. Do notpoolNF4/BF16historicalpoliciesforPPOorresumestaleadvantages.


### D250 - Frozen local critic fit and held-out flights (2026-10-15)
User authorized critic-only local fitting with the new accepted-goal reward, no AWS. Actual NF4 VLA runtime matches recorded D228 identity; first action-token probability exactly replays all 41 contexts; before/after model parameter fingerprints identical. Fitted zero-initialized linear critic for 200 fixed AdamW steps, lr1e-4, on 23 calls from two seed-7 terminal flights (one success, one failure). Three separate seed-8 terminal flights /18 calls held out until checkpoint save; three censored timeouts excluded. Fixed protocol before fitting; no hyperparameter selection on held-out results and no second fit. Targets are undiscounted Monte Carlo returns under v8, no bootstrap. No AWS, new flights or actor updates.
All three held-out terminal raw and train-centered signs correct: successful turn +0.010844, successful vertical descent +1.837319, actual premature descent -0.324079. Narrow declared diagnostic passes train-mean baseline and zero exact context overlap. However training MSE0.000008085 versus held-out0.935382 shows poor transfer; held-out beats train-mean constant1.417259 but is worse than zero0.869006. Successful vertical-descent stop value -0.847463 versus observed +0.989856. Training advantage std0.002915 yields held-out train-scaled terminal values around +630/-111; diagnostic only, not production PPO normalization or actor update. Do not equate correct signs with calibrated values or all-state continuation credit. paid_training_ready=false; original report passed flag preserved as narrow preregistered criterion, not readiness.
Independent saved-head reload, MC/GAE recomputation, split and checkpoint/frozen/runtime hash checks pass. Evidence reports/d250_local_critic_fit/{protocol.json,report.json,predictions.json,verification_local.json,REPORT.md,frozen_features.pt,critic.pt}; source scripts/rl/fit_local_stopping_critic.py. Recommended next local work: establish balanced same-movement successful/failed training coverage, separate tuning and untouched final-test flights before refit; current held-out set now development evidence. Do not repeatedly tune against it or authorize AWS based on terminal signs. Same-scene five-flight NF4 diagnostic, not BF16/generalization proof. Production workflow unchanged.


### D251 - Locked critic gate and conditional AWS pilot (2026-10-15, in progress)
User required complete acceptance criteria before fitting and authorized AWS only after all pass. Locked protocol and human-readable CRITERIA at reports/d251_critic_gate before model selection or fresh test flights. Seven fixed gates cover exact frozen model/runtime/replay, independent whole-flight outcome coverage, equal-flight held-out MSE at least10% better than both zero and training-mean constants with per-kind protection, terminal value calibration, actual stop sign margins, state/continuation credit, advantage scale and regressions. Missing coverage is failure/inconclusive, not pass. No post-test threshold change/refit/new seeds. Conditional pass triggers existing-instance AWS pilot capped at2instance-hours/fouractorupdates, same NF4 policy/reward, ordinary identity/replay/KL/frozen/reload runtime checks, matched turn/altitude comparison and unconditional review stop; no automatic retry/continuation/landing/benchmark.
Repair: flight-balanced linear ridge critic on fixed normalized real VLA context features. All five D250 terminal flights are development data, no longer held-out. Alpha chosen from fixed six-value grid by leave-one-whole-flight-out validation; refit all five (41calls). Selectedalpha0.001, developmentCV_MSE0.6025482, rawadvantagestd0.1914916 versus D2500.0029153. This is development evidence only. Checkpoint locked before six new local test flights on three existing numeric practice tasks, seeds101/102, max128moves,45min collectioncap. Six ridge tests pass and nine reward control preferences plus existing regression suites pass. No actor updates or AWS calls. Final outcome pending in result.json; do not infer pass from repair/development metrics.


### D251 outcome - Fixed gate failed; AWS not started (2026-10-15)
Completed all six new local flights (48calls) in1057.328s with frozen original NF4 action model; five successful terminal stops and one actual premature stop, no horizons. Full coverage gate passed. Locked critic evaluated once in46.093s; all48 first-token log probabilities replay exactly, model parameter fingerprints unchanged, no exact image/context overlap with development. No actor updates or AWS calls.
All seven predeclared criterion verdicts: identity PASS, coverage PASS, value calibration FAIL, actual-stop direction FAIL, state/continuation FAIL, advantage scale FAIL, regressions PASS. Equal-flight held-out MSE0.929892 versus zero1.050875 and training-mean constant0.959543: improvement11.51% over zero but only3.09% over training mean, failing required10% against both. Turn MSE0.705844; altitude1.041916 versus altitude training-mean0.988698. Terminal MAE0.630850 (limit0.25), maxerror1.401310 (limit0.50). Five successful terminal credits positive; actual premature stop reward-0.999709, V=-1.076366 gives wrong positive advantage+0.076657. Successful steep-descent stop V=-0.404714 versus reward+0.996596 gives+1.401310, showing poor calibration. State stop-now direction correct outside acceptance87.5% (required95%), accepted100%; qualifying measured continuation correct8/15=53.33% (required90%), with3turn/12altitude calls. Test rawadvantage std0.861027 / development0.191492=4.4964 (limit4), maxtrain-scaled magnitude9.23452 (limit10).
Implemented regularized_critic.py and development-only d251_fit_critic.py, held-out collection/evaluation/report scripts.57 local tests,9reward branch controls,3optional initial-critic-load tests and5existing engine tests pass. train_path_ppo.py gained explicit --initial-critic exact loading and warmup0 only when supplied; default unchanged, never executed against actual action model. This candidate repair is not approved for PPO. Same selected checkpoint, no reward changes, no extra seeds/refit/threshold revisions after test. Numerical gates were fixed before fitting; failed candidate preserved and AWS authorization condition not met.
Independent CPU reload verified all48 predictions/MCtargets/GAE and source/checkpoint/protocol hashes. Reporting-only Turn/Rotate->Turn alias correction fixes per-kind grouping/counts; original result preserved, every gate verdict unchanged. Evidence reports/d251_critic_gate/{CRITERIA.md,protocol.json,fit.json,result.json,result_initial_unmapped_kind.json,verification_local.json,REPORT.md,state_checks.json,test_features.pt,critic.pt,test_flights/}. Stop here under the declared failure rule; no automatic more fitting, flights, reward patch or cloud work.


### D252 - Local critic failure cause and observed-state input candidate (2026-10-15)
User requested cause diagnosis/local repair without AWS or changed pass criteria. Reconstructed all48 D251 locked ridge predictions from old targets/features to max2.45e-7; all89 MC targets and48 test GAEs recompute exactly. Premature steep-descent terminal reward-0.999709 minus V=-1.076366 gives positive+0.076657. Prediction below minimum development target-1.007580, signed target-weight mass negative-0.544606; two failed steep-descent development flights contribute-0.903402/-0.090764. Data are5flights/41calls/2560features: turn2success/0failure, vertical1success/0failure, steep0success/2failure. Task/outcome confounding and extrapolation observed; no causal retraining ablation or same-state policy continuation expectation measured. No evidence of reward/GAE/load arithmetic bug.
Reward depends on execution history absent as explicit critic input. Old near-goal failed descent: endpoint0.0514m/heading0.824deg but prior route deviation3.6091>0.75; new successful descent prior deviation0.7241. Failed new pre-action state2.9677m short/15.521deg, quietcount8;2furtherquietmovesstop. Added local-only critic_observation.py exposing measured prefix geometry, acceptance history, quiet count/budget; no future action/return/outcome input, value clamp or actor change. Eight focused causal/collector/geometry tests pass; extractor ran onall89savedstates. Not integrated into PPO, no candidate critic fitted/validated. Fromexactfailedprefix, constructed valid20movecontinuation thenholds earns remaining+1.337193 versus stationary-1 under unchangedreward; geometrycontrol only, notmodel/physicsflight or policyexpectedreturn.
D251 remainsFAIL. All original protocol/CRITERIA/result/checkpoint, reward andproductiontraining hashes unchanged. NoAWS/newflights/VLAforwards/actorupdates/newcriticfit. Evidence reports/d252_local_critic_cause/{REPORT.md,diagnosis.json,critic_observations.json,geometry_control.json,verification.json,audit.log}; scriptd252_diagnose_critic.py. Recommendation: local critic repair with pre-action observed state and actual within-movement success/failure development coverage, retaining allsevenfixedcriteria and independentvalidation. Existing D251 data nowdiagnostic, never relabel untouchedtest. No extra readiness criteria, changedthresholds, automaticseedsearch orpaidcontinuation.


### D253 - Authorized local observed-state critic refit and separate final flights (2026-10-15, in progress)
User authorizes local-only refit with new inputs and successful/failed actual flights, frozen action model, unchanged pass criteria and noAWS. Protocol saved before model selection: original D251 C1-C7 strings copied verbatim, canonical criteria SHA25624990581782a42591da9fbcf2bd7434597637cb2f0261b8270a7445d35620c28. All11 previously inspected D250/D251 terminal flights/89calls are now development only (8success/3failure, including steep-descent success); turn/vertical have no failed terminal development examples, explicitly recorded. Fixed state-aware linear ridge candidate over frozen VLA features plus22causal observed-prefix inputs; train-fold-only standardization, fixed6alpha x3state-weight grid, leave-one-whole-flight-outCV, then lock. No value clamp/sign target changes. Six new final flights on same3numeric practice tasks, seeds201/202,128moves,45minute collectioncap, oneevaluation. Missingcoverage remains fail/inconclusive; noseedsearch/refit aftertest. Noactortraining/AWS evenifpass. Sources/report under reports/d253_state_critic; finaloutcome pending.


### D253 status - Local collection interrupted, final evaluation unavailable (2026-10-15)
Independent status check found only1/6 saved flights, last server activity08:16:08Israel, recorded server/worker absent and original session unavailable. No completion/failure/cleanup marker; interruption cause unknown. Critic fit saved, final result absent. Unchanged C2 coverage not met; C4 early/target-stop and C5 continuation credit untested, not a pass or proof critic repair failed. Preserve original protocol/checkpoint/evidence; no restart/newflight/refit/AWS in this check. Evidence reports/d253_state_critic/{STATUS.md,interruption_status.json}. Prior collecting status was stale and user-facing running statement corrected.


### D253 recovery authorization - Final bounded local assessment (2026-10-15)
User explicitly authorized completing interrupted assessment without further questions, within90minutes, noAWS. Goal began15:07:16UTC; immutable end16:37:16UTC. Windows System events41/6008 confirm an unclean reboot, not exact interruption time/cause. Preserve first completed flight byte-for-byte; collect only five missing fixed task/seed pairs, with same frozen actor/critic, original scene seed201, limits/reward/criteria. Recovery uses three simulator sessions rather than original one; retain this procedural difference and all raw evidence. No outcome-based rerolls. Existing evaluator/model source hashes and protocol remain unchanged. Recovery script bounds collection to40minutes and evaluation/verification to remaining budget, cleans its child processes; no more refits/flights if final assessment fails. Image files namespaced to avoid overwriting first-flight evidence. reports/d253_state_critic/recovery/start.json records pins and deadline. Final outcome pending.


### D253 final assessment - NO-GO; partial evidence independently verified (2026-10-15)
Authorized bounded recovery ended at18:51Israel collection cap; no extension/retry. Five of six fixed flights saved:3successful terminal stops,2horizons128moves, sixthunfinished. First flight byte-identical; remaining evidence recovered in three simulator sessions with same scene seed201, procedural difference recorded. Windows events41/6008 indicate unclean reboot before recovery, exact interruptioncause/timeuncertain. No premature terminalstop; C2coverageFAIL. Original protocol/criteria/model/sourcepinsunchanged; no rewardpatch/refit/newseeds.
Derived partial evaluator preserves original numerical metrics and explicitly fails coverage; original evaluator/result namespace preserved, no full result fabricated.53calls exact firsttokenreplayerror0, unchangedactor, zeroexactcontextoverlap. MCmetrics use only3successfulterminalflights/21calls; horizonsnotinventedreturns. Actualsuccessfulstopcredits+0.123503,+0.415891,+0.168258. Hypotheticalstopnow outside13/13negative, accepted8/8positive; actualprematurestopunvalidated. Usefulcontinuation6/8=75%<90%:FAILC5. Two qualifyingturnMCsurpluses-0.323084,-0.611695 despite continuation-minus-stopgaps1.351147,.938169. This is agreedMCdiagnostic, notproofeverynegativeindividualadvantageisabug or PPOinvalid.
SubsetcalibrationMSE.184645 versus zero1.007599/trainmean.521892, terminalMAE.235884,max.415891; subsetscale std.346042/dev.387555,maxscaled2.142565. C1/C3/C4/C6/C7 computedtrueonavailabledata, neverfullvalidation; C2/C5false. Oldcritic SAMEsubsetMSE.170741, so no generalcalibrationimprovementclaim; newcriticcorrectstwoobservedoldturnstopnegativecredits. IndependentCPU predictionreload/state/MC/GAE discrepancies0;73tests+9controls retained. FinaldecisionNO-GOforAWS/currentPPOprocedure, stop incrementalpatch/retryloop. Zeroactorupdates/noAWS. Localcollection/evaluationexited. Evidence reports/d253_state_critic/{DECISION.md,conclusion.json,partial_result.json,partial_state_checks.json,partial_verification_local.json,partial_REPORT.md,recovery/}.


### D254 - User-authorized closeout, preserve step8000 and disks (2026-10-15)
User requested stopping both paid machines, retaining disks, verifying checkpoints/reports backups, and preserving step8000 for later HuggingFace upload. AWSCore independently found Windows i-03a3c8632314bf5fc and training i-05f21d6200f5c1045 already stopped; no stop/start/termination needed, disks preserved. No ElasticIPs inus-east-1. gp3250+150GiB at verifiedUSD.08/GB-month givesUSD20+12=32/month combined (~1.07/day/30days), excluding othercharges; stoppedcompute does not removeEBScharges.
S3 d207/stage2/adapter_s8000 containsweights/config/README; freshlydownloadedweights/configexactSHA256matchlocalD207step8000:12e43d18489531d7446219bc88ed5ede0371935649b1464674a41e1cc155ac8e and5869411e81804b7a85195b80e9affd8485989f62f566997de92a35303bf03127. Resume state645101597bytes exists atd207/stage2/checkpoint/state.pt; D240step4checkpoint/evidence also present. Step8000 isLoRAadapter, notfullbasemodel; preservedmetadatahaslocalbasepath/templateREADME, so portableHFexportneedsmetadata/modelcard/inferenceassets; notpublished. RetainD208baseline; DAgger/PPO/criticfailures summarizedinreports/d254_training_closeout/CLOSEOUT.md withoutoverstatingendpointmetricsorpartialD253evidence. Fullreports/sourcearchivepreparednext; uploadverificationpending. No moretrainingauthorized.


### D254 storage reduction review (2026-10-15)
User questions retaining both full disks and wants only durable restart essentials. Original D207 launch script dependencies checked against S3 HeadObject: official store tar 75371663360 bytes, boxes 9823300, two extra-rollout archives 50314897 and111288326, original code800917, step8000 state645101597, report202377, validation split13608. These objects exist; full dataset was not redownloaded or a fresh restore performed. Step8000 weight/config byte hashes were verified separately. Both stopped root disks are reinstall convenience, not necessary durable training artifacts once restore assets are verified. Proposed removal of250GiB Windows and150GiB Linux disks saves verified32USD/month storage; deletion not performed because prior explicit preserve-disks instruction awaits clear superseding authorization. Reports/source backup archive3893851668bytes SHA25631669287a6f2d9990600f88af059e49c7702015a6bca8ebadba3d2e63ba33ab5 passes ZIP CRC; S3 upload underway. See reports/d254_training_closeout/restore_inventory.json.


### D255 - GitHub source backup and complete VLA release preparation (2026-10-15)
User requested source on GitHub, remaining restart artifacts in S3, and a complete VLA rather than an adapter-only HF download. Remote main is an ancestor of local HEAD (25 existing local commits ahead); normal fast-forward push planned, no history rewrite. Source/docs/tests scanned for credential patterns and private bucket names without findings. Raw reports/checkpoints remain S3 artifacts, not newly committed Git blobs.
Built local D:/drone_vla_pilot/release/d207_s8000_complete:184files,9210455119bytes before manifest/checks; includes pinned original Qwen base revision ebb281ec70b05090aa6165b016eac8ec08e71b17, verified step8000 adapter, tokenizer/processor, four-channel action normalization and repository inference code. Base+adapter are both included in one package, not merged or quantized. Export adapter metadata uses public base ID, original metadata separately retained; original trained weights unchanged. serve.py sets local asset paths/offline mode and D208 inference flags. Publication pending: no HF destination or upload selected; full GPU inference export smoke and upstream licensing notices not yet verified. Existing dataset-license uncertainty remains disclosed. No training/flights/cloud instance starts. Full reports/source S3 upload still underway; never report pending upload as verified.


### D255 source push verified and backup transfers (2026-10-15)
GitHub main independently matches77801a28e6f0e71731aad74c0c0d3b0755852e19; normal fast-forward fromb156123, including25 prior commits and new165-file closeout commit. Credential/private-bucket scan of staged files passed; all Python sources parse. Existing extra-blank-line whitespace warnings preserved, not functional failures. Complete VLA processor loads offline and4-channel action tokenizer decodes8x4; launcher dry-run resolves all local assets. No full GPU inference check performed. S3 complete-package assets ZIP4226178bytes upload HTTP200; trained adapter copied server-side into d255/full_vla/adapter/. Two original base shards and reports/source archive are uploading with full-object SHA256 validation. Initial tiny-block archive transfer was cancelled and replaced with buffered transfer; not a training retry. No compute start or disk deletion. Upload status and expected checksums are in reports/d254_training_closeout; completion awaits independent S3 HeadObject verification.


### D255 backup verification: complete VLA saved, reports upload failed (2026-10-15 17:52 UTC)
Independent S3 HeadObject checks confirm both full base shards and package-assets ZIP match local full-object SHA256 and byte counts; closeout supplement also matches. Trained step8000 adapter is present in the complete-package prefix (original weights separately hash-verified). Full VLA backup therefore contains base, trained adapter and inference assets. No HF publication or full GPU smoke test claimed.
The separate3893851668-byte reports_source.zip transfer exited with ConnectionError/write TimeoutError; S3 returns404 for its destination. Local archive remains intact. Overall backup is INCOMPLETE; do not delete disks on the basis of this closeout. No automatic retry, compute start or deletion. Save verification_s3.json and backup_status.json; notify actionable transfer failure and close monitor. Recommended recovery is resumable multipart or smaller archive parts, not another blind full-file upload.
