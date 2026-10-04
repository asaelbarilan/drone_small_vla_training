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
