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

(to be filled, one section per item above)
