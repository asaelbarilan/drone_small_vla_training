# Release checklist: paper, Hugging Face, GitHub (2026-10-03)

Full package: D:/drone_vla_pilot/release/qwen3vl4b_uavflow_vla_progress/ and
s3://<bucket>/release/ (package tar, g5_logs.tgz, lineage/).

## To continue training later (all saved, g5 not needed)
- [x] Checkpoint at 31,000: adapter + optimizer state + step + report (release/lineage/long_run_checkpoint_s31000/)
- [x] Every saved adapter (d171/run/adapter_s2000 ... s31000, adapter_best; phase 0-2 adapters)
- [x] Prepared training data, 75 GB (d170/store.tar) + action ranges + splits + held-out lists
- [x] Exact trainer version and every launch script (git; d179/train_uav_flow_vla.py; g5_logs.tgz)
- [x] Hyperparameters of every phase (training/phases.json)
- [ ] Not saved (re-download): base model Qwen/Qwen3-VL-4B-Instruct, raw UAV-Flow shards
- Resume = new GPU machine, unpack store.tar, put the checkpoint in OUT/checkpoint, run the
  trainer with --resume and the phase-5 arguments (about 1 h of downloads first).

## Hugging Face model card
- [x] Adapter weights (adapter_s31000), model card README.md, action ranges, results, limitations
- [ ] A short standalone inference example (load base + adapter, build the prompt, decode the
      progress line and the 8 moves) - the server is the only runnable path today
- [ ] GGUF version of the adapter for llama.cpp (the 8 GB deployment path, D157 showed it converts)
- [ ] Final licence wording (base Apache-2.0; UAV-Flow declares no data licence: research use)

## GitHub
- [x] All code in the repo (trainer, data prep, server, scorer, RL loop, AWS/Windows scripts)
- [ ] A clean standalone repository: only the VLA code, a README with "reproduce in 5 steps"
      (data prep, the six training phases, evaluation on UAV-Flow-Eval, the inference options)
- [ ] requirements file (d170/requirements_g5.txt on S3 + the Windows venv list)

## Data sets (rule 1)

| Set | Made from | Used for |
|---|---|---|
| **Train** | UAV-Flow real + simulator training flights (and the 200 practice tasks built from them, `scripts/rl/build_rl_tasks.py`) | learning only: supervised training, DAgger, RL |
| **Validation** | the 504 held-out UAV-Flow-Sim flights (D169, `release/qwen3vl4b_uavflow_vla/sim_val_flights.json`) -> 307 flyable tasks (`build_rl_tasks.py --validation`) | every development decision and error analysis |
| **Test** | the 273 UAV-Flow-Eval benchmark tasks | untouched; flown once at the end for the paper. 100 of them were used during development before 2026-10-03 (disclosed) |

## Paper: have
- [x] Main table (69% / 0.449 vs OpenVLA-UAV 67% / 0.395), per class, task-weighted estimate
- [x] Ablations: no line / line / line + memory / + deadband; 18,000 vs 31,000
- [x] Training curves, schedule, compute and cost; harness audit; benchmark flaws (tilted
      starts, Pass has no distance); the memory error analysis (REPORT sections 11, 12, 19-21)

## Paper: still missing (ranked)
1. [ ] Camera controls for the final model (gray photo / swapped photo / text only) - the
       repo's standing rule; shows the model uses vision. About $2-4 on the Windows box.
2. [ ] All 273 test tasks instead of 100, for a number directly comparable with the papers.
       About $4.
3. [ ] Confidence intervals (bootstrap over tasks) - free, from the existing scores.
4. [ ] The 8 GB claim measured: llama.cpp memory and latency of base + adapter next to the VLM.
5. [ ] A matched ablation of the progress line (same extra updates without it) - the
       18,000 -> 31,000 comparison mixes the line with more training. About $10 + a test.
6. [ ] RL rounds (planned), if they help.
