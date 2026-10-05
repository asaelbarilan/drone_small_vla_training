# Decision log (D158 onward)

Copied from the original working repository; each entry is one decision with its evidence. Later entries supersede earlier ones where they conflict.

## D158 - 2026-09-22 - official UAV-Flow format ported; action-range clipping found

- scripts/prepare_uav_flow_official.py: official OpenVLA-UAV data format (4-D
  drone-frame actions, state in prompt, every frame, first/last x5). 0/500
  mismatches vs the official loop; xy re-integration matches to 5 mm; raw vs
  preprocessed altitude differ (median 0.19 m, max 6.8 m) - inherited by the
  official code. Official trains on `instruction`, D155 used `instruction_unified`.
- train_uav_flow_vla.py: --format official, --instruction, --chunk K (1/8/16).
  scripts/score_uav_flow_official.py: open-loop rollout, floor/real/gray/swap,
  seconds+tokens per call, text baselines recomputed on the same endpoints.
- FINDING: the official q01/q99 action range clips fast motion. Floor endpoint
  error (perfect predictions through clip+256 bins), 62 held-out flights:
  median 0.10 m, p90 4.0 m, max 8.4 m; clipping alone explains all of it.
  24.5% of steps have a clipped channel. Caps: yaw 0.105 rad/step (30 deg/s),
  forward 0.46 m/step (2.3 m/s); max yaw step 3.1 rad. Plausible cause of
  OpenVLA-UAV Rotate 20% SR.
- Planned single-GPU tests (g5, same update budget each): 1) action range
  q01/q99 vs q0.1/q99.9 or max at K=1; 2) horizon K=1/8/16; 3) instruction
  wording official vs both; 4) data 1/5/10 shards.

- D157/D155 shard run SCORED (adapter_s1000, Qwen3-VL-4B bf16, D155 format,
  8-step chunks, no state): held-out median endpoint error 6.474 m on real
  frames - IDENTICAL on gray and on swapped frames (57/55/58 distinct endpoints,
  sign tests p=0.75 / p=1.0). Text-only bar is 4.496 m, no-text 6.868 m, so the
  model is worse than the text baseline and still ignores the camera. Train
  subset 3.199 m, also identical across conditions. Representation floor 0.081 m.
  Conclusion: a 30x longer run on a 16x larger model in full precision did not
  fix camera-blindness -> the cause is the setup (format/prompt/data), which is
  what the official-recipe port (D158) changes. Instance stopped after scoring.
  Known defect: the floor condition failed to parse 52/62 flights (its last
  partial chunk decodes to None); the floor numbers come from the flights whose
  length is a multiple of the chunk. Fix before E0 scoring.

- D159 (2026-09-23): official-format K=1 run on shard 1 (Qwen3-VL-4B bf16, 800
  updates x 32 = 25.6k examples = 1 epoch, lr 5e-4 cosine, loss 16.72 -> 2.96;
  wandb asael/vla training/official_k1_shard1). Mirror probe (20 val flights x 3
  frames = 60 comparisons) per checkpoint: s200 -, s400 3/60 (5%), s600 0/60,
  s800 10/60 (16.7%); sideways sign flipped 1/60; yaw never. Reference: released
  OpenVLA-UAV changed in 16/20 first frames, 6/20 sideways sign flipped.
  Also D159: base Qwen3-VL scores 24/24 on a pasted red rectangle (left/right) at
  512 px, so the probe and the VLM's perception are sound; but it answers the
  "which side is the instruction's target" question the same under mirroring at
  256/512/896 px (flip 2/12 at every size) - resolution is NOT the bottleneck.
  83% of flights name a visual target, so instruction filtering is pointless.

- D160 (2026-09-23): 10-shard run official_k8_10shard (K=8, both wordings,
  mirror, cap 0.1/99.9, 2,500 updates x 32, held-out loss 11.02 -> 2.566) scored
  on 150 unseen-site flights, batched (0.35 s/call vs ~1 s sequential):
  floor 0.072 m | real 3.067 m | gray 3.115 m | swap 3.121 m | text-only NN 4.533 m
  | no-text mean 4.208 m. FIRST adapter to beat both text baselines, by ~1.1 m.
  Camera contribution is ~0.05 m and not significant (gray sign test 70/127
  p=0.29; swap 71/122 p=0.085). Mirror probe: 42/180 changed (23%), sideways
  sign flipped 1/180. So the gain comes from state + instruction, not the image.
  Batched generation verified: left padding correct; batch-vs-single logit noise
  ~0.56 (same with zero padding), flips only near ties. Scorer and preparation
  now batched/threaded. Instance stopped after scoring.

- D161 (2026-09-23): mirror-probe trend across checkpoints of official_k8_10shard
  (60 held-out flights x 3 frames = 180 comparisons, batched): s250 0%, s500 11%,
  s1000 19%, s1500 14%, s2000 12%, s2500 26%. Rising overall but noisy, no sign
  of saturation - weak support for more data/training. (Earlier sequential
  s2500 run gave 23%; batched 26% - the gap is batch-shape noise near ties.)
  probe_mirror_vla.py now takes several --adapter paths, loads the base once and
  batches generation. Instance stopped afterwards.

- D162 (2026-09-24): UAV-Flow-Eval closed loop running LOCALLY on the 4060.
  Simulator: UnrealZoo Collection_WinNoEditor_0424_25.zip (51,216,498,779 bytes,
  ModelScope UnrealZoo/UnrealZoo-UE4, user-approved) downloaded with a
  24-connection resumable range downloader (~23 MB/s; single connection ~1 MB/s),
  extracted to D:/drone_vla_pilot/simulators/Collection_WinNoEditor_0424_25 (49 GB).
  Eval code: D:/drone_vla_pilot/simulators/uav_flow_repo (git clone of
  buaa-colalab/UAV-Flow), venv D:/drone_vla_pilot/venv_uaveval (Python 3.11,
  gym 0.10.9, numpy pinned <2 because their track.py breaks on numpy 2).
  Local changes: DowntownWest.json env_bin_win -> our path; batch_run_act_all.py
  sends `vrun t.MaxFPS $UE_MAX_FPS` (default 10, used 5) after reset, since at
  full frame rate Unreal saturates the GPU and each policy call took 13-40 s
  (now ~1.3 s). Original kept as batch_run_act_all.orig.py.
  Protocol facts: POST /predict {image 224px PNG, proprio [x,y,z cm, yaw deg] in
  the START frame, instr} -> {action: [[x,y,z cm, yaw rad], ...]} poses in the
  start frame; the simulator is in CENTIMETRES (OpenVLA-UAV "sim" action q99 is
  48.7 forward per step), our model is in metres -> x100 / /100 in the server.
  273 test tasks, 10 classes; official metric = nDTW per class (success rate is
  human-judged in the paper). The evaluator ends a task after 100 steps or 10
  near-still steps.
  scripts/uav_flow_eval_server.py serves openvla-uav (NF4, eager, D156 mask fix;
  official server is bf16 + flash-attn) or our qwen adapter (K steps -> K poses).
  scripts/score_uav_flow_sim.py = official nDTW functions + end distance/yaw.
  Smoke: "Turn to the direction of the person" -25.6 vs ref -29.7 deg (good);
  "Rotate 105 degrees to the right" turned the wrong way for 100 steps.
  FULL RUN of OpenVLA-UAV started 10:00 local: flights in
  D:/drone_vla_pilot/runs/sim_eval/openvla_uav/flights, log eval.log.
  OUR MODEL IS BLOCKED: adapter_s2500 exists only on the stopped AWS instance and
  the AWS connector is disconnected; needs the user to reconnect, then start the
  instance just long enough to scp runs/official_k8_10shard/adapter_s2500 and
  the reports/uav_flow_official_10shard manifest (action stats) to D:.

- D162 update (2026-09-24): the local closed-loop run was STOPPED at the user's
  request (it froze the laptop: simulator + 6 GB model on one 8 GB GPU). The
  simulator had also crashed once at task 70 (unrealcv image request timeout).
  69/273 OpenVLA-UAV flights completed, scored in
  D:/drone_vla_pilot/runs/sim_eval/openvla_uav/score_69.json: mean nDTW 0.537 over
  Turn 0.18 (8/15), Move 0.11 (10/15), Shift 0.70 (28/49), Rotate 0.40 (8/15),
  Surround 0.76 (6/12), Ascend/Descend 0.78 (9/19); Approach/Retreat/Pass/Land
  not reached (the task order is alphabetical by timestamp, not by class).
  Partial and class-biased - not a result. The evaluator resumes where it left
  off (skips tasks with plots); scripts in D:/drone_vla_pilot/runs/sim_eval/.
  UnrealZoo also ships a Linux build (Collection_v4_LinuxNoEditor.zip, 45 GB),
  so the evaluation can move to the g5.

- D162 AWS run INVALID (2026-09-24): OpenVLA-UAV 177/273 flights on the Linux
  v4 build (mean nDTW 0.278; Land 0.04, Approach 0.07) and our adapter's first
  12 flights are NOT usable. The first camera frame of every task is the SAME
  image regardless of start pose (contact sheet
  D:/drone_vla_pilot/runs/sim_eval_aws/first_frames_sheet.png): the camera does
  not follow the drone on Collection_v4_LinuxNoEditor, probably tied to the
  camera-count behaviour that forced the base_env.py remove_agent timeout patch.
  Colours are also swapped (orange sky): unrealcv returns BGR and the official
  evaluator passes it straight to PIL.Image.fromarray - check whether the
  Windows build does the same before calling it our bug. Both evaluations were
  stopped; the instance was left running per the user. Next: make the camera
  follow the drone (compare camera ids/poses on Linux vs the Windows build), or
  evaluate on the Windows build on a Windows GPU instance.

- D163 (2026-09-24): closed-loop eval on an AWS WINDOWS GPU box (the supported
  build; Linux v4 camera was frozen). Instance <instance-id>, launched
  g5.2xlarge but started as g6.xlarge (no g5 capacity in us-east-1d), Windows
  Server 2022, 250 GB gp3, <security-group> (no inbound), IAM role/instance
  profile vla-eval-windows-role (SSM core + read ec2-windows-nvidia-drivers +
  RW s3://<bucket>, private bucket). Controlled only via
  SSM RunCommand (no RDP/password). NVIDIA GRID 596.86 from the AWS bucket.
  Gotchas: fresh Windows needs UE4PrereqSetup_x64.exe (VC++/DirectX) or
  Collection.exe exits silently; SSM runs in session 0 so the evaluator must
  pass offscreen=True (UAV_EVAL_OFFSCREEN=1 patch); never Start-Process with
  -RedirectStandardOutput from an SSM command (the agent blocks until the child
  exits); ModelScope CDN needs SSL_CERT_FILE=certifi on fresh Windows; the
  qwen venv also needs pydantic+pyyaml (uavlab imports). Driver:
  scripts/win_eval/run_eval.ps1 (stratified 100 tasks = first 10 per class).
  Camera verified: first frames differ per task and show the targets. Colours
  are BGR-as-RGB in the OFFICIAL evaluator too (orange sky) - left as-is.
  RESULT OpenVLA-UAV (bf16, 100 tasks, 5,463 calls): mean nDTW 0.395; Turn 0.18,
  Move 0.12, Shift 0.67, Rotate 0.35, Surround 0.75, Ascend/Descend 0.78,
  Approach 0.39, Retreat 0.29, Pass 0.25, Land 0.17; median end dist 0.61 m.
  Files D:/drone_vla_pilot/runs/sim_eval_win/openvla/{flights,score.json}.
  OUR adapter (qwen k8 10-shard s2500): first attempt INVALID (server died on
  missing pydantic, 0 calls). Rerun launched ~17:10 UTC via C:
unsinish.ps1:
  runs qwen, zips both results to s3://<bucket>/results/
  {openvla,qwen}.zip + run_eval.log, then Stop-Computer (instance STOPS itself).

- D163 update: user asked to shut down before our adapter's rerun finished.
  Windows instance STOPPED ~17:15 UTC mid-run (qwen had 0 completed tasks). Both
  instances stopped. To finish later: start <instance-id> (g6.xlarge or
  g5.2xlarge), then via SSM run C:
unsinish.ps1 again (it reinstalls deps,
  reruns qwen from scratch, uploads results, and stops the instance itself).

- D163 update 2: user changed their mind ("if it will run and shut down, never
  mind"). Instance restarted 17:45 UTC as g6.xlarge and C:
unsinish.ps1
  relaunched: qwen rerun -> results/{openvla,qwen}.zip + run_eval.log to S3 ->
  Stop-Computer. Expected done ~19:30 UTC. Next session: confirm the instance is
  STOPPED, then download and score results/qwen.zip.

- D164 (2026-09-25): OUR adapter (Qwen3-VL-4B bf16, official format K=8, 10
  shards + mirror, s2500) on the same 100 Windows closed-loop tasks: mean nDTW
  0.129 vs OpenVLA-UAV 0.395; theirs better on 77/100 paired flights. Per class
  ours/theirs: Turn .154/.176, Move .044/.121, Shift .106/.675, Rotate .110/.349,
  Surround .001/.753, Ascend/Descend .057/.775, Approach .320/.389,
  Retreat .355/.293, Pass .134/.252, Land .004/.167. Ours ends much earlier
  (median 26 vs 50 steps) and further from the reference end (2.62 vs 0.61 m).
  Only Retreat is better. 404 model calls, run valid. Caveats: ours was trained
  on REAL flights only and tested in SIM (their checkpoint's action stats are the
  "sim" key); our outputs are scaled m->cm by assumption; the evaluator's
  10-near-still-steps rule may end our slower flights early. Results:
  D:/drone_vla_pilot/runs/sim_eval_win/{openvla,qwen}/score.json. Both
  instances confirmed STOPPED.

- D165 (2026-09-25): real + SIMULATOR training run launched (user approved).
  Overlap check first (logs of all 21 UAV-Flow-Sim shards read via HF range
  requests, index D:/drone_vla_pilot/data/uav_flow_sim_index.json): 10,109 sim
  flights; only 1 of 273 test tasks has a sim flight with the same start (<0.5 m)
  AND the same instruction; 54 test tasks have a sim flight starting <0.5 m away;
  238/273 test instructions occur verbatim (templated wording). Excluded every
  sim flight starting <0.5 m from any test start: 168 flights, list
  D:/drone_vla_pilot/data/uav_flow_sim_excluded.json -> 9,941 sim flights kept
  (312k frames, median 25 frames / 5 m). Sim data shares the DowntownWest town
  with the test tasks: report as same-environment, disjoint-trajectory.
  scripts/prepare_uav_flow_sim.py: official maths after cm -> m (x,y,z of raw and
  preprocessed logs /100), all sim to train, appended to the 10-shard store
  (original kept as episodes.real_only.jsonl), action stats NOT recomputed.
  Trainer: --init-adapter (fresh optimiser). Run: from adapter_s2500, K=8, both
  wordings, mirror, bf16, batch 32, lr 2e-4 cosine, 2,000 updates, ckpt every
  500, wandb asael/vla training/official_k8_realsim, out
  ~/runs/official_k8_realsim.
  The user's home IP changed (<ip> -> <ip>), so SSH to the
  Linux box is blocked by its SG; instead vla-eval-windows-role was attached to
  it and it is driven by SSM + S3 (no SG change). Pipeline
  scripts/aws/sim_pipeline_d165.sh: hard auto-stop +6 h (04:36 UTC), downloads
  sim, prepares, trains, uploads s3://<bucket>/d165/
  realsim_adapter.tgz + sim_pipeline.log, then shuts down. NEXT: when the tarball
  is in S3, start the Windows box and run run_eval.ps1 -Models qwen with the new
  adapter (the driver needs its adapter path parameterised), compare with D164.

- D165 training DONE 2026-09-25 02:12 UTC (instance stopped itself): 2,000
  updates from adapter_s2500 on real+sim (1,121,502 train examples incl. mirror),
  lr 2e-4 cosine; train loss 2.23 -> 1.19; held-out (REAL unseen split) 2.565 ->
  2.666 (250) -> 2.582 (2000), i.e. real performance roughly unchanged. Adapter
  s3://<bucket>/d165/realsim_adapter.tgz, local
  D:/drone_vla_pilot/runs/d165/adapter_s2000. Windows closed-loop eval of it
  launched 07:56 UTC (C:
uns
ealsim.ps1, same 100 tasks, output
  C:
uns\qwen_realsim), uploads results/qwen_realsim.zip and stops itself.

- D166 (2026-09-25): real+sim adapter (D165, s2000) on the same 100 Windows
  closed-loop tasks: mean nDTW 0.333 (0.336 over the 99 non-empty flights) vs
  0.128 real-only and 0.395 OpenVLA-UAV. Better than real-only on 69/100, worse
  than OpenVLA-UAV on 62/100. Per class (theirs/real/real+sim): Shift
  .675/.106/.664, Surround .753/.001/.612, A/D .775/.057/.721, Retreat
  .293/.355/.467, Approach .389/.320/.343, Rotate .349/.110/.259, Land
  .167/.004/.051, Pass .252/.134/.142, Move .121/.044/.020, Turn .176/.154/.120.
  Median steps 34, end distance 1.29 m. Domain gap confirmed as the main cause.
  Report section 9 in reports/uav_flow_closed_loop_20260925/REPORT.md. Both
  instances stopped.

## D167 (2026-09-28): why our closed-loop flights stop early

- Harness audit from run logs: state chaining exact 567/567 calls; evaluator flies
  exactly our poses in 99/100 flights; left/right conventions agree real vs sim.
  8/273 test files have a tilted initial_pos (bent reference path; 2 in our 100).
- 89/100 of our real+sim flights end by the evaluator's 10-tiny-moves rule after
  the model returns a whole all-zero 8-move chunk ("stop"); Land/Approach stop on
  the 3rd call. Flown/ref length: Land 3.3/10.2 m, Pass 7.9/15.7, Approach 3.6/6.0.
- Training data (10-shard store, Linux g5, ~10 min): no hover pauses mid-flight
  (<=5% of flights); all-zero-chunk targets are 15-27% of samples on short sim
  flights (last frame + 5 repeats). Sim train speeds 0.20-0.43 m/step, so our
  0.28-0.30 m/move is in range.
- Likely mechanism: with K=8 one "stop" answer = 8 still moves, so one wrong stop
  ends the flight; OpenVLA-UAV (K=1) must answer stop ~10 times in a row. Next
  test (no training): execute one move of an all-zero chunk and re-ask.
- Research note: docs/research/VLA_AFTER_SFT_20260925.md (camera-use framing is a
  hypothesis; early stopping is the measured failure).

## D168 (2026-09-28): stop-confirm closed-loop run launched

- scripts/uav_flow_eval_server.py --stop-confirm: if every step of a chunk is
  still (< 3 sim units, < 1 deg, the evaluator's own threshold) only the first
  pose is returned, so a flight ends only if the model keeps answering "stop"
  (10 still moves), as a one-step model must. Offline: 183/567 D166 calls were
  all-still chunks. The evaluator is unchanged.
- Windows <instance-id> (g6.xlarge; capacity on 3rd try) started 10:3x UTC,
  hard shutdown +4 h. C:\runs\stopconfirm.ps1: backs up the old server
  (uav_flow_eval_server.py.d166), pulls s3://<bucket>/
  d168/uav_flow_eval_server.py, runs the same 100 tasks with the D165 real+sim
  adapter_s2000 into C:\runs\qwen_stopconfirm, uploads results/qwen_stopconfirm.zip
  + results/run_eval_stopconfirm.log, then Stop-Computer. Verified: server
  process runs with --stop-confirm, 100 tasks selected, model loaded.
- To finish: presign-download results/qwen_stopconfirm.zip, score with
  scripts/score_uav_flow_sim.py, compare per class with qwen_realsim (0.333) and
  openvla (0.395); confirm the instance is stopped.
- D168 partial (12:00 UTC, 48/100 flights, D:/drone_vla_pilot/runs/sim_eval_win/
  qwen_stopconfirm_partial): the finished tasks are the classes run first (Turn,
  Move, Shift, Rotate, Surround, Ascend/Descend); the target classes (Approach,
  Retreat, Pass, Land) have not run yet. Same 48 tasks: stop-confirm 0.377, old
  ours 0.410, OpenVLA-UAV 0.463. 41 identical, 1 better, 6 worse - all turns
  (Turn 0.15 -> 0.08, Rotate 0.27 -> 0.15). 6 flights hit the 100-move cap.
- D168 RESULT (100/100, run ended 13:07 UTC; D:/drone_vla_pilot/runs/sim_eval_win/
  qwen_stopconfirm): mean nDTW 0.316 vs 0.333 without stop-confirm and 0.395
  OpenVLA-UAV. Better on 10 flights, worse on 17; 12 hit the 100-move cap.
  Per class new/old/theirs: Land .05/.05/.17, Approach .33/.34/.39, Pass
  .17/.14/.25, Retreat .47/.47/.29, Turn .06/.12/.18, Rotate .13/.26/.35, others
  unchanged. Conclusion: when asked again the model keeps answering "stop" - it
  believes it is done. The early stop is the model's judgment, not the 8-move
  format; the fix belongs in training. Stop-confirm is not adopted.
- Windows box did NOT stop itself (Stop-Computer after Stop-Transcript had no
  effect); found running at 13:36 UTC and stopped manually. Future scripts: put
  Stop-Computer before Stop-Transcript and keep the shutdown /s timer as backup.
- D168 Land data check (Linux g5, stopped after): the training data is FINE.
  Sim Land 539 flights: 98% descend >0.5 m, median -2.26 m, dz actions and state
  agree; real Land 21 flights: 81%, median -2.61 m. The descent comes at the END
  of land flights, so our drone (which stops at a fixed ~3 m) never reaches that
  phase. Side note: 29% of sim Land dz steps are below the token range (-0.133
  m/step) and get clipped. Next agreed step: longer training on the existing
  real+sim data (last run saw ~6% of it once), e.g. 10k updates, ~15 h, ~$18.

## D169 (2026-09-28): long run with simulator validation and early stop

- Trainer: --sim-val-per-kind (504 held-out UAV-Flow-Sim flights, 56 per motion
  type; list saved as sim_val_flights.json), --sim-val-examples 1024, real
  validation 1,024 examples (was 128), per-type loss in wandb (sim_val/<type>),
  --early-stop-patience (on simulator loss; adapter_best keeps the best point).
  scripts/aws/long_run_d169.sh: self-stopping, uploads s3://.../d169/, hard stop 20 h.
- First launch (lr 2e-4 cosine, 17:22 UTC, wandb official_k8_d169): validation
  step 1 sim 0.737 / real 2.621; step 500 sim 0.767 / real 2.663, every type
  slightly worse - the high-learning-rate bump also seen in D165. Early stop
  would likely cut it off during that bump, so it was stopped at ~update 560.
  Step-1 sim loss by type: Pass 2.19, Land 1.49, Approach/Move 1.19, Retreat 1.03,
  Shift 0.44, Ascend/Descend 0.39, Surround 0.36, Turn/Rotate 0.20.
- D169b relaunch 18:37 UTC at lr 5e-5 (same data, validation, early stop), out
  ~/runs/official_k8_d169b, wandb official_k8_d169b. The stale wandb entries
  official_k8_d169 and wandb_demo_b8 are dead processes, not live runs.
- D169b stopped by the user at update ~2,700 (23:24 UTC) to save budget for a
  larger run with more real shards (waiting on the P-quota reply). Validation
  (real / sim): s1 2.6207/0.7372, s500 2.6272/0.7431, s1000 2.6292/0.7375,
  s1500 2.6321/0.7365, s2000 2.6224/0.7325, s2500 2.6258/0.7316 (best).
  s3://<bucket>/d169/d169_adapters.tgz (895 MB) holds
  adapter_s1000, adapter_s2000, adapter_best (= update 2500), report.json,
  sim_val_flights.json; log d169/long_run_d169.log. Machine stopped itself.
  Research: docs/research/VLA_PLATEAU_20260929.md (loss is a poor gauge; far less
  training than OpenVLA-UAV's 200k steps; LoRA LR 5e-5 is 10x below the recipe).

## D170 (2026-09-29): scripts for the bigger run (all real shards), ready, not launched

- scripts/prepare_uav_flow_more.py: appends real shards train-00010..00053 to the
  10-shard store. Validation SITES stay fixed (new flights at those sites -> val),
  action statistics are not recomputed, flights already present are skipped,
  appends per shard. --instruction-sweep optional (the old rule drops ~44% of
  flights whose wording appears in validation). Verified locally: 15 flights
  removed from a copy of the 1-shard store and re-added from shard 00000 come back
  identical (site, split 8/3/4, instructions, actions, proprio, frames).
- Trainer: validation now also reports teacher-forced action-token accuracy
  (exact and within one bin; OpenVLA's measure), overall and per motion type;
  every rank lists every motion type so multi-GPU reductions line up.
- scripts/aws/prep_more_d170.sh (g5): grows the root filesystem (volume must be
  enlarged first via the API), 3-update trainer smoke test, 44 shards streamed one
  at a time through the NVMe scratch disk, store streamed to s3://.../d170/store.tar,
  code tarball + pip freeze, then stops. ~5 h, ~$6 (estimate).
- scripts/aws/train_big_d170.sh (p4d, after the P-quota reply): data from S3 to
  NVMe, 20-update multi-GPU smoke test (DDP never ran before), run sized from the
  measured speed to HOURS (default 5), one warmup + cosine at recipe lr 5e-4,
  validation every 1,000 with token accuracy, no early stop on loss, adapters
  synced to S3 every 30 min, wandb offline (no credentials copied), self-stop.
- D170 prep DONE (2026-09-29, g5 09:48-11:22 UTC, stopped itself; root volume
  now 400 GB): shards 00010-00053 all added, 21,795 new real flights = 18,430
  train + 3,365 val (fixed val sites 11031_1227, 11160_717, 11207_2618,
  11207_2619), 0 dropped (no instruction sweep, user decision), 0 skipped; 0.8% of
  new steps lie outside the action range. Store streamed to
  s3://<bucket>/d170/store.tar (75.4 GB) + vla_code.tgz +
  requirements_g5.txt + prep log. Trainer smoke test passed; first token-accuracy
  numbers (32/16-example smoke validation, D165 adapter): real exact 0.50 /
  within-1-bin 0.68, sim 0.90 / 0.91 (Pass 0.52, Approach 0.82, Land 0.92).
  OpenVLA trained until TRAIN token accuracy > 95%. Next: big run on the p4d when
  the P quota (case <support-case>) is granted: scripts/aws/train_big_d170.sh.

## D171 / D172 (2026-09-30): long g5 run launched; RL pipeline built

- D171: long single-GPU run on all 54 real shards + sim (scripts/aws/
  train_long_g5_d171.sh), 26,000 updates, lr 5e-4, one warmup + cosine, from
  D169b adapter_best; validation every 1,000 with token accuracy; adapters to
  s3://.../d171/run/ hourly; self-stop, hard stop 50 h. Started 15:29 UTC,
  6 s/update, 12 of 30 GB RAM. wandb asael/vla training/official_k8_d171.
- Costs while stopped (Cost Explorer + inventory): EBS 400 GB + 250 GB gp3 about
  $1.7/day, S3 about $1.8/month; September usage $56.03 fully covered by credits.
  Small unrelated items on the bill (t3.micro, Bedrock DeepSeek tokens,
  CloudFront, Secrets Manager) did not come from this work.
- D172 RL (WorldVLN recipe, docs/research/VLA_AFTER_SFT_20260925.md):
  scripts/rl/reward_uav_flow.py (0.5 x [0.85 dense end reward + 0.15 success at 3 m /
  10 deg] + 0.5 x official nDTW, matched to 5e-5 on 300 flights; reproduces 67/50/38% success on the three existing runs, reference
  flight scores 1.0); scripts/rl/build_rl_tasks.py (200 practice tasks, 25 per
  motion type, from simulator TRAINING flights in the benchmark format, person/
  car parked underground, instructions naming spawned objects dropped (2,874),
  self-check worst end mismatch 32 cm; no Surround - all orbit instructions name
  a person); rollout mode in scripts/uav_flow_eval_server.py
  (--rollout-temperature: action-token-only sampling, saves image, prompt, ids,
  log-probs per call); scripts/rl/rollout_samples.py (flight <-> episode matching,
  group advantages; 299/300 existing flights matched); scripts/rl/train_grpo_vla.py
  (PPO-clipped per-token GRPO, old log-probs recomputed, expert SFT anchor);
  scripts/rl/collect_rollouts.ps1 (Windows, PASSES x tasks, uploads, stops).
  Tasks packaged at D:/drone_vla_pilot/data/rl/rl_tasks.zip. The GPU parts
  (sampling server, GRPO trainer) are NOT yet run: only one G machine may run at
  a time, and the g5 is training until ~2026-10-02.

## D173 (2026-09-30): all post-SFT options from the research note added (not yet run)

User asked to add every option from docs/research/VLA_AFTER_SFT_20260925.md:
- Expert path in each RL group (EG-GRPO / WorldVLN 12:1 replay): the SFT anchor
  in train_grpo_vla.py (--sft-weight, default 0.1).
- Retry failed tasks (RecoverFly): scripts/rl/select_tasks.py builds the next
  round's task zip, half the worst unsolved tasks so far, half new ones spread
  over motion types; collect_rollouts.ps1 -TaskZipKey <zip> -PerKind 0 flies it.
- Preference pairs (AeroDPO / GRAPE): train_grpo_vla.py --objective dpo, best vs
  worst flight per task (reward gap >= 0.1), trajectory log-prob = sum over its
  calls, rollout adapter as reference, exact gradient computed call by call.
- Contrastive decoding (PCD / VCD) against a gray photo: server --contrast-alpha,
  with a plausibility cut (tokens below 0.1 x max real probability excluded).
- Re-ask sooner: server --execute-steps N (fly N of the 8 moves, then new photo).
All GPU parts are untested until the g5 run ends (one G machine at a time).
Plan after D171: flight test of the best checkpoint plain / --execute-steps 4 /
--contrast-alpha 0.5 (about $1.30 each), then RL round 1 (GRPO) and, on the same
rollouts, a DPO update, both flight-tested.

## D174 (2026-09-30): research - what the papers do not solve; goal is now beating WorldVLN

User goal (new, supersedes the D159 "unrealistic" note): surpass WorldVLN's 79.1%.
Note: docs/research/VLA_BEYOND_WORLDVLN_20260930.md; PDFs papers/aerial_vla_open_problems/.
No code changed.
- The paper success rate is task-weighted (Land 54, Shift 49, Approach 42, Pass 40 of
  273). WorldVLN's per-class table reproduces 79.1% = 216/273 that way. Our re-scored
  per-class results give OpenVLA-UAV about 68% and ours (real + sim) about 47%.
- WorldVLN's four worst classes are Pass 40%, Rotate 46.7%, Surround 58.3%, Turn 60%;
  OpenVLA-UAV's worst are Rotate 1/10, Pass 4/10. Ours: Pass 0/10, Land 1/10, Rotate
  3/10, Turn 4/10.
- Measured from the task files: Rotate is pure numeric yaw (15-180 deg); Turn ends at
  exactly the bearing to the object (median 14 deg change, 10 deg tolerance); Pass ends
  7.1-14.7 m past the object with no distance in the instruction.
- Our flights: all 10 Turn flights run 34 steps whatever the angle; Rotate flights end
  at 18/42/74 steps (same count for 90 and 150 deg). The state (yaw vs start, degrees)
  is in the prompt, but the model does not compare it with the instruction.
- Shared failure = extent/termination grounding. WorldVLN has no pose input.
  Proposal ranked 1: progress tokens (predict remaining displacement before the 8 moves,
  free labels). 2: goal memory in the prompt. 3: RL at WorldVLN scale (multi-round driver,
  dense per-call reward, clip 0.02-0.05). 4: sample-and-verify.
- RL pipeline: complete for one round, untested end to end; missing a multi-round
  driver, a dense reward, and a conservative clip/KL.

## D175 (2026-09-30): progress tokens, goal memory and RL loop built (code only, nothing launched)

From docs/research/VLA_BEYOND_WORLDVLN_20260930.md (#1-#3). No AWS used.
- Progress tokens (train_uav_flow_vla.py --progress): before its moves the model writes
  "Left +FF.F,+SS.S,+UU.U,+YYY" = displacement left to the flight's end in its own frame
  (metres; yaw = summed per-frame turn in degrees, so a full circle reads 360), always 21
  tokens. Labels from the stored per-frame state. Mirror copies negate sideways and yaw.
  Checked locally on 300 real flights: labels equal the re-integrated actions (xy median
  1 mm, yaw 0.00 deg; z differs by the known raw/preprocessed altitude gap, p95 0.94 m).
  Token accuracy in validation now counts move tokens only (identical for old runs).
- Server (uav_flow_eval_server.py --progress [--goal-memory]): phase 1 writes the line
  (greedy), phase 2 the moves after it; malformed line -> answer without it. Goal memory:
  the first line fixes a goal in the start frame; later calls get the line recomputed
  from the drone's pose (round trip exact to 1e-13). Line, parse and goal are logged per call.
- RL: per-move reward (rollout_samples.call_gain: closeness gained per call, telescopes to
  the flight's total, checked on 99 logged flights), flight advantage + 0.5 x call
  advantage; PPO clip default 0.2 -> 0.05; the server's progress line is context in GRPO.
- rl_loop.ps1: several rounds on the Windows box alone (rollouts, GRPO update, next tasks
  via select_tasks.py), each round's adapter to S3, machine off at the end + 24 h timer.
  collect_rollouts.ps1 gained -NoShutdown, -TaskZipFile, -ServerArgs.
- export_anchor_store.py: the practice tasks' expert flights (episodes + frames) as one tar,
  so the Windows loop has the SFT anchor without the 75 GB store.
- Fixed a latent trap: the repo trainer pointed at the 1-shard action ranges and every
  machine sed-edited it. The 10-shard manifest (used by every adapter since D160) is now
  in the repo and the default (override: UAV_FLOW_OFFICIAL_REPORT).
- scripts/aws/train_progress_d175.sh: after D171 - anchor store export, 30-update smoke +
  server smoke (smoke_progress_server.py), then the fine-tune (PROGRESS=0 gives the matched
  "without" run). Needs d175/vla_code.zip uploaded first. Not run.

## D176 (2026-10-01): D171 continues "until real accuracy stops improving", with progress tokens

User decisions: do not end at a fixed 26,000 updates while accuracy still rises; merge the
progress tokens into the remaining training instead of a separate fine-tune afterwards
(cheaper, more practice; the clean with/without comparison is given up).
- Trainer: --schedule wsd (hold the learning rate; after --plateau-patience 3 validations
  with no new best real-flight token accuracy, exact or within one bin, a cosine decay over
  --decay-updates 3000, then stop; the decay also starts 3,000 before the --updates cap);
  --reset-plateau judges the plateau only on validations after the resume.
- scripts/aws/switch_d176.sh, started 2026-10-01 19:24 UTC on the g5 as user ubuntu (a first
  start as root was killed within a minute: wandb's login belongs to ubuntu). It waits for
  adapter_s18000 (the run was at 16,170, just past 16,000), stops the D171 script and trainer,
  replaces the D171 hard stop with +4620 min, runs a 30-update --progress smoke from that
  adapter + the server smoke (greedy + goal memory, sampling), then resumes the same run
  (same folder, optimiser, data stream, wandb run) at the learning rate D171 had at that step
  (no new warm-up), with --progress --reset-plateau if the smoke passed, plain otherwise.
  Cap 56,000 updates (about $85 more at most). Machine off at the end.
- Mid-run numbers (wandb, about 15,000): real token accuracy 51.6% (start) -> 53.3%, within
  one bin 71.4% -> 73.5%; simulator back to its starting 91.0%, within one bin 93.1% (best).

## 2026-10-02: quotas
- P quota (L-417A185B) approved at 64 vCPU, not 96: no p4d/p4de (96). Usable: p5.4xlarge
  (1x H100 80 GB, 16 vCPU, $6.88/h on demand, offered in all us-east-1 zones). Estimated
  about 5x a g5 per update (not measured), so similar cost per update; not used for D176.
- G quota (L-DB2E81BA) increase 8 -> 16 requested 2026-10-02 07:35 UTC, request
  4dd23f43a0bd4749a7ee0b54a980aa63wOLQAA7I, PENDING. Purpose: run the Windows test box
  (g6.xlarge, 4 vCPU) while the g5 (8 vCPU) trains, to flight-test checkpoints mid-run.

## D177 (2026-10-02): stop rule also counts simulator accuracy

At about 24,000 updates real token accuracy was nearly flat (54.0% for 4 checks, +0.1 point
per check) while simulator accuracy (the benchmark domain) still rose (91.5%, Pass 62.8% ->
63.9% since the progress tokens). The D176 rule watched real accuracy only and would likely
have started the decay while the simulator still improved. User approved: count both.
- Trainer --plateau-sim: real exact / within-one-bin and simulator exact / within-one-bin;
  any new best resets the counter; bests resume from the run's history (no reset).
- scripts/aws/switch_d177.sh started 2026-10-02 10:35 UTC as ubuntu (run at 24,200). At
  adapter_s26000 it stops the D176 script and trainer and resumes the same run: same
  learning rate 1.142e-4, --progress --plateau-sim, cap 56,000. D176 hard stop
  (2026-10-05 about 03:35 UTC) unchanged.

## D178 (2026-10-02): flight-test script for several variants (built, not run)
- scripts/win_eval/eval_variants.ps1: -Variants "name|adapter S3 prefix|server flags;...".
  Pulls the current code bundle (server with --progress / --goal-memory), downloads each
  adapter from S3 (the run's hourly sync puts adapter_s<step> under d171/run/), flies the
  same 100 tasks as D163-D168, uploads d178/<name>.zip per variant, machine off (+7 h timer).
  Parse-checked; variant parsing tested.
- score_uav_flow_sim.py now also reports success (end within 3 m and 10 deg, WorldVLN's
  rule) overall and per class; re-scoring the D166 real+sim flights gives 50%, as before.

## D179 (2026-10-02): end the run with the decay (user decision)
User asked to stop training; chose the proper ending (decay) and to wait for the 28,000
checkpoint so nothing is thrown away. D177 had resumed at 26,000 (14:03 UTC) with
--plateau-sim; the run was at 27,060 at 16:08 UTC.
- Trainer --decay-now: on --resume, the wsd decay starts at the checkpoint step.
- scripts/aws/decay_d179.sh started 16:08 UTC as ubuntu: waits for adapter_s28000 (about
  18:00 UTC), stops the D177 script + trainer, sets a 9 h hard stop, resumes with
  --decay-now (cosine 1.142e-4 -> 0 over 3,000 updates, progress tokens on), final adapter
  adapter_s31000 (about 23:45 UTC), S3 sync to d171/run/, machine off.
- wandb drops re-logged steps it has already seen (warning only).
- RL prerequisite done while the g5 trains (no extra machine): rl_tasks.zip uploaded to
  d172/rl_tasks.zip; export_anchor_store.py on the g5 packed all 200 practice tasks' source
  flights (200 episodes, 5,625 frames, 0 missing, 156 MB) to d175/anchor_store.tar, the
  key rl_loop.ps1 reads. Training was unaffected (27,180 at the time).

## D180 (2026-10-02): RL trial round prepared (not run)
- rl_loop.ps1 -AdapterKey <S3 prefix> downloads the start adapter (e.g. d171/run/adapter_s31000)
  and refuses to start without one. Code bundle key for both Windows scripts: d180/vla_code.zip
  (to be built from the repo's scripts/ and uploaded right before launch).
- Trial: -Rounds 1 -Passes 2 -PerKind 1 -Count 8, server and trainer with --progress, Tag
  rl_trial: 8 tasks (one per motion type) x 2 flights, GRPO update on the same box, about $1.
  What it checks: rollouts recorded with progress lines and log-probs, flights matched to
  calls, expert_examples > 0 (anchor store split field), adapter_grpo written, S3 upload,
  machine off.

## D179 result (2026-10-03): training finished, g5 off
Decay 28,000 -> 31,000 finished (report status complete, resumed at 18,000 / 26,000 / 28,000);
log uploaded 23:43 UTC 2026-10-02; g5 stopped. Adapters adapter_s2000 ... adapter_s31000 and
adapter_best on s3://.../d171/run/. Teacher-forced validation (move tokens only):

| Step | Real exact | Real within 1 bin | Sim exact | Sim within 1 bin | Pass | Land | Approach |
|---|---|---|---|---|---|---|---|
| 1 (start) | 51.6% | 71.4% | 91.1% | 93.1% | 62.3% | 77.1% | 81.9% |
| 18,000 (progress tokens on after this) | 53.3% | 74.1% | 91.1% | 93.3% | 62.8% | 76.8% | 82.3% |
| 28,000 (decay starts) | 53.8% | 74.9% | 91.6% | 93.8% | 64.0% | 77.7% | 83.2% |
| 31,000 (final) | 54.5% | 75.5% | 91.7% | 94.1% | 64.9% | 78.0% | 83.4% |

Not yet flight-tested. G quota still 8 (request pending); with the g5 off the Windows box fits.

## D180 launched (2026-10-03, user-approved, about $5)
Windows box <instance-id> started 07:45 UTC; scripts/win_eval/run_d180.ps1 (from
d180/run_d180.ps1, code bundle d180/vla_code.zip) runs: flight test on the 100 tasks of
s18000 (no flags), s31000 --progress, s31000 --progress --goal-memory (results d178/<name>.zip),
then the RL trial (rl_loop.ps1 -Rounds 1 -Passes 2 -PerKind 1 -Count 8, start adapter
d171/run/adapter_s31000, --progress; results d175/rl_trial_r1.grpo.zip + d172/rl_trial_r1.zip).
8 h shutdown timer (15:47 UTC) plus Stop-Computer at the end. First variant's server loading
at 07:47.
- D180 first result (s18000, no progress tokens; 100 tasks, 71 min): nDTW 0.3665, success 52/100
  (D166 adapter 0.336 / 50; OpenVLA-UAV 0.395 / 67). Per class success: Turn 5, Move 7, Shift 9,
  Rotate 5, Surround 1, A/D 9, Approach 5, Retreat 8, Pass 1, Land 2. Long training alone gave
  +0.03 nDTW and +2 points; Land and Pass still fail. Score: D:/drone_vla_pilot/runs/sim_eval_win/
  d178_s18000/score.json. s31000_progress was at 19/100 at 09:23 UTC (slower: about 2 h per 100).
- D180 second result (s31000 --progress, 100 tasks, finished 10:58 UTC): nDTW 0.3834, success
  61/100 (s18000 0.367 / 52; D166 0.336 / 50; OpenVLA-UAV 0.395 / 67). Per class success:
  Turn 4, Move 8, Shift 9, Rotate 8 (s18000: 5; OpenVLA-UAV: 1), Surround 1, A/D 9, Approach 6,
  Retreat 9, Pass 1, Land 6 (s18000: 2; OpenVLA-UAV: 5). Median flight 41 steps (was 34).
  Progress lines: 0 unparsed. Caveat: s31000 also has 13,000 more updates and the decay, so the
  gain is not attributable to the progress line alone; the gains sit on the targeted classes
  (Rotate, Land). Pass (1/10) and Turn (4/10) remain. Score: .../d178_s31000_progress/score.json.
- D180 third result (s31000 --progress --goal-memory, finished 12:45 UTC): nDTW 0.409 (best so
  far, above OpenVLA-UAV 0.395), success 61/100 (same as line only). Per class: Turn 3, Move 10,
  Shift 7, Rotate 6, Surround 1, A/D 7, Approach 7, Retreat 10, Pass 6 (line only: 1; OpenVLA-UAV:
  4), Land 4 (line only: 6). Memory fixes Pass (object leaves view) and helps path shape, but locks
  in a wrong first estimate on numeric tasks (Rotate, Shift, A/D, Land drop).

## D181 (2026-10-03): two Windows-script bugs (found in D180)
- The RL trial never ran: eval_variants' finally uploaded eval.log while its own transcript held
  it (IOException, terminating) -> the wrapper's catch -> skipped the trial.
- The box did not turn off: Stop-Computer refuses while a `shutdown /s /t` timer is scheduled
  ("A system shutdown has already been scheduled"). It idled from 12:46 UTC until the 8 h timer
  (15:47 UTC), about 3 h, about $3 wasted. The same pattern was in rl_loop / collect_rollouts.
- Fix in all four scripts: Stop-Transcript (try) -> upload log (try) -> shutdown /a ->
  shutdown /s /f /t 5. Parse-checked.

## D180 RL trial result (2026-10-03, 14:36-14:58 UTC, about $0.40): pipeline works end to end
- 8 tasks x 2 flights collected with the sampling server (--progress); all 16 flights matched
  to their calls; 103 samples with signal; recomputed old log-probs vs rollout: max diff 0.115,
  mean 0.003. Expert anchor found 171 examples from the 8 source flights (anchor store OK).
- 3 GRPO updates (clip 0.05: ratio about 1.000, 1.7-5.5% clipped; SFT loss 0.73-0.86);
  adapter_grpo + report uploaded (d175/rl_trial_r1.grpo.zip); select_tasks built the next zip
  (2 retried, 6 fresh); machine shut itself down (D181 fix works).
- Finding: 4 of 8 tasks gave two IDENTICAL flights at temperature 1.0 (zero advantage, no
  signal). The policy is very peaked; RL needs more exploration (higher temperature, more
  passes) - SimpleVLA-RL also raises the temperature.
- Timing: about 10 min per pass of 8 tasks, so round 1 at 48 tasks x 8 passes is about 7 h
  (about $7-9), not the $5 estimated before.

## D182 (2026-10-03): why goal memory broke Rotate/Shift/Land, and a fix (offline analysis, no spend)
scripts/analyze_progress_lines.py scores, on the 615 logged calls of the goal-memory flight
test, the model's fresh line and the remembered line against the true remainder to the
reference end (same drone states).
- The memory line is MORE accurate almost everywhere (median position error 0.08 m vs 0.43 m,
  turn 0.32 vs 0.86 deg).
- The failures come from arrival: once the drone is at or slightly past the goal, memory says a
  small remainder or an overshoot ("-0.2 m", "-2 deg"). Training lines only count down to
  exactly 0, so the model never learned to stop on those and keeps moving; the error grows
  ("Turn 90 deg": -2 -> -22 deg; "Move 5 m left at 50 deg": flew 10.6 m past, timeout; "Descend
  9 m at 30 deg": 10.3 m past). The fresh line says 0 there and the model stops.
- Other cases: Land - the first estimate from the start photo is short (6.8 vs 9.4 m), memory
  freezes it; visual Turn - the first bearing estimate is worse than later ones.
- Fix: --memory-deadband 1.0,5 on the server: with goal memory, a remainder under 1 m and 5 deg
  becomes the 0 line. Offline "line says stop exactly when truly arrived" agreement: Shift
  74 -> 97%, Rotate 79 -> 95%, A/D 60 -> 99%, overall 79 -> 85%. Needs one flight test (~$2).
- D182 flight test launched (user-approved, about $2): Windows box started 16:16 UTC,
  eval_variants.ps1 variant s31000_memdead (--progress --goal-memory --memory-deadband 1.0,5),
  4 h shutdown timer, result d178/s31000_memdead.zip; machine off at the end (D181 fix).
- D182 RESULT (s31000 --progress --goal-memory --memory-deadband 1.0,5; finished 17:54 UTC,
  box stopped itself): **success 69/100, nDTW 0.449** - above OpenVLA-UAV (67 / 0.395) on both,
  first time. Per class: Turn 3, Move 10, Shift 9, Rotate 10, Surround 1, A/D 9, Approach 8,
  Retreat 10, Pass 4, Land 5. Deadband recovered Rotate (6 -> 10), Shift (7 -> 9), A/D (7 -> 9),
  Approach 7 -> 8; Pass fell 6 -> 4. Task-weighted (paper style, using these per-class rates)
  about 68% vs WorldVLN 79.1%. 100 tasks: about +-5 points of noise.

## D183 (2026-10-03): release package for the model
User asked to save everything needed for GitHub, Hugging Face and the paper.
- Full package (weights + everything): D:/drone_vla_pilot/release/qwen3vl4b_uavflow_vla_progress/
  (669 MB), backed up as s3://<bucket>/release/qwen3vl4b_uavflow_vla_progress.tar.
  Contents: adapter_s31000 (final) and adapter_s18000 (pre-progress ablation); training/
  (phases.json with every phase's data/init/schedule/updates/outcome, report_long_run.json with
  every update's loss and every validation, curves.png, logs of phases 1, 2, 4b, 5 and the data
  prep); data/ (action ranges, held-out simulator flights, excluded near-test flights, added
  shards, RL practice tasks); eval/ (100-task list, per-flight scores + flights + server logs for
  every variant and OpenVLA-UAV, results.json); rl_trial/; code/ (scripts snapshot + commit);
  README.md = Hugging Face model card.
- In git: release/qwen3vl4b_uavflow_vla/ (model card, phases, results, curves, task list, data
  lists) and scripts/release/build_release.py (derives phases.json, curves.png, results.json).
- Not in the package: the phase 3/4 console logs and the watcher log, which exist only on the
  g5 disk (the same numbers are in report_long_run.json and in wandb, project "vla training",
  runs official_k8_10shard, official_k8_realsim, official_k8_d169b, official_k8_d171).
- g5 logs exported (user-approved, g5 started about 10 min, 2026-10-03 18:25 UTC): every console
  log and launch script in /home/ubuntu, the report.json/logs of all runs (phase 0
  official_k8_10shard ... official_k8_d171, plus probes and smoke runs) and the local wandb run
  folders (without the .wandb binary) -> s3://<bucket>/release/g5_logs.tgz
  (0.9 MB). g5 stopped again; its disk <volume-id> is KEPT (user: do not delete yet).
- g5 disk contents secured before freeing it (2026-10-03 18:35 UTC): long-run checkpoint at
  31,000 (optimizer state, 923 MB; needed only to resume this exact run), the phase-0 adapter
  (official_k8_10shard adapter_s2500) and the report.json of every earlier run ->
  s3://.../release/lineage/ (1.29 GB). Already on S3: the 75 GB training store (d170/store.tar,
  written 2026-09-29 10:38:09, 3 s after the store's last change - identical), all later
  adapters, logs. Re-downloadable, not copied: base model (8.3 GB), raw shards, raw sim (5.5 GB).
  g5 stopped; terminate instance + delete <volume-id> only after user confirmation.

## D184-D185 (2026-10-03): DAgger and PPO built, switchable (code only, nothing run)
Research D184 (docs/research/VLA_CHEAPER_THAN_GRPO_20261003.md): of seven methods, DAgger and
PPO are both better and cheaper than our GRPO. Built so one can be used and the others off:
rl_loop.ps1 -Method grpo|ppo|dagger, -Kinds "Land,Pass".
- DAgger: scripts/rl/dagger_relabel.py relabels every logged call with the recorded expert
  flight - progress line to the recorded end, and 8 moves that follow the path from its nearest
  point with the pose's offset removed over the chunk. Checks: on the recorded path the labels
  equal the trainer's (max diff 0.0000), the moves equal the path's steps (0.00000); from a pose
  1 m off the path and 20 deg rotated the 8 moves land exactly back on it; on the RL trial's
  16 real flights: 103 rows, photos found, trainer encodes them (21 + 33 labelled tokens).
  Server --save-call-images (photos also when greedy); collector -Temperature 0 uses it;
  trainer --extra-rows/--extra-repeat and a validation fallback for stores without val flights.
  Loop: greedy pass on the same tasks each round, rows aggregated over rounds, supervised update
  (300 updates, lr 2e-5 by default) from the current adapter.
- PPO: train_grpo_vla.py --objective ppo: zero-initialised value head on the last hidden state
  at the prompt end, rollout_samples.ppo_advantages = GAE over each flight's calls (reward =
  call_gain, + flight reward on the last call; gamma 0.99, lambda 0.95), normalised; loss =
  clipped policy loss + 0.5 x value MSE in one backward; value_head.pt carried across rounds;
  all calls kept (no zero-spread filter). GAE checked on the trial data (advantages fall from
  about 2.3 to 0.2 along a flight). The GPU parts are untested until a Windows round.


## D185 check (2026-10-03): DAgger and PPO run end to end on the simulator box (about $0.60)
One tiny round each on 4 Land/Pass practice tasks, best server setup (progress line + goal
memory + deadband), start adapter = the final model (update 31,000).
- DAgger: greedy pass, every photo saved; 4 flights -> 32 relabelled rows; 20 updates (loss
  0.72 -> 0.57); adapter saved and uploaded; box continued to the PPO check.
- PPO (temperature 1.3, 2 passes): 8 flights, all 4 tasks with reward spread (at 1.0 the GRPO
  trial had half its groups flat); 71 samples; old log-probs match the rollout (max 0.08);
  GAE advantages computed; expert anchor 138 examples; 2 updates; adapter + value_head.pt saved;
  next tasks selected; box shut itself down.
- Bug found and fixed: value loss 3.5 -> 211 after one update - the head reads raw last-layer
  hidden states (large values), so one Adam step moved V wildly. The head now reads
  layer-normalised hidden states.


## D187 (2026-10-03): rule 1 - the 273 test tasks are off limits

> **RULE 1 (user, 2026-10-03): the 273 UAV-Flow-Eval benchmark test tasks are NOT flown again,
> for any purpose (tuning, checks, ablations), until the user decides otherwise.** The 100 of them
> flown so far (first 10 per class) were used for development decisions (goal memory, deadband);
> the paper must say so. Decisions from now on use validation tasks built from the 504 held-out
> UAV-Flow-Sim flights (D169, `release/qwen3vl4b_uavflow_vla/sim_val_flights.json`).


## D188 (2026-10-03 21:00 UTC): DAgger round 1 launched, decided on validation (user-approved, about $6.30)
Windows box, scripts/win_eval/run_d188.ps1 (7 h timer, machine off at the end):
1. before - best setup (adapter_s31000 + progress line + goal memory + deadband) on the 73 Land +
   Pass VALIDATION tasks (d187/val_tasks.zip; measured only) -> S3 d188_before/before.zip
2. DAgger round 1 - one greedy pass over the 50 Land + Pass practice (TRAIN) tasks, every call
   relabelled from the recorded flight, 300 updates at lr 2e-5 from adapter_s31000 -> S3
   d188/dagger_r1_adapter/
3. after - the DAgger adapter on the same 73 validation tasks -> S3 d188_after/after.zip
4. PPO check - 4 practice tasks x 2 flights, one PPO update (value-head fix on the GPU)
The 273 benchmark test tasks are not used (rule 1). Started 20:59 UTC, 73 validation tasks loaded.

D188 step 1 result (22:38 UTC), "before" on the 73 Land + Pass VALIDATION tasks, scored with
scripts/score_tasks.py (success = end within 3 m and 10 deg):

| kind | flights | success | nDTW |
|---|---|---|---|
| Land | 41 | 26/41 | 0.328 |
| Pass | 32 | 19/32 | 0.234 |
| all | 73 | 61.6% | 0.287 |

This is the number DAgger round 1 has to beat on the same tasks (step 3). Step 2 (DAgger
greedy pass over the 50 practice tasks) started 22:39 UTC.

D188 result (run done 02:21 UTC 10-04, box off). Same 73 Land + Pass VALIDATION tasks:

| | Land | Pass | all | nDTW |
|---|---|---|---|---|
| before (adapter_s31000) | 26/41 | 19/32 | 61.6% | 0.287 |
| after DAgger round 1 | 36/41 | 18/32 | 74.0% | 0.305 |

Paired per task: 11 tasks fixed (all Land), 2 broken (1 Land, 1 Pass); sign test p = 0.02.
DAgger: 50 practice (train) flights -> 394 relabelled rows, 300 updates, loss 0.74 -> 0.51.
DAgger fixes Land (stopping at the ground) but not Pass (stopping after passing the object).
PPO check: 2 updates, value loss 3.54 -> 2.51 (was 3.5 -> 211 before the layer-norm fix), clip
fraction 1-3%, ratio ~1.0: PPO is stable. Adapter: S3 d188/dagger_r1_adapter/.

D188 Pass analysis (validation flights, after DAgger): all 14 Pass failures stop SHORT (2.9-10 m
before the end, sideways error <2.5 m). Successes have paths 7.8-13 m, failures mostly 13-20 m.
The model's first progress line for Pass is ~10 m on every task (correlation with the true
length 0.04): it does not read the distance from the photo, it says the training mean, and goal
memory then locks that guess for the whole flight. More DAgger on Pass is unlikely to fix this
(training already had the true lengths). DAgger round 2 held back pending a Pass fix.

Per-class error analysis (successes out of 10, 100 TEST tasks flown before rule 1; re-scored from
saved score files, nothing flown; also in docs/research/VLA_BEYOND_WORLDVLN_20260930.md section 6):

| | Turn | Move | Shift | Rotate | Surround | Asc/Desc | Approach | Retreat | Pass | Land | All |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Memory, no fix (D180) | 3 | 10 | 7 | 6 | 1 | 7 | 7 | 10 | 6 | 4 | 61% |
| Memory + snap to 0 (D182) | 3 | 10 | 9 | 10 | 1 | 9 | 8 | 10 | 4 | 5 | 69% |
| OpenVLA-UAV 7B | 10 | 10 | 9 | 1 | 0 | 10 | 9 | 9 | 4 | 5 | 67% |

Biggest gap: Turn (3 vs 10). Land and Pass tied with OpenVLA-UAV; Rotate our win.

## D189 (2026-10-04): can Turn / Surround training flights become practice tasks? (local, $0)
UAV-Flow-Sim logs hold only drone poses + instruction (no object position), so the 733 Turn and
165 Surround training flights were dropped from the practice tasks (D172). Local check on the
user's PC (local simulator, no model; scripts/rl/fetch_sim_flights.py, replay_object_check.py):
- Replaying recorded poses reproduces the recorded photos (same scene, same framing).
- Surround: a least-squares circle through the path gives the centre; fitted radii 6.0 / 3.0 /
  6.0 m equal the instructions exactly, and the placed person stands where the recorded one does.
- Objects: the evaluator's use_obj 2 is a CAR. People and dogs are both BP_Character_21:
  appearance 1-19 = people, 20-35 = robot dogs; appearance 0 is ignored when switching.
- Turn: Qwen3-VL-4B (local, NF4) boxes the dog / person in the recorded last photo; the foot row
  and centre column give distance and bearing (90 deg FOV, camera height = drone z, ground z 0).
  Dogs: 2 of 3 placed exactly (same spot and size in first and last frame), 1 roughly.
  People: the person mesh sits about 1 m from its actor location (depends on its rotation), so
  people render closer and to the side - needs a one-off offset calibration.
- Fetching from Hugging Face reads only the needed parquet row groups (ids are sorted; row-group
  id ranges are in the metadata).
- D189 build (local, $0, about 2.5 h on the user's PC): scripts/rl/build_object_tasks.py ->
  Surround: circle fit (radius within 25 % of the stated radius, all passed); Turn: Qwen3-VL-4B
  box on the last recorded photo -> bearing + distance (max of ground-plane and size estimate;
  person 1.75 m, dog box 1.0 m). Cross-check: one dog seen in a Turn and a Surround recording
  2 s apart is placed 0.5 m apart by the two methods. scripts/rl/verify_object_tasks.py renders
  every Turn task with and without the object and keeps it only if the object is visible
  (>= 0.2 % of the image) within 0.08 image width (about 9 deg) of the photo's box.
  | Set | Surround | Turn built | Turn kept (person / dog) | dropped (no box / range / not visible / too small / bearing) |
  |---|---|---|---|---|
  | train | 165 | 679 | 492 (375 / 117) | 48 / 6 / 49 / 88 / 50 |
  | validation | 55 | 33 | 23 (15 / 8) | 2 / 0 / 1 / 6 / 3 |
  Dogs fail most (uneven ground: placed at z 0 they sink or hide). Files:
  D:/drone_vla_pilot/data/rl/object_tasks_{train,val}.zip (verified only), verify.json per set,
  spot-check sheets object_tasks_train_spotcheck*.png. Not flown with the model yet.

## D190 (2026-10-04 12:05 UTC): DAgger round 2 launched (user-approved, about $10)
Windows box, scripts/win_eval/run_d190.ps1 (13 h timer, machine off at the end), code d190/vla_code.zip.
1. before - round-1 model (d188/dagger_r1_adapter + line + memory + deadband) on the 100-task
   VALIDATION check set d190/check_tasks.zip (Land 15, Pass 15, Shift 15, A/D 10, Approach/Move 10,
   Rotate 10, Surround 10, Turn 12, Retreat 3) -> d190_before/before.zip
2. DAgger round 2 - greedy pass over 150 TRAINING tasks d190/dagger_tasks.zip (50 new Land, 75
   Turn, 25 Surround with reconstructed objects, D189), relabel, 300 updates lr 2e-5 on the new rows
   + round 1's 394 rows (rl_loop -PriorRows) -> d190/dagger_r2_adapter/
3. after - same 100 validation tasks -> d190_after/after.zip. Reject the round if any kind drops.
No flight is in both sets (checked). At 12:08 UTC: 100 tasks selected, "before" server loading.
- D190 bug seen at 12:30 UTC (24/100 before flights): 3 flights ended at once with "Response
  'action' is empty" - e.g. "Orbit the dog clockwise at a 5.5-meter radius": the progress line
  says "Left +00.0,+00.1,+00.0,-360", then phase 2 does not yield 32 action tokens, the server
  returns [] and the evaluator stops (flight "no data"). Same server before and after, so the
  comparison stays fair; to investigate after the run (likely Surround in D182 too: 1/10).

## D191 (2026-10-04): the orbit bug - greedy decoding was not restricted to action tokens (fixed, local, $0)
Reproduced locally (round-1 adapter, NF4, the 20 Surround + Rotate check tasks rendered as the
evaluator sends them): on both "Orbit the dog clockwise at a 5.5-meter radius." tasks the line is
"Left +00.0,+00.1,+00.0,-360" and phase 2 writes 8 x [dx, dy, dz, token 151022]: the yaw slot is a
token OUTSIDE the 256 action tokens. Greedy decoding was unconstrained and dropped non-action
tokens afterwards -> 24 of 32 left -> decode None -> the server answered no moves -> the evaluator
ended the flight at once. The other 18 tasks were well-formed.
Fix (scripts/uav_flow_eval_server.py): greedy phase 2 generates exactly 4 x chunk tokens with the
same ActionTokensOnly mask that sampling already used. Re-run on all 20 tasks: every answer has 8
moves; the dog orbits now yaw like the person orbits (token 151490 vs 151492); answers that were
already well-formed are unchanged. --unconstrained-greedy reproduces the old behaviour.
NOT in the running D190 (its code bundle d190/vla_code.zip is left unchanged so before and after
use the same server). Earlier Surround scores (test 1/10) were partly this bug.
- D190 step 1 result (round-1 model, old server, 100 validation check tasks): success 67%, nDTW
  0.383. Per kind: Land 13/15, Pass 8/15, Shift 13/15, A/D 9/10, Approach/Move 5/10, Rotate 10/10,
  Surround 2/10, Turn 4/12, Retreat 3/3. Step 2 at 137/150 DAgger flights at 16:21 UTC.
- D190 RESULT (run done 19:06 UTC, box off; same 100 validation tasks, old server both times):

  | Kind | before (round 1) | after (round 2) |
  |---|---|---|
  | Land | 13/15 | 15/15 |
  | Pass | 8/15 | 9/15 |
  | Shift | 13/15 | 13/15 |
  | Ascend/Descend | 9/10 | 9/10 |
  | Approach/Move | 5/10 | 5/10 |
  | Rotate | 10/10 | 9/10 |
  | Surround | 2/10 | 2/10 |
  | Turn | 4/12 | 4/12 |
  | Retreat | 3/3 | 3/3 |
  | All | 67% / nDTW 0.383 | 69% / nDTW 0.380 |

  Paired: 3 fixed, 1 broken, p = 0.63 - no clear gain. Turn did not move despite 75 Turn practice
  tasks; Surround is still capped by the orbit bug (old server). Rotate lost one task (noise at n=10,
  but by the stated rule a kind got worse). DAgger: 147/150 flights -> 934 rows (+394 from round 1),
  300 updates, loss 0.83 -> 0.52; adapter d190/dagger_r2_adapter/.
- D190 Turn analysis (validation, both models): on all 12 Turn flights the FIRST progress line says
  a right turn of +24 deg (one says +22), whatever the scene; the true turns range -30 .. +25 deg.
  Goal memory then locks that guess and every flight ends at about +22 deg. The 4 successes are the
  tasks whose true turn happens to be about +20 deg. Same pattern as Pass (first distance always
  ~10 m, D188): the model does not read the target's bearing / distance from the photo; it writes
  the training average. DAgger round 2 (75 Turn tasks) did not change the first line. The teacher's
  label is right for Turn from any state (the object does not move), so the teacher is not the cause.

## D192 (2026-10-04): research - making the model use the photo (NO CODE CHANGED)
docs/research/VLA_VISUAL_GROUNDING_20261004.md, papers/visual_grounding/. Found two input
mismatches in our pipeline: training photos are RGB at 256 px, the evaluator sends BGR at 224 px.
Ranked: (1) fix both in the server and test locally on the Turn / Pass start photos ($0);
(2) ECoT-style grounded reasoning - the model writes the target's bbox_2d (Qwen's native format)
before the progress line, labels from base Qwen3-VL as in D189, can be dropped at test time
(ECoT-Lite). LIT's goal-bottleneck is what our progress line + goal memory already is.
- D192 check (local, $0): asked "Locate the person/dog ... bbox_2d" on the 12 Turn validation start
  photos (correct colours, 256 px). Base Qwen3-VL-4B: boxes the person 8/8; the robot dog 1/4 with
  the word "dog" (D189 found them when prompted "a four-legged robot dog"). Our round-1 model: 0/12 -
  it answers EVERY question with a progress line and move tokens ("Left +00.6,+01.0,...") whatever
  is asked. The fine-tune has replaced the base model's grounding output entirely (forgetting), so
  a box-first design must co-train boxes (labels from base Qwen) - it cannot be switched on by prompt.

## D193 (2026-10-05): box-first training - give the model back its grounding (user decision)
User: "don't do the test, do the training to return the box, and insert what we have learned".
Built (code, not yet run): scripts/label_boxes.py (base Qwen3-VL-4B boxes the object the
instruction refers to; simulator frames first: 0, every 8th, last; real flights: first, middle,
last; time-boxed); trainer --boxes (box line `{"bbox_2d": [x1, y1, x2, y2]}` in Qwen's native
format before the progress line; unlabelled frames train without it, ECoT-Lite style),
--box-repeat, mirror_box, --photo-aug (random BGR swap + 224 px); server --box (phase 0 writes
the box after a forced prefix, then the line, then the moves); scripts/aws/train_box_d193.sh
(g5: setup, DAgger rows rebuilt from the round-1/2 zips, labels, 30-update smoke, 4,000 updates
lr 1e-4 cosine from d190/dagger_r2_adapter, S3, machine off). Checklist of what every run keeps:
docs/PLAN.md "D193 checklist". Checks: box helpers + mirror unit-checked; an encoded example
reads `{"bbox_2d": [...]}\nLeft ...\n` + 32 moves (80 answer tokens); server --box runs end to
end locally.
- D193 labeller check (local, 4-bit, 24 real frames): 24/24 got a box; on a drawn sample of 8 the
  box sits on the instructed tree in 7, one boxed a bench. Noisy but usable auto-labels (ECoT also
  trains on auto-generated boxes); simulator frames (clearer objects) are labelled first.

## D194 (2026-10-05): research - what else to train in the same run; "dreaming in box space" added
docs/research/VLA_RETRAIN_WHAT_ELSE_20261005.md. In the run: box first (D193), photo augmentation
(D192), and --next-box: after the box the model writes where the target will be after its 8
moves ({"bbox_2d_next": [...]}, the labelled frame 8 steps ahead) - a box-sized visual subgoal
(CoT-VLA +17 % real / +6 % sim with subgoal images; OneWM-VLA future latents 47.9 -> 61.5 %).
Not in this run: a future-embedding head (cost x1.2-3), general VQA co-training (external data,
not a measured failure), history frames (2x image tokens). Server --box reads both lines.
Checks: an example encodes as box + next box + Left + 32 moves (107 answer tokens); mirror flips
both boxes; the server parses both lines.
D193 machine: g6.2xlarge i-02a16e9098bf8eec4 (L4), launched 09:2x UTC 2026-10-05 from the AWS
deep-learning AMI; training waits for d193/go.txt.
- D193 run (g6.2xlarge i-05f21d6200f5c1045, us-east-1c; the first g6 i-02a16e9098bf8eec4 is
  STOPPED, kept, after three fixed start-up bugs: CRLF script, store tar without a top folder, the
  AMI's broken transformer_engine breaking `import peft`; then no capacity to restart it).
  Setup 10:03, DAgger rows 394 + 934 rebuilt, box labels 10:13-13:13 UTC (3 h budget, 6.4/s):
  81,424 frames, 61,899 with a box (all simulator frames first). go.txt read: --next-box.
  Smoke (30 updates) passed - move-token accuracy kept: real 54.7 % / 75.7 % within 1 bin
  (s31000: 54.5 / 75.5), sim 91.5 / 93.9 (91.7 / 94.1). Training started 13:23 UTC: 4,000
  updates, about 9.2 s each, ends about 23:40 UTC; adapters to s3://.../d193/run/, machine off.
