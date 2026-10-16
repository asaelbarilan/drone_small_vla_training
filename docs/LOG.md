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
- D193 restart with wandb (user request): the first training start was stopped at update 210
  (14:01 UTC; pipeline script killed first so the machine stayed on). User logged wandb in on the
  machine (Session Manager). scripts/aws/train_only_d193.sh restarted training 14:17 UTC, same
  flags + --wandb-project "vla training": run box_d193
  (https://wandb.ai/asael/vla%20training/runs/box_d193). Ends about 00:30 UTC, then S3 + machine off.

## D193 result (2026-10-06): training done; the model writes boxes but not grounded ones
Run complete (4,000 updates, adapter s3://.../d193/run/adapter_s4000, machine off). Move-token
accuracy kept: real 54.4 % / 75.5 % within 1 bin, sim 91.7 % / 94.1 % (start 53.5 / 90.2);
sim held-out loss 1.71 -> 0.49 (box lines learned as a format). Local test on the 12 Turn
validation start photos (as the evaluator sends them, and in normal colours - same result):
- own box: nearly the same box every time (around the image centre) - the average again, now in
  box form; first turn within 10 deg: 6/12 (old model 4/12).
- base Qwen (adapter off) writes the box, our adapter continues: its next box (dream) moves the
  person toward the centre correctly (left target -> next box right of it, and vice versa), but the
  Left line still says about -18 deg on almost every task: 5/12 within 10 deg.
- turn computed straight from the base box's centre column (atan of the offset, 90 deg FOV):
  right direction 11/11, within 10 deg 7/11 (one photo without a box).
Reading: grounding exists in the base model and the adapter can use a given box for the next box,
but the progress line / moves still follow the text prior. Candidate fix without training: for
"turn/face toward <object>" set the goal's yaw from the base box (server rule), then validate.

## D195 (2026-10-06): research - how to make the model ground the target itself (NO CODE CHANGED)
docs/research/VLA_FIX_GROUNDING_20261006.md. Our labels: 68 % of simulator boxes sit within
100/1000 of the image centre (the drone flies toward the target), so an average box is "almost
right"; the box loss went flat at update 500 (format learned, grounding never) and our validation
counted only move tokens. Field: ST4VLA (Qwen2.5-VL) shows exactly this - action-only training
destroys grounding, plain co-training only partly keeps it; staged grounding-first training then
spatially guided action training gives +18 pts. Ranked: (1) measure generated-box error and
line-vs-box consistency during training; (2) balanced box data + box-token loss weight, continued
training (~$10-15), stop early if grounding does not move; (3) staged retrain ST4VLA-style if not.
User ruled out a second model / base-Qwen box at inference.

## D196 (2026-10-06): plan - staged retrain (user decision), written to docs/PLAN.md
User chose the evidence-backed staged retrain over continuing D193. Stage 0 code + metrics ($0),
stage 1 grounding-only LoRA from the base model (~$5), stage 2 actions on top with the box kept and
a lower vision learning rate (~$40-50), stage 3 validation flights (~$3). Total ~$50-60.

## D197 (2026-10-06): research - more lessons before the staged retrain (NO CODE CHANGED)
docs/research/VLA_STAGED_RECIPE_LESSONS_20261006.md. ST4VLA appendix: best grounding:action loss
ratio in stage 2 is 1:10 (1:1 drops WidowX 71.7 -> 47.2; D193 had box lines on ~42 % of rows);
grounding pre-training pays off only from ~2M pairs (0.5M/1M: no gain) and robot-domain labels
beat general ones - our 81k labels are far too few for stage 1; both spatial prompting and
spatial pre-training add (61.1 -> 67.4 -> 73.2). VLANeXt: temporal history hurt (we leave it
out), state in the VLM helps (we do), bin-token actions are the weakest design (deployment
trade-off, user decision, not in this retrain). Fold into D196: many more stage-1 labels,
1:10 mixing in stage 2, no 3x box repeat, slower vision learning rate.

## D198 (2026-10-06): step 0 of the staged retrain - code + checks (local, $0)
- scripts/box_eval.py: lets the model WRITE its box on held-out rows and scores it: centre error,
  IoU, spread ratio, x correlation with the label, the error of a dumb "always the centre" box
  (baseline), empty-box accuracy, next-box error, Turn line-vs-box consistency; balanced row
  pickers (left / centre / right / empty).
- Trainer: --grounding-only (stage 1: answer = box + next box only, balanced), --grounding-mix
  (stage 2: box-only examples per flight example, ST4VLA 1:10), --sim-share (simulator share of
  flight rows), --vision-lr-mult (slower vision LoRA), --box-val N (written-box checks at every
  validation, logged to wandb as box/*).
- Gate passed - the check flags the D193 model on 60 held-out simulator photos (balanced):
  x correlation -0.02 (boxes do not follow the target at all), centre error 265 vs 251 for the
  always-centre box, IoU 0.19; spread ratio 0.98 (spread alone would have missed it).
- Local smoke tests (NF4, local 1-shard store, 24 local labels): stage 1 (3 updates, loss 2.11 ->
  1.60, 48 balanced box examples, vision LoRA 208 tensors at 0.1x); stage 2 from that adapter
  (5,010 box-only + 50,106 flight examples = 1:10, photo aug, loss 8.2 -> 5.5).

## D199 (2026-10-06 14:37 UTC): step 1 launched - boxes first (user-approved, ~$3)
scripts/aws/train_stage1_d199.sh on i-05f21d6200f5c1045 (no g6 capacity in us-east-1c, switched to
g5.2xlarge / A10G; wandb login and packages kept). Fresh LoRA on the base model, --grounding-only
--next-box --photo-aug --mirror, 81k D193 labels balanced 30/30/30/10, 1,000 updates lr 2e-4
cosine, written-box checks on 120 balanced held-out simulator rows every 250 updates (wandb box/*),
adapters every 250 to s3://.../d199/stage1/, machine off at the end (5 h hard stop).
Gate for step 2: box/box_x_corr clearly > 0 and box/box_centre_err < box/centre_err_const.

## D200 (2026-10-06): research - action head instead of action tokens (NO CODE CHANGED)
docs/research/VLA_ACTION_HEAD_20261006.md, papers/action_heads/. OpenVLA-OFT: bin tokens 76.5 % ->
parallel chunk decoding 90.2 % -> continuous L1 MLP head 95.3 % (26x faster); VLANeXt: bin tokens
are the weakest design; Qwen-VLA (Qwen3.5-4B + 1.15B DiT) and FLIGHT (UAV, Qwen2.5-VL-3B + DiT-B)
use separate heads. Recommendation: an OFT-style MLP head (~10 MB, L1, all 8 moves in one pass)
in stage 2; box and Left lines stay text (goal memory / deadband); stage 1 unchanged. 8 GB still
fits (one VLM + tiny head) but deployment needs hidden states from llama.cpp, not text only.

## D201 (2026-10-06): continuous action head built (code, local smoke only)
scripts/action_head.py: OpenVLA-OFT-style head - LayerNorm + 4-layer MLP (2560 -> 1024 -> 1024 ->
1024 -> 4, 4.7M parameters, ~19 MB) on the last hidden states of k=8 slot tokens appended after
the box / progress lines; tanh output in the tokenizer's normalised range; L1 loss. Deviation from
OFT: slots attend causally (Qwen3-VL attention / M-RoPE untouched). Slot token = the middle
action-bin id. Trainer --action-head (--head-weight, --init-action-head): total loss = text loss
(box + line tokens) + weight x L1; validation accuracy bins the head's moves exactly like tokens
(comparable with old runs); action_head.pt saved next to every adapter. Server: if the adapter
folder has action_head.pt, the moves come from one forward pass (no 32-token generation).
Local smoke (NF4, 3 updates from the stage-1 smoke adapter, 1:10 grounding mix, vision 0.1x):
loss 4.22 -> 2.44, validation ran, head saved; the server loaded it and returned 8 moves.

## D199 progress (2026-10-06): step-1 box checks (120 balanced held-out simulator rows)
| Update | centre error | always-centre | IoU | x_corr | next-box error | empty correct |
|---|---|---|---|---|---|---|
| 1 (base) | 117 | 254 | 0.29 | 0.848 | - | 0.64 |
| 250 | 61 | 265 | 0.61 | 0.864 | 118 | 0.90 |
| 500 | 87 | 265 | 0.60 | 0.713 | 142 | 0.90 |
| 750 | 59 | 265 | 0.65 | 0.844 | 126 | 0.90 |
| 1000 (final, d199/stage1/adapter_s1000) | 55 | 265 | 0.66 | 0.857 | 127 | 0.90 |
Gate passed (x_corr >> 0, error far below always-centre). The wandb sim_acc/* curves fall during
step 1 (0.25-0.45 -> 0.12-0.25): they score the 8 move tokens, which step 1 never trains (box-only
answers end after the box); expected, not a bug. sim_acc/Pass = 0: no Pass rows in that set.

## D202 (2026-10-06): step 2 pipeline - flying on top of the boxes (user-approved "ok")
scripts/aws/train_stage2_d202.sh on i-05f21d6200f5c1045 (g5.2xlarge) after step 1 stops it.
From d199/stage1/adapter_s1000; full answer box + next box (labelled photos, x3) + Left line +
8 moves from the D201 action head (L1, weight 1). 1 box-only example per 10 flight examples,
simulator = half of the flight rows (all 54 real shards in the store), DAgger rows rounds 1+2
(x20), mirror, photo-aug, both wordings, vision LoRA at 0.1x. 20-update smoke, then 8,000
updates lr 2e-4 cosine (3 % warmup); validation (move accuracy + box/*) and adapters every 1,000
to s3://.../d202/stage2/; wandb "vla training" / stage2_d202; machine off at the end (28 h hard
stop). About 20 h, about $25. Gate at every 1,000: move accuracy rising AND box/* not falling.
Started 17:32 UTC 2026-10-06 (setup OK: peft 0.21.0, transformers 5.17.0, wandb 0.30.0).
Try 1 failed at the smoke: the code zip held only scripts/ (missing reports/.../manifest.json);
machine stopped itself (~1 h). Try 2 (zip = scripts + reports + release) 18:30 UTC: smoke passed
(loss 2.39 -> 0.47 in 20 updates; box checks kept: centre error 61, x_corr 0.87). Training from
19:07 UTC, 7.5 s/update (~17 h). Data: 1.32M sim + 1.32M real flight rows, 264k box-only rows,
DAgger 1,328 x 20; head 4.73M parameters. Update 1 box checks: error 55, IoU 0.66, x_corr 0.86.
Box checks during step 2 (always-centre baseline 265):
| Update | centre error | IoU | x_corr | next-box error | Turn line-vs-box |
|---|---|---|---|---|---|
| 1 | 55 | 0.66 | 0.86 | 125 | - |
| 1000 | 74 | 0.60 | 0.80 | 138 | 0.75 |
| 2000 | 75 | 0.62 | 0.82 | 138 | 0.80 |
| 3000 | 69 | 0.64 | 0.83 | 128 | 0.80 |
| 4000 | 67 | 0.61 | 0.80 | 115 | 0.75 |
| 5000 | 73 | 0.61 | 0.77 | 113 | 0.75 |
Grounding dropped a little in the first 1,000 updates, then held (no collapse like D193). Move
accuracy (wandb, 5k): sim exact ~0.47 / within 1 bin ~0.86, still rising. NOT comparable with the
old token runs' 91.7 / 94.1: those were teacher-forced over 32 tokens (each move token sees the
true earlier ones; an untrained-on-moves model scored 25-45 % by copying in D199), while the head
predicts all 8 moves at once with nothing given. The fair comparison is validation flights.

## D203 (planned, user-approved "ok" 2026-10-07): step 3 - validation flights of the step-2 model
After D202 ends (~12:30 UTC): Windows box i-03a3c8632314bf5fc, scripts/win_eval/eval_variants.ps1
-Tag d203 -CodeZipKey d202/vla_code.zip -TaskZipKey d190/check_tasks.zip (the 100-task VALIDATION
check set; rule 1) -Variants "stage2|d202/stage2/adapter_s8000|--progress --goal-memory
--memory-deadband 1.0,5 --box" (action head loaded from action_head.pt in the adapter folder).
Compared per kind with D190 (round-2 token model, same 100 tasks). Then decide on more step-2
training from adapter_s8000 (user: "keep training while it goes up").
D202 optimizer state: the trainer's checkpoint/ (adapter + action_head.pt + state.pt with Adam
moments, 977 MB) lived only on the wiped NVMe. 09:43 UTC: /home/ubuntu/ckpt_watch.sh on the
machine syncs it to s3://.../d202/checkpoint/ whenever state.pt changes (6000 done; 7000, 8000
follow). Trainer --resume now also reloads the head from checkpoint/action_head.pt, so more
step-2 training can continue exactly (copy d202/checkpoint to OUT/checkpoint, --resume,
--updates N).

## D202 result (2026-10-07 14:05 UTC): step 2 done, 8,000 updates, machine stopped itself
| Update | real exact / 1 bin | sim exact / 1 bin | box error | x_corr | next-box error | Turn line-vs-box |
|---|---|---|---|---|---|---|
| 6000 | 39.3 / 59.3 | 72.3 / 89.6 | 56 | 0.89 | 111 | 0.83 |
| 7000 | 41.9 / 59.4 | 73.9 / 90.2 | 59 | 0.80 | 115 | 0.83 |
| 8000 | 41.6 / 59.5 | 76.2 / 90.4 | 61 | 0.79 | 108 | 0.80 |
Sim per kind at 8000 (exact): Approach/Move 70, Ascend/Descend 68, Land 59, Pass 46, Retreat 77,
Shift 84, Surround 86, Turn/Rotate 80, other 57. Move accuracy = the head's continuous moves
rounded to the 256 bins, nothing given (not teacher-forced, unlike the old 32-token numbers).
Adapter + head: s3://.../d202/stage2/adapter_s8000/. Optimizer checkpoint on S3 is update 7000
(the 8000 copy was cut by the shutdown); a continuation can resume exactly from 7000 or start
from adapter_s8000 with a fresh optimizer.
D203 flights blocked 14:10 UTC: no g6 (L4) capacity in us-east-1d at xlarge / 2xlarge / 4xlarge;
g5.xlarge started but the Windows NVIDIA driver does not see the A10G ("No CUDA GPUs"), so it was
stopped and set back to g6.xlarge. Retrying the L4 start.
D203 flights started 16:11 UTC 2026-10-07 (L4 capacity back after ~1.5 h of retries; machine on
g6.xlarge as before). Server log: "action head loaded" - moves come from the head. Pace ~3 min
per flight (~13 s per call: box line + progress line generated, then one head pass), so ~5 h
for 100; the 6 h hard-stop timer was reset at 17:04 UTC to 6 h from then (results upload only
at the end of the variant). Machine turns itself off when done.

## D203 result (2026-10-07 20:52 UTC): step-2 model FLIES BADLY - 12 % vs 69 % (D190)
100-task validation check set, same server flags as D190 plus --box, moves from the action head.
| Kind | D203 head (step 2) | D190 tokens (round 2) |
|---|---|---|
| Approach/Move | 1/10 | 5/10 |
| Ascend/Descend | 0/10 | 9/10 |
| Land | 0/15 | 15/15 |
| Pass | 0/15 | 9/15 |
| Retreat | 0/3 | 3/3 |
| Shift | 0/15 | 13/15 |
| Surround | 0/10 | 2/10 |
| Turn | 7/12 | 4/12 |
| Turn/Rotate | 4/10 | 9/10 |
| all | 12 % (nDTW 0.05) | 69 % (nDTW 0.38) |
Raw evidence (server_calls.jsonl, D:/drone_vla_pilot/runs/d203/stage2): EVERY flight runs the full
104 steps - the drone never stops. (1) The head yaws ~0.4 deg every step on straight tasks, from
the first call with a perfect line, so goal memory's yaw error grows past the 5-degree deadband
(-13 deg after 4 calls on "descend 5 m and land") and the line never becomes 0 = never "arrived".
(2) The head does not follow the line: on that Land task dz stays at the q01 bound (-13.3 cm per
step, tanh saturated) even when the line says go UP 2-7 m; the drone ends 12 m down. (3) Even
with line 0 it keeps moving 4-6 cm per step, above the evaluator's 3 cm "still" rule. The old
token model on the same task: line 0 at -4.2 m, stop on the next call.
Offline validation said sim 76 % exact / 90 % within one bin - so either the head is off on the
simulator's own photos (domain gap) or something differs between trainer validation and server.
Next: find which, locally (no AWS): run the trainer's head on held-out rows as the evaluator sends
them, check yaw bias and line-following.
D203 diagnosis (local, free, 2026-10-08): scripts/diag/diag_head.py runs the step-2 head on 60
held-out real VAL rows exactly as in training (trainer encode, ground-truth line, dataset photo;
NF4 on the local GPU). The same faults appear there, so it is the trained head, NOT the server:
| Rows | mean error dx / dy / dz cm, yaw deg | predicted step / yaw | true step / yaw |
|---|---|---|---|
| still (true chunk all 0) | -4.0 / 0.0 / -0.3, +0.61 | 4.2 cm, +0.62 deg | 0, 0 |
| straight (no turn) | +0.9 / +0.4 / -0.6, +0.48 | 27 cm, +0.50 deg | 22 cm, 0 |
| turning | -3.4 / +0.6 / -0.4, -0.64 | 4.3 cm, -4.98 deg | 2.7 cm, -4.92 deg |
Still rows pass the evaluator's still rule 0/20. The head is ~4-5 bins off near zero (a
normalised error of 0.03-0.05; one bin is 0.008): it cannot say "exactly still" or "exactly no
turn", which the bin tokens did for free (the zero bin). The "90 % within one bin" offline number
hid this (dy / dz are near-trivial channels). OFT trains its head 50-150k steps; ours had 8k.
D203 trend over training (same 60 VAL rows, scripts/diag/diag_head.py, NF4 local):
| Update | still rows passing the still rule | still: step / yaw | straight: yaw bias | turning: pred vs true yaw |
|---|---|---|---|---|
| 2000 | 13/20 | 1.1 cm / +0.05 deg | -0.01 deg | -2.5 vs -4.9 deg |
| 4000 | 0/20 | 3.6 cm / +0.53 deg | +0.27 deg | -4.4 vs -4.9 deg |
| 6000 | 0/20 | 3.8 cm / +0.58 deg | +0.48 deg | -4.7 vs -4.9 deg |
| 8000 | 0/20 | 4.2 cm / +0.62 deg | +0.50 deg | -5.0 vs -4.9 deg |
NOT lack of training: the head could nearly hold still at 2,000 updates and got WORSE with more
training (a growing backward / right-turn bias near zero) while turns got more accurate. More
updates of the same recipe would not fix it. Cause of the drift not yet known (n = 20 rows per
group, NF4 - direction is clear, exact numbers rough).

## D204 (2026-10-08, user-approved "ok"): step 2 again with move TOKENS instead of the head
scripts/aws/train_stage2_d204.sh = D202 exactly, minus --action-head: from d199/stage1/adapter_s1000,
box + next box + Left line + 32 bin tokens; 1:10 box-only mix, sim share 0.5, vision LoRA 0.1x,
DAgger rows x20, mirror, photo-aug; 20-update smoke, 8,000 updates lr 2e-4 cosine; validation +
adapters every 1,000; the optimizer checkpoint is now synced to S3 too. ~20 h, ~$20, machine off at
the end. Then the same 100-task validation flights (D203 setup, no head).
D204 restarted 21:27 UTC (still in setup) with adapters every 500 updates. Early-warning plan
(user: "see there is an improvement on few updates"): scripts/diag/diag_moves.py (free, local
GPU, no teacher forcing - the way the server answers) on adapter_s500, s1000, s1500, s2000 ...;
judge the TREND toward the calibration numbers of the old token model that flew well
(d188 dagger_r1_adapter), not one pass/fail number. Stop early only if the trend is flat or wrong.
diag_moves.py calibration (60 VAL rows, NF4): old token model that flew well (d188 dagger_r1):
still rule 20/20, still step 0.28 cm, straight yaw bias -0.02 deg (truth -0.03), turning -4.0 vs
true -4.6 deg. D202 head s2000: still 13/20, step 1.1 cm (matches diag_head.py - the check works).
Target for D204's trend: still -> 20/20, straight yaw bias -> ~0, turning yaw -> ~-4.6.
D204 early check, update 500 (diag_moves.py): still rule 20/20 (step 0.28 cm - same as the old
flyer), straight yaw bias -0.16 deg (old flyer -0.02), turning -0.4 vs true -4.6 deg (turns not
learned yet). Tokens hold still from the start, unlike the head. Next points: 1000, 1500, 2000.
| D204 update | still rule | straight yaw bias | turning yaw (true -4.6) | teacher-forced real / sim exact | box error / x_corr |
|---|---|---|---|---|---|
| 500 | 20/20 | -0.16 deg | -0.4 deg | - | - |
| 1000 | 20/20 | -0.09 deg | -3.2 deg | 52.0 / 89.0 % (1 bin 69.7 / 91.2) | 65 / 0.83 |
| old flyer d188 | 20/20 | -0.02 deg | -4.0 deg | (s31000: 54.5 / 91.7) | - |
Trend right on every point: still kept, straight drift shrinking, turns being learned, move
accuracy already near the old model after 1,000 updates, boxes holding.
| 1500 | 20/20 | -0.14 deg | -4.0 deg | - | - |
(update 1500: turning now equals the old flyer (-4.0); straight drift -0.09 -> -0.14 is within
the noise of 20 rows. Trend still good - training continues to 8,000.)
| 2000 | 20/20 | +0.03 deg | -4.1 deg | 52.7 / 89.8 % (1 bin 72.1 / 92.1) | 66 / 0.83 (Turn line-vs-box 5/5) |
Update 2000 matches the old flyer on every check (still 20/20, straight drift ~0, turning -4.1 vs
-4.0); move accuracy at the old s31000 level after 2,000 updates; boxes holding. Sim per kind
(exact): Approach/Move 81, Ascend/Descend 99, Land 75, Pass 61, Retreat 86, Shift 99, Surround 97,
Turn/Rotate 99. Gate passed - training continues to 8,000, then the 100 validation flights.

## D205 (2026-10-08, user-approved): validation flights of D204 update 2000, training continues
Windows box (g6.xlarge), eval_variants.ps1 -Tag d205, code d204/vla_code.zip, the same 100-task
validation check set (d190/check_tasks.zip), server --progress --goal-memory --memory-deadband
1.0,5 --box, adapter d204/stage2/adapter_s2000 (move tokens, D191 constrained greedy). Started
~09:48. Chosen over update 4000: passed every diag_moves check, boxes better (x_corr 0.83 vs ~0.81
later). If it beats D190's 69 %, stop D204 early (saves ~$10, keeps the boxes).
D205 peek at 10:29 UTC (23 of 100 flown, ~1.8 min per flight - flights end normally again):
16/23 vs D190 15/23 on the same tasks; Turn 4/5 vs 2/5, others equal (Turn/Rotate 0/1 vs 1/1).
D204 stopped early at update ~5,100 (11:13 UTC, user-approved): move accuracy 2000 -> 5000 only
+1.5 points (sim 89.8 -> 91.3, real 52.7 -> 54.1) while box x_corr 0.83 -> 0.81. Saved ~$9.
Everything on S3: adapter_s500 ... adapter_s5000, and checkpoint/ (update 5000, with the
optimizer state) - training can resume exactly from 5000 with --resume. Machine stopped itself.
| D204 update | 4000 | 5000 |
|---|---|---|
| real / sim exact | 53.4 / 90.7 % | 54.1 / 91.3 % |
| box error / x_corr / next-box error | 69 / 0.80 / 131 | 62 / 0.81 / 106 |
Next: D205 flights of s2000 finish, then fly s5000 on the same 100 validation tasks.
D206 queued on the Windows box (11:20 UTC): C:\runs\d206_queue.ps1 waits for D205's ALL_DONE, aborts
D205's end-of-run shutdown for 90 s, then flies d204/stage2/adapter_s5000 on the same 100 tasks
(eval_variants -Tag d206, own 7 h stop timer; machine off at the end). Avoids a stop/start (L4
capacity was short on 2026-10-07).
D205 peek at 12:22 UTC (89 of 100): 62 vs D190 64 on the same tasks. Turn 10/12 vs 4/12, Surround
5/10 vs 2/10, A/D 10/10 vs 9/10; Land 9/14 vs 14/14, Turn/Rotate 3/10 vs 9/10, Pass 6/9 vs 7/9.
Written to RESEARCH.md 51e (staged retrain results) and 44b (head vs tokens).

### D206 interim verification (85/100)
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
D240 capacity monitor d240-resume-and-comparison created in this chat, every5minutes. Capacity-only start attempts limited through2026-10-13 15:52UTC; after a successful start use immutable2hour deadline. Instance independently stopped15:21:46UTC; no dispatch or new update yet. Local launch_status.json records pending state. Final comparison is still outstanding.

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


### D256 - Resumable reports backup verified; cloud disk removal ready (2026-10-16)
User authorized uploading reports in resumable parts and checking readiness to remove both machines/disks, not deletion itself. The3893851668-byte reports archive was split into30 independently resumable128MiB parts (last1537556bytes). All30uploads succeeded; independent S3 HeadObject sizes/fullSHA256 match every part. Ordered parts reconstruct archiveSHA25631669287a6f2d9990600f88af059e49c7702015a6bca8ebadba3d2e63ba33ab5, rehashed locally before transfer. Restore manifest atd256/reports_parts/manifest.json; scriptd256/restore_report_parts.py validates every part and finalarchive. Concatenation and corrupt-part rejection tests passed. Former single-object upload is absent and superseded, not a remaining blocker.
Downloaded original D207 resume state645101597bytes, SHA256fd0cf93d2f5b77935cb5e411dc5d0e72a0c7b1b80905ad6d631e5d52c4c84082: step8000,712optimizer states/moments in2groups. Checkpoint has no savedRNG state: future training can resume weights/optimizer, not promise bit-identical random continuation. Original codeZIP CRC passes107files, normalization manifests present. Rechecked required75GB trainingstore, boxes, extra rollouts, splits/report/config, fullVLA base/adapter/assets and D240checkpoint. Full model checksums unchanged. No bucket lifecycle expiry configuration. Dataset archive was not fully re-downloaded and no freshGPU training/restore was run.
Required inference/training-resume/research assets are now preserved outside the cloud disks. It is safe to remove Windows i-03a3c8632314bf5fc withvol-02e8231bb8920b3a9 (250GiB) and Linux i-05f21d6200f5c1045 withvol-01459786d91f4e270 (150GiB) for this scope; reinstall OS/dependencies/simulator later rather than expect a full-machine image. Both independently stopped. No instance start/deletion/termination or HF publication. Final metadata/code supplement d256/final_closeout.zip; verification_final.json and S3_RESTORE.md document restore.
