# What the UAV-Flow papers (WorldVLN included) do not solve, and where we can win (2026-09-30)

**NO CODE CHANGED.** Research note (D174). PDFs: `papers/aerial_vla_open_problems/`.
Goal set by the user on 2026-09-30: surpass WorldVLN (79.1% success on UAV-Flow-Sim).
This supersedes the D159 note that beating WorldVLN was "unrealistic".

## 1. Where the score is lost: per class, weighted by task count

The success rate in the papers is averaged over tasks, not classes. The 273 test tasks are
unevenly spread: Land 54, Shift 49, Approach 42, Pass 40, A/D 19, Turn 15, Move 15,
Rotate 15, Retreat 12, Surround 12. WorldVLN's Table 1 per-class numbers reproduce its
79.1% exactly when weighted this way (216 of 273 tasks). To beat it we need 217.

| Class (tasks) | WorldVLN (paper) | OpenVLA-UAV (ours, 3 m + 10°) | Ours, real + sim |
|---|---|---|---|
| Land (54) | 92.6% | 5/10 | **1/10** |
| Shift (49) | 85.7% | 9/10 | 9/10 |
| Approach (42) | 97.6% | 9/10 | 7/10 |
| Pass (40) | **40.0%** | 4/10 | **0/10** |
| Ascend/Descend (19) | 94.7% | 10/10 | 9/10 |
| Turn (15) | **60.0%** | 10/10 | 4/10 |
| Move (15) | 100% | 10/10 | 7/10 |
| Rotate (15) | **46.7%** | 1/10 | 3/10 |
| Retreat (12) | 91.7% | 9/10 | 10/10 |
| Surround (12) | 58.3% | 0/10* | 0/10* |
| Task-weighted | **79.1%** | about 68% | about 47% |

\* Our automatic rule scores orbits by end point only; OpenVLA-UAV's orbit nDTW is 0.75.
OpenVLA-UAV's paper number is 65.6%, which matches our re-scoring within the 10-per-class
sample noise.

Two conclusions:
- **Land decides the race.** It is 20% of the benchmark; we pass 1 of 10. Bringing Land to
  WorldVLN's level alone is worth about +16 points task-weighted (47% → 63%). Pass (15% of
  the tasks) is next: 0 → 80% would add about +12.
- **Everyone fails the same four classes**: Pass, Rotate, Turn and Surround are WorldVLN's
  four worst, and Rotate and Pass are OpenVLA-UAV's two worst. These are the classes
  where the field has nothing to copy from. That makes them the place to win.

## 2. Measured: what the hard classes actually require

Computed from the 273 benchmark task files (`test_jsons`, `target_pos`, reference paths):

| Class | What ends the flight correctly | What the instruction says |
|---|---|---|
| Rotate | Pure yaw of 15–180° (median 90°), no object, no camera needed | The number: "Turn right by 150 degrees" |
| Turn | Final yaw = bearing to a person or dog, exactly (0.0° residual); median change 14°, success needs 10° | Nothing numeric: "Face toward the dog" |
| Pass | End 7.1–14.7 m (median 10.4 m) *past* the object; object starts 5.5–11 m away | No distance: "Advance past the car from the right side" |
| Land | End about 2 m from the object, lower | Object only |
| Move | End 2–5 m from the object | The distance: "a point 3.5 meters away from the dog" |

And what our model does on them (flight logs of the real + sim adapter):
- **Turn:** all 10 flights run exactly 34 steps, whatever the correct angle (7° to 30°);
  final errors 5–52°. It overshoots, and it also drifts forward.
- **Rotate:** flights end after 18, 42 or 74 steps. The same step count for 90° and 150°
  "right" (74 steps, errors 56° and 46°). The drone's yaw relative to the start is in the
  prompt in degrees, and the requested angle is in the instruction in degrees. The model
  has all the information and does not compare the two.
- **Pass:** stops at 5.8 m, before the object (median 7.6 m away).

So the shared failure is **extent and termination grounding**: knowing how far to go and
when to stop. That covers numeric targets (Rotate, Shift, Move), visual alignment (Turn),
and an object that leaves the camera view (Pass, Land at the end, Surround). It is the same
early-stop failure measured in REPORT section 12, and it is the one thing the papers don't
address.

## 3. Why the papers don't solve it

- **WorldVLN gets no pose.** Its inputs are the image, the past frames and the executed
  action history. Rotating by 150° means integrating its own past actions inside an
  8B video latent. It has no number to compare with the instruction. Its authors give
  no analysis of Pass or Rotate and list long-horizon and onboard execution as the open
  problems.
- **ImagineUAV** reports weakness on "yaw-dominant and orbit-style maneuvers" (Surround
  48.3%) and attributes it to viewpoint drift and the target leaving the view.
- **OpenVLA-UAV** gets the pose as text but predicts one step at a time with no history.
  It is reactive: once the object is behind it (Pass) or the angle is numeric (Rotate,
  1/10), nothing tells it to stop.
- **Field-wide:** VLAs inject the pose late and weakly. The 2026 proprioception study
  (2608.03052) and ThinkProprio (2602.06575) find that how the state enters matters.
  VLMs decode numbers poorly without explicit steps (counting/arithmetic papers).
  ProgressVLA (2603.27670) and VLA-SCT (2602.01811) exist because VLAs have "no
  progress awareness" and rely on heuristics to stop. DreamFly (2608.12308, OpenFly
  benchmark) adds a separate stop estimator (LiteStop) and causal memory.
- **Part of Pass is benchmark noise:** the instruction never says how far past the
  object to go (7–15 m in the references). No model can beat a ceiling there.
  Predicting the median gets within 3 m on most tasks, but not all.

## 4. What we can do that they cannot (ranked)

| # | Idea | Addresses | Evidence | Cost | Recommendation |
|---|---|---|---|---|---|
| 1 | **Progress tokens** (reasoning-lite): before its 8 moves, the model writes the *remaining* displacement to the flight's end (dx, dy, dz, dyaw, local frame, coarse bins or short numbers). Labels are free: the recorded end pose. | Rotate/Shift/Move/A-D become "target minus state" arithmetic done in visible tokens; Land/Pass/Approach become "where is the goal" | ECoT: +28% on OpenVLA with intermediate reasoning (2407.08693); CognitiveDrone 59.6→77.2% with a reasoner; CosFly-VLA uses a think block before 8-step UAV chunks; VLA-AN predicts target waypoints | Trainer + server change (about a day); one fine-tune from the D171 adapter (about 5k updates, about 8 h, about $10); +4–8 tokens per call | **Do first.** Novel on UAV-Flow, fits 8 GB, and hits exactly the classes where WorldVLN is weakest. Paper angle: "coordinate-space imagination instead of video imagination" |
| 2 | **Goal memory**: carry the first confident goal estimate forward through odometry and put it in the prompt next to the state | Pass/Land/Surround after the object leaves the view | CronusVLA and DreamFly (memory helps); VLA-AN aggregates frames | Server only, once #1 exists | Add with #1 as an inference-time option; test separately |
| 3 | **RL at WorldVLN scale** with a dense per-call reward | Everything, after SFT saturates (WorldVLN: +10 points) | WorldVLN code: 8 rollouts per task, 12 GRPO + 1 expert per group, action-MSE-to-expert dense reward plus terminal reward, 3,000 iterations, PPO clip 0.02, KL 0.9 | Loop automation; many Windows rounds ($5 per round of 384 flights) | After #1. See section 5 |
| 4 | **Sample-and-verify at test time** (best of N chunks, scored by a learned value or progress head) | Precision classes (Turn) | RoboMonkey +9% in-distribution, +25% out of distribution (2506.17811) | N× inference time per call | Later; latency budget is tight on 8 GB |

Deliberately not taken:
- **Video imagination** (WorldVLN's 8B InfinityStar, ImagineUAV's 1.3B Wan): does not
  fit next to the VLM on 8 GB (decision 09-28).
- **A separate stop network** (DreamFly LiteStop): progress tokens give the same signal
  inside the one model.
- **Hand-parsing numbers from the instruction** into the prompt: it would fix Rotate, but
  it is instruction-specific engineering, not a VLA result.

## 5. Is the RL pipeline complete? For one round, yes; for WorldVLN-scale RL, no

Built (REPORT section 16): practice tasks, reward (end + nDTW), sampled rollouts with
log-probabilities, GRPO with expert anchor, DPO pairs, RecoverFly task picker, dynamic
sampling (zero-spread groups get no advantage). It has **not run end to end yet**. The
gaps against WorldVLN's released code:

1. **Scale and iteration.** WorldVLN runs 3,000 iterations. Ours does one round per
   Windows boot, by hand (collect → g5 update → new adapter → collect). A multi-round
   driver script is missing, and each round costs a machine start on both boxes.
2. **Dense reward.** WorldVLN adds a per-clip action-MSE-to-expert reward (decay 0.9)
   to the terminal reward. Ours is trajectory-level only (every call in a flight shares
   one advantage). Progress tokens (#1) would give a natural per-call signal: did the
   remaining distance shrink?
3. **Conservative update.** WorldVLN uses PPO clip 0.02 and KL 0.9 at lr 8e-7. Ours:
   clip 0.2, lr 1e-5, no KL (the SFT anchor instead). Ours is 10× more aggressive; start
   at clip 0.05 or lower.
4. **No Surround tasks** (flight records carry no object placement), so RL cannot help
   that class.

## Sources

- WorldVLN 2605.15964 (https://arxiv.org/abs/2605.15964); code github.com/EmbodiedCity/WorldVLN.code
- ImagineUAV 2606.01205 (https://arxiv.org/abs/2606.01205)
- UAV-VLN survey 2604.13654 (https://arxiv.org/abs/2604.13654)
- ECoT 2407.08693 (https://arxiv.org/abs/2407.08693)
- CognitiveDrone 2503.01378 (https://arxiv.org/abs/2503.01378)
- VLA-AN 2512.15258 (https://arxiv.org/abs/2512.15258)
- CosFly-VLA 2607.15004 (https://arxiv.org/abs/2607.15004)
- DreamFly 2608.12308 (https://arxiv.org/abs/2608.12308)
- CronusVLA 2506.19816 (https://arxiv.org/abs/2506.19816)
- ProgressVLA 2603.27670 (https://arxiv.org/abs/2603.27670)
- VLA-SCT 2602.01811 (https://arxiv.org/abs/2602.01811)
- Proprioception in VLAs 2608.03052 (https://arxiv.org/abs/2608.03052)
- ThinkProprio 2602.06575 (https://arxiv.org/abs/2602.06575)
- RoboMonkey 2506.17811 (https://arxiv.org/abs/2506.17811)
- SimpleVLA-RL 2509.09674 (https://arxiv.org/abs/2509.09674)
- WorldFly 2606.06147 (https://arxiv.org/abs/2606.06147), own urban-canyon benchmark, no UAV-Flow number
