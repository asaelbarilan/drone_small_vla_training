# What to do after SFT: making our drone VLA use its camera (2026-09-25)

**NO CODE CHANGED.** Research note only. PDFs not yet downloaded (links below).

## The measured problem (our numbers, not the literature's)

- Closed loop, same 100 UAV-Flow-Eval tasks, mean nDTW: ours real+sim **0.333**,
  OpenVLA-UAV **0.395**, ours real-only 0.128 (REPORT.md section 9).
- Ours is at or above theirs on classes that can be flown from the words alone
  (Shift 0.66, Surround 0.61, Ascend/Descend 0.72, Retreat 0.47 vs 0.29) and far
  below on classes that need the camera: Turn-to-target, Land-by-object,
  Move-to-object, Pass-through-object.
- Open loop the camera adds ~0.05 m (real 3.067 m vs gray 3.115 m, swap 3.121 m;
  text-only 4.533 m). Mirroring the photo changes 26% of answers (s2500).
- The test harness was audited from the run logs (2026-09-25): state chaining
  exact in 567/567 calls, the evaluator flies exactly our poses in 99/100 flights
  (1 empty reply), left/right sign conventions agree between real data and the
  simulator. So the gap is the model, not the harness. Side finding: 8/273 test
  files have a tilted `initial_pos`, which bends their reference path (2 in our
  100; both models score 0 on the worst one).

In the field's vocabulary this is **visual under-reliance / language (text-prior)
shortcut** (LIBERO-Plus, LIBERO-PRO), and on the execution side
**closed-loop reactivity vs action-chunk horizon** (Bidirectional Decoding).

## The three papers trained on our data (local PDFs, `.local/drone_vla_papers_20260919/`)

Read first; they are the only evidence measured on UAV-Flow itself.

- **UAV-Flow** (`uavflow.pdf`, 2505.15725) splits tasks into motion-level
  ("move 5 m at 45 degrees") and perception-grounded ("fly to the right side of
  the marker") and stresses the critical role of spatial grounding. Our result
  follows that split exactly: we match or beat OpenVLA-UAV on motion-level
  classes and lose on perception-grounded ones.
- **WorldVLN** (`worldvln.pdf`, 2605.15964), 79.1% SR. Two mechanisms:
  1. The policy is conditioned on past observations and past actions
     (a_t ~ pi(. | o<=t, a<t, l)) and predicts the *next latent world state*
     before decoding actions. Its largest gains over OpenVLA-UAV are on exactly
     our weak classes: Approach 45 -> 98%, Land 46 -> 93% (Table 1).
  2. Action-aware GRPO after SFT: segment reward = gamma^(j-1) * (trajectory
     reward 1/(1+||a - a*||) against the expert segment + task reward
     1/(1+terminal distance) + reference-policy reward), temporal decay
     weighting early segments. Over 10 points after SFT had saturated (Fig. 4).
     Simulator rollouts ran on ONE RTX 4090; training on 8x A800.
- **ImagineUAV** (`imagineuav.pdf`, 2606.01205), 70.9% SR with 1.3B parameters.
  It imagines instruction-conditioned future views and reads the motion off
  them. Ablation: removing instruction-conditioned imagination drops SR 70.9 ->
  40.2%. The future-view prediction carries the gain.

Common thread: both winners force the model to predict **what the camera will
see next**. That is a training-time reason to use the image, which our
action-only cross-entropy does not give.

## What the field found, including what did NOT work

| Finding | Source | Why it matters to us |
| --- | --- | --- |
| VLAs ignore an input channel; SR 95% -> <30% under moderate perturbation; models replay memorised trajectories instead of using visual feedback | [LIBERO-Plus 2510.13626](https://arxiv.org/abs/2510.13626) | Our gray/swap result is the drone version of this. Not unique to us. |
| **RL improves execution and semantics but NOT vision robustness** (on par with SFT); PPO beats GRPO and DPO | [RL4VLA 2505.19789](https://arxiv.org/abs/2505.19789) (NeurIPS 2025) | Negative result: RL alone is unlikely to fix camera use. It may fix stopping / overshoot. |
| Plain GRPO on a UAV VLA: 18.7 -> 26.4% SR; adding **one expert trajectory per group**: 43.1% (overall 26.1 -> 55.6) | [EG-GRPO 2606.02313](https://arxiv.org/abs/2606.02313) (OpenVLA-OFT, UAV-Flow data, 4-D actions, LLM-judge reward) | The closest paper to us. Our UAV-Flow-Sim reference paths are exactly the "expert" member of each group. Code not released. |
| GRPO from 1 demo/task: 17.3 -> 91.7% (LIBERO-Long) | [SimpleVLA-RL 2509.09674](https://arxiv.org/abs/2509.09674), [code](https://github.com/PRIME-RL/SimpleVLA-RL) | Evidence RL scales from little SFT data (efficiency angle). Manipulation, 8+ GPUs. |
| Drop groups where all rollouts get the same reward (dynamic sampling); RLOO+PPO, no critic | [RIPT-VLA 2505.17016](https://arxiv.org/abs/2505.17016), [code](https://github.com/Ariostgx/ript-vla) | Cheap trick that saves most rollouts when tasks are too easy/hard. |
| Rollout diversity matters more than rollout count | [ExToken 2607.12931](https://arxiv.org/abs/2607.12931) | With a slow simulator, fewer, more diverse rollouts. |
| Aerial RL post-training: revisiting failure cases, +3-8 pts SR on TravelUAV | [RecoverFly 2608.09467](https://arxiv.org/abs/2608.09467) | Failure-replay curriculum; teleport sim makes resets to failure states free. |
| Auto-built preference pairs by simulator rollback; 2B matches 7B | [AeroDPO 2608.07557](https://arxiv.org/abs/2608.07557) | Efficiency angle, small model. RL4VLA found DPO weakest, so lower priority. |
| **Training-free contrastive decoding**: contrast action distributions of the original vs object-masked image; OpenVLA +up to 50.6% in sim | [PCD 2505.13255](https://arxiv.org/abs/2505.13255) (ICLR 2026) | Directly targets under-reliance on vision, works on discrete action tokens, no training. Needs object masks; we would use gray / mirrored images instead. |
| Mirrored demonstration pairs + reflection-equivariant prior | [MirrorDuo 2606.20048](https://arxiv.org/abs/2606.20048) | Manipulation only, no language side-word swap. We already train with mirror pairs (`--mirror`). |
| Plan K, execute 1, replan | [DreamFly 2608.12308](https://arxiv.org/abs/2608.12308) (OpenFly); [BID 2408.17355](https://arxiv.org/abs/2408.17355) | We execute all 8 of 8 moves blind. No UAV paper ablates this. |
| World-model RL: cheap rollouts, but imagined rollouts hallucinate and the policy exploits them | [WMPO / World-Env / HaWMPO 2609.09941](https://arxiv.org/abs/2609.09941) | Out of scope: we already have a real simulator, and it is cheaper to use it. |
| Pilot chain-of-thought + async VLM/action model; 59% vs OpenVLA-UAV 10.5% on long-horizon | [FLIGHT 2606.06836](https://arxiv.org/abs/2606.06836) | Reasoning helps long multi-stage tasks; UAV-Flow tasks are single-stage. |

## Ranked options (effort vs expected effect)

| # | Option | Effort | Cost | Evidence | Expected effect on our weak classes |
| --- | --- | --- | --- | --- | --- |
| 1 | **Receding horizon**: execute 2-4 of the 8 predicted moves, then re-ask | server flag | ~$1 eval | BID, DreamFly | Helps only if the model uses new photos; also a diagnostic |
| 2 | **Visual contrastive decoding**: logits = (1+a)·real - a·gray (and a mirror variant) | server change, 2 forward passes | ~$1-2 eval | PCD (+50% OpenVLA) | Directly amplifies whatever the camera contributes; zero training |
| 3 | **Rejection-sampling fine-tuning** on UAV-Flow-Sim train tasks (sample N flights, keep the best by nDTW, SFT on them) | rollout script on the Windows box | few $ | RIPT, SimpleVLA-RL (as a weaker cousin) | Fixes execution (stop, overshoot); vision unclear (RL4VLA) |
| 4 | **Expert-in-group GRPO** (EG-GRPO) with the reference path as the expert, reward nDTW | trainer + rollout loop, L4 24 GB | $20-60 | EG-GRPO 18.7 -> 43.1 | Largest published UAV gain; vision unclear |
| 5 | **"Blind means unsure" loss**: extra term pushing p(action \| gray image) toward the action prior, so the answer cannot be stored in the text | trainer change, ~2k updates | ~$6 | none direct (new) | Forces camera use at training time; risk: hurts text-only classes |
| 6 | **Self-grounding**: the base Qwen3-VL labels each training frame with target direction (left/centre/right, near/far); the VLA predicts that token before its actions | label pass + trainer | ~$10 | ECoT-style grounding; our D157 probe: Qwen3-VL finds a target 24/24 | Gives the camera a supervised job; uses the shared-backbone angle |
| 7 | **Lite world-action loss** (WorldVLN / ImagineUAV, cheap form): add history (first + previous frame, E5) and an auxiliary head that predicts the vision-encoder embedding of the frame K moves ahead | trainer change, target embeddings precomputed once | ~$10 | WorldVLN Approach 45 -> 98%, Land 46 -> 93%; ImagineUAV 70.9 -> 40.2% without it | Strongest on-dataset evidence for our weak classes; no video generator needed |

Recommendation: 1 and 2 first (no training, ~$2 together, one Windows run),
because they also *measure* how much the camera can contribute. Then 7 (the
mechanism the two UAV-Flow winners share), with 5 or 6 as cheaper
alternatives. RL (3/4) last, in WorldVLN's reward form. Doing RL first would,
per RL4VLA, most likely improve Shift/Surround-type execution while leaving the
camera problem where it is; WorldVLN also applied its GRPO only after SFT had
saturated.

## Paper ideas (new, and compatible with the efficiency angle)

1. **"Does the drone look?"** A camera-reliance audit of aerial VLAs in closed
   loop: gray / swap / mirror / text-only controls per motion class, for
   OpenVLA-UAV and a 4B model. LIBERO-Plus did this for manipulation; nobody has
   for UAV-Flow. Our finding so far: the classes where the 4B model loses are
   exactly the camera-dependent ones.
2. **Mirror-contrastive decoding.** PCD needs an object segmenter. A drone
   instruction has a natural counterfactual with no segmenter: mirror the photo
   and swap left/right words; the mirrored answer mapped back should agree.
   Contrast (or average) the two distributions at inference. Training-free,
   discrete-token, works on an 8 GB GPU.
3. **Blind-uncertainty training** (option 5): make the model's confidence depend
   on the image by construction. Cheap, one extra forward pass per example.
4. **One backbone, two jobs.** The VLA is a LoRA on the testbed's own VLM
   (D157: switchable per request in llama.cpp). Self-grounding (option 6) lets
   the VLM supervise its own action adapter - no extra model, no human labels.
5. **World-action modelling without a video model.** WorldVLN (8B, 8x A800)
   and ImagineUAV (video diffusion) both win by predicting future views. Test
   whether predicting only the *embedding* of the future frame gives a 4B VLA
   the same gain on the camera-dependent classes, at SFT cost on one GPU.
6. **Cheap RL in a teleport simulator.** Teleporting makes resets free, so
   group rollouts from the same start and failure-state replay (RecoverFly)
   cost only inference. Expert-in-group GRPO with the dataset's reference path.

## Out of scope, deliberately

- World-model RL (WMPO, World-Env, VLA-RFT): we have the real simulator;
  learned simulators add hallucination that the policy exploits (HaWMPO).
- Flow-matching / diffusion action heads: they break llama.cpp weight sharing
  with the VLM (D157), which is the deployment requirement.
- The full 30k-flight official recipe on a p4d: user decision, last resort.
- RL or any tuning on the 273 UAV-Flow-Eval test tasks: training tasks must come
  from UAV-Flow-Sim minus the 168 excluded flights (`uav_flow_sim_excluded.json`).

## WorldVLN code read (2026-09-28, github.com/EmbodiedCity/WorldVLN.code, CC BY 4.0)

- Imagination is not portable: an 8B InfinityStar video model generates the next
  16-frame latent segment and a TimeSformer visual-odometry decoder reads the
  moves off it (train/TRAINING.md: infinity_qwen8b, 49 frames, lr 1e-5, 10 epochs,
  data dir uavflow_49f_*). No validation split in the released code; progress was
  tracked by closed-loop success. A "lite" embedding-prediction head on our 4B
  VLA would be our own untested idea, not their method.
- Their RL IS portable to a token VLA (Worldmodel/runtime/tools/GRPO/):
  Stage A collects K_CAND=8 simulator rollouts per task on TRAINING tasks
  (uavflow_tasks/select_from_train_jsons, not the test set); Stage B trains on the
  replay. Each group = 12 GRPO samples + 1 SFT anchor (expert) sample
  (build_hybrid_replay_meta_12grpo1sft.py), i.e. expert-in-group as in EG-GRPO.
  Reward (reward_uavflow.py): action MSE to the expert clip (group z-score ->
  exp), clip decay 0.9 over 3 clips of 16; terminal task reward
  0.85 * 1/(1 + (pos_err/2 m)^2 + (yaw_err/10 deg)^2) + 0.15 * success;
  success = final position within 3 m AND yaw within 10 deg; CE term 0.3.
  Paper settings: 3,000 iterations, lr 8e-7, KL 0.9, PPO clip 0.02.
- Their success rule (3 m, 10 deg) gives us an automatic success rate to report
  beside nDTW.
