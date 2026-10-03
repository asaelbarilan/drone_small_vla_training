# Cheaper / faster than GRPO for our drone VLA (2026-10-03)

**NO CODE CHANGED.** Research note (D184). PDFs: `papers/vla_post_training/`.

## Our situation (measured)

- Model: Qwen3-VL-4B LoRA, 256-bin action tokens, 8 moves per answer, progress line ("Left ...").
  Best setup 69% / nDTW 0.449 on 100 tasks (D182). Remaining failures: where to stop on Land and
  Pass (Turn cannot be practised: its tasks need spawned people/dogs the flight records lack).
- Cost driver: flying, not updating. About 10 min per pass of 8 tasks on the Windows box;
  one GRPO round of 24 tasks x 8 flights is about 3.5 h / $4 and gives about 40 updates.
- RL trial (D180): 4 of 8 tasks gave two identical flights at temperature 1.0, so half of the
  flying produced no learning signal. GRPO gives ONE number per whole flight.
- Budget: about $50 of credits left.

## The key fact we have not used: a free, exact teacher

Every practice task is a recorded training flight. For ANY state the drone visits, we can
compute exactly (a) the correct progress line - the displacement to the flight's recorded end
(D182 already computes it: `scripts/analyze_progress_lines.py`, `truth_left`) - and (b) a
correct next move toward the recorded path. So every call of every flight can get a dense,
correct label instead of one reward per flight. That is DAgger (Ross et al. 2011): fly the
student, let the expert relabel the states the student actually reached, train on them, repeat.
It fixes exactly the "student drifts to states the demonstrations never show" problem - which is
our arrival/overshoot failure (D182: the model never saw "slightly past the goal" states).

## Options ranked by expected gain per dollar

| # | Method | What it needs | Signal per flight | Cost for one useful round | Evidence | Verdict |
|---|---|---|---|---|---|---|
| 1 | **DAgger / on-policy distillation with the geometric teacher**: fly greedily on practice tasks (save each call's photo), relabel every call with the true progress line and the reference's next moves from that pose, SFT on them (mixed with normal data) | server saves photos in greedy mode (small change); a relabel script; the existing trainer | one exact label per call (about 7 per flight) | one pass of ~100 Land/Pass practice tasks (about $2) + SFT on the Windows GPU (about $1) | DAgger theory; VLA-OPD 2026: on-policy distillation "significantly improves sample efficiency over RL" and avoids forgetting; RLinf ships embodied DAgger | **Build first** |
| 2 | Filtered self-imitation (RFT / RIPT-style): sample N flights per task, keep the successful ones, SFT on them | the existing RL collector + trainer | only successes count | needs N passes like GRPO (about $4 per round) | RIPT-VLA: sparse success reward, dynamic sampling, works from 1 demo | Fallback if 1 fails |
| 3 | GRPO / PPO (built) | built and tested | one number per flight | about $4 per round, many rounds | WorldVLN +10 points after about 3,000 iterations; RL4VLA: PPO > GRPO > DPO | Only after 1, with what money is left |
| 4 | Trajectory DPO (built) | built | one pair per task | cheapest update | RL4VLA: weakest of the three | Not first |
| 5 | Value-guided test-time selection (V-GPS) | a trained value function; N samples per call | - | adds N x latency on the 8 GB GPU | V-GPS: +gains across 5 policies, no fine-tuning | Later |
| 6 | Offline RL with critics (CO-RFT, ConRFT, EXPO-FT) | new critic heads; built for continuous action heads | - | big engineering | sample-efficient in manipulation | Out of scope now |
| 7 | World-model RL | a video model | - | does not fit 8 GB | - | Out of scope (decision 09-28) |

## Why 1 should beat GRPO here

- **Dense and exact**: GRPO learns from "flight A scored 0.6, flight B 0.4"; DAgger learns "at
  this photo and pose the line should read +2.3 m, and the moves should be these".
- **No wasted flights**: identical sampled flights (half the trial) carry no GRPO signal; every
  greedy flight gives DAgger labels.
- **Targets the measured failure**: the overshoot and early-stop states are exactly the states
  the student visits and the demonstrations never contain.
- **Cheap update**: ordinary supervised training, same trainer, runs on the Windows L4.

Limits: the teacher is the recorded demonstration's end and path, so it cannot make the drone
better than the demonstrations (RL can); the Pass ambiguity (no stated distance) remains;
labels only exist on practice tasks (training flights), never on test tasks.

## What to build (about half a day, then about $3 per round)

1. Server: save each call's photo also in greedy mode (today only with --rollout-temperature).
2. `scripts/rl/dagger_relabel.py`: match calls to flights (rollout_samples.match), compute for
   every call the true progress line and the reference's next 8 moves from the logged pose
   (nearest point on the recorded path, then the recorded steps after it, in the drone frame).
3. Train: the existing trainer with --progress on (relabelled examples + an equal share of the
   original data, to avoid forgetting), a few hundred updates at a low learning rate.
4. Round = fly 100 Land/Pass practice tasks once (greedy, best setup) -> relabel -> train ->
   flight-test on the 94 Land + Pass benchmark tasks (about $3). Repeat while it improves.
