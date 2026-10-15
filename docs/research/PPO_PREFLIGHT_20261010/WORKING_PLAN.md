# PPO working plan

**Goal:** find out whether a small PPO update improves actual flight completion without damaging other behavior or spending a day on an invalid run.

**Status (D216): pre-training mechanics and live-data tests passed.** Full D208/BF16 update/reload passed; all 73 live/probe action chunks replayed exactly; real timeout bootstrap passed. Eight practice flights produced 3/8 geometry completions, including one 178-move Surround completion. Reward v2 fixes a heading-overshoot control; 72 local tests pass. AWS confirmed stopped at 16:34:21 UTC on 2026-10-10 after testing; no learning pilot started. See the [short result report](../../../reports/d216_pretraining/README.md) and [route plots](../../../reports/d216_pretraining/routes.png).

## 1. The approach we will test

Start from **D208, step8000**. Collect new stochastic practice flights, score what the drone actually did, and use PPO to favor better actions. Keep the existing model, action tokens and eight-action chunks for this first experiment.

The proposed design keeps D208's box/progress generation fixed and trains a separate action adapter, with vision initially frozen. This isolates action learning from changes to the context it receives. It needs a small technical check before commitment: both adapters must reproduce the baseline initially, and switching between them must fit memory and preserve correct probabilities. Evaluate this same combined policy; do not silently export it as a single adapter.

We will adapt our existing scripts. GRPO, a new framework, larger action steps and further DAgger work are deferred so the pilot answers one question at a time.

## 2. Do the work in this order

### Step 1 — Validate a geometry-first reward

Following the user's D214 decision, the first reward measures **the demonstrated route itself**: ordered progress, distance from the path and completion. The reference is continuous geometry; photo/waypoint spacing does not set action size. Images support diagnosis, not the reward calculation. Independent object geometry is not a prerequisite for this route-following pilot.

Implemented candidate checks pass on all100 practice references; all25 circle controls rank completion above partial travel above standing still. The full CPU suite has40 passing tests. See the [candidate report](../../../reports/d214_path_reward/REPORT.md). Reference quality, discounted incentives, tolerances and recovery behavior still need checking before rollout integration.

**Deliverable:** a tested reward with interpretable progress/deviation components. **Proceed only when** shortcut controls and executed-transition integration pass. This work needs no AWS training.

### Step 2 — Make the PPO script mathematically trustworthy

Repair exact input replay, including box/progress context; distinguish timeout from real termination; define the probability of the whole action chunk; account for every training sample; and save complete optimizer/checkpoint state.

Test these on tiny CPU examples with known answers. Test failed processes and interrupted saves too. Then use saved observations for a small GPU check: reproduce sampling probabilities, perform one update, verify frozen weights and measure memory.

**Deliverable:** a correctness report with actual pass/fail results. **Proceed only when** the trainer updates the same policy that produced the samples, fits the hardware and reloads correctly. A script that merely starts has not passed.

### Step 3 — Prove the simulator provides useful learning data

Run a small, fixed practice subset before any PPO flight training. Check resets, images, actual movement, completion scores and stopping. Include both easy tasks and the problematic kinds. Compare D208 with D209 where useful, under identical limits.

Measure stochastic outcomes, geometric progress/deviation reward variation and time per flight. Use the D214 geometry-first reward with its completion bonus. If nothing succeeds, diagnose the horizon, exploration or action limitations before collecting more.

**Deliverable:** evidence of useful reward variation plus a measured runtime/cost estimate. **Proceed only when** the simulator and scoring agree and a bounded pilot is affordable.

### Step 4 — Run one small learning pilot

One bounded round with at most **four updates**, with fresh practice flights, conservative updates and a fixed development check before/after each round. Set explicit episode, update, time and money ceilings before launch. Save results and shut down on completion or failure.

Stop on probability mismatch, invalid data, numerical failure, excessive policy change or clear behavior regression. Do not automatically restart or extend training.

**Deliverable:** a short before/after report. **Proceed only when** the run is technically valid and shows enough promise to justify broader evaluation.

### Step 5 — Decide from matched validation

Compare the candidate with D208 on the same validation tasks, using identical scoring and limits. Report individual wins/losses, results by task kind, completed paths, stopping and grounding retention. A small aggregate gain alone is insufficient.

**Deliverable:** keep, reject or investigate recommendation. The 273 benchmark test tasks remain untouched. More training is a separate decision.

## 3. What happens next

Latest D217r2 outcome: three real PPO updates saved; per-microbatch policy-change guard stopped preparation for update4. Final frozen/reload checks and candidate flights were skipped by failure handling, so no improvement result exists. Artifacts hash-verified; AWS stopped18:25:26UTC. Next is offline guard diagnosis and a separately authorized saved-candidate verification/evaluation, not additional training. See reports/d217_ppo_pilot/r2/README.md.

D217 failed before sampling/training because its deployment omitted the normalization manifest. Zero flights/updates. Logs recovered; EC2 confirmed stopped17:02:42UTC. Local repaired bundle passes extracted-archive asset verification, but no restart/retry occurred. See reports/d217_ppo_pilot/README.md.

**Next task: verify the fixed-seed development runner, then one bounded learning pilot.** The 72-chunk practice batch is prepared with v2 rewards, but admission flags remain false. Use four separate practice tasks for matched before/after checks. Proposed limits: 90 minutes training, 3 hours total instance session, no automatic extension. Keep AWS running between stages within that future session; stop after its final evidence upload. See [pilot proposal](../../../reports/d216_pretraining/pilot_proposal.json).

DAgger can stay paused. Its separate repair is: phase-aware path matching → sensible target spacing → validated corrections and stopping examples → audited replay. PPO avoids needing those action labels, but still depends on correct scoring and feasible actions.

## Detail lookup

| When needed | Supporting material |
|---|---|
| Scoring, action spacing and horizon | Long plan, sections 3–4 |
| Model, probabilities and optimizer | Long plan, sections 5–7 |
| Exact tests, stop conditions and budget | Long plan, sections 8–10 |
| Why this approach; alternatives and sources | [Synthesis](SYNTHESIS.md), [evidence](EVIDENCE.md) |
