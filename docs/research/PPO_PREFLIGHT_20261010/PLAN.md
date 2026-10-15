# D212 — PPO design and preflight plan

**Start with [WORKING_PLAN.md](WORKING_PLAN.md).** It defines the working sequence. This longer document is the supporting technical reference; conditional alternatives are not simultaneous implementation requirements.

**Status: design only; NOT ready to launch. 2026-10-10.** No instance, rollout, training job, external package installation or test-benchmark run is authorized by this planning task. The purpose is to eliminate identifiable invalid experiments before spending on a bounded pilot. No plan can guarantee that PPO improves the model.

Read [SYNTHESIS.md](SYNTHESIS.md) for the evidence and alternatives decision; [EVIDENCE.md](EVIDENCE.md) and [RESOURCES.md](RESOURCES.md) for source-level claims and resource limitations. E-numbers below refer to that ledger. Numerical guardrails below are proposed engineering defaults, not constants established by the literature. Freeze them in a manifest after calibration, before the paid pilot.

## 1. The proposed experiment in plain language

Start from D208's step8000 adapter. Let it attempt practice tasks with controlled randomness. Measure what the drone actually did using independently tested task-completion rules. Increase the probability of sampled actions that were better than the value model predicted, while keeping each update small and checking that other behaviors are retained.

This avoids asking our geometric DAgger script for the correct action at every image. It does NOT avoid deciding what a successful flight means, checking units and timing, recording the sampled action probabilities, or validating the simulator. PPO can efficiently optimize a bad reward, which would make the result worse while the training curve looks better.

The first pilot isolates action learning: use a frozen D208 model adapter for box/progress prefixes and a separate trainable action adapter initialized from the same D208 checkpoint. Freeze vision parameters initially. This retains the deployed staged generation interface while preventing the greedy prefix policy from drifting outside the action PPO objective. It is a composite policy and must be evaluated and deployed as such. A later single-adapter consolidation would be a separate experiment, not an automatic export step.

## 2. What is already known, and what remains unknown

Historical endpoint/nDTW scores are D206 81/100 and0.3792, D208 82/100 and0.4255, D209 83/100 and0.4350. D209 has five gains and four losses against D208; these counts alone do not establish a reliable improvement. The three apparent Surround losses were stationary D208 endpoint passes, not completed circles. Both adapters remain useful controls.

The 60-frame box check worsened after D209: centre error65.8494 to70.9322, IoU0.5932 to0.5857, correlation0.837 to0.810. Those are comparisons to fixed pseudo-labels, not certified human ground truth. Shared-parameter interference is plausible, not established as the cause. The D209 supervised training memory/time measurement is NOT a PPO resource estimate.

| Local finding | Consequence | Required response |
|---|---|---|
| Trainer reconstructs progress but not sampled box prefix | PPO may optimize probabilities under a different observation/context | Save and replay exact multimodal input and prefix tokens |
| Reward favors stationary circles over a correct partial orbit in saved controls | RL can learn premature stopping | Versioned semantic reward with negative controls |
| Reference nDTW uses only first20 stride5 samples | Long-circle score is not full-path fidelity | Separate full-path diagnostic, preserve historical metric |
| GAE treats final collected call as terminal | Timeouts incorrectly erase future value | Terminal/truncation/final-observation contract |
| Existing PPO surrogate clips individual token ratios | Not the same objective as joint chunk PPO | Choose and name the action probability/objective explicitly |
| Remainder minibatches can be dropped; tiny input can yield zero updates | Report may imply training without intended updates | Exhaustive sample accounting and zero-update failure |
| Actor-only gradient clipping; no full atomic optimizer resume | Critic instability and misleading recovery | Clip correct parameter groups; complete checkpoint state |
| Teleported simulator poses, no verified dynamics/collision interface | Geometric success may hide infeasible moves | State fidelity limits and check swept paths where supported |

Evidence: scripts/rl/train_grpo_vla.py, rollout_samples.py, reward_uav_flow.py; scripts/uav_flow_eval_server.py; reports/d210_dagger_audit/ and reports/d209_completion_20261010/. These are local code/artifact findings, not conclusions inferred from papers.

Unresolved: exact reliable target geometry per kind, collision observability, stop semantics, stochastic exploration quality, likelihood precision across server/trainer, actual PPO GPU fit, simulator reset throughput and cost. These become gates below, not assumptions hidden in a launch script.

## 3. Separate the four frequencies

The recording rate, reference waypoint spacing, commanded movement distance and observation/replanning rate are separate choices. A photograph is an observation, not a mandatory control timestep. A reference point describes geometry, not necessarily the next command target.

Example: a20m loop represented by200 points spaced0.1m apart needs about200 movement commands if each command follows one increment. At eight actions per model call, that is roughly25 calls. Commands averaging0.4m would nominally need50 moves, but only if the action representation, curvature, visibility and clearance support that spacing. Multiplying all outputs by four is not a validated fix: it changes turns, clipping and recovery. Stopping and vertical/yaw-only tasks also differ.

Write an action contract with: coordinate frame; metres versus centimetres at each boundary; yaw units/sign/wrapping; four action dimensions;256-bin decoding and saved bounds; eight actions/32 tokens per call; quantization; actual execution order; stop detection; reset behavior; and exact maximum-step enforcement. Round-trip tests must cover zero, minimum/maximum bins, rotations at179/-179 degrees, vertical motion and mixed translation/yaw. Compare requested and measured poses.

Keep the existing action representation and chunk length for the first pilot. Change horizon through an explicitly versioned diagnostic protocol, not an accidental chunk overshoot. A100-step budget must execute at most100 moves, not104. A500-step diagnostic is a proposed ceiling for examining long practice trajectories, not a claim that500 is optimal or that the learned policy completes them. Choose the training horizon from practice-only feasibility and cost measurements; freeze it before heldout comparison. If the real application requires100 moves, improvement at500 must not be marketed as meeting that requirement.

## 4. Reward and success specification before optimization

**D214 update:** the user selected reference-route geometry/progress as the first reward. This supersedes the earlier binary semantic-success-first proposal below for the initial pilot. See WORKING_PLAN.md and reports/d214_path_reward/REPORT.md. The detailed object predicates below remain possible later semantic evaluation work, not prerequisites to implementing the agreed geometric candidate.

### 4.1 Independent task predicates

Implement reward v2 alongside the historical scorer. Never overwrite old reported scores. A task specification must identify the intended object/region, permitted tolerances, progress state, completion event and failure conditions. Ambiguous tasks are quarantined with reasons; do not infer convenient semantics from whichever trajectory gets a high score.

| Kind | Completion evidence to specify | Negative control |
|---|---|---|
| Surround | Correct target; ordered signed angular coverage around its centre, sufficient sweep, radial/altitude corridor, completion and appropriate stop | Stationary at start, yaw-only spin, half-circle, reverse orbit, wrong object, repeated small arc |
| Pass | Cross the intended gate/region from the correct side to the other, with clearance | Stop near endpoint without crossing; pass wrong opening |
| Approach/Move | Reach intended target region with required orientation/visibility where specified | Move toward unrelated nearby object |
| Retreat | Increase separation in intended direction to specified extent while satisfying relevant facing condition | Turn away without translating |
| Shift | Required lateral displacement relative to the specified frame/target | Forward translation that happens to approach final coordinates |
| Ascend/Descend | Correct signed altitude change and final tolerance | Horizontal proximity without altitude completion |
| Land | Intended landing region, altitude and any supported contact/settling requirement | Hover above region; teleport through surface |
| Turn and Turn/Rotate | Signed requested rotation with unwrapped progress where needed | Wrong-direction wrap shortcut or translation-only endpoint match |

These are specification templates, not a claim that every current task contains adequate metadata. For Surround, winding must come from drone position around the intended centre; camera yaw alone is insufficient. A jump across a large arc must not count as traversing it. Use ordered sectors and a movement/clearance check; tune tolerances from training annotations and simulator precision. If reliable centre/gate/contact information is unavailable, explicitly limit the initial task scope or annotate a small practice development set. Do not substitute a guessed success predicate and call it validated.

An independent reviewer should inspect representative trajectory-and-camera examples for every included kind, all ambiguous cases and counterexamples. I can implement/check the geometry and produce the review pack; that is not independent certification of my own interpretation. Human agreement on the concrete examples is a useful final semantic check. Disagreements become excluded tasks or revised specifications before training, not post-hoc score adjustments.

### 4.2 Start with the smallest defensible reward

Use independently validated binary completion as the primary objective. Give terminal success once; do not pay repeatedly for remaining inside a region. Distinguish task failure from simulator infrastructure failure. A disconnected renderer is not a negative example of drone control.

Before adding dense shaping, measure whether stochastic practice attempts produce both success and failure. If signal exists, first test outcome reward alone. If almost every attempt fails, do not run hours of PPO on zero-information batches. Inspect whether exploration, horizon or representation prevents success. A tested progress term can then be a separately versioned intervention.

For a dense-progress candidate, use bounded task-specific progress computed from actual executed trajectory history, not distance to the final point for every kind. Surround progress must encode ordered orbit coverage; moving away from the start can be correct. Test oscillation, reversing, waiting, spinning and repeated crossings. Progress differences must allow negative changes; clipping all negative increments to zero creates an oscillation exploit. Any time/length penalty must be small enough that stopping immediately is not preferable to feasible completion.

Do not claim reward shaping preserves the original optimum without checking the required assumptions. If using a discounted potential difference, define the augmented state/history, terminal potential and truncation treatment explicitly; terminal cancellation can change intuitive partial-trajectory rankings. Validate the actual summed discounted return, not just individual rewards. An episodic coverage bonus changes the objective and must be named as such. Select weights using a frozen table of hand-constructed controls and training-only examples, never to make D209 look better on validation.

Required ordering tests include completed valid orbit above stationary, correct orbit above wrong-object orbit, no benefit from repeating the same arc, and no completion credit from yaw-only spinning. Partial-versus-failure ranking is an explicit design decision. Save each component and total for every fixture. The saved D210 stationary/reference examples are mandatory regression fixtures.

### 4.3 Full-path and historical reporting

Compute a separate full-path diagnostic with both trajectories represented under a declared spatial/angular sampling rule. Do not truncate the reference to its first95 indices. Handle zero-distance yaw-only sections, duplicates, different trajectory lengths and angle wrap. Report semantic completion, coverage, wrong direction, actual moves/calls, path length, endpoint error and full-path similarity separately. A path-similarity score is not a substitute for completion. Keep historical100-step endpoint/nDTW columns with their limitations for continuity.

## 5. Policy architecture and exact sampling

### 5.1 Frozen prefix, trainable action branch

Pin base-model revision, processor, action statistics and D208 adapter hash. Both prefix and action adapters initially equal D208. Generate box/progress/goal-memory context with the frozen adapter using the original rules; condition the trainable action adapter on those exact tokens. The critic observes the same context available before action sampling.

With one GPU, adapter switching on one base model may avoid two full model copies. However, prefix and action hidden-state/KV caches cannot be reused across adapters as if their weights were identical after updates. Recompute the action prompt under the action adapter. Clear adapter-specific caches and test simultaneous-server access; initially serialize requests. Confirm no stale adapter selection, optimizer ownership or value-head attachment after a switch. If this is too slow or cannot be made reliable, stop and revise the design before collecting paid rollouts.

At initialization, this composite policy must match D208 under identical inputs and deterministic decoding within declared numerical tolerances. After updates, evaluate exactly this composite inference path. A box check on the frozen prefix should be invariant; it is not evidence that the trainable action adapter's standalone grounding improved. Optionally measure the latter as a diagnostic, clearly separated. Freeze vision LoRA parameters initially; verify gradients and optimizer parameter lists rather than trusting a flag name.

### 5.2 Behavior distribution

Use stochastic categorical action sampling for training, deterministic decoding for the primary evaluation and a separately labelled stochastic diagnostic. Set temperature and token restrictions explicitly; avoid top-p/top-k initially because changing support complicates probability replay. If any transform is used, old/new log probabilities must use the same transformed distribution and permitted-token mask. Sampling only valid action bins must be reproduced in training.

Save all32 sampled action-token IDs and per-token behavior log probabilities. The initial objective is PPO on the joint autoregressive chunk: sum log probabilities of the sampled action tokens, exponentiate the new-minus-old sum, and clip that ratio. This is different from averaging32 independently clipped token objectives. Joint ratios may be noisy; measure their tails before proceeding. If they are unusable, a per-token surrogate is an explicitly named alternative experiment with its own tests, not a silent numerical replacement.

Treat the sampled full chunk as the macro-action, even if a terminal event prevents executing its suffix; unused sampled dimensions add variance but do not justify an arbitrary likelihood mask. Record execution masks for diagnostics. If implementing marginal likelihood over an executed prefix later, prove that truncation/stop selection does not depend on the unexecuted suffix. Always exclude padding and never include tokens from a new reset episode. Record whether stopping depends on progress, action values or the whole proposed chunk.

Before any update, compare server-recorded log probabilities with trainer recomputation for the exact same processed image, prompt, box/progress prefix, action history and adapter. Disable dropout during both inference and gradient evaluation if necessary for a deterministic probability contract; eval mode still permits gradients. Measure BF16/backend variation, then require tolerances small relative to the PPO clip range. A diagnostic warning is insufficient: parity failure blocks training. Do not replace old probabilities merely to hide mismatch.

## 6. Critic, returns and PPO update design

Use a small normalized value head on a pre-action context representation. Do not let it see the sampled current action or future reward. For the first pilot, detach backbone features for the value loss and train the head separately; actor updates still change its features, so track prediction quality on fresh data. This is a conservative local design, not an assertion that shared-backbone actor–critic training is always wrong. Prior local raw-value instability makes normalization and bounded diagnostics important.

Warm up the value head on the first collected batch with the actor frozen. Preserve the exact behavior policy and calculate advantages from a clearly identified value snapshot. Report value loss against a constant-return predictor and explained variance; on a near-constant reward batch, explained variance may be undefined and should not be dressed up as success. Do not warm up indefinitely to fit noise.

For each decision t, calculate delta = reward + gamma * bootstrap_value - value; apply GAE with explicitly defined continuation masks. A true terminal has zero bootstrap. An artificial collection/time cutoff bootstraps from its actual final observation with the frozen prefix and critic. It must not bootstrap from the next episode's reset image. Recurrence never crosses an episode reset. If a finite deadline is part of the task, expose time remaining and treat it according to that finite-horizon objective, rather than pretending it is an external truncation (E05/E27).

Initial proposed gamma0.99 and lambda0.95 are per model decision/macro-action, not per photograph or metre. Changing chunk length changes the effective discount convention. Record this explicitly; duration-adjusted discounting is a later option requiring new tests. Compute advantages once for the collected policy snapshot; normalize across the declared training batch, guard tiny variance, and log raw and normalized distributions.

Conservative starting configuration for calibration: one PPO epoch per fresh batch, clip0.05 on the JOINT ratio, actor learning rate1e-6 to1e-5 chosen by a one-update diagnostic, value head up to1e-4 only if its warmup tests pass, gradient accumulation sized from measured memory. Do not inherit a successful manipulation paper's learning rate. All values go into the manifest; changing one after a failure creates a new experiment version.

Clip actor and critic parameter groups deliberately, log each norm, and reject nonfinite loss/gradient/update. Log joint approximate KL to behavior, ratio quantiles, clip fraction, token entropy, advantage/reward distribution and parameter changes. A sampled KL estimate can be noisy or negative; distinguish it from exact categorical KL. If measuring frozen-reference KL, compute it on the same contexts and masks; adapter switching must not alter caches. PPO clipping alone is not a guaranteed trust region (E01–E04).

Use only fresh stochastic rollouts from the exact current behavior snapshot for PPO. D209 deterministic logs are valuable fixtures and diagnosis data, not on-policy experience for the new adapter. Do not mix replay into PPO ratios with fabricated old log probabilities. Verified training-only SFT retention examples can enter a separately weighted supervised loss, with distinct reporting.

Retention anchors must be balanced across flight kinds and include independently reviewed grounding examples. Do not assume every old correction is trustworthy. The60 heldout box frames remain evaluation-only. A grounding mix ratio from SFT is not equivalent to a PPO anchor-loss coefficient. Calibrate coefficients using gradient magnitudes and retention checks; record tokens/masks and examples seen. Avoid multiplying memory by loading multiple base models without profiling.

## 7. Data contract and script structure

### 7.1 Every rollout must be reconstructible

Use a versioned schema with immutable run/iteration/episode/call IDs. Store:

- Task ID, kind, split and provenance hashes; simulator build/map/seed; base/adapter/processor/action-statistics hashes; code revision plus dirty patch hash; reward/evaluator versions.
- Raw image path/hash and processed-image shape/hash or deterministic preprocessing metadata; exact instruction/state prompt; input token IDs, attention masks and image positions.
- Raw box and progress text/tokens, effective goal-memory-adjusted progress, memory state and update rule; prefix adapter hash; action adapter hash; sampling seed, temperature and support mask.
- Proposed action tokens, decoded actions, behavior per-token and joint log probability, pre-action value, requested and actual poses, execution mask, stop reason and step/call counters.
- Each reward component, semantic progress state, termination versus truncation, infrastructure-error code, actual final observation/context and bootstrap value provenance.
- Timings for reset, rendering, prefix generation, action generation, execution, serialization and upload; GPU peak allocation/reservation and disk space; schema validation status.

Avoid secrets and bucket names in repository reports. Fail on missing required fields; do not silently reconstruct an absent box prefix from a rounded parsed box. Reject duplicate episode IDs, mixed behavior adapter versions, broken call ordering, missing images, split contamination and inconsistent terminal records before training.

### 7.2 Implementation map

Names marked NEW are proposed files, not existing delivered implementations. This request delivers the design; it does not make these changes or claim their tests have passed.

| File/component | Required change | Acceptance evidence |
|---|---|---|
| scripts/uav_flow_eval_server.py | Exact prefix/token/log-prob logging, explicit sampling transforms, frozen-prefix/action adapter selection, cache isolation | Deterministic D208 parity and stochastic log-prob replay report |
| scripts/rl/rollout_samples.py | Schema-driven reconstruction, pre-action value context, final observation and terminal/truncation masks | Hand-computed return/GAE fixtures |
| scripts/rl/train_grpo_vla.py | Extract/clarify PPO path; joint objective, detached value warmup, full parameter clipping, remainder accounting, hard parity gate | Tiny categorical-policy update tests plus one real-model update |
| scripts/rl/reward_uav_flow.py | Preserve historical behavior as v1; route version explicitly | Historical scores unchanged |
| NEW scripts/rl/reward_v2.py | Independent per-kind completion and tested optional progress components | Golden positive/negative trajectory suite |
| NEW scripts/rl/rollout_schema.py | Typed/versioned records and validation | Missing/mixed/corrupted record rejection tests |
| NEW scripts/rl/ppo_math.py | Pure ratio, loss, masks, returns and GAE functions | CPU analytic and numerical tests |
| NEW scripts/rl/preflight_ppo.py | Run gates, aggregate evidence, refuse launch without valid signed-off manifest | Deliberately failed gates block runner |
| NEW scripts/rl/run_ppo_round.py | Synchronous stage machine, deadlines, atomic outputs and recovery | Fault-injection recovery report |
| scripts/rl/collect_rollouts.ps1 and Windows evaluation launchers | Explicit horizon, native exit marker, simulator readiness/reset checks, no duplicate process | Exit0/exit7, timeout and duplicate-launch tests |
| NEW scripts/rl/score_semantic.py | Versioned full-path and task-semantic reports alongside v1 | Independent fixture and saved-flight comparison |
| NEW tests/rl/ and reports/<run>/ | Unit/integration fixtures, stage reports and review pack | Machine-readable pass/fail plus human-readable report |

Prefer extracting small testable functions over rewriting the whole stack. Preserve existing uncommitted work and historical artifacts. Pin dependencies only after checking Windows/CUDA compatibility and actual package licenses. Do not migrate to RLinf, continuous action heads or a world model during this first implementation.

### 7.3 Orchestration state machine

Use explicit states: CREATED → PREFLIGHT_PASSED → COLLECTING → VALIDATING_ROLLOUTS → VALUE_WARMUP → UPDATING → CANARY → CHECKPOINTED → COMPLETE. FAILED/ABORTED are terminal until an explicit reviewed recovery decision. Each transition writes an atomic JSON record containing timestamps, input/output hashes, counts and exit status.

Only one process owns the run lock. A second launch must report the existing run and exit. A stopped EC2 instance is not evidence of training completion; a completion marker must identify verified checkpoint/report uploads. Historical failure markers must remain scoped to their attempt, avoiding the D209 confusion.

Checkpoint actor, critic, optimizer states, scheduler, RNG states (Python/NumPy/Torch/CUDA), scaler if used, iteration/sample position, behavior snapshot, normalization statistics and manifest. Write to a temporary destination then atomically commit locally; validate and checksum remote uploads before announcing completion. Never claim exact resume from adapter weights alone. A partial update is not automatically retried on the same data. Test interruption recovery and document when a run must restart from the last committed iteration instead.

Windows wrappers must propagate the child Python native exit code reliably. Test actual child exit0 and exit7, exceptions, forced termination and missing marker. Set hidden windows for background launch. No automatic retry of failed optimization, no fallback instance type and no extending the budget silently.

## 8. Pre-validation gates: order, deliverables and stop rules

The default state of every gate is NOT RUN. Prior D210 analysis supplies fixtures and known failures; it does not satisfy the new acceptance suite. The following sequence is intended to make cheap failures happen before paid compute.

### Gate A — Offline semantics and provenance; zero cloud spend

1. Build train/development/validation/test ID manifests and assert disjointness. Development tasks used to tune rewards cannot later be described as untouched validation. The273 test tasks remain untouched.
2. Reproduce saved D210 stationary-versus-reference reward counterexamples. Historical v1 scores must remain reproducible.
3. Implement independent semantic fixtures for all included kinds: valid completion, no movement, wrong direction/object, partial completion, excessive jump, near-boundary tolerances and stop timing. Add at least20 varied Surround cases including loop start/end aliasing, yaw wrap and off-centre paths.
4. Audit action units/frame/bin integration with synthetic and saved traces, including clipped non-Surround components. Test raw versus tokenized trajectories; report quantization error rather than assuming it is negligible.
5. Produce a compact review pack: instruction, camera samples, trajectory, reference/target geometry, predicted completion, and explanation. Include all ambiguous/failed controls and stratified ordinary examples.
6. Record which predicates are verified, which rely on approximate metadata, and excluded tasks. If no trustworthy success definition exists for a kind, it cannot enter reward v2 training.

**Pass:** all deterministic invariants pass; stationary-circle exploit fails under v2; intended completed controls succeed; no split leakage; review disagreements resolved or quarantined. **Stop:** any reward exploit, unsupported metadata assumption or action-contract mismatch. Deliver reports/preflight_ppo/semantics.json and review pack.

### Gate B — CPU-only optimizer and orchestration tests

- A tiny categorical policy with analytically known action probabilities must match joint log-probability and PPO loss; one gradient step must favor the positive-advantage action. Check finite differences, clipping boundaries, entropy, padding and batch weights.
- Hand-compute a short terminal episode and a truncated episode with a nonzero final value. GAE implementation must match exactly within numerical tolerance; no recurrence crosses episodes. Test timeout with reset observation deliberately different from final observation.
- Feed0,1,17,31,32,33 samples and non-multiple accumulation sizes. Each accepted sample must be accounted for; remainder updates scaled correctly. Empty batches and zero actual optimizer steps are explicit non-success outcomes.
- Critic warmup must leave actor weights bitwise unchanged. Frozen vision/prefix parameters must never enter optimizer groups. Check that gradient clipping covers intended trainable groups.
- Detect NaN/Inf, corrupted images, missing token IDs, mixed adapter hashes, duplicate calls, invalid action tokens and schema-version mismatch before an optimizer step.
- Inject process exit7, disk-full/write failure, interrupted checkpoint, missing upload, duplicate launch and expired deadline using mocks. Failure must save diagnostics and prevent a completion marker. Test checkpoint/resume equivalence on a tiny deterministic model, including RNG.

**Pass:** all meaningful invariants pass with test evidence. **Stop:** any silent skip, incorrect mask, ambiguous exit status or incomplete recovery. These tests exercise failure modes, not merely mirror implementation outputs.

### Gate C — Real model on saved observations; GPU needed, no simulator flight yet

Use a local GPU if available and suitable. Otherwise this is the first separately authorized, hard-budgeted GPU smoke test—not a full training run.

1. Load the pinned D208 baseline and new composite policy; compare deterministic output on a stratified saved practice set.
2. Sample and replay at least32 contexts spanning box-empty/nonempty, long/short instructions, progress-memory changes and action extremes. Compare token IDs/context hashes and old log probabilities before updates. Initial engineering tolerance proposal: maximum joint-log-prob error0.01 nats; calibrate backend precision on repeated identical inputs first. If this cannot be met relative to clip0.05, unify kernels/precision or change the implementation. Do not casually loosen the gate.
3. Verify stochastic repeatability with restored seeds and diversity with different seeds. Zero temperature is forbidden for PPO collection. Check support, entropy and decoded action distributions for unreasonable jumps or pervasive clipping.
4. Run critic warmup and one tiny actor update; inspect actual parameter deltas, loss terms, ratio/KL distribution and all frozen weights. Use fresh saved-context sampling for this plumbing test; no flight-improvement claim follows.
5. Measure inference and backward peak allocated/reserved memory, not only model loading. Leave at least15–20% practical headroom as a proposed target; test the longest allowed prompt. If it does not fit, reduce microbatch/sequence bounds or use recomputation deliberately. Never silently truncate instructions or prefixes.
6. Save/reload the complete checkpoint and reproduce evaluation outputs. Exercise adapter switches repeatedly to catch stale-cache state.

**Pass:** exact context parity, acceptable probability precision, nonzero correct update, stable memory and reproducible reload. **Stop:** OOM, support mismatch, unexpected frozen-parameter change or unexplained probability drift. Deliver parity.json, memory_profile.json, update_smoke.json.

### Gate D — Small simulator control study before training

After separate authorization, run a fixed small practice subset through baseline D208, D209 and simple controls where feasible, under the same v2 protocol. Include Surround, Pass, Retreat and a simple altitude/turn task; do not cherry-pick only easy successes. Start with approximately8–12 tasks, bounded by the approved wall-time/cost cap. Run long-horizon diagnostics only where needed to answer feasibility.

Compare actual path, semantic completion, stop reason, clipping and runtime. Test reset reproducibility, renderer freshness, object metadata and any collision/swept-path interface. A stale frame or failed reset invalidates the episode. If collision detection is unavailable, state that the pilot measures kinematic simulated task execution and restrict risky geometric cases; do not claim physical safety.

Collect a few stochastic attempts on the same development tasks to measure reward variance and exploration. If all successes or all failures, examine task selection and reward resolution. Do not repeatedly resample only successful tasks until the reported batch looks healthy. Bounded filtering must log rejected attempts and their compute cost.

**Pass:** environment and predicates agree on controls, useful variation exists, and measured throughput supports a small pilot. **Stop:** no useful signal, unresolved completion disagreements, broken reset, or infeasible actions. Only now calculate a credible runtime and AWS budget.

### Gate E — Bounded learning pilot, not a day-long run

Proposed initial scope: at most two synchronous collect/update rounds, at most16 fresh practice episodes per round and at most1024 model calls per round, with a smaller target of roughly256 calls if feasible. Complete or explicitly truncate episodes according to the recorded contract. Effective batch64 calls is a starting point subject to memory tests; one epoch, at most16 optimizer updates per round. These are ceilings, not quotas to fill after the wall budget expires.

Use a fixed stratified practice schedule and record actual task exposure. Long episodes otherwise contribute more calls; report this weighting and avoid claiming equal task weighting. Do not silently drop difficult long episodes. Keep a reserved fixed development canary set distinct from optimization tasks. Snapshot and evaluate before and after each round.

Hard limits are the earliest of episode, call, optimizer-update, wall-time and monetary ceilings. The monetary ceiling must be explicitly provided/approved at launch time using measured throughput and current instance pricing; this document does not invent a dollar authorization. A smoke test can be capped around30–45 minutes and an initial pilot around2–4 hours only if measured throughput makes the requested comparisons feasible. Otherwise shrink scope or stop; do not extend the deadline to finish a nominal schedule.

Acceptance: data/probability/optimization checks remain valid; intended training behavior improves on useful diagnostics; no severe canary retention loss; outputs are reproducible and complete. A flat or worse pilot is a valid result. Stop and diagnose instead of automatically increasing updates from32 to300.

### Gate F — Evaluation and promotion

Only after the pilot is technically valid and promising, compare a frozen candidate with D208 and, where useful, D209 on the same100 validation tasks under an agreed v2 horizon/semantic protocol. Baselines must be run/scored comparably; a500-step candidate cannot be compared as if equivalent to a100-step baseline. Historical v1 numbers remain separate. If saved traces cannot support v2 semantics, report missing evidence rather than inventing a rescore.

Report per-task and per-kind wins/losses, semantic completion, full-path diagnostics, horizon/calls, failures, reward components and the60-frame grounding check. Use paired task bootstrap or another stated paired uncertainty method; distinguish variation across tasks from training-seed uncertainty. With one trained seed, call the result a pilot. Multiple independent training seeds require a later budget decision, ideally at least three before robust performance claims. Do not select reward weights or best checkpoints repeatedly on the heldout set and then call it untouched.

Promotion requires a documented decision, not merely the maximum aggregate reward. Preserve baseline and candidate artifacts. Do not automatically chain DAgger, GRPO, another PPO round or benchmark test execution.

## 9. Monitoring thresholds and failure response

Thresholds must be calibrated and frozen before optimization. Suggested initial guards:

| Signal | Proposed action |
|---|---|
| Context/hash/support mismatch; nonfinite loss/gradients; mixed behavior adapters | Immediate abort before further updates |
| Pre-update joint-log-prob parity outside calibrated tolerance | Abort, fix inference/replay contract |
| Mean joint approximate KL near0.02 | Warning/reduce remaining update exposure under predefined rule |
| Joint KL above0.05 or extreme ratio tails | Stop current update epoch; inspect before another collection round |
| Clip fraction above0.3 persistently, or above0.5 with large KL | Warning then stop; clipping is not proof of safety |
| Entropy falls over30% from paired baseline without completion gains | Pause pilot for collapse diagnosis |
| Critic no better than constant predictor after bounded warmup | Diagnose target/mask/features; do not spend indefinitely fitting it |
| All-zero reward/advantage batches | No learning claim; stop or invoke explicitly planned diagnostic, not blind retries |
| Prefix grounding changes despite frozen adapter | Implementation bug; abort |
| Standalone/action retention centre error worsens over5% or IoU drops0.02 | Review/rollback trigger, not a statistically proven cutoff |
| Renderer stale/reset invalid, simulator crash | Infrastructure failure, exclude from learning with logged reason |
| Budget/deadline/disk/memory reserve reached | Stop collection, save committed state and logs, shut down as authorized |

Monitor both sampled approximate KL and richer distribution diagnostics. Small averages can conceal rare huge ratios. Do not use a threshold as a substitute for inspecting per-kind collapse. No failed optimizer is automatically restarted.

## 10. Compute, shutdown and cost controls

Measure separately p50/p95 reset time, prefix/action inference, per-move simulation, training microbatch, checkpoint and upload. Estimate total cost from those measured components and the intended episodes/calls/updates, including baseline/control runs and rejected attempts. Quote a range with contingency, not a paper's GPU-hours transplanted to L4.

Reserve roughly20% of the approved wall budget for saving, verification and shutdown. Set an independent instance shutdown deadline before lengthy setup, plus an earlier runner deadline. The runner must stop starting episodes when the remaining time cannot cover a p95 episode plus upload reserve. Use the approved existing instance only; no implicit larger GPU or second instance.

If simulator and trainer share one GPU, initially run collection and training sequentially. Explicitly release renderer/model resources as needed and test restart overhead; do not assume simultaneous graphics and backward memory fit. If separate collection/training machines become necessary, design artifact transfer, behavior-version locking and cost caps before requesting that expanded experiment.

On success/failure: save logs and a truthful status manifest, verify checkpoint/artifact sizes and hashes, request shutdown through the wrapper, then externally verify EC2 stopped. An uploaded failure file alone is not proof that billing stopped. Readiness failure and upload failure need bounded handling and notification; do not leave an idle machine waiting indefinitely. Use role-based access/environment configuration without printing credentials.

## 11. Mistakes to avoid and the evidence behind them

| Mistake | Evidence or local example | Prevention |
|---|---|---|
| Treat paper benchmark gains as a promise for our drone | E07–E12 use different tasks/representations/compute | Local controls and bounded pilot |
| Optimize a proxy that rewards standing still | D210 direct stationary-circle counterexample | Adversarial reward fixtures before learning |
| Treat timeout as task termination | E05/E27; current rollout conversion | Final-observation bootstrap tests |
| Copy PPO without validating implementation details | E03/E04/E26 | Analytic math, probability parity and sample accounting |
| Assume GRPO always has a useful ranking | E08/E09/E22 discuss group signal/filtering | Measure within-task variance and rejected-rollout cost |
| Increase learning rate/epochs to force visible progress | E07/E10 and PPO implementation studies | One epoch, small calibrated update, KL/retention stops |
| Ignore action chunk execution and observation cadence | E23/E24; local100→104 overshoot | Explicit macro-action/horizon contract |
| Trust a synthetic expert because labels look plausible | D210/D211; E06 assumes access to an expert | Independent teacher validation if DAgger resumes |
| Use aggregate success as the only evidence | E18–E21; D208 stationary passes | Per-task traces, semantic predicates and paired uncertainty |
| Assume SFT anchors or frozen vision protect everything | E14 supports retention strategy, not our exact setting | Gradient ownership tests and behavior/grounding canaries |
| Adopt a large framework before knowing bottlenecks | E13 systems design versus our Windows constraints | Profile first; reuse concepts before migrating |
| Call a process launch or adapter export successful training | D209 Windows exit-code failure; current checkpoint gaps | Native exit markers, update counts, atomic complete state |

## 12. Work packages and decision points

1. **Offline correctness package:** semantics/action specification, v2 controls, split/provenance manifests, review pack. Deliverable is an accepted measurement contract, not a trained model.
2. **Trainer repair package:** schema, prefix parity, PPO math, critic/timeouts, sample accounting, checkpoint and orchestration tests. Deliverable is CPU evidence plus a GPU smoke-test recipe.
3. **Feasibility package:** real-model parity/memory and a small simulator control study under separately approved compute. Deliverable is measured throughput, reward signal and a priced pilot proposal.
4. **Pilot package:** at most two bounded rounds with canary evaluation and complete artifacts. Deliverable is a go/no-go diagnosis, even if there is no gain.
5. **Validation package:** matched heldout comparison and uncertainty, then an explicit promotion/next-experiment decision.

Do not promise these packages fit in a half-day. Predicate metadata or exact likelihood replay can expose structural work. After each package, stop if its assumptions fail; this avoids spending a full training day to discover a problem that a fixture could reveal.

DAgger repair remains separate: geometric/phase-aware target sampling, raw/tokenized teacher rollout checks, terminal/recovery examples and complete replay provenance. PPO does not require completing that labeler repair first. Both paths require valid task semantics and an action interface capable of expressing the behavior. If the8-action/256-bin representation cannot produce feasible trajectories, that is a representation problem to address explicitly before either method.

## 13. Final launch checklist and current readiness

A launch manifest must name: accepted task/reward/action versions; completed gate reports; exact code/model/data hashes; baseline and composite-policy definition; seed/sampling settings; critic/GAE/PPO/anchor configuration; allowed task lists; hardware; current pricing; episode/call/update/wall/cost ceilings; checkpoint destinations; shutdown deadline; rollback/stop rules; and responsible completion review.

**Current readiness: RED.** The known reward/context/timeout defects are not repaired by this document. The test specifications above are not executed tests. No claim is made that PPO fits L4, that reward v2 is implemented, that all DAgger labels were audited, or that a new run will improve success. The next authorized implementation request should begin with Gates A and B; paid compute only follows concrete evidence and a bounded launch decision.
