# Evidence synthesis and decision

## What the evidence supports

PPO is a plausible experiment, not a demonstrated solution for this drone. RL4VLA (E07) favors PPO in its comparisons; SimpleVLA-RL (E08) obtains strong results with GRPO, and RIPT-VLA (E09) uses an RLOO-style approach. These are different models, tasks, rewards and compute regimes. Their results do not identify a universal winner. Manipulation success percentages cannot be compared directly with our endpoint scores.

The strongest common lesson is to validate the environment, policy probability calculation and evaluation before scaling. PPO implementation studies (E03, E04, E26) show that implementation choices materially affect results. Time-limit work (E05, E27) distinguishes true termination from artificial truncation. Our local omissions—box context in replay, timeout handling and full-orbit reward semantics—are concrete reasons not to launch the current trainer.

Our D210 stationary-circle counterexample already falsifies the current reward as a sufficient Surround objective. This was not the objective used by supervised D209 and cannot explain its training outcome. UAV-Flow (E15) also uses semantic human assessment; our endpoint proxy must not be presented as equivalent to that assessment. Full-path geometry and independently reviewed completion predicates must accompany automated scores.

## Taxonomy and transfer limits

| Family | Evidence | Useful transfer | What does not transfer automatically |
|---|---|---|---|
| On-policy actor–critic | E01–E05, E07, E10, E26–E27 | Clipped policy updates, value estimation, replay parity, timeout handling | Drone reward, categorical chunk stability, L4 fit |
| Group-relative / leave-one-out | E08–E09, E22 | Avoid separate critic; compare stochastic attempts | Cheap rollouts, nonzero within-task reward variance, exact reuse of an LLM trainer |
| Demonstration / correction learning | E06, E11, E14 | Aggregation, retention anchors, intervention and recovery | An automatically correct geometric teacher |
| Action/model representation | E23–E25 | Chunk cadence, deployment consistency, action-head implications | Safe larger moves or a free switch from token actions to continuous control |
| Systems / embodied navigation | E12–E13 | Profiling, worker boundaries, reproducibility | Distributed Linux stacks fitting our Windows simulator and one L4 |
| Benchmark and statistical validity | E15–E21 | Independent task predicates, domain diversity, paired evidence and uncertainty | A trustworthy result from a single aggregate score or one seed |

## Reuse, adapt or build

| Option | Benefit | Cost/risk | Decision for first experiment |
|---|---|---|---|
| Run existing PPO flag unchanged | Least initial work | Known reward, context, timeout and checkpoint defects | Reject |
| Repair existing small trainer and collection interface | Preserves token actions, simulator and known deployment | Must explicitly test probability math and orchestration | Preferred bounded implementation, conditional on gates |
| Adopt RLinf / a full distributed VLA stack | Mature separation of workers and broader scaling tools | Integration, OS, model/action and resource mismatch; license/dependency audit | Reuse design ideas now; reconsider after profiling demonstrates need |
| Use GRPO first | No learned value head | Multiple expensive stochastic flights per task; equal-reward groups give no ranking; reward bug remains | Keep as a later alternative if critic fails and group signal is demonstrated |
| Repair DAgger first | Potentially efficient supervised updates | Requires a phase-aware validated expert and replay audit | Optional separate track; not prerequisite for PPO action labels |
| Replace action representation / train a world model | May address longer-term control or throughput limits | Changes several variables, additional data/compute and validation | Outside first pilot |

## Recommendation and falsification

Build a small, synchronous PPO pilot around the existing action-token policy, with a frozen D208 prefix generator and conservative action-adapter updates. This is an engineering proposal to isolate action learning; none of the papers proves this exact composite policy will improve our drone. Keep vision frozen initially, preserve exact behavior probabilities, and evaluate the composite policy actually trained. Do not silently merge it into a single adapter.

The recommendation is falsified or postponed if: task completion cannot be measured reliably; no sampled practice flight supplies useful signal at a feasible horizon; action likelihoods cannot be replayed accurately; one update cannot fit the available hardware with margin; or simulator/action representation prevents the desired behavior. GRPO solves none of the first, third or fifth problems. A failed critic-specific gate can motivate a GRPO/RLOO comparison, but only after measuring within-task reward variance and rollout cost.

Success of the pilot means the system learns the intended behavior without unacceptable retention loss, under a recorded budget. A smoke test only establishes plumbing. A small positive pilot is not a publication-quality improvement; E19–E21 motivate paired task analysis and later independent seeds. See PLAN.md for executable acceptance specifications and stop conditions.
