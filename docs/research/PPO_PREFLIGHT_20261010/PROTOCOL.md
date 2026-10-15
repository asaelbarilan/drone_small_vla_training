# D212 research protocol — 2026-10-10

Decision: Is a bounded PPO pilot preferable to repairing DAgger or running GRPO for our grounded Qwen3-VL-4B token-action drone policy, and what evidence must precede any paid rollout/update?
Scope: simulator-only, Windows L4 collection, existing LoRA/action-token server and Python trainer; existing artifacts first; no new training, cloud launch or benchmark-test execution authorized by this planning task.
Acceptance: evidence-backed design, explicit alternatives and uncertainties, file-level implementation map, executable acceptance-test specifications, stage gates with stop conditions, budget formula and bounded pilot, reproducibility and shutdown/recovery requirements. Never promise improvement.

Search protocol (written before new searches):
- Lanes: VLA PPO/GRPO and UAV navigation RL; reviews/citation discovery; PPO/TRPO/GAE/DAgger/reward shaping; recent 2023–2026 VLA post-training; UAV-Flow/LIBERO/RoboTwin datasets; official code/model cards/licenses; implementation failures/reward hacking/truncation/statistical evaluation; reusable training systems.
- Primary queries include RL4VLA, RIPT-VLA, SimpleVLA-RL, VLA-RL, RLinf, WorldVLN, ConRFT, PPO implementation details, timeout bootstrapping, GRPO zero-variance groups, action chunks PPO likelihood, UAV-Flow evaluator.
- Date range: foundations unrestricted; recent lane 2023 through search date2026-10-10. Record actual publication year; do not invent unavailable future work.
- Include primary methods/results with relevance to our constraints, official repos and benchmark protocols; exclude marketing-only results, duplicate versions as separate works, and unverified aggregate benchmarks. Backward/forward citation search from direct VLA sources.
- Candidate ledger >=30 works; retain >=20 deeply inspected works (target25), >=15 primary, >=3 direct solutions and >=3 dataset/benchmark sources. Abstract-only candidates cannot substantiate central recommendations.
- Inspect source methods/results and resources; mark unknown hardware/licensing/weights as not_reported/unverified. Missing license verification prohibits adopting that component until resolved.
- Stopping rule: coverage counts and schema validator pass, competing explanations/options considered, critical recommendations traced to evidence; unresolved local/hardware questions become preflight gates, never assumed resolved.

No paper result on manipulation is evidence of success for our drone. Numerical thresholds in the plan will be proposed engineering guardrails, not literature-established constants. The validator checks ledger structure, not the truth of each claim.
