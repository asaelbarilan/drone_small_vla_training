# Aerial VLA open problems — reading list

Gathered for `docs/research/VLA_BEYOND_WORLDVLN_20260930.md` (D174): what the UAV-Flow
papers do not solve, and how to beat WorldVLN. PDFs are git-ignored; re-fetch from arXiv.

| file | what it is | what we took |
|---|---|---|
| `2026_WorldVLN_autoregressive-world-action-model.pdf` | 8B video world-action model, 79.1% on UAV-Flow | Per-class table: Pass 40%, Rotate 46.7%, Surround 58.3%, Turn 60%. No pose input |
| `2026_ImagineUAV_world-action-kinodynamic.pdf` | 1.3B Wan world-action model + planner, 70.9% | States its own weakness: yaw-dominant and orbit tasks |
| `2026_survey_uav-vln-progress-challenges-roadmap.pdf` | UAV-VLN survey | Field's open list: grounding, long horizon, memory, sim-to-real, onboard compute |
| `2024_ECoT_embodied-chain-of-thought.pdf` | Intermediate reasoning tokens before actions, +28% on OpenVLA | The basis for progress tokens |
| `2025_CognitiveDrone_vla-reasoning-uav.pdf` | UAV VLA + reasoner, 59.6 → 77.2% | Reasoning before acting pays off on drones |
| `2026_CosFly-VLA_reasoning-trace-uav-tracking.pdf` | UAV VLA, think block then 8-step 4-DoF chunk | Same action format as ours, with reasoning |
| `2025_VLA-AN_onboard-aerial-vla-waypoints.pdf` | Onboard aerial VLA predicting target waypoints + yaw | Predicting the goal, not only the next move |
| `2026_DreamFly_causal-memory-litestop.pdf` | Aerial VLN with causal memory and a stop estimator (OpenFly) | Termination needs its own signal; memory of past frames |
| `2025_CronusVLA_multi-frame-vla.pdf` | Single-frame VLA extended to multi-frame | Cheap history for when the object leaves the view |
| `2026_ProgressVLA_progress-guided-policy.pdf` | Progress estimator guiding a VLA | VLAs lack progress awareness |
| `2026_VLA-SCT_self-correction-termination.pdf` | Training-free termination and self-correction | Same diagnosis, manipulation |
| `2026_proprio-state-in-vlas.pdf` | Where and how the robot state should enter a VLA | Our state-as-text is one of five options they compare |
| `2026_ThinkProprio_state-as-tokens.pdf` | State as vocabulary tokens, used early | Supports state as text tokens |
| `2025_RoboMonkey_test-time-sampling-verification.pdf` | Sample N actions, VLM verifier picks | Test-time option, later |
| `2025_SimpleVLA-RL_scaling-vla-rl.pdf` | GRPO for VLAs, dynamic sampling, higher temperature | RL settings; we already skip zero-spread groups |
| `2026_WorldFly_world-model-uav-vla.pdf` | World-model UAV VLA, own benchmark | No UAV-Flow number; not comparable |
