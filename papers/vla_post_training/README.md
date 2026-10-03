# VLA post-training cheaper than GRPO - reading list

For `docs/research/VLA_CHEAPER_THAN_GRPO_20261003.md` (D184). PDFs are git-ignored.

| file | what it is | what we took |
|---|---|---|
| `2011_DAgger_reduction-imitation-to-no-regret.pdf` | DAgger: expert relabels the states the student visits | The core idea: our recorded flights are a free exact expert |
| `2026_VLA-OPD_on-policy-distillation.pdf` | Teacher gives token-level labels on the student's own trajectories | On-policy distillation beats RL on sample efficiency, avoids forgetting |
| `2025_RL4VLA_what-rl-brings-vla.pdf` | Empirical study of RL for VLAs | PPO > GRPO > DPO; RL helps execution, not vision robustness |
| `2025_RIPT-VLA_interactive-post-training.pdf` | RL post-training from sparse success, dynamic sampling | Filtered self-imitation as fallback; skip groups without spread |
| `2024_V-GPS_value-guided-policy-steering.pdf` | Re-rank sampled actions with a value function, no fine-tuning | Test-time option for later |
| `2025_CO-RFT_chunked-offline-rl.pdf` | Offline RL with action chunks from few demos | Needs critic heads; out of scope now |
| `2026_EXPO-FT_sample-efficient-rl-finetuning.pdf` | Sample-efficient online RL fine-tuning of VLAs | Needs actor-critic changes; out of scope now |
