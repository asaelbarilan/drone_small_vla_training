# D212 resource inventory

These are resource inspections, not installation or reproduction certificates. Repository badges do not license third-party assets or model weights. No new dependency was adopted. A missing/unverified license or compatibility detail must be resolved before reuse.

| Evidence | Code / resources | License status | Weights / data | Compute and compatibility |
|---|---|---|---|---|
| E01 | not_reported_in_inspected_passage | unverified; no dependency adoption authorized | not_reported_in_inspected_passage; not_reported_in_inspected_passage | not_reported_in_inspected_passage; no local throughput inference |
| E02 | not_reported_in_inspected_passage | unverified; no dependency adoption authorized | not_reported_in_inspected_passage; not_reported_in_inspected_passage | not_reported_in_inspected_passage; no local throughput inference |
| E03 | not_reported_in_inspected_passage | unverified; no dependency adoption authorized | not_reported_in_inspected_passage; not_reported_in_inspected_passage | not_reported_in_inspected_passage; no local throughput inference |
| E04 | not_reported_in_inspected_passage | unverified; no dependency adoption authorized | not_reported_in_inspected_passage; not_reported_in_inspected_passage | not_reported_in_inspected_passage; no local throughput inference |
| E05 | not_reported_in_inspected_passage | unverified; no dependency adoption authorized | not_reported_in_inspected_passage; not_reported_in_inspected_passage | not_reported_in_inspected_passage; no local throughput inference |
| E06 | not_reported_in_inspected_passage | unverified; no dependency adoption authorized | not_reported_in_inspected_passage; not_reported_in_inspected_passage | not_reported_in_inspected_passage; no local throughput inference |
| E07 | https://github.com/gen-robot/RL4VLA | MIT code; underlying model/data licenses separate | https://huggingface.co/collections/gen-robot/rlvla; ManiSkill/SimplerEnv tasks in repo | Paper v2: shared 44.4GB vs separate81.3GB; about42h onA100. Not an L4 estimate. |
| E08 | https://github.com/PRIME-RL/SimpleVLA-RL | MIT code; benchmark/model terms separate | not_verified; LIBERO/RoboTwin integration | Distributed training/inference/rendering; local Windows compatibility unverified |
| E09 | https://github.com/Ariostgx/ript-vla | unverified; no license text found in retrieved README | README offers OpenVLA-OFT workflow; checkpoint access not tested; LIBERO | not_reported_in_inspected_passage |
| E10 | Paper Section3.5 describes implementation; official repository link not verified | unverified | not_verified; LIBERO | BF16; dedicated inference GPU plus training GPUs;48GPUh reported, not local estimate |
| E11 | https://github.com/cccedric/conrft | unverified | not_verified; real manipulation/human intervention | Paper45-90min/task robot interaction; not our total compute cost |
| E12 | https://github.com/EmbodiedCity/WorldVLN.code | CC-BY-4.0 repository badge; underlying components separate | HuggingFace link exists in README; binary not fetched; UAV-Flow/IndoorUAV links | Paper8xA80080GB training,RTX4090 rollout |
| E13 | https://github.com/RLinf/RLinf | Apache-2.0 code; dependencies separate | Model adapters rather than our checkpoint; ManiSkill/LIBERO examples | Cluster performance only; Windows Qwen3-VL integration unverified |
| E14 | https://github.com/RLinf/RLinf | Apache-2.0 framework; specific experiment assets unverified | not_verified; Four sim-real tasks | not_reported_in_inspected_passage |
| E15 | https://github.com/buaa-colalab/UAV-Flow | Apache-2.0 code; data terms must be separately checked; project records research-use restriction | OpenVLA-UAV release linked; no binary checked; HF real/sim downloads linked in README | UnrealCV simulator; no transferable local PPO timing |
| E16 | https://github.com/Lifelong-Robot-Learning/LIBERO | MIT code; CC-BY-4.0 datasets per README | not_applicable for benchmark; Download scripts and HF linked | Simulator dependencies/version-sensitive |
| E17 | https://github.com/RoboTwin-Platform/RoboTwin | MIT code; asset licenses separate and unverified | not_applicable for benchmark; RoboTwin-OD download linked by project | GPU simulation; local feasibility not tested |
| E18 | not_reported_in_inspected_passage | unverified; no dependency adoption authorized | not_reported_in_inspected_passage; not_reported_in_inspected_passage | not_reported_in_inspected_passage; no local throughput inference |
| E19 | https://github.com/google-research/rliable | unverified in this review | not_applicable; Benchmark score analysis | CPU statistical analysis; actual runtime not measured |
| E20 | not_reported_in_inspected_passage | unverified; no dependency adoption authorized | not_reported_in_inspected_passage; not_reported_in_inspected_passage | not_reported_in_inspected_passage; no local throughput inference |
| E21 | not_reported_in_inspected_passage | unverified; no dependency adoption authorized | not_reported_in_inspected_passage; not_reported_in_inspected_passage | not_reported_in_inspected_passage; no local throughput inference |
| E22 | not_reported_in_inspected_passage | unverified; no dependency adoption authorized | not_reported_in_inspected_passage; not_reported_in_inspected_passage | not_reported_in_inspected_passage; no local throughput inference |
| E23 | Author project link in paper: https://tonyzhaozh.github.io/aloha/ | unverified | not_verified; ALOHA demonstrations | 50Hz collection/control in paper; no local timing inference |
| E24 | https://github.com/moojink/openvla-oft | MIT code; underlying weights terms separate | HF checkpoint used in official README example; LIBERO/ALOHA | README inference16-18GB;training27-80GB across1-8GPUs; notQwenPPO |
| E25 | https://github.com/openvla/openvla | Code/model/dataset terms not jointly verified; do not assume a repository grants all rights | https://huggingface.co/openvla/openvla-7b; OpenX mixture,970K trajectories official README | BF16/LoRA example; no localPPO feasibility assertion |
| E26 | https://github.com/vwxyzjn/ppo-implementation-details | unverified | not_applicable; Reference reproduction environments | not_reported_in_inspected_passage |
| E27 | https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/ | unverified; conceptual API use only | not_applicable; not_applicable | not_applicable |
