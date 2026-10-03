# Drone VLAs under 7B with downloadable weights (checked 2026-10-02)

**NO CODE CHANGED.** Checked on the Hugging Face API (searches: uav, drone, aerial, uav-flow,
openfly, worldvln, exp2vla, aerialvla, racevla, travel-uav, citynav, ...) and the papers.

## Downloadable, under 7B, and actually output drone actions

| Model | Size | Data | Training | Results published? | Fits our test? |
|---|---|---|---|---|---|
| [LaZeAsh/qwen2_5_vl-uav-flow-200_000](https://huggingface.co/LaZeAsh/qwen2_5_vl-uav-flow-200_000) | Qwen2.5-VL-3B (7.9 GB) | UAV-Flow (LeRobot copy) | 200,000 x 32, **only an action head; vision and language frozen** | No (community run, empty card) | Yes, same data; needs a LeRobot adapter in our server |
| [LaZeAsh/smolvla-uav-flow-expert-200_000](https://huggingface.co/LaZeAsh/smolvla-uav-flow-expert-200_000) | SmolVLA ~0.45B (0.9 GB) | UAV-Flow | 200,000 x 32, action expert only | No | Same |
| [LaZeAsh/pi05-uav-flow-pilot](https://huggingface.co/LaZeAsh/pi05-uav-flow-pilot) | pi0.5 ~3B (9.4 GB) | UAV-Flow | only 3,000 x 32 ("pilot") | No | Same |
| [UPB-RAT-VLA/Exp2VLA-Pi05-MOv1](https://huggingface.co/UPB-RAT-VLA/Exp2VLA-Pi05-MOv1) (already on D:) | pi0.5 ~3B (7.5 GB) | Own Isaac Lab data, NOT UAV-Flow | paper: 84.1% on their own task | Yes, on their benchmark | Different action space and world; not comparable without retraining |

All three LaZeAsh models use 7-D actions, 10- or 50-step chunks and a 6-D state, through the
LeRobot library (dataset Smolbrainer/UAV-Flow-lerobot). None reports a closed-loop result.

## Not what we need

- 7B or larger, open: OpenVLA-UAV (7B, our reference, on D:), openfly-agent-7b, AerialVLA/AeroVLA
  (OpenVLA-7B), RaceVLA (OpenVLA-based), UAV-Flow-Qwen2.5-VL-7B, **WorldVLN** (8B backbone + action
  decoder, 73 GB, released: EmbodiedCity/WorldVLN, so a same-harness comparison is possible on a
  big GPU).
- Small but NOT action models: Miril-DroneVLM-2B-2 (Gemma-4 E2B; captions/QA/pointing on overhead
  images), Qwen3.5-4B-aerialsim-rl (text/JSON decisions in an AirSim safety environment),
  qwen2.5-vl-3b-drone-action-qlora (no card, unknown format).
- Small, published, but weights NOT released: AeroDPO (2B), LiteVLA-H (256M), GRaD-Nav++,
  VLA-AN, CognitiveDrone-R1 (reasoner).
- kekeabab/fastwam-uavflow: 12 GB raw .pt checkpoint, no card or code reference.

## Answer

The earlier claim ("no drone VLA under 7B") was too strong. Correct version: **no published,
benchmarked drone VLA under 7B has open weights trained on UAV-Flow.** Three unpublished community
runs on UAV-Flow exist (0.45B, 3B frozen-backbone, 3B pilot). They can be flight-tested on our
100 tasks if we add a LeRobot policy adapter to uav_flow_eval_server.py (lerobot package on the
Windows box; map their 7-D action chunk to our poses). That would give the paper a same-data,
same-size baseline that nobody has reported. Effort: about a day of adapter work + $1.30 per model.

## Decision (user, 2026-10-02)
Not used. No published results, and the two longer runs trained only an action head on a frozen
model. Baselines stay OpenVLA-UAV (7B, flown on our 100 tasks) and the published WorldVLN /
ImagineUAV numbers.
