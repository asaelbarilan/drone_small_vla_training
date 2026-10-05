# What else to train in the box-first run (D194, 2026-10-05)

**Code changed: `--next-box` (trainer) and the server's box phase.** Question (user): while we
retrain for boxes anyway, what else is worth training in the same run - VLM abilities,
"dreaming" (world-model prediction), other ideas from earlier notes - given little money left.

## Ranked (effect on our measured failures vs extra cost in this run)

| # | Option | Evidence | Extra cost in this run | Decision |
|---|---|---|---|---|
| 1 | **Box first** (grounding back) | ECoT +28 pts (2407.08693); ECoT-Lite (2505.08243) | already in D193 | **in** |
| 2 | **Photos as the evaluator sends them** (BGR, 224 px) | own finding D192 | none | **in** |
| 3 | **Dreaming in box space**: after the box, the model writes where the target will be after its 8 moves (`{"bbox_2d_next": [...]}`, the box in the frame 8 steps ahead) | CoT-VLA predicts a subgoal image before acting: +17 % real, +6 % sim (2503.22020); OneWM-VLA predicts one latent per future frame: MT50 47.9 -> 61.5 %, and removing the latent loss drops 58.1 -> 21.6 % (2605.07931); UAV-Flow's leaders both "imagine" (WorldVLN, ImagineUAV) | about 15 more answer tokens on labelled frames; labels come free from the same labeller (simulator frames are labelled every 8th frame, exactly t and t + 8) | **in** (`--next-box`) |
| 4 | Dreaming as a future-embedding head (JEPA / OneWM style) | VLA-JEPA, VLAFlow, OneWM-VLA; VLANeXt: world-model objectives help but nearly triple training time | a new regression head, a second image pass per example (+20-200 % time), new failure modes | later, if #3 helps |
| 5 | General VLM co-training (VQA / captions) against forgetting | pi0.5-style co-training; VLA instruction tuning (2507.17520) | external data to download and mix; does not target a measured failure | no - #1 brings back the one VLM skill we measured as missing (grounding) |
| 6 | History frames (first + previous photo) | VLA-AN, CronusVLA | doubles the image tokens per example | no - goal memory already carries the past |
| 7 | Rebalance language vs vision (LangForce 2601.15197, ReViP) | manipulation | architecture change | no |

## Why #3 fits

- It is a visual subgoal: the model must look at where the target is AND predict how its own
  moves will shift it - the same coupling WorldVLN / ImagineUAV get from video prediction, at
  text cost.
- It addresses Pass directly: the next box running out of view says "keep going past it".
- Same ECoT-Lite dropout: frames without labels train without the lines, so the model can also
  fly without them; the server writes both lines (`--box`).

## Sources

- CoT-VLA 2503.22020 (https://arxiv.org/abs/2503.22020)
- OneWM-VLA "One token per frame" 2605.07931 (https://arxiv.org/abs/2605.07931)
- VLA-JEPA 2602.10098 (https://arxiv.org/abs/2602.10098)
- VLAFlow 2607.01586 (https://arxiv.org/abs/2607.01586)
- VLANeXt 2602.18532 (https://arxiv.org/abs/2602.18532)
- VLA instruction tuning 2507.17520 (https://arxiv.org/abs/2507.17520)
- LangForce 2601.15197 (https://arxiv.org/abs/2601.15197)
- ECoT 2407.08693, ECoT-Lite 2505.08243 (see VLA_VISUAL_GROUNDING_20261004.md)
