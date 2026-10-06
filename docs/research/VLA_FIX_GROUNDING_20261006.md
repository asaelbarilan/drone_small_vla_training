# Fixing the model so it finds the target itself (D195, 2026-10-06)

**NO CODE CHANGED.** Research note. PDFs: `papers/visual_grounding/` (README there).

## What we measured (D193 result, our data)

- After box-first co-training (4,000 updates from the round-2 model), the model writes a box but
  nearly the SAME box on every Turn photo (about [500, 506, 572, 700], the image centre) - the
  average again, in box form. Normal colours or BGR: same.
- Given the base Qwen's (correct) box, our model's next box moves the right way, but its Left
  line still says about -18 deg on almost every Turn task. The line does not follow the box.
- The labels explain why the average is "almost right": 68 % of simulator box labels have their
  centre within 100/1000 of the image middle (the drone usually flies toward the target); only
  about 700 Turn first frames (target off-centre) among more than a million boxed rows.
- Our validation could not see this: accuracy counts move tokens only; the box tokens are only in
  the loss, which fell 1.71 -> 0.50 by update 500 and then stayed flat (the format was learned in
  500 updates, grounding never). More updates of the same would not help.

## What the field found (same problem)

| Paper | Finding | Mechanism for us |
|---|---|---|
| **ST4VLA** 2602.10109 (Qwen2.5-VL, closest) | action-only training drives grounding (RefCOCO) to near random by 20k steps; **plain co-training** of grounding + actions only partly keeps it and oscillates; **staged training** - grounding pre-training first, then action post-training conditioned on the spatial output, with the action gradient into the planner damped (x0.5) - keeps 70 % of grounding and lifts success 66.1 -> 84.6 % (Google Robot) and 54.7 -> 73.2 % (WidowX); robot-specific grounding data alone 54.9 -> 73.1 % | exactly our history: 31k action-only updates, then co-training (D193) |
| **Why grounding hurts medical VQA** 2604.27720 | answer-only SFT silently removes box output (the model answers box questions with text); format rehearsal + mixed grounding supervision restore it | what we saw in D192 (our model answers every question with a flight answer) |
| **ECoT-Lite** 2505.08243 | reasoning PRE-training (reasoning first, then actions) gave the second-biggest gain (+5.4 % LIBERO-90); co-training less | staged order matters |
| **BeTTER** 2604.18000 | VLAs exploit sensorimotor shortcuts; reasoning outputs do not drive actions under interventions; causes named: capacity compression, myopic downsampling | our line ignoring the box |
| **VLA reasoning faithfulness** 2605.17268 | reasoning-action consistency only 53 % in a driving VLA | measure line-vs-box consistency explicitly |

## Options, ranked (effect on the measured failure vs cost)

| # | Option | Why | Cost | Recommend |
|---|---|---|---|---|
| 1 | **Measure grounding during training**: on held-out simulator frames, let the model WRITE the box (not teacher-forced) and report the centre error / IoU against the label, plus Turn consistency (sign of the line's turn = side of the box) | the loss could not show the failure; without this we would repeat it | code only | **must** |
| 2 | **Balanced box data + box loss weight**: sample box rows so off-centre targets are as frequent as centred ones (bins by box centre), first frames of object tasks oversampled, box tokens weighted x3-5 in the loss | removes the "average is almost right" shortcut that the label statistics invite | code + continued training, about $10-15 | first, with #1 to stop early if grounding does not move within about 1k updates |
| 3 | **Staged retraining (ST4VLA)**: stage 1 grounding only (instruction -> target box, balanced, our sim + real frames, plus general grounding data) on a fresh LoRA from the base model; stage 2 action training on top with the box prefix, box still supervised, the action gradient into the vision path damped (lower lr or frozen vision LoRA) | the only approach with strong published evidence (+18 points) for exactly "co-training loses grounding" | stage 1 about $5; stage 2 needs most of the action training again (about 15-30k updates, $40-100) | if #2 does not move grounding |
| 4 | More updates of D193 as is | box loss flat from update 500 | - | no |
| 5 | Higher image resolution (more image tokens) | small-object accuracy drops 60-70 % from large to tiny objects in VLMs; but the base Qwen finds our targets at 256 px, so resolution is not the first limit | x2-4 compute | later |
| 6 | A second model / base-Qwen box at inference | user decision: not a fix, the model itself must ground | - | no |

## Sources

- ST4VLA 2602.10109 (https://arxiv.org/abs/2602.10109)
- Why does grounding hurt medical VQA? 2604.27720 (https://arxiv.org/abs/2604.27720)
- ECoT-Lite 2505.08243 (https://arxiv.org/abs/2505.08243)
- BeTTER, Unmasking the illusion of embodied reasoning 2604.18000 (https://arxiv.org/abs/2604.18000)
- Is VLA reasoning faithful? 2605.17268 (https://arxiv.org/abs/2605.17268)
