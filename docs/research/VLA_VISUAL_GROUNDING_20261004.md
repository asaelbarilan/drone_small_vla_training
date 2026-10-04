# Making the VLA use the photo: how others solve visual under-reliance, and on Qwen (2026-10-04, D192)

**NO CODE CHANGED.** Research note. PDFs: `papers/visual_grounding/` (README there).

## The measured problem (our numbers)

- Turn (D190, 12 validation flights, both models): the FIRST progress line is "+24 deg" on
  every task (one says +22), while the true turns range -30 .. +25 deg. Goal memory locks it.
- Pass (D188): the first distance guess is about 10 m on every task (correlation 0.04 with the
  true length).
- Earlier open-loop camera controls (D157, D160): gray / swapped photos changed almost nothing.
- So the model writes the training average where it should read the target's bearing or
  distance from the photo. In the field's vocabulary: **visual under-reliance / vision-action
  shortcut / causal confusion in imitation learning** (de Haan 2019, "copycat").

## Two input mismatches found in our own pipeline (checked in the code, 2026-10-04)

1. **Colour:** training photos are loaded with `Image.open(...).convert("RGB")`
   (`scripts/train_uav_flow_vla.py` `load_image`); the official evaluator sends the simulator's
   raw BGR array as if it were RGB (`batch_run_act_all.py`: `Image.fromarray(image)`, no
   conversion). Every flight test since D163 - test, validation and the DAgger practice
   flights - saw red and blue swapped. The same is true for OpenVLA-UAV and the published papers.
2. **Size:** training photos are thumbnailed to 256 px (`load_image`); the evaluator sends 224 px.
   Qwen3-VL turns 256 px into 8 x 8 = 64 visual tokens and 224 px into 7 x 7 = 49, so the test
   image is coarser than anything seen in training, and a dog 5 m away covers about one token.

Both are hypotheses for the Turn / Pass failure, not proven causes: OpenVLA-UAV sees the same
swapped colours and still gets Turn 10/10.

## What the field does (ranked by effort vs expected effect for us)

| # | Fix | Who / evidence | What it means for us | Cost | Recommend |
|---|---|---|---|---|---|
| 1 | Remove train/test input mismatches (colour, size) | standard practice; our code shows both | server converts BGR -> RGB and resizes to 256 before the model (the evaluator stays unmodified) | $0 to test locally on the Turn / Pass start photos (does the first line follow the target?) | **First** |
| 2 | Grounded reasoning: write the target's **bounding box before the action** | ECoT, +28 points on OpenVLA generalization; ECoT-Lite: the gain comes mostly from better representations, and the box can be dropped at test time | before the "Left" line the model writes `bbox_2d` of the instruction's object, in Qwen's native format (Qwen3-VL was pre-trained to output exactly this). Labels: the base Qwen3-VL boxes the object in the training photos, as D189 already did for Turn | labelling local ($0, hours of GPU); one fine-tune ($10-20) | **Second, if #1 is not enough** |
| 3 | Goal-pose bottleneck: action prior without images, vision only through the goal | LIT (2609.12641): blocking direct visual access raised LIBERO-Plus OOD 67.7 -> 71.9 % | our progress line + goal memory already is this design; the weak link is predicting the goal from the photo, which #2 targets | - | no new build |
| 4 | Separate target head (box) next to the action head | CosFly-VLA (Qwen3.5 backbone, UAV tracking) | a second head = architecture change; #2 gets the same supervision through tokens | larger | later, only if #2 fails |
| 5 | Separate grounding model before the VLA | OBEYED-VLA, DroneVLA (Grounding DINO) | a second model at inference; conflicts with the one-model 8 GB goal | larger | no |
| 6 | Rebalance state vs vision | ReViP (2601.16667), 2608.03052 | our Turn failure happens at the first call, where the state is all zero; state is not the cause here | - | no |

What the field tried and dropped: state dropout (zero-masking the state, p = 0.8) did not pay
off; pose supervision alone gave only +1.8 points in LIT. Plain text chain-of-thought (sub-task
plans without boxes) is weaker than grounded reasoning (ECoT).

## Out of scope

- Re-training at a higher resolution (448 px, 196 tokens): plausible for small objects, but it
  multiplies the cost of every step; test #1 first.
- A Grounding-DINO-style detector at inference: a second model, against the 8 GB one-model goal.

## Sources

- ECoT 2407.08693 (https://arxiv.org/abs/2407.08693)
- ECoT-Lite 2505.08243 (https://arxiv.org/abs/2505.08243)
- LIT 2609.12641 (https://arxiv.org/abs/2609.12641)
- ReViP 2601.16667 (https://arxiv.org/abs/2601.16667)
- How should VLAs use proprioceptive state? 2608.03052 (https://arxiv.org/abs/2608.03052)
- CosFly-VLA 2607.15004 (https://arxiv.org/abs/2607.15004)
- OBEYED-VLA 2512.22519 (https://arxiv.org/abs/2512.22519)
- Causal confusion in imitation learning 1905.11979 (https://arxiv.org/abs/1905.11979)
- Qwen3-VL grounding (https://github.com/QwenLM/Qwen3-VL)
