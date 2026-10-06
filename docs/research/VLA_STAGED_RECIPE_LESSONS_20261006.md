# More lessons before the staged retrain (D197, 2026-10-06)

**NO CODE CHANGED.** Research note. Read in full: ST4VLA (2602.10109, incl. appendix C) and
VLANeXt (2602.18532, the design-trajectory figure). PDFs in `papers/visual_grounding/`.

## ST4VLA - the numbers that change our plan

| Finding | Their numbers | What it means for D196 |
|---|---|---|
| Both parts count: spatial prompting during action training AND grounding pre-training | WidowX: co-training 61.1 -> + spatial prompting 67.4 -> + spatial pre-training 73.2 | keep the box-first answer (our "spatial prompt") AND add stage 1 |
| **Grounding:action loss ratio in stage 2** | 1:1 -> 47.2 %, 1:5 -> 58.3, **1:10 -> 71.7**, 1:15 -> 71.8, 1:20 -> 68.3 (WidowX); more grounding weight keeps grounding better but hurts actions | stage 2 mixes grounding-only examples at about 1 per 10 action examples. D193 had box lines on about 42 % of rows - far heavier than their best ratio |
| **Grounding data volume** | pre-training 0 / 0.5M / 1M / 2M / 3M pairs -> average 61.4 / 61.0 / 63.4 / 71.0 / 77.9 | gains appear only from about 2M pairs. Our 81k labels are far below; stage 1 needs many more labels (and robot-domain labels matter most: general grounding data 54.9 -> 65.2, robot grounding -> 73.1) |
| Damp the action gradient into the VLM (factor 0.5) | part of the recipe; not ablated alone | our equivalent: slower learning for the vision part in stage 2 (plan already has it) |
| Not just faster - a higher ceiling | at 100k steps the baselines still plateau lower | worth doing even if stage 2 has to be long |
| Images 224 x 224 | used to match prior work | our 224 / 256 mix is fine |
| A weaker VLM also gains | Florence-2: 46.1 -> 67.9 | the recipe does not depend on model size |

## VLANeXt - which design choices matter (LIBERO-Spatial, step by step)

From a RT-2-style baseline (actions as bin tokens, like ours) to their final model:
- 19.8: bin-token baseline.
- 30.2: + policy head.
- 64.4: + more query tokens.
- 74.6: + action chunking.
- 80: + flow-matching loss.
- 90: + stronger VLM.
- 92: + soft connection.
- 50 when adding temporal history: it HURT, so they dropped it.
- 87.7: proprioception into the VLM (helps).
- World modelling helped (90.3) but was left out because it triples training time.
- 93.1: final, + time-series forecasting.

For us:
- **Temporal history hurts.** This confirms we were right to leave out history frames (D194 #6).
- **Our state goes into the VLM prompt.** That is the variant that helped.
- **Our action output (bin tokens) is the weakest design in their study.** The big jumps come
  from a policy head with continuous actions and flow matching.
  - This conflicts with our deployment goal: one LoRA on the shared VLM, run by llama.cpp,
    tokens only (D157).
  - It is the largest untapped lever, but an architecture decision for the user, not for this
    retrain.

## Changes to fold into D196

1. **Stage 1 needs far more grounding labels.**
   - Label many more of our own frames (simulator first, then real; every 2nd-4th frame).
   - Grow from 81k toward 0.5-1M pairs, about 25-45 h of labelling on one L4 ($25-45).
   - Or accept a smaller stage 1 and measure.
   - General datasets (RefCOCO etc.) help less than robot-domain labels, per their Table 6.
2. **Stage 2 mixing.**
   - About 1 grounding-only example per 10 action examples.
   - The action examples still carry the box line where labelled; the box tokens get a normal
     weight.
   - Do NOT repeat boxed rows 3x as in D193.
3. Keep the slower vision learning rate (their gradient damping).
4. Keep: no history frames; state in the prompt; images at 224 / 256.
5. Out of scope for this retrain (user decision needed): a continuous action head / flow
   matching. It is the biggest gain in VLANeXt, but it breaks the tokens-only llama.cpp
   deployment.

## Sources

- ST4VLA 2602.10109 (https://arxiv.org/abs/2602.10109), appendix C (ablations)
- VLANeXt 2602.18532 (https://arxiv.org/abs/2602.18532), Fig. 2
