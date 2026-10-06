# Action head instead of action tokens? (D200, 2026-10-06)

**NO CODE CHANGED.** Research note. PDFs: `papers/action_heads/` (README there).

## What we do now

The 8 moves are 32 text tokens: each value is one of 256 bins on the last vocabulary ids.
They are generated one by one after the box line(s) and the Left line, and decoded greedily over
action tokens only (D191). The goal behind this was one LoRA on the shared VLM, running in
llama.cpp on 8 GB (D157).

## Evidence

| Paper | Change | Gain |
|---|---|---|
| OpenVLA-OFT 2502.19645 (7B, LIBERO, LoRA r32) | bin tokens one by one (76.5 %) -> all 8 steps in one pass from empty action slots with bidirectional attention (90.2 %) -> continuous values from a 4-layer MLP, L1 loss (95.3 %); diffusion head 95.4 % but 10x slower | +18.8 points; 26x faster action generation (0.24 s -> 0.07 s for a chunk) |
| VLANeXt 2602.18532 | bin tokens 19.8 -> policy head 30.2 -> more query tokens 64.4 -> chunking 74.6 -> flow matching 80 | largest single lever in their study |
| Qwen-VLA 2605.30280 | Qwen3.5-4B + 1.15B DiT flow-matching decoder | Qwen's own VLA uses a separate continuous decoder |
| FLIGHT 2606.06836 | Qwen2.5-VL-3B LoRA + DiT-B head, UAV | 59 % vs OpenVLA-UAV 10.5 % (long-horizon UAV) |

## What fits us (ranked)

| # | Option | Fits 8 GB / one VLM? | Effort | Expected effect |
|---|---|---|---|---|
| 1 | **OFT-style MLP head**: after the box and Left line, 8 empty action slots read in one pass; a small MLP (about 2560 -> 1024 -> 32, about 3M parameters, about 10 MB) outputs the 8 moves as continuous numbers; L1 loss | yes: llama.cpp can return the hidden states, and the MLP is tiny (CPU or GPU) | trainer + server change; the bidirectional mask over the slots needs care in HF Qwen3-VL (causal slots are a fallback) | the OFT ablation: +5 for continuous / L1, +14 for parallel decoding. Also removes bin clipping (D158: 24.5 % of steps clipped a channel) |
| 2 | Flow-matching / DiT head (pi0, FLIGHT, Qwen-VLA) | a DiT-B is about 100M+ parameters; Qwen-VLA's is 1.15B - against the 8 GB goal | larger | best published numbers, but OFT shows L1 is equal at a fraction of the cost |
| 3 | Keep bin tokens | yes | none | known ceiling |

What stays the same in every option: the box line(s) and the Left line remain text (goal memory
and the deadband need the line), and the boxes-first stage 1 (D199) is needed anyway.

## Cost and timing

- If chosen, the head replaces the move tokens in stage 2 of the staged retrain. Stage 2 has to
  train the actions again anyway, so the head adds no extra training run, only code (about a
  day) and probably more updates: OFT fine-tunes for 50-150k steps from an action-pretrained
  OpenVLA. Measure first with the planned ~8k and the checks.
- Server: one forward pass for the moves instead of 32 generated tokens, so flight calls get
  faster.
- 8 GB / llama.cpp: needs a small custom step (read the hidden states, run the MLP). This is no
  longer a pure "LoRA in llama.cpp" deployment, but it is still one VLM plus a 10 MB head.

## Recommendation

Option 1 (OFT-style MLP head, L1) in stage 2, decided before stage 2 starts. Stage 1 (D199) does
not change either way.

## Sources

- OpenVLA-OFT 2502.19645 (https://arxiv.org/abs/2502.19645)
- VLANeXt 2602.18532 (https://arxiv.org/abs/2602.18532)
- Qwen-VLA 2605.30280 (https://arxiv.org/abs/2605.30280), https://github.com/QwenLM/Qwen-VLA
- FLIGHT 2606.06836 (https://arxiv.org/abs/2606.06836)
